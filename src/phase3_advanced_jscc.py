import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

# ==========================================
# 1. SETUP AND DATA PREPARATION
# ==========================================
# Define robust absolute paths so the script works from any directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

print("Initializing Phase 3 Advanced JSCC...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We DO NOT upscale to 224x224 here. We want to transmit the raw 32x32 image to save bandwidth.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# 2. NOVELTY: FiLM (Feature-wise Linear Modulation)
# ==========================================
class FiLMLayer(nn.Module):
    def __init__(self, num_features):
        super(FiLMLayer, self).__init__()
        # Takes a 1D scalar (SNR) and outputs gamma and beta for each feature map
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_features * 2)
        )
        self.num_features = num_features

    def forward(self, x, snr):
        params = self.fc(snr) # shape: (batch_size, num_features * 2)
        gamma = params[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)
        beta = params[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * x + beta

# ==========================================
# 3. NOVELTY: CBAM (Convolutional Block Attention Module)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==========================================
# 4. ENCODER, CHANNEL, DECODER
# ==========================================
class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(SemanticEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1) # 16x16
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 8x8
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        
        self.film1 = FiLMLayer(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 4x4
        self.bn3 = nn.BatchNorm2d(256)
        self.cbam3 = CBAM(256)
        
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x, snr):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.film1(x, snr) # Dynamic Modulation!
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        
        x = self.flatten(x)
        latent = self.fc_latent(x)
        return latent

class AWGNChannel(nn.Module):
    def __init__(self):
        super(AWGNChannel, self).__init__()

    def forward(self, x, snr_db):
        signal_power = torch.mean(x**2, dim=1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

class SemanticDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(SemanticDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.film_dec = FiLMLayer(256)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 8x8
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 16x16
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1) # 32x32

    def forward(self, x, snr):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        
        x = self.film_dec(x, snr)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x) 
        return x

class AdvancedJSCC(nn.Module):
    def __init__(self, latent_dim=512):
        super(AdvancedJSCC, self).__init__()
        self.encoder = SemanticEncoder(latent_dim)
        self.channel = AWGNChannel()
        self.decoder = SemanticDecoder(latent_dim)
        
    def forward(self, x, snr_db):
        snr_tensor = snr_db.view(-1, 1).to(x.device)
        latent = self.encoder(x, snr_tensor)
        noisy_latent = self.channel(latent, snr_tensor)
        reconstructed = self.decoder(noisy_latent, snr_tensor)
        return reconstructed

# ==========================================
# 5. HYBRID LOSS & EVALUATOR SETUP
# ==========================================
# Load the Phase 2 ResNet18 as our frozen Task Evaluator
print("Loading frozen Task Evaluator (ResNet18)...")
task_model = models.resnet18(weights=None)
task_model.fc = nn.Linear(task_model.fc.in_features, 10)
task_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet18_cifar10_phase2.pth"), map_location=device))
task_model = task_model.to(device)
task_model.eval() # Freeze weights

# Upscaler for ResNet
upscale_for_resnet = transforms.Resize((224, 224))

mse_criterion = nn.MSELoss()
task_criterion = nn.CrossEntropyLoss()

def hybrid_loss(reconstructed, original, labels, alpha=0.5, beta=0.5):
    # 1. Pixel Loss
    loss_mse = mse_criterion(reconstructed, original)
    
    # 2. Task Loss (ResNet)
    # We must upscale the 32x32 reconstruction to 224x224 for ResNet
    recon_upscaled = upscale_for_resnet(reconstructed)
    task_preds = task_model(recon_upscaled)
    loss_task = task_criterion(task_preds, labels)
    
    # Total Hybrid Loss
    loss_total = (alpha * loss_mse) + (beta * loss_task)
    return loss_total, loss_mse, loss_task

# ==========================================
# 6. TRAINING LOOP
# ==========================================
jscc_model = AdvancedJSCC(latent_dim=512).to(device)
optimizer = optim.Adam(jscc_model.parameters(), lr=1e-3)

EPOCHS = 5

for epoch in range(EPOCHS):
    jscc_model.train()
    running_total = 0.0
    running_mse = 0.0
    running_task = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Random SNR between -5 and 20 dB for dynamic training
        random_snr = torch.empty(images.size(0)).uniform_(-5, 20).to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        reconstructed = jscc_model(images, random_snr)
        
        # Calculate Loss
        loss_total, loss_mse, loss_task = hybrid_loss(reconstructed, images, labels, alpha=0.5, beta=0.5)
        
        # Backprop
        loss_total.backward()
        optimizer.step()
        
        running_total += loss_total.item()
        running_mse += loss_mse.item()
        running_task += loss_task.item()
        
        progress_bar.set_postfix(
            Loss=running_total/(progress_bar.n+1),
            MSE=running_mse/(progress_bar.n+1),
            TaskL=running_task/(progress_bar.n+1)
        )
        
print("Training Complete! Ready for evaluation.")
