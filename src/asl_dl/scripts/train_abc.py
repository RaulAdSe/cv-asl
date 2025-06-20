#!/usr/bin/env python3
"""
ASL A-B-C Training Script

Downloads the Kaggle ASL dataset and trains a model with only A, B, C letters.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asl_dl.data.kaggle_downloader import download_abc_dataset
from asl_dl.models.mobilenet import MobileNetV2ASL
from asl_dl.data.dataset import ASLDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 ASL A-B-C Training")
    logger.info("=" * 40)
    
    # 1. Download dataset (A, B, C only)
    logger.info("📥 Downloading Kaggle ASL dataset (A, B, C)...")
    try:
        dataset_path = download_abc_dataset()
        logger.info(f"✅ Dataset ready: {dataset_path}")
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1
    
    # 2. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Device: {device}")
    
    # 3. Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Load dataset
    logger.info("📊 Loading dataset...")
    full_dataset = ASLDataset(dataset_path, transform=transform)
    
    # 5. Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"📊 Dataset split: {train_size:,} train, {val_size:,} validation")
    
    # 6. Data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. Model
    num_classes = len(full_dataset.classes)
    model = MobileNetV2ASL(num_classes=num_classes, pretrained=False).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model parameters: {total_params:,}")
    
    # 8. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    logger.info(f"🏃 Starting training for {num_epochs} epochs...")
    
    # 9. Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        # Print epoch results
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
    
    # 10. Save model
    total_time = time.time() - start_time
    logger.info(f"⏱️ Training completed in {total_time:.1f} seconds")
    
    model_path = Path("models") / "asl_abc_model.pth"
    model_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': full_dataset.classes,
        'num_classes': num_classes,
        'final_val_acc': val_acc
    }, model_path)
    
    logger.info(f"💾 Model saved: {model_path}")
    logger.info(f"✅ Final validation accuracy: {val_acc:.2f}%")
    
    return 0

if __name__ == "__main__":
    exit(main()) 