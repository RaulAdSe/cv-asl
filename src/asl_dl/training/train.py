"""
ASL Hand Sign Classification Training Module

This module implements MobileNetV2-based lightweight model architectures for real-time
ASL hand sign classification, optimized for 30 FPS performance.

Author: CV-ASL Team
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import json
import logging
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    """Dataset class for ASL images"""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing class folders with images
            transform: Torchvision transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        self.classes = []
        
        self._load_images()
        
    def _load_images(self):
        """Load all images and create class mappings"""
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()  # Ensure consistent ordering
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            
            # Get all image files in class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in class_dir.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            for image_file in image_files:
                self.images.append(str(image_file))
                self.labels.append(class_idx)
        
        logger.info(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MobileNetV2ASL(nn.Module):
    """MobileNetV2-based ASL classifier"""
    
    def __init__(self, num_classes: int, pretrained: bool = True, 
                 dropout_rate: float = 0.2, input_size: int = 224,
                 width_mult: float = 1.0):
        super(MobileNetV2ASL, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.input_size = input_size
        
    def forward(self, x):
        return self.backbone(x)

class MobileNetV2Lite(nn.Module):
    """Lightweight MobileNetV2 for faster inference"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(MobileNetV2Lite, self).__init__()
        
        # Use MobileNetV2 with reduced width multiplier
        self.backbone = models.mobilenet_v2(pretrained=pretrained, width_mult=0.5)
        
        # Replace classifier  
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ASLTrainer:
    """Training manager for ASL models"""
    
    def __init__(self, device: str = "auto"):
        """Initialize trainer"""
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Training transforms
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Create model based on type"""
        if model_type == "mobilenetv2":
            model = MobileNetV2ASL(num_classes=num_classes)
        elif model_type == "mobilenetv2_lite":
            model = MobileNetV2Lite(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model.to(self.device)
    
    def create_datasets(self, data_dir: str, model_type: str, 
                       train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets"""
        
        # Create full dataset
        full_dataset = ASLDataset(
            data_dir=data_dir,
            transform=self.train_transforms
        )
        
        # Split into train and validation
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply different transforms to validation set
        val_dataset.dataset = ASLDataset(data_dir=data_dir, transform=self.val_transforms)
        
        return train_dataset, val_dataset
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """Validate model"""
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        return val_loss, accuracy
    
    def train(self, data_dir: str, model_type: str = "mobilenetv2",
             epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001,
             save_dir: str = "models", early_stopping_patience: int = 5) -> Dict:
        """
        Train ASL model
        
        Args:
            data_dir: Directory containing training data
            model_type: Type of model to train
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save trained model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training with {model_type}")
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(data_dir, model_type)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=2)
        
        # Get number of classes from dataset
        num_classes = len(train_dataset.dataset.classes)
        classes = train_dataset.dataset.classes
        
        # Create model
        model = self.create_model(model_type, num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Early stopping and best model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                save_path = Path(save_dir)
                save_path.mkdir(exist_ok=True)
                
                model_filename = f"asl_{model_type}_best.pth"
                model_path = save_path / model_filename
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': num_classes,
                    'classes': classes,
                    'model_type': model_type,
                    'history': history,
                    'model_config': {
                        'input_size': 224,
                        'dropout_rate': 0.2 if model_type == 'mobilenetv2' else 0.1,
                    }
                }, model_path)
                
                logger.info(f"  💾 Saved best model to {model_path}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Training summary
        results = {
            'model_type': model_type,
            'best_val_acc': best_val_acc,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'final_model_path': str(model_path),
            'classes': classes,
            'num_classes': num_classes,
            'history': history
        }
        
        logger.info(f"\n🎉 Training completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"Total training time: {total_time/60:.2f} minutes")
        logger.info(f"Model saved to: {model_path}")
        
        return results

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ASL Recognition Model")
    parser.add_argument("--data", required=True, help="Path to training data directory")
    parser.add_argument("--model", default="mobilenetv2", 
                       choices=["mobilenetv2", "mobilenetv2_lite"],
                       help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-dir", default="models", help="Directory to save models")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ASLTrainer(device=args.device)
    
    # Train model
    results = trainer.train(
        data_dir=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    # Save training results
    results_path = Path(args.save_dir) / f"training_results_{args.model}.json"
    with open(results_path, 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, list, dict, bool)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Training results saved to: {results_path}")

if __name__ == "__main__":
    main()
