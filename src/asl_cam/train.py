"""
ASL Hand Sign Classification Training Module

This module implements multiple lightweight model architectures for real-time
ASL hand sign classification, with a focus on efficiency and accuracy.

Author: CV-ASL Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mediapipe as mp
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, mobilenet_v3_small
import logging
from typing import List, Tuple, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    """Dataset class for ASL hand signs with support for multiple data formats"""
    
    def __init__(self, data_dir: str, sequence_length: int = 16, transform=None, 
                 use_mediapipe: bool = False):
        """
        Initialize ASL dataset
        
        Args:
            data_dir: Path to directory containing ASL data
            sequence_length: Number of frames per sequence for video models
            transform: Data augmentation transforms
            use_mediapipe: Whether to extract MediaPipe hand landmarks
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.use_mediapipe = use_mediapipe
        
        # Initialize MediaPipe if needed
        if self.use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
        
        # Load dataset
        self.samples = self._load_samples()
        self.classes = sorted(list(set([sample['label'] for sample in self.samples])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logger.info(f"Loaded {len(self.samples)} samples with {len(self.classes)} classes")
    
    def _load_samples(self) -> List[Dict]:
        """Load samples from Kaggle ASL dataset format"""
        samples = []
        
        # Expect structure: data_dir/train_images/A/image1.jpg, etc.
        for class_dir in self.data_dir.glob("*"):
            if class_dir.is_dir() and class_dir.name != '.DS_Store':
                class_name = class_dir.name
                for img_path in class_dir.glob("*.jpg"):
                    samples.append({
                        'path': img_path,
                        'label': class_name
                    })
        
        return samples
    
    def _extract_mediapipe_features(self, image: np.ndarray) -> np.ndarray:
        """Extract MediaPipe hand landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        features = np.zeros((21 * 2 * 2))  # 21 landmarks × 2 coords × 2 hands
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                    base_idx = hand_idx * 42 + landmark_idx * 2
                    features[base_idx] = landmark.x
                    features[base_idx + 1] = landmark.y
        
        return features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_mediapipe:
            # Extract MediaPipe features
            features = self._extract_mediapipe_features(image)
            return torch.FloatTensor(features), self.class_to_idx[sample['label']]
        else:
            # Standard image processing
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            return image, self.class_to_idx[sample['label']]

class EfficientNetLSTM(nn.Module):
    """EfficientNet backbone with LSTM for temporal modeling"""
    
    def __init__(self, num_classes: int, sequence_length: int = 16, 
                 hidden_size: int = 128, num_layers: int = 2):
        super(EfficientNetLSTM, self).__init__()
        
        # EfficientNet backbone (lightweight)
        self.backbone = efficientnet_b0(pretrained=True)
        # Remove final classifier
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame through backbone
        x = x.view(-1, *x.shape[2:])  # Flatten batch and sequence dimensions
        features = self.backbone(x)  # Extract features
        features = features.view(batch_size, seq_len, -1)  # Reshape back
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Use last timestep output
        output = self.classifier(lstm_out[:, -1, :])
        
        return output

class MediaPipeClassifier(nn.Module):
    """Lightweight classifier using MediaPipe hand landmarks"""
    
    def __init__(self, num_classes: int, input_dim: int = 84):
        super(MediaPipeClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class ASLTrainer:
    """Main training class for ASL models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data transforms
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Create model based on specified type"""
        
        if model_type == "efficientnet_lstm":
            model = EfficientNetLSTM(
                num_classes=num_classes,
                sequence_length=self.config.get('sequence_length', 16)
            )
        
        elif model_type == "mediapipe":
            model = MediaPipeClassifier(num_classes=num_classes)
        
        elif model_type == "mobilenet":
            model = mobilenet_v3_small(pretrained=True)
            # Modify final layer
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """Validate model"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, data_dir: str, model_type: str = "efficientnet_lstm") -> Dict:
        """Main training function"""
        logger.info(f"Starting training with model type: {model_type}")
        
        # Create datasets
        use_mediapipe = (model_type == "mediapipe")
        
        full_dataset = ASLDataset(
            data_dir=data_dir,
            transform=self.train_transforms if not use_mediapipe else None,
            use_mediapipe=use_mediapipe
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4
        )
        
        # Create model
        num_classes = len(full_dataset.classes)
        model = self.create_model(model_type, num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.config.get('scheduler_step', 10), 
            gamma=0.5
        )
        
        # Training loop
        best_accuracy = 0.0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        num_epochs = self.config.get('num_epochs', 50)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_accuracy = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_accuracy:.2f}%'
            )
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'classes': full_dataset.classes,
                    'model_type': model_type
                }, f'models/best_{model_type}_model.pth')
                
                logger.info(f'New best model saved with accuracy: {val_accuracy:.2f}%')
        
        return {
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'model_type': model_type,
            'classes': full_dataset.classes
        }

def compare_models(data_dir: str, config: Dict) -> Dict:
    """Compare different model architectures"""
    results = {}
    models_to_test = ["efficientnet_lstm", "mediapipe", "mobilenet"]
    
    for model_type in models_to_test:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type}")
        logger.info(f"{'='*50}")
        
        trainer = ASLTrainer(config)
        result = trainer.train(data_dir, model_type)
        results[model_type] = result
        
        logger.info(f"{model_type} - Best Accuracy: {result['best_accuracy']:.2f}%")
    
    return results

if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 30,  # Start with fewer epochs for testing
        'weight_decay': 1e-4,
        'scheduler_step': 10,
        'sequence_length': 16
    }
    
    # Path to your ASL dataset (modify this to match your setup)
    data_dir = "data/raw/asl_dataset/train_images"
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Compare different models
    logger.info("Starting model comparison...")
    results = compare_models(data_dir, config)
    
    # Print final comparison
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS COMPARISON")
    logger.info("="*60)
    
    for model_type, result in results.items():
        logger.info(f"{model_type:20} - Accuracy: {result['best_accuracy']:6.2f}%")
    
    # Save results
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to 'model_comparison_results.json'")
