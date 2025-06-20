"""
ASL Hand Sign Classification Training Module

This module implements MobileNetV2-based lightweight model architectures for real-time
ASL hand sign classification, optimized for 30 FPS performance.

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
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import logging
from typing import List, Tuple, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    """Dataset class for ASL hand signs with support for multiple data formats"""
    
    def __init__(self, data_dir: str, transform=None, use_mediapipe: bool = False):
        """
        Initialize ASL dataset
        
        Args:
            data_dir: Path to directory containing ASL data
            transform: Data augmentation transforms
            use_mediapipe: Whether to extract MediaPipe hand landmarks
        """
        self.data_dir = Path(data_dir)
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

class MobileNetV2ASL(nn.Module):
    """MobileNetV2 backbone optimized for 30 FPS ASL classification"""
    
    def __init__(self, num_classes: int, input_size: int = 224, width_mult: float = 1.0):
        super(MobileNetV2ASL, self).__init__()
        
        # MobileNetV2 backbone with adjustable width multiplier for speed
        self.backbone = mobilenet_v2(pretrained=True, width_mult=width_mult)
        
        # Remove final classifier
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier optimized for ASL
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features with MobileNetV2
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output

class MobileNetV2Lite(nn.Module):
    """Ultra-lightweight version for maximum speed (targeting 60+ FPS)"""
    
    def __init__(self, num_classes: int):
        super(MobileNetV2Lite, self).__init__()
        
        # Use width_mult=0.5 for 2x speedup
        self.backbone = mobilenet_v2(pretrained=True, width_mult=0.5)
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Minimal classifier for speed
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

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
    """Main training class for ASL models optimized for 30 FPS"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Optimized transforms for 30 FPS (smaller input size for speed)
        input_size = config.get('input_size', 224)
        
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Create model based on specified type"""
        
        if model_type == "mobilenetv2":
            model = MobileNetV2ASL(
                num_classes=num_classes,
                input_size=self.config.get('input_size', 224),
                width_mult=self.config.get('width_mult', 1.0)
            )
        
        elif model_type == "mobilenetv2_lite":
            model = MobileNetV2Lite(num_classes=num_classes)
        
        elif model_type == "mediapipe":
            model = MediaPipeClassifier(num_classes=num_classes)
        
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
    
    def benchmark_model(self, model: nn.Module, input_size: Tuple[int, int, int], 
                       num_runs: int = 100) -> Dict:
        """Benchmark model inference speed"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        fps = 1000 / avg_time
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': fps,
            'times': times
        }
    
    def train(self, data_dir: str, model_type: str = "mobilenetv2") -> Dict:
        """Main training function"""
        logger.info(f"Starting training with model type: {model_type}")
        logger.info(f"Target: 30 FPS real-time performance")
        
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
        
        # Create data loaders with optimized settings for speed
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 64),  # Larger batch for efficiency
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 64),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Create model
        num_classes = len(full_dataset.classes)
        model = self.create_model(model_type, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
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
        
        num_epochs = self.config.get('num_epochs', 25)
        
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
                
                # Benchmark speed
                input_size = (3, self.config.get('input_size', 224), self.config.get('input_size', 224))
                benchmark = self.benchmark_model(model, input_size)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'classes': full_dataset.classes,
                    'model_type': model_type,
                    'config': self.config,
                    'benchmark': benchmark,
                    'total_params': total_params
                }, f'models/best_{model_type}_model.pth')
                
                logger.info(f'New best model saved!')
                logger.info(f'Accuracy: {val_accuracy:.2f}%')
                logger.info(f'Inference speed: {benchmark["fps"]:.1f} FPS')
                logger.info(f'Inference time: {benchmark["avg_inference_time_ms"]:.1f}ms')
        
        # Final benchmark
        input_size = (3, self.config.get('input_size', 224), self.config.get('input_size', 224))
        final_benchmark = self.benchmark_model(model, input_size)
        
        return {
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'model_type': model_type,
            'classes': full_dataset.classes,
            'benchmark': final_benchmark,
            'total_params': total_params
        }

def compare_models(data_dir: str, config: Dict) -> Dict:
    """Compare different model architectures optimized for 30 FPS"""
    results = {}
    
    # Models to test, ordered by expected speed
    models_to_test = [
        ("mediapipe", "MediaPipe Features (Ultra Fast)"),
        ("mobilenetv2_lite", "MobileNetV2 Lite (Fast)"),
        ("mobilenetv2", "MobileNetV2 (Balanced)")
    ]
    
    for model_type, description in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {description}")
        logger.info(f"Target: 30+ FPS real-time performance")
        logger.info(f"{'='*60}")
        
        # Adjust config for each model type
        model_config = config.copy()
        
        if model_type == "mediapipe":
            model_config['batch_size'] = 128  # Faster training for simple model
            model_config['num_epochs'] = 15
        elif model_type == "mobilenetv2_lite":
            model_config['input_size'] = 192  # Smaller input for speed
            model_config['batch_size'] = 80
        else:  # mobilenetv2
            model_config['input_size'] = 224
            model_config['width_mult'] = 1.0
        
        trainer = ASLTrainer(model_config)
        result = trainer.train(data_dir, model_type)
        results[model_type] = result
        
        logger.info(f"\n{description} Results:")
        logger.info(f"Best Accuracy: {result['best_accuracy']:.2f}%")
        logger.info(f"Inference Speed: {result['benchmark']['fps']:.1f} FPS")
        logger.info(f"Parameters: {result['total_params']:,}")
        
        # Check if meets 30 FPS target
        if result['benchmark']['fps'] >= 30:
            logger.info("✅ MEETS 30 FPS TARGET!")
        else:
            logger.info("⚠️  Below 30 FPS target")
    
    return results

if __name__ == "__main__":
    # Optimized configuration for 30 FPS target
    config = {
        'batch_size': 64,           # Larger batch for efficiency
        'learning_rate': 0.002,     # Slightly higher LR for faster convergence
        'num_epochs': 25,           # Reasonable number for good results
        'weight_decay': 1e-4,
        'scheduler_step': 8,
        'input_size': 224,          # Standard size, can be reduced for speed
        'width_mult': 1.0,          # Full width for accuracy
        'num_workers': 4            # Parallel data loading
    }
    
    # Path to your ASL dataset
    data_dir = "data/raw/asl_dataset/unified/train_images"
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Compare different models optimized for 30 FPS
    logger.info("Starting ASL model training - Target: 30 FPS real-time")
    results = compare_models(data_dir, config)
    
    # Print final comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS COMPARISON - 30 FPS TARGET")
    logger.info("="*80)
    
    for model_type, result in results.items():
        fps = result['benchmark']['fps']
        accuracy = result['best_accuracy']
        params = result['total_params']
        status = "✅ MEETS TARGET" if fps >= 30 else "⚠️  BELOW TARGET"
        
        logger.info(f"{model_type:20} - Acc: {accuracy:6.2f}% | FPS: {fps:6.1f} | Params: {params:8,} | {status}")
    
    # Save results
    with open('model_comparison_30fps.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to 'model_comparison_30fps.json'")
    logger.info("Ready for 30 FPS real-time ASL recognition! 🚀")
