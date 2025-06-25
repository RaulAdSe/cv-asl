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
# MediaPipe import (optional for landmark-based training)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import logging
from typing import List, Tuple, Dict, Optional
import time
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Removed complex TrainingMonitor class for simplicity

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
        
        # Initialize MediaPipe if needed and available
        if self.use_mediapipe:
            if not MEDIAPIPE_AVAILABLE:
                raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
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
        
        logger.info(f"📁 Dataset loaded: {len(self.samples):,} samples, {len(self.classes)} classes")
        
        # Print class distribution
        class_counts = {}
        for sample in self.samples:
            class_counts[sample['label']] = class_counts.get(sample['label'], 0) + 1
        
        logger.info("📊 Class distribution:")
        for cls, count in sorted(class_counts.items()):
            logger.info(f"   {cls}: {count:,} samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load samples from collected data or Kaggle dataset format"""
        samples = []
        
        # Check if it's the collected data format (data/raw with hand_*.jpg files)
        raw_images = list(self.data_dir.glob("hand_*.jpg"))
        if raw_images:
            logger.info("📁 Loading from collected hand detection data")
            # For collected data, we'll need manual labeling or use filename patterns
            # For now, create a single 'hand' class
            for img_path in raw_images:
                samples.append({
                    'path': img_path,
                    'label': 'hand'
                })
        else:
            # Standard dataset format: class_dir/images
            logger.info("📁 Loading from standard dataset format")
            for class_dir in self.data_dir.glob("*"):
                if class_dir.is_dir() and class_dir.name not in ['.DS_Store', '__pycache__']:
                    class_name = class_dir.name
                    for img_path in class_dir.glob("*.jpg"):
                        samples.append({
                            'path': img_path,
                            'label': class_name
                        })
                    for img_path in class_dir.glob("*.jpeg"):
                        samples.append({
                            'path': img_path,
                            'label': class_name
                        })
                    for img_path in class_dir.glob("*.png"):
                        samples.append({
                            'path': img_path,
                            'label': class_name
                        })
        
        if not samples:
            raise ValueError(f"No samples found in {self.data_dir}. Collect data first with: python -m src.asl_cam.collect")
        
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
        self.backbone = mobilenet_v2(pretrained=False, width_mult=width_mult)
        
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
        self.backbone = mobilenet_v2(pretrained=False, width_mult=0.5)
        
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
    """Main training class for ASL models with advanced monitoring"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Device: {self.device}")
        
        # Setup TensorBoard
        self.setup_tensorboard()
        
        # Optimized transforms for training
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
    
    def setup_tensorboard(self):
        """Setup TensorBoard logging"""
        import datetime
        log_dir = Path("logs") / f"asl_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        logger.info(f"📊 TensorBoard logs: {log_dir}")
        logger.info(f"   Run: tensorboard --logdir={log_dir}")
    
    def estimate_training_time(self, dataset_size: int, batch_size: int, num_epochs: int) -> str:
        """Estimate total training time based on dataset size and hardware"""
        # Rough estimates based on typical performance
        samples_per_second = {
            'cuda': 1000,  # With GPU
            'cpu': 100     # CPU only
        }
        
        sps = samples_per_second.get(self.device.type, 100)
        seconds_per_epoch = dataset_size / sps
        total_seconds = seconds_per_epoch * num_epochs
        
        return str(datetime.timedelta(seconds=int(total_seconds)))
    
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
    
    def train(self, data_dir: str, model_type: str = "mobilenetv2") -> Dict:
        """Enhanced training function with comprehensive monitoring"""
        logger.info(f"🚀 Starting ASL training with {model_type}")
        logger.info(f"💾 Data directory: {data_dir}")
        
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
        
        logger.info(f"📊 Dataset split: {train_size:,} train, {val_size:,} validation")
        
        # Estimate training time
        batch_size = self.config.get('batch_size', 64)
        num_epochs = self.config.get('num_epochs', 25)
        estimated_time = self.estimate_training_time(train_size, batch_size, num_epochs)
        
        logger.info(f"⏱️  Estimated training time: {estimated_time}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
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
        logger.info(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Initialize simple progress tracking
        logger.info(f"📊 Training will process {train_size:,} samples in {train_size//batch_size} batches per epoch")
        
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
        
        # Start training
        start_time = time.time()
        best_accuracy = 0.0
        training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            logger.info(f"\n📈 Epoch {epoch+1}/{num_epochs}")
            
            # Train epoch
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_accuracy = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Update history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"⏱️  Epoch time: {epoch_time:.1f}s | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                
                # Benchmark speed for final model
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
                    'total_params': total_params,
                    'training_history': training_history
                }, f'models/best_{model_type}_model.pth')
                
                logger.info(f"💾 New best model saved! Inference: {benchmark['fps']:.1f} FPS")
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Training completed in {total_time/60:.1f} minutes")
        
        # Close TensorBoard writer
        self.writer.close()
        
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
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        # Simple progress with tqdm
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/(batch_idx+1):.4f}'
            })
        
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

def main():
    """Main training script with different modes."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ASL Model Training Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        default="custom", 
        choices=["custom", "kaggle_abc", "compare"], 
        help=(
            "Select training mode:\\n"
            "  - custom: Train on a specified data directory (default).\\n"
            "  - kaggle_abc: Train specifically on the Kaggle A,B,C dataset.\\n"
            "  - compare: Train and compare multiple model architectures."
        )
    )
    parser.add_argument("--data-dir", default="data/raw", help="Data directory (for 'custom' mode).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "mobilenetv2_lite", "mediapipe"], help="Model architecture for 'custom' mode.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    
    args = parser.parse_args()
    
    # --- Base Config ---
    config = {
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'num_workers': min(os.cpu_count(), 4) # Safe default for num_workers
    }
    
    # --- Mode-specific Logic ---
    if args.mode == 'kaggle_abc':
        logger.info("🚀 Training Mode: Kaggle A, B, C")
        data_dir = "data/raw/kaggle_asl/train"
        model_type = "mobilenetv2"
        
        # Verify dataset
        if not Path(data_dir).exists():
            logger.error(f"❌ Kaggle ABC dataset not found at {data_dir}. Please run the download.")
            return

        config.update({
            'batch_size': 32,
            'weight_decay': 1e-4,
            'scheduler_step': 7,
            'input_size': 224,
            'width_mult': 1.0
        })
        
        trainer = ASLTrainer(config)
        trainer.train(data_dir, model_type)

    elif args.mode == 'custom':
        logger.info(f"🚀 Training Mode: Custom Dataset")
        data_dir = args.data_dir
        model_type = args.model
        
        config.update({
            'batch_size': 32,
            'weight_decay': 1e-4,
            'scheduler_step': 5,
            'input_size': 224 if model_type == "mobilenetv2" else 192,
        })
        
        trainer = ASLTrainer(config)
        trainer.train(data_dir, model_type)
        
    elif args.mode == 'compare':
        logger.info("🚀 Training Mode: Compare Models on Kaggle ABC dataset")
        data_dir = "data/raw/kaggle_asl/train"
        
        # Verify dataset
        if not Path(data_dir).exists():
            logger.error(f"❌ Kaggle ABC dataset not found at {data_dir} for comparison. Please run the download.")
            return
            
        compare_models(data_dir, config)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        main()
        logger.info("🎉 Training script finished.")
    except KeyboardInterrupt:
        logger.info("🎉 Training interrupted by user.")
    except Exception as e:
        logger.error(f"❌ An error occurred during training: {e}", exc_info=True)
