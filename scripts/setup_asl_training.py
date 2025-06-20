#!/usr/bin/env python3
"""
Quick Setup Script for ASL Training with MobileNetV2

This script sets up the ASL dataset and prepares the environment for 30 FPS training.

Usage:
    python scripts/setup_asl_training.py
    
Author: CV-ASL Team
Date: 2024
"""

import os
import sys
import subprocess
import shutil
import zipfile
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_kaggle_api():
    """Check if Kaggle API is available and configured"""
    try:
        import kaggle
        return True
    except ImportError:
        logger.warning("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.warning(f"Kaggle API not configured: {e}")
        logger.info("Please configure Kaggle API:")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Create API token (downloads kaggle.json)")
        logger.info("3. Place in ~/.kaggle/kaggle.json")
        logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_asl_dataset():
    """Download ASL dataset from Kaggle"""
    dataset_name = "ayuraj/asl-dataset"
    output_dir = Path("data/raw")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {dataset_name}...")
    
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=str(output_dir),
            unzip=True
        )
        
        logger.info("Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def setup_directory_structure():
    """Create necessary directories for training"""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "results",
        "scripts"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def check_dataset_structure():
    """Check and organize dataset structure"""
    raw_dir = Path("data/raw")
    
    # Look for the downloaded dataset
    dataset_folders = [
        raw_dir / "asl_dataset",
        raw_dir / "asl-dataset", 
        raw_dir
    ]
    
    # Find the actual dataset location
    dataset_path = None
    for folder in dataset_folders:
        if folder.exists():
            # Look for train folder or similar
            possible_train_dirs = [
                folder / "train",
                folder / "Train", 
                folder / "training",
                folder / "train_images"
            ]
            
            for train_dir in possible_train_dirs:
                if train_dir.exists() and any(train_dir.iterdir()):
                    dataset_path = folder
                    break
                    
            if dataset_path:
                break
    
    if not dataset_path:
        logger.error("Dataset not found! Please download manually:")
        logger.info("1. Go to https://www.kaggle.com/datasets/ayuraj/asl-dataset")
        logger.info("2. Download and extract to data/raw/")
        return None
    
    logger.info(f"Found dataset at: {dataset_path}")
    
    # Create unified structure if needed
    unified_dir = raw_dir / "asl_dataset" / "unified"
    if not unified_dir.exists():
        unified_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to find and organize training data
        for possible_train in ["train", "Train", "training", "train_images"]:
            source_train = dataset_path / possible_train
            if source_train.exists():
                target_train = unified_dir / "train_images"
                if not target_train.exists():
                    logger.info(f"Organizing training data: {source_train} -> {target_train}")
                    shutil.copytree(source_train, target_train)
                break
        
        # Try to find and organize test data
        for possible_test in ["test", "Test", "testing", "test_images"]:
            source_test = dataset_path / possible_test
            if source_test.exists():
                target_test = unified_dir / "test_images"
                if not target_test.exists():
                    logger.info(f"Organizing test data: {source_test} -> {target_test}")
                    shutil.copytree(source_test, target_test)
                break
    
    # Count classes and samples
    train_dir = unified_dir / "train_images"
    if train_dir.exists():
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        total_samples = sum(len(list((train_dir / cls).glob("*.jpg"))) + 
                           len(list((train_dir / cls).glob("*.png"))) 
                           for cls in classes)
        
        logger.info(f"Dataset organized:")
        logger.info(f"  Classes: {len(classes)}")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Training path: {train_dir}")
        
        return str(train_dir)
    
    return None

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing training dependencies...")
    
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "transformers>=4.21.0",
        "mediapipe>=0.10.0",
        "timm>=0.9.0",
        "tensorboard>=2.13.0",
        "seaborn>=0.12.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0"
    ]
    
    try:
        for req in requirements:
            logger.info(f"Installing {req}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", req
            ], check=True, capture_output=True)
        
        logger.info("Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def create_training_config():
    """Create optimized training configuration for 30 FPS"""
    config = {
        "model_configs": {
            "mobilenetv2": {
                "batch_size": 64,
                "learning_rate": 0.002,
                "num_epochs": 25,
                "input_size": 224,
                "width_mult": 1.0,
                "weight_decay": 1e-4,
                "scheduler_step": 8,
                "num_workers": 4
            },
            "mobilenetv2_lite": {
                "batch_size": 80,
                "learning_rate": 0.003,
                "num_epochs": 20,
                "input_size": 192,
                "width_mult": 0.5,
                "weight_decay": 1e-4,
                "scheduler_step": 6,
                "num_workers": 4
            },
            "mediapipe": {
                "batch_size": 128,
                "learning_rate": 0.005,
                "num_epochs": 15,
                "weight_decay": 1e-3,
                "scheduler_step": 5,
                "num_workers": 2
            }
        },
        "target_fps": 30,
        "confidence_threshold": 0.7
    }
    
    import json
    config_path = Path("configs/training_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to: {config_path}")
    return config_path

def create_quick_start_script():
    """Create a quick start training script"""
    script_content = '''#!/usr/bin/env python3
"""
Quick Start ASL Training - 30 FPS Optimized

Run this script to start training MobileNetV2 for real-time ASL recognition.
"""

import sys
import os
sys.path.append('src')

from asl_cam.train import ASLTrainer, compare_models
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load optimized configuration  
    with open('configs/training_config.json', 'r') as f:
        configs = json.load(f)
    
    # Use MobileNetV2 configuration
    config = configs['model_configs']['mobilenetv2']
    
    # Dataset path
    data_dir = "data/raw/asl_dataset/unified/train_images"
    
    # Verify dataset exists
    if not os.path.exists(data_dir):
        logger.error(f"Dataset not found at: {data_dir}")
        logger.info("Please run: python scripts/setup_asl_training.py")
        return
    
    logger.info("Starting MobileNetV2 ASL training for 30 FPS...")
    logger.info(f"Dataset: {data_dir}")
    logger.info(f"Configuration: {config}")
    
    # Create trainer and start training
    trainer = ASLTrainer(config)
    results = trainer.train(data_dir, model_type="mobilenetv2")
    
    logger.info(f"Training completed!")
    logger.info(f"Best accuracy: {results['best_accuracy']:.2f}%")
    logger.info(f"Model speed: {results['benchmark']['fps']:.1f} FPS")
    
    if results['benchmark']['fps'] >= 30:
        logger.info("✅ Model meets 30 FPS target!")
    else:
        logger.warning("⚠️ Model below 30 FPS target")
    
    logger.info("Model saved in: models/best_mobilenetv2_model.pth")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/quick_train.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"Quick start script created: {script_path}")
    return script_path

def main():
    """Main setup function"""
    logger.info("🚀 Setting up ASL Training Environment for 30 FPS")
    logger.info("=" * 60)
    
    # Step 1: Setup directories
    logger.info("1. Setting up directory structure...")
    setup_directory_structure()
    
    # Step 2: Check dependencies 
    logger.info("2. Installing dependencies...")
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Step 3: Download dataset
    logger.info("3. Checking dataset...")
    if check_kaggle_api():
        if not download_asl_dataset():
            logger.warning("Manual dataset download required")
    else:
        logger.info("Please download dataset manually:")
        logger.info("https://www.kaggle.com/datasets/ayuraj/asl-dataset")
        logger.info("Extract to: data/raw/")
    
    # Step 4: Organize dataset
    logger.info("4. Organizing dataset structure...")
    dataset_path = check_dataset_structure()
    
    if not dataset_path:
        logger.error("Dataset setup incomplete")
        return False
    
    # Step 5: Create configuration
    logger.info("5. Creating training configuration...")
    config_path = create_training_config()
    
    # Step 6: Create quick start script
    logger.info("6. Creating quick start script...")
    script_path = create_quick_start_script()
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ ASL Training Environment Setup Complete!")
    logger.info("=" * 60)
    logger.info(f"Dataset ready at: {dataset_path}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Quick start: {script_path}")
    logger.info("\nNext steps:")
    logger.info("1. Start training: python scripts/quick_train.py")
    logger.info("2. Or compare models: python -m src.asl_cam.train")
    logger.info("3. Target: 30 FPS real-time performance 🎯")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 