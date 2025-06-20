"""
ASL Dataset Setup and Management

This module handles downloading and setting up the ASL dataset from Kaggle
for training deep learning models.

Author: CV-ASL Team
Date: 2024
"""

import os
import sys
import zipfile
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLDatasetManager:
    """Manager for ASL dataset download and setup"""
    
    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset information
        self.kaggle_dataset = "ayuraj/asl-dataset"
        self.dataset_dir = self.base_dir / "asl_dataset"
        
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured"""
        try:
            import kaggle
            return True
        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
            return False
        except OSError as e:
            if "credentials" in str(e).lower():
                logger.error(
                    "Kaggle API credentials not found. Please:\n"
                    "1. Go to https://www.kaggle.com/account\n"
                    "2. Create API token (kaggle.json)\n"
                    "3. Place it in ~/.kaggle/kaggle.json\n"
                    "4. Run: chmod 600 ~/.kaggle/kaggle.json"
                )
                return False
            raise e
    
    def download_kaggle_dataset(self) -> bool:
        """Download ASL dataset from Kaggle"""
        if not self.check_kaggle_setup():
            return False
        
        try:
            import kaggle
            
            logger.info(f"Downloading {self.kaggle_dataset} to {self.dataset_dir}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=str(self.dataset_dir),
                unzip=True
            )
            
            logger.info("Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def download_manual(self) -> None:
        """Provide manual download instructions"""
        logger.info(
            "\n" + "="*60 + "\n"
            "MANUAL DOWNLOAD INSTRUCTIONS\n"
            "="*60 + "\n"
            f"1. Go to: https://www.kaggle.com/datasets/{self.kaggle_dataset}\n"
            f"2. Download the dataset ZIP file\n"
            f"3. Extract to: {self.dataset_dir}\n"
            f"4. Run this script again\n"
            "="*60
        )
    
    def verify_dataset_structure(self) -> bool:
        """Verify the dataset has the expected structure"""
        expected_structure = [
            "Train",
            "Test"
        ]
        
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Check for main directories
        for dir_name in expected_structure:
            dir_path = self.dataset_dir / dir_name
            if not dir_path.exists():
                logger.error(f"Expected directory not found: {dir_path}")
                return False
        
        # Count classes and samples
        train_dir = self.dataset_dir / "Train"
        test_dir = self.dataset_dir / "Test"
        
        train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        test_classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Train classes: {len(train_classes)}")
        logger.info(f"Test classes: {len(test_classes)}")
        
        # Count samples per class
        total_train_samples = 0
        total_test_samples = 0
        
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                samples = len(list(class_dir.glob("*.jpg")))
                total_train_samples += samples
                
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                samples = len(list(class_dir.glob("*.jpg")))
                total_test_samples += samples
        
        logger.info(f"Total train samples: {total_train_samples}")
        logger.info(f"Total test samples: {total_test_samples}")
        
        return True
    
    def create_unified_structure(self) -> bool:
        """Create a unified train/val structure for easier training"""
        if not self.verify_dataset_structure():
            return False
        
        # Create unified directory
        unified_dir = self.dataset_dir / "unified"
        unified_dir.mkdir(exist_ok=True)
        
        train_unified = unified_dir / "train_images"
        test_unified = unified_dir / "test_images"
        
        train_unified.mkdir(exist_ok=True)
        test_unified.mkdir(exist_ok=True)
        
        # Copy train data
        train_source = self.dataset_dir / "Train"
        for class_dir in train_source.iterdir():
            if class_dir.is_dir():
                dest_class_dir = train_unified / class_dir.name
                if not dest_class_dir.exists():
                    shutil.copytree(class_dir, dest_class_dir)
        
        # Copy test data
        test_source = self.dataset_dir / "Test"
        for class_dir in test_source.iterdir():
            if class_dir.is_dir():
                dest_class_dir = test_unified / class_dir.name
                if not dest_class_dir.exists():
                    shutil.copytree(class_dir, dest_class_dir)
        
        logger.info("Unified structure created successfully!")
        return True
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information"""
        info = {
            'dataset_path': str(self.dataset_dir),
            'unified_path': str(self.dataset_dir / "unified"),
            'train_path': str(self.dataset_dir / "unified" / "train_images"),
            'test_path': str(self.dataset_dir / "unified" / "test_images"),
            'classes': [],
            'class_counts': {},
            'total_samples': 0
        }
        
        train_dir = self.dataset_dir / "unified" / "train_images"
        
        if train_dir.exists():
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    sample_count = len(list(class_dir.glob("*.jpg")))
                    
                    info['classes'].append(class_name)
                    info['class_counts'][class_name] = sample_count
                    info['total_samples'] += sample_count
            
            info['classes'] = sorted(info['classes'])
        
        return info
    
    def setup(self, force_download: bool = False) -> bool:
        """Complete dataset setup process"""
        logger.info("Starting ASL dataset setup...")
        
        # Check if dataset already exists
        if self.dataset_dir.exists() and not force_download:
            logger.info("Dataset directory already exists")
            if self.verify_dataset_structure():
                logger.info("Dataset structure verified")
                self.create_unified_structure()
                return True
        
        # Try to download from Kaggle
        if not self.download_kaggle_dataset():
            logger.warning("Kaggle download failed, providing manual instructions")
            self.download_manual()
            return False
        
        # Verify and create unified structure
        if self.verify_dataset_structure():
            self.create_unified_structure()
            return True
        
        return False

def main():
    """Main function for dataset setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup ASL dataset for training")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download even if dataset exists")
    parser.add_argument("--info", action="store_true",
                       help="Display dataset information")
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    dataset_manager = ASLDatasetManager()
    
    if args.info:
        # Display dataset information
        info = dataset_manager.get_dataset_info()
        
        print("\n" + "="*50)
        print("ASL DATASET INFORMATION")
        print("="*50)
        print(f"Dataset path: {info['dataset_path']}")
        print(f"Train path: {info['train_path']}")
        print(f"Number of classes: {len(info['classes'])}")
        print(f"Total samples: {info['total_samples']}")
        print("\nClasses:")
        for class_name in info['classes']:
            count = info['class_counts'][class_name]
            print(f"  {class_name}: {count} samples")
        print("="*50)
        
    else:
        # Setup dataset
        success = dataset_manager.setup(force_download=args.force)
        
        if success:
            logger.info("Dataset setup completed successfully!")
            
            # Display dataset info
            info = dataset_manager.get_dataset_info()
            logger.info(f"Ready to train with {len(info['classes'])} classes")
            logger.info(f"Train data path: {info['train_path']}")
            
        else:
            logger.error("Dataset setup failed")
            sys.exit(1)

if __name__ == "__main__":
    main() 