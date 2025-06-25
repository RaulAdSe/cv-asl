"""
ASL Dataset class for loading and preprocessing ASL hand sign images.
"""

import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    """Clean ASL dataset class for image loading"""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize ASL dataset
        
        Args:
            data_dir: Path to directory containing class folders
            transform: Optional transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load samples
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
    
    def _load_samples(self):
        """Load image samples from class directories"""
        samples = []
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
                
            class_name = class_dir.name
            
            # Load all image files
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in class_dir.glob(ext):
                    samples.append({
                        'path': img_path,
                        'label': class_name
                    })
        
        if not samples:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label index
        label = self.class_to_idx[sample['label']]
        
        return image, label 