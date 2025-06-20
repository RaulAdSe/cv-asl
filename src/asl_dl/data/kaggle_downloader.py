"""
Kaggle ASL Dataset Downloader

Downloads and organizes the ASL dataset from Kaggle using kagglehub.
"""

import kagglehub
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def download_kaggle_asl(target_dir: str = "data/raw/kaggle_asl", letters: list = None) -> Path:
    """
    Download ASL dataset from Kaggle using kagglehub
    
    Args:
        target_dir: Target directory for the dataset
        letters: List of letters to keep (e.g., ['A', 'B', 'C']). If None, keeps all.
    
    Returns:
        Path to the organized dataset
    """
    logger.info("🚀 Downloading ASL dataset from Kaggle...")
    
    # Download using kagglehub
    try:
        path = kagglehub.dataset_download("ayuraj/asl-dataset")
        logger.info(f"✅ Downloaded to: {path}")
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise
    
    # Organize dataset
    source_path = Path(path)
    target_path = Path(target_dir)
    
    logger.info(f"📁 Organizing dataset to: {target_path}")
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    train_path = target_path / "train"
    
    if train_path.exists():
        logger.info("🗑️ Removing existing dataset...")
        shutil.rmtree(train_path)
    
    # Find the actual dataset directory
    train_source = None
    
    # Look for asl_dataset directory first
    asl_dataset_dir = source_path / "asl_dataset"
    if asl_dataset_dir.exists():
        train_source = asl_dataset_dir
    else:
        # Look for train directory
        for potential_dir in source_path.rglob("*"):
            if potential_dir.is_dir() and potential_dir.name in ['train', 'Train']:
                train_source = potential_dir
                break
        
        if not train_source:
            # Use source path directly if it contains numbered/letter directories
            numbered_dirs = [d for d in source_path.iterdir() if d.is_dir() and (d.name.isdigit() or len(d.name) == 1)]
            if numbered_dirs:
                train_source = source_path
            else:
                raise ValueError(f"Could not find dataset directory in {source_path}")
    
    logger.info(f"📂 Source directory: {train_source}")
    
    # Copy and filter dataset
    train_path.mkdir(exist_ok=True)
    
    # Map from numbers to letters for Kaggle dataset
    number_to_letter = {
        '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
        '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L',
        '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X',
        '24': 'Y', '25': 'Z'
    }
    
    total_samples = 0
    
    for class_dir in train_source.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
            
        # Convert directory name to letter
        if class_dir.name.isdigit():
            # Numbered directories (0=A, 1=B, 2=C, etc.)
            letter = number_to_letter.get(class_dir.name)
            if not letter:
                continue
        elif len(class_dir.name) == 1:
            # Single letter directories (a, b, c, etc.)
            letter = class_dir.name.upper()
        else:
            # Skip other directories
            continue
        
        # Filter letters if specified
        if letters and letter not in letters:
            continue
        
        # Create target directory
        target_class_dir = train_path / letter
        target_class_dir.mkdir(exist_ok=True)
        
        # Copy images
        copied = 0
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_file in class_dir.glob(ext):
                shutil.copy2(img_file, target_class_dir / img_file.name)
                copied += 1
        
        total_samples += copied
        logger.info(f"   📝 {letter}: {copied:,} samples")
    
    logger.info(f"\n✅ Dataset organized:")
    logger.info(f"   📁 Location: {train_path}")
    logger.info(f"   📊 Total samples: {total_samples:,}")
    
    return train_path

def download_abc_dataset() -> Path:
    """Download only A, B, C letters for quick testing"""
    return download_kaggle_asl(letters=['A', 'B', 'C']) 