"""
Preprocessing utilities for classical computer vision hand detection.
This module will be expanded with additional preprocessing techniques
as we progress through the roadmap.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

def normalize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Normalize frame size and format for consistent processing.
    
    Args:
        frame: Input BGR frame
        target_size: Target (width, height) for resizing
        
    Returns:
        Normalized frame
    """
    if frame.shape[:2][::-1] != target_size:
        frame = cv2.resize(frame, target_size)
    
    return frame

def enhance_lighting(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance frame lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        frame: Input BGR frame
        clip_limit: CLAHE clip limit
        
    Returns:
        Enhanced frame
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def gaussian_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        frame: Input frame
        kernel_size: Size of Gaussian kernel (must be odd)
        
    Returns:
        Blurred frame
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def adaptive_threshold_mask(frame: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Create adaptive threshold mask for hand segmentation.
    
    Args:
        frame: Input BGR frame
        block_size: Size of neighborhood for threshold calculation
        C: Constant subtracted from mean
        
    Returns:
        Binary mask
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, C
    )
    
    return mask

# These functions will be expanded as we implement the roadmap stages
# Stage B: Background subtraction utilities
# Stage C: Motion detection utilities  
# Stage D: Data augmentation utilities
