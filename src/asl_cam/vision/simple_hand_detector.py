"""
Simple and effective hand detection using skin color and motion.

This detector provides a baseline for identifying hand regions using
traditional computer vision techniques with OpenCV. It's designed to be
lightweight and understandable.

Key Improvements:
- Heuristic-based confidence scoring.
- More robust contour filtering.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

from .skin import SkinDetector
from .background_removal import BackgroundRemover

logger = logging.getLogger(__name__)

class SimpleHandDetector(SkinDetector):
    """Simple hand detector using skin color segmentation and motion analysis."""
    
    def __init__(self, 
                 min_area: int = 3000, 
                 max_area: int = 40000,
                 motion_threshold: int = 500):
        super().__init__()
        
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=30, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.motion_threshold = motion_threshold
        
        # --- NEW: Link to the advanced background remover ---
        self.bg_remover = BackgroundRemover()
        
    def _calculate_confidence(self, contour: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """
        Calculate a heuristic-based confidence score for a hand contour.
        
        Args:
            contour: The contour of the potential hand.
            frame_shape: The (height, width) of the frame.
            
        Returns:
            A confidence score between 0.0 and 1.0.
        """
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Score 1: Area confidence (peaks at a 'normal' hand size)
        ideal_area = 20000
        area_confidence = 1.0 - abs(area - ideal_area) / (self.max_area - self.min_area)
        area_confidence = max(0, area_confidence)

        # Score 2: Aspect ratio confidence (hands are usually not too skinny or wide)
        aspect_ratio = w / float(h) if h > 0 else 0
        ideal_aspect_ratio = 1.0
        aspect_confidence = 1.0 - abs(aspect_ratio - ideal_aspect_ratio) / 1.5
        aspect_confidence = max(0, aspect_confidence)
        
        # Score 3: Solidity (hands are complex shapes, not perfect blobs)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        # Lower solidity is better for hands with fingers
        solidity_confidence = 1.0 - solidity if solidity > 0.6 else 0.8 
        
        # Combine scores
        total_confidence = (area_confidence * 0.4 + 
                            aspect_confidence * 0.4 + 
                            solidity_confidence * 0.2)
                            
        return min(1.0, max(0.0, total_confidence))

    def detect_hands(self, frame: np.ndarray, max_hands: int = 1) -> List[Dict]:
        """
        Detects hands using the foreground mask from the background remover.
        This is more robust than the previous skin/motion method.
        
        Args:
            frame: BGR input frame.
            max_hands: Maximum number of hands to detect.
            
        Returns:
            A list of dictionaries, where each dict contains 'bbox' and 'confidence'.
        """
        # --- Use the advanced background remover to get a clean foreground mask ---
        _, fg_mask = self.bg_remover.remove_background(frame)

        if not self.bg_remover.bg_model_learned:
            # If the background isn't learned yet, we can't detect hands.
            return []

        # Find contours on the clean foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Use a slightly more forgiving min_area since the mask is cleaner
            if self.min_area * 0.5 < area < self.max_area:
                bbox = cv2.boundingRect(contour)
                confidence = self._calculate_confidence(contour, frame.shape[:2])
                
                valid_hands.append({
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        # Sort by confidence and return the top N hands
        valid_hands.sort(key=lambda h: h['confidence'], reverse=True)
        
        return valid_hands[:max_hands]
    
    def reset(self):
        """Resets the detector's state, including the background model."""
        self.bg_remover.reset()
        logger.info("SimpleHandDetector reset.")
    
    def detect_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Detects the skin mask from the frame.
        
        Args:
            frame: BGR input frame.
            
        Returns:
            A binary mask where skin pixels are white and non-skin pixels are black.
        """
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range of skin color in HSV
        lower = np.array([0, 40, 80])
        upper = np.array([20, 255, 255])
        
        # Threshold the HSV image to get only skin colors
        mask = cv2.inRange(hsv, lower, upper)
        
        return mask 