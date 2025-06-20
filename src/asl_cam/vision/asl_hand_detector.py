#!/usr/bin/env python3
"""
ASL-Optimized Hand Detector

Enhances the SimpleHandDetector with features specifically for ASL,
such as stability tracking and improved image cropping for the ML model.
It now uses a confidence score to filter for reliable detections.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .simple_hand_detector import SimpleHandDetector
from .background_removal import BackgroundRemover

logger = logging.getLogger(__name__)

class ASLHandDetector(SimpleHandDetector):
    """
    Extends SimpleHandDetector with ASL-specific optimizations.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 stability_frames: int = 3):
        """
        Initializes the ASL hand detector.
        
        Args:
            min_detection_confidence: The minimum confidence score to consider a detection valid.
            stability_frames: The number of frames to check for detection stability.
        """
        super().__init__()
        
        self.min_detection_confidence = min_detection_confidence
        self.stability_frames = stability_frames
        self.detection_history = []
        self.bg_remover = BackgroundRemover()
        
        logger.info(f"🤚 ASL Hand Detector initialized with min confidence: {self.min_detection_confidence}")
    
    def _is_detection_stable(self, new_detection_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Checks if a detection's position is stable over recent frames.
        """
        if len(self.detection_history) < self.stability_frames:
            return False
        
        recent_bboxes = [det['bbox'] for det in self.detection_history[-self.stability_frames:]]
        
        # Calculate variance of bbox centers
        centers = [(x + w // 2, y + h // 2) for x, y, w, h in recent_bboxes + [new_detection_bbox]]
        center_variance = np.var(centers, axis=0)
        
        # A stable hand shouldn't move erratically
        return np.all(center_variance < 150) # Increased variance tolerance
    
    def _enhance_hand_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extracts and enhances the hand crop for better ML model performance.
        """
        x, y, w, h = bbox
        padding = 20 # Reduced padding
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
        
        hand_crop = frame[y1:y2, x1:x2]
        
        if hand_crop.size == 0:
            return hand_crop
        
        # Lightweight enhancement: resize and basic contrast adjustment
        # Note: CLAHE is too slow for real-time on CPU.
        # A simple brightness/contrast adjustment is much faster.
        try:
            hand_crop = cv2.convertScaleAbs(hand_crop, alpha=1.1, beta=5)
        except cv2.error:
            logger.debug("Could not enhance hand crop.")
            
        return hand_crop

    def detect_hands_asl(self, frame: np.ndarray) -> List[Dict]:
        """
        Detects hands with ASL optimizations, filtering by confidence.
        
        Args:
            frame: The input BGR frame.
            
        Returns:
            A list of detected hands, each with enhanced information.
        """
        # Get raw detections from the parent SimpleHandDetector
        raw_hands = super().detect_hands(frame, max_hands=2)
        
        enhanced_hands = []
        for hand in raw_hands:
            # Critical step: Filter by confidence score
            if hand['confidence'] < self.min_detection_confidence:
                continue
            
            bbox = hand['bbox']
            
            # Enhance the detection with more info
            hand['is_stable'] = self._is_detection_stable(bbox)
            hand['enhanced_crop'] = self._enhance_hand_crop(frame, bbox)
            
            enhanced_hands.append(hand)
        
        # Update stability history only with high-confidence detections
        self.detection_history.extend(enhanced_hands)
        self.detection_history = self.detection_history[-self.stability_frames:]
        
        return enhanced_hands
    
    def reset(self):
        """Resets detector state."""
        super().reset()
        self.detection_history.clear()
        self.bg_remover = BackgroundRemover()
        logger.info("ASL Hand Detector state has been reset.") 