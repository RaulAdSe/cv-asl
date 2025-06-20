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
from .tracker import HandTracker

logger = logging.getLogger(__name__)

class ASLHandDetector(SimpleHandDetector):
    """
    An enhanced hand detector for ASL that uses tracking for stability.
    It finds a hand, then tracks it smoothly, only searching for a new
    hand if the track is lost.
    """
    
    def __init__(self, min_detection_confidence: float = 0.6):
        """
        Initialize the ASL hand detector.
        
        Args:
            min_detection_confidence: The minimum confidence from the simple
                                      detector to consider a hand found.
        """
        super().__init__()
        
        self.min_detection_confidence = min_detection_confidence
        self.tracker = HandTracker()
        
        logger.info(f"🤚 ASL Hand Detector initialized with min confidence: {self.min_detection_confidence}")

    def detect_and_track(self, frame: np.ndarray) -> Optional[Dict]:
        """
        The main method to detect and track a hand in a frame.
        
        Strategy:
        1. If we have a tracked hand, update or predict its position.
        2. If we don't have a track, run a full detection.
        3. If a high-confidence hand is found, initialize the tracker.
        
        Args:
            frame: The input BGR frame.
            
        Returns:
            A dictionary containing the smoothed 'bbox' and 'status' 
            ('TRACKED', 'PREDICTED', or 'NEW_DETECTION'), or None if no hand.
        """
        hand_info = None

        if self.tracker.initialized:
            # --- We are already tracking a hand ---
            detections = super().detect_hands(frame, max_hands=1)
            
            if detections:
                # We see a hand, update the tracker with the best detection
                best_detection = detections[0]
                smooth_bbox = self.tracker.update(best_detection['bbox'])
                hand_info = {'bbox': smooth_bbox, 'status': 'TRACKED'}
            else:
                # We don't see a hand, predict its position
                predicted_bbox = self.tracker.predict()
                if predicted_bbox:
                    hand_info = {'bbox': predicted_bbox, 'status': 'PREDICTED'}
                else:
                    # Tracker has lost the hand
                    hand_info = None
        else:
            # --- We are not tracking, so run a new detection ---
            detections = super().detect_hands(frame, max_hands=1)
            if detections:
                best_detection = detections[0]
                if best_detection['confidence'] >= self.min_detection_confidence:
                    # High-confidence detection, initialize tracker
                    self.tracker.initialize(best_detection['bbox'])
                    hand_info = {'bbox': best_detection['bbox'], 'status': 'NEW_DETECTION'}
        
        return hand_info

    def get_hand_crop(self, frame: np.ndarray, hand_bbox: Tuple[int, int, int, int], crop_size: int) -> Optional[np.ndarray]:
        """
        Crops the hand region from the frame, ensuring it's square.
        
        Args:
            frame: The full frame.
            hand_bbox: The bounding box of the hand.
            crop_size: The desired square dimension of the output crop.
            
        Returns:
            A square-cropped image of the hand, or None if the bbox is invalid.
        """
        x, y, w, h = hand_bbox
        
        # Ensure the bounding box is valid
        if w <= 0 or h <= 0:
            return None
        
        # Get the center and the largest dimension
        center_x = x + w // 2
        center_y = y + h // 2
        max_dim = max(w, h)
        
        # Define the square crop region
        crop_x1 = max(0, center_x - max_dim // 2)
        crop_y1 = max(0, center_y - max_dim // 2)
        crop_x2 = min(frame.shape[1], center_x + max_dim // 2)
        crop_y2 = min(frame.shape[0], center_y + max_dim // 2)
        
        hand_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if hand_crop.size == 0:
            return None
        
        # Resize to the final square size
        return cv2.resize(hand_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    def enhance_hand_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Enhances the hand crop for better recognition by the model.
        This now only applies background removal and basic enhancements.
        
        Args:
            hand_crop: The square image of the hand.
            
        Returns:
            The enhanced hand image.
        """
        # The new background remover works on the full frame, but we can refine
        # the crop here if needed. For now, basic enhancements are enough.
        
        # Convert to grayscale for some operations
        gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        
        # Fast contrast enhancement
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Convert back to BGR for the model
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def detect_and_process_hand(self, frame: np.ndarray, crop_size: int) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        A high-level function that combines detection, tracking, and processing.
        
        Args:
            frame: The input BGR frame.
            crop_size: The final size for the model input crop.
            
        Returns:
            A tuple of (processed_hand_crop, hand_info)
        """
        hand_info = self.detect_and_track(frame)
        
        if not hand_info:
            return None, None
            
        hand_crop = self.get_hand_crop(frame, hand_info['bbox'], crop_size)
        
        if hand_crop is None:
            return None, hand_info
            
        enhanced_crop = self.enhance_hand_crop(hand_crop)
        
        return enhanced_crop, hand_info
    
    def reset(self):
        """Resets detector state."""
        super().reset()
        self.tracker.reset()
        logger.info("ASL Hand Detector state has been reset.") 