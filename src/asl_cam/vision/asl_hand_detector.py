#!/usr/bin/env python3
"""
ASL-Optimized Hand Detector

Enhanced hand detection specifically optimized for ASL recognition.
Builds on the simple hand detector with improvements for:
- Better hand segmentation for ASL poses
- Stability for letter recognition
- Optimized preprocessing for the ML model

Author: CV-ASL Team
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ MediaPipe not available - falling back to basic hand detection")

from .simple_hand_detector import SimpleHandDetector
from .background_removal import BackgroundRemover

logger = logging.getLogger(__name__)

class ASLHandDetector(SimpleHandDetector):
    """
    ASL-optimized hand detector
    
    Extends SimpleHandDetector with ASL-specific optimizations:
    - Improved hand segmentation
    - Stability filtering for consistent detection
    - Better preprocessing for ML models
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 stability_frames: int = 3):
        """
        Initialize ASL hand detector
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection (unused with simple detector)
            min_tracking_confidence: Minimum confidence for hand tracking (unused with simple detector)
            stability_frames: Number of frames for stability filtering
        """
        
        # Initialize parent class (SimpleHandDetector doesn't take parameters)
        super().__init__()
        
        # ASL-specific settings
        self.stability_frames = stability_frames
        self.detection_history = []
        self.stable_detection = None
        
        # Background remover for better hand isolation
        self.bg_remover = BackgroundRemover()
        self.bg_initialized = False
        
        logger.info("🤚 ASL Hand Detector initialized")
    
    def _is_stable_detection(self, new_detection: Dict) -> bool:
        """
        Check if detection is stable across multiple frames
        
        Args:
            new_detection: New hand detection
            
        Returns:
            True if detection is stable
        """
        
        if len(self.detection_history) < self.stability_frames:
            return False
        
        # Check if hand position is consistent
        new_bbox = new_detection['bbox']
        recent_bboxes = [det['bbox'] for det in self.detection_history[-self.stability_frames:]]
        
        # Calculate position variance
        centers = []
        for bbox in recent_bboxes + [new_bbox]:
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        # Check if centers are close together (stable)
        center_variance = np.var(centers, axis=0)
        max_variance = 50  # pixels
        
        return all(var < max_variance for var in center_variance)
    
    def _enhance_hand_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Enhance hand crop for better ASL recognition
        
        Args:
            frame: Input frame
            bbox: Hand bounding box (x, y, w, h)
            
        Returns:
            Enhanced hand crop
        """
        
        x, y, w, h = bbox
        
        # Add padding for better hand capture
        padding = max(20, min(w, h) // 4)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract hand crop
        hand_crop = frame[y1:y2, x1:x2].copy()
        
        if hand_crop.size == 0:
            return hand_crop
        
        # Apply enhancement filters
        try:
            # 1. Contrast enhancement
            lab = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            hand_crop = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Slight blur to reduce noise
            hand_crop = cv2.GaussianBlur(hand_crop, (3, 3), 0)
            
            # 3. Sharpen edges
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]]) * 0.1
            hand_crop = cv2.filter2D(hand_crop, -1, kernel)
            
        except Exception as e:
            logger.warning(f"Hand enhancement failed: {e}")
        
        return hand_crop
    
    def detect_hands_asl(self, frame: np.ndarray, use_background_removal: bool = True) -> List[Dict]:
        """
        Detect hands with ASL-specific optimizations
        
        Args:
            frame: Input BGR frame
            use_background_removal: Whether to use background removal
            
        Returns:
            List of detected hands with enhanced information
        """
        
        processed_frame = frame.copy()
        
        # Optional background removal for better hand isolation
        if use_background_removal:
            if not self.bg_initialized:
                self.bg_remover.initialize_background(frame)
                self.bg_initialized = True
            
            try:
                # Get foreground mask
                fg_mask = self.bg_remover.get_foreground_mask(frame)
                
                # Apply mask to frame
                processed_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
                
                # Fill masked areas with neutral color to avoid artifacts
                processed_frame[fg_mask == 0] = [128, 128, 128]
                
            except Exception as e:
                logger.warning(f"Background removal failed: {e}")
                processed_frame = frame
        
        # Detect hands using parent method
        hands = super().detect_hands(processed_frame)
        
        # Enhance detected hands with ASL-specific information
        enhanced_hands = []
        
        for hand in hands:
            enhanced_hand = hand.copy()
            
            # Enhance hand crop
            enhanced_crop = self._enhance_hand_crop(frame, hand['bbox'])
            enhanced_hand['enhanced_crop'] = enhanced_crop
            
            # Add stability information
            self.detection_history.append(enhanced_hand)
            
            # Keep only recent history
            if len(self.detection_history) > self.stability_frames * 2:
                self.detection_history = self.detection_history[-self.stability_frames:]
            
            # Check stability
            enhanced_hand['is_stable'] = self._is_stable_detection(enhanced_hand)
            
            # Update stable detection if current detection is stable
            if enhanced_hand['is_stable']:
                self.stable_detection = enhanced_hand
            
            enhanced_hands.append(enhanced_hand)
        
        return enhanced_hands
    
    def get_stable_hand_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get a stable hand crop for ASL recognition
        
        Args:
            frame: Input frame
            
        Returns:
            Stable hand crop or None if no stable detection
        """
        
        hands = self.detect_hands_asl(frame)
        
        if not hands:
            return None
        
        # Use the most stable hand
        stable_hands = [h for h in hands if h.get('is_stable', False)]
        
        if stable_hands:
            # Use the largest stable hand
            stable_hand = max(stable_hands, key=lambda h: h['bbox'][2] * h['bbox'][3])
            return stable_hand.get('enhanced_crop')
        
        # If no stable hands, use the largest hand
        largest_hand = max(hands, key=lambda h: h['bbox'][2] * h['bbox'][3])
        return largest_hand.get('enhanced_crop')
    
    def reset_stability(self):
        """Reset stability tracking"""
        self.detection_history.clear()
        self.stable_detection = None
        logger.info("🔄 ASL hand detector stability reset")
    
    def reset_background(self):
        """Reset background model"""
        self.bg_initialized = False
        self.bg_remover.reset()
        logger.info("🔄 Background model reset")
    
    def reset(self):
        """Reset all detector states"""
        super().reset()
        self.reset_stability()
        self.reset_background() 