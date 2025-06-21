#!/usr/bin/env python3
"""
ASL-Optimized Hand Detector

Integrates background removal and a Kalman Filter tracker to provide a stable,
reliable hand bounding box for ASL gesture recognition.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from .background_removal import BackgroundRemover
from .tracker import HandTracker

logger = logging.getLogger(__name__)

class ASLHandDetector:
    """
    Detects and tracks a single hand robustly using background subtraction and
    a Kalman filter.
    
    The strategy is stateful:
    1.  Wait for a stable background model.
    2.  Perform an initial, full-frame search for the hand.
    3.  Once found, initialize a tracker and only search within a small
        region of interest (ROI) around the hand's predicted position.
    4.  If the track is lost for several frames, revert to a full-frame search.
    """
    
    def __init__(self, min_contour_area=3000, search_roi_scale=1.5):
        self.bg_remover = BackgroundRemover()
        self.tracker = HandTracker(patience=5) # 5 frames of patience
        self.min_contour_area = min_contour_area
        self.search_roi_scale = search_roi_scale
        
        logger.info(f"🤚 ASL Hand Detector initialized.")

    def _find_best_hand_contour(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Finds the largest contour in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        return cv2.boundingRect(largest_contour)

    def _get_search_roi(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Calculates an expanded search ROI around a given bounding box."""
        x, y, w, h = bbox
        
        # Expand the ROI to give the tracker some breathing room
        new_w = int(w * self.search_roi_scale)
        new_h = int(h * self.search_roi_scale)
        
        # Center the new ROI
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        
        # Ensure the ROI is within frame bounds
        new_x2 = min(frame_shape[1], new_x + new_w)
        new_y2 = min(frame_shape[0], new_y + new_h)
        
        return new_x, new_y, new_x2 - new_x, new_y2 - new_y

    def _preprocess_hand(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], target_size: int) -> Optional[np.ndarray]:
        """Crops, squares, and enhances the hand image for the model."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0: return None
        
        center_x, center_y, max_dim = x + w // 2, y + h // 2, max(w, h)
        
        crop_x1 = max(0, center_x - max_dim // 2)
        crop_y1 = max(0, center_y - max_dim // 2)
        crop_x2 = min(frame.shape[1], center_x + max_dim // 2)
        crop_y2 = min(frame.shape[0], center_y + max_dim // 2)
        
        hand_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if hand_crop.size == 0: return None
        
        # Resize to square
        resized = cv2.resize(hand_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Basic enhancement
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Return in BGR format
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def detect_and_process_hand(self, frame: np.ndarray, target_size: int) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        The main pipeline for detecting, tracking, and processing the hand.
        This version ensures that a "ghost" bounding box is not shown when the
        hand is temporarily lost, and that full-frame detection resumes reliably.
        """
        _, fg_mask = self.bg_remover.remove_background(frame)
        hand_bbox = None
        status = "LOST" # Default status if nothing is found

        if self.tracker.is_tracking:
            # --- TRACKING STATE ---
            # We believe a hand is present. Let's try to track it.
            predicted_bbox, pred_status = self.tracker.predict()

            if pred_status == "LOST":
                # The tracker gave up. It is now reset.
                # We'll let the logic fall through to the SEARCHING state below.
                pass
            else:
                # Tracker is still active. Search for the hand in the predicted ROI.
                roi_bbox = self._get_search_roi(predicted_bbox, frame.shape)
                x, y, w, h = roi_bbox
                
                # We need to handle the case where the ROI is empty
                if w > 0 and h > 0:
                    best_in_roi = self._find_best_hand_contour(fg_mask[y:y+h, x:x+w])
                else:
                    best_in_roi = None

                if best_in_roi:
                    # SUCCESS: We found the hand. Update the tracker with the new location.
                    gx, gy, gw, gh = best_in_roi
                    hand_bbox = (gx + x, gy + y, gw, gh)
                    self.tracker.update(hand_bbox)
                    status = "TRACKED"
                else:
                    # MISS: No hand in the ROI. Tell the tracker.
                    # It will start its patience countdown. No box will be shown.
                    self.tracker.update(None)
                    status = "LOST"
        
        if not self.tracker.is_tracking:
            # --- SEARCHING STATE ---
            # The tracker is not active, so we search the whole frame.
            status = "LOST"
            hand_bbox = self._find_best_hand_contour(fg_mask)
            if hand_bbox:
                # SUCCESS: We found a new hand. Initialize the tracker.
                self.tracker.initialize(hand_bbox)
                status = "NEW_DETECTION"

        # --- Prepare Output ---
        processed_hand, hand_info = None, None
        if hand_bbox is not None:
            # Only create info dict if we have a box to draw.
            hand_info = {"bbox": hand_bbox, "status": status}
            processed_hand = self._preprocess_hand(frame, hand_bbox, target_size)
            
        return processed_hand, hand_info

    def reset(self):
        """Resets the state of the detector and its components."""
        self.bg_remover.reset()
        self.tracker.reset()
        logger.info("ASL Hand Detector state has been reset.") 