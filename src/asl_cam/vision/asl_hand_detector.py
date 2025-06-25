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
        """Crops, squares, removes background, and enhances the hand image for the model."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0: return None
        
        center_x, center_y, max_dim = x + w // 2, y + h // 2, max(w, h)
        
        crop_x1 = max(0, center_x - max_dim // 2)
        crop_y1 = max(0, center_y - max_dim // 2)
        crop_x2 = min(frame.shape[1], center_x + max_dim // 2)
        crop_y2 = min(frame.shape[0], center_y + max_dim // 2)
        
        hand_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if hand_crop.size == 0: return None
        
        # Apply background removal first, then resize to model input size
        # This is what the user wants: Background Removal → Resize → Model Input
        crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
        hand_crop_bg_removed = self._remove_background_from_crop(hand_crop, crop_coords)
        
        # Now just resize the background-removed image (no additional filters!)
        resized = cv2.resize(hand_crop_bg_removed, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def get_skin_mask_for_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Get the skin detection mask for visualization purposes.
        
        Args:
            hand_crop: Hand crop image in BGR format
            
        Returns:
            Simple binary skin mask (0-255)
        """
        return self._simple_skin_detection(hand_crop)
    
    def get_mog2_mask_for_crop(self, hand_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the MOG2 foreground mask for visualization purposes.
        TEMPORARILY DISABLED: To avoid tracking interference.
        
        Args:
            hand_crop: Hand crop image in BGR format
            
        Returns:
            None (temporarily disabled to avoid tracking issues)
        """
        # TEMPORARILY DISABLED to avoid tracking interference
        # TODO: Implement with separate MOG2 instance
        return None
    
    def _remove_background_from_crop(self, hand_crop: np.ndarray, crop_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Simple, effective background removal using the original working approach.
        Falls back to MOG2+skin combination when background is available.
        """
        # Primary: Try MOG2+skin combination if background is available
        if self.bg_remover.background_image is not None:
            try:
                return self._mog2_skin_combination(hand_crop, crop_coords)
            except Exception as e:
                logger.warning(f"MOG2+skin combination failed: {e}")
        
        # Fallback: Use the original simple skin detection that was working well
        return self._simple_skin_removal(hand_crop)
    
    def _mog2_skin_combination(self, hand_crop: np.ndarray, crop_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Enhanced background removal using MOG2 static background + skin detection.
        This is the advanced version that should work better when background is learned.
        """
        result = hand_crop.copy()
        
        # 1. Get the corresponding crop from the static background image
        x1, y1, x2, y2 = crop_coords
        background_crop = self.bg_remover.background_image[y1:y2, x1:x2]

        # Ensure the background crop has the same size as the hand crop
        if background_crop.shape != hand_crop.shape:
            raise ValueError("Background crop and hand crop shape mismatch")

        # 2. Compute the absolute difference to find what's changed (the hand)
        diff = cv2.absdiff(background_crop, hand_crop)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 3. Threshold the difference to create a "motion mask"
        _, motion_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        
        # 4. Get simple skin mask
        skin_mask = self._simple_skin_detection(hand_crop)

        # 5. Combine Motion Mask + Skin Mask (AND operation for precision)
        combined_mask = cv2.bitwise_and(motion_mask, skin_mask)
        
        # 6. If combined mask is too restrictive, use OR operation
        if np.sum(combined_mask > 0) < 0.3 * np.sum(skin_mask > 0):
            combined_mask = cv2.bitwise_or(motion_mask, skin_mask)

        # 7. Simple morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Apply the mask
        result[combined_mask == 0] = [0, 0, 0]
        return result
    
    def _simple_skin_removal(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Original simple skin-based background removal that was working well.
        This is the fallback that should be reliable and fast.
        """
        result = hand_crop.copy()
        
        # Get simple skin mask
        skin_mask = self._simple_skin_detection(hand_crop)
        
        # Apply mask - set non-skin pixels to black
        result[skin_mask == 0] = [0, 0, 0]
        
        return result
    
    def _simple_skin_detection(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Simple, fast skin detection using BGR ratios and HSV ranges.
        This was the original approach that worked well.
        """
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # BGR components
            b, g, r = cv2.split(hand_crop)
            
            # Method 1: Simple BGR-based skin detection
            bgr_mask = ((r > 95) & (g > 40) & (b > 20) & 
                       (r > g) & (r > b)).astype(np.uint8)
            
            # Method 2: HSV-based skin detection
            hsv_mask = ((h >= 0) & (h <= 20) & 
                       (s >= 30) & (s <= 150) & 
                       (v >= 60) & (v <= 255)).astype(np.uint8)
            
            # Combine both methods (OR operation)
            skin_mask = (bgr_mask | hsv_mask).astype(np.uint8) * 255
            
            # Simple morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            return skin_mask
            
        except Exception as e:
            logger.warning(f"Simple skin detection failed: {e}")
            # Ultimate fallback: basic threshold
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            return mask
    


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