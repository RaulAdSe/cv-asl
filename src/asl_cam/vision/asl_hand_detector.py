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
        
        # Apply background removal to the hand crop
        hand_crop_bg_removed = self._remove_background_from_crop(hand_crop)
        
        # Resize to square
        resized = cv2.resize(hand_crop_bg_removed, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def get_skin_mask_for_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """Get the skin mask used for background removal (for visualization)."""
        hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        return skin_mask
    
    def _remove_background_from_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Fast background removal from hand crop, optimized for real-time performance.
        Uses efficient skin detection instead of slow GrabCut.
        """
        return self._fast_skin_based_removal(hand_crop)
    
    def _fast_skin_based_removal(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Enhanced background removal using improved skin color detection.
        Optimized for better quality while maintaining >15 FPS performance.
        """
        # Pre-allocate result to avoid copy
        result = hand_crop.copy()
        
        try:
            # Convert to multiple color spaces for better skin detection
            hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2YCrCb)
            lab = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2LAB)
            b, g, r = cv2.split(hand_crop)
            
            # Method 1: BGR ratio-based detection (fast)
            bgr_mask = ((r > 60) & (r > g) & (g > b) & (r > b + 15)).astype(np.uint8)
            
            # Method 2: HSV-based detection (more accurate for skin tones)
            h, s, v = cv2.split(hsv)
            # Improved HSV ranges for better skin detection
            hsv_mask1 = ((h >= 0) & (h <= 20) & (s >= 30) & (s <= 255) & (v >= 60) & (v <= 255)).astype(np.uint8)
            hsv_mask2 = ((h >= 160) & (h <= 179) & (s >= 30) & (s <= 255) & (v >= 60) & (v <= 255)).astype(np.uint8)
            hsv_mask = hsv_mask1 | hsv_mask2
            
            # Method 3: YCrCb-based detection (excellent for different lighting conditions)
            y, cr, cb = cv2.split(ycrcb)
            # Fine-tuned YCrCb ranges for better skin detection
            ycrcb_mask = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8)
            
            # Method 4: LAB color space (good for consistent lighting)
            l_lab, a_lab, b_lab = cv2.split(lab)
            lab_mask = ((l_lab >= 50) & (l_lab <= 200) & (a_lab >= 120) & (a_lab <= 150) & (b_lab >= 130) & (b_lab <= 160)).astype(np.uint8)
            
            # Combine all four methods for robust detection
            combined_mask = ((bgr_mask | hsv_mask | ycrcb_mask | lab_mask)).astype(np.uint8) * 255
            
            # NEW: Apply adaptive threshold to handle lighting variations
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            
            # Use adaptive threshold to find strong edges (likely hand boundaries)
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Invert so hand regions are white
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
            
            # Combine with skin detection (AND operation to be more conservative)
            combined_mask = cv2.bitwise_and(combined_mask, adaptive_thresh)
            
            # Enhanced morphological operations for cleaner result
            # First: close small gaps in skin regions
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Larger kernel
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Second: remove small noise and non-skin artifacts
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Larger kernel
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            
            # Third: find largest connected component (assume it's the hand)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
            
            if num_labels > 1:  # If we found components
                # Get the largest component (excluding background)
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                combined_mask = (labels == largest_component).astype(np.uint8) * 255
            
            # Fourth: dilate to ensure we capture hand edges properly
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=3)
            
            # Fifth: smooth edges with Gaussian blur
            combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 2)
            
            # Apply mask with black background
            result[combined_mask == 0] = [0, 0, 0]
            
        except Exception as e:
            logger.warning(f"Enhanced skin detection failed, using fallback: {e}")
            # Fallback to simple BGR-only detection for performance
            b, g, r = cv2.split(hand_crop)
            simple_mask = ((r > 60) & (r > g) & (g > b)).astype(np.uint8) * 255
            result[simple_mask == 0] = [0, 0, 0]
        
        return result
    
    def get_skin_mask_for_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Get the skin detection mask for visualization purposes.
        
        Args:
            hand_crop: Hand crop image in BGR format
            
        Returns:
            Binary skin mask (0-255)
        """
        try:
            # Use the same enhanced detection as the removal method
            hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2YCrCb)
            b, g, r = cv2.split(hand_crop)
            
            # Method 1: BGR ratio-based detection
            bgr_mask = ((r > 60) & (r > g) & (g > b) & (r > b + 15)).astype(np.uint8)
            
            # Method 2: HSV-based detection (same improved ranges)
            h, s, v = cv2.split(hsv)
            hsv_mask1 = ((h >= 0) & (h <= 20) & (s >= 30) & (s <= 255) & (v >= 60) & (v <= 255)).astype(np.uint8)
            hsv_mask2 = ((h >= 160) & (h <= 179) & (s >= 30) & (s <= 255) & (v >= 60) & (v <= 255)).astype(np.uint8)
            hsv_mask = hsv_mask1 | hsv_mask2
            
            # Method 3: YCrCb-based detection (same improved ranges)
            y, cr, cb = cv2.split(ycrcb)
            ycrcb_mask = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8)
            
            # Combine all methods
            combined_mask = ((bgr_mask | hsv_mask | ycrcb_mask)).astype(np.uint8) * 255
            
            # Apply same enhanced morphological operations
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
            
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=2)
            
            # Apply same blur for consistency
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 1)
            
            return combined_mask
        except Exception as e:
            logger.warning(f"Failed to generate skin mask: {e}")
            return np.zeros(hand_crop.shape[:2], dtype=np.uint8)

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