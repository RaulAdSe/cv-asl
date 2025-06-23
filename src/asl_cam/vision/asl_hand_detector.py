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
        
        # Pass the crop coordinates along with the hand crop
        crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
        hand_crop_bg_removed = self._remove_background_from_crop(hand_crop, crop_coords)
        
        # Resize to square
        resized = cv2.resize(hand_crop_bg_removed, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def get_skin_mask_for_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Get the skin detection mask for visualization purposes.
        
        Args:
            hand_crop: Hand crop image in BGR format
            
        Returns:
            Binary skin mask (0-255)
        """
        return self._get_skin_detection_mask(hand_crop)
    
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
        Robust background removal using the static learned MOG2 background image
        combined with advanced skin detection.
        """
        # If the MOG2 background model isn't ready, fall back to skin detection.
        if self.bg_remover.background_image is None:
            return self._fast_skin_based_removal(hand_crop)

        result = hand_crop.copy()
        try:
            # 1. Get the corresponding crop from the static background image
            x1, y1, x2, y2 = crop_coords
            background_crop = self.bg_remover.background_image[y1:y2, x1:x2]

            # Ensure the background crop has the same size as the hand crop
            if background_crop.shape != hand_crop.shape:
                logger.warning("Background crop and hand crop shape mismatch, falling back.")
                return self._fast_skin_based_removal(hand_crop)

            # 2. Compute the absolute difference to find what's changed (the hand)
            diff = cv2.absdiff(background_crop, hand_crop)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # 3. Threshold the difference to create a "motion mask"
            _, motion_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

            # 4. Get the skin mask
            skin_mask = self._get_skin_detection_mask(hand_crop)

            # 5. Combine Motion Mask + Skin Mask for ultimate precision
            # A pixel must be BOTH motion AND skin-colored.
            combined_mask = cv2.bitwise_and(motion_mask, skin_mask)

            # 6. Refine the final mask
            # Find largest connected component (assume it's the hand)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
            if num_labels > 1:
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                final_mask = (labels == largest_component).astype(np.uint8) * 255
            else:
                final_mask = combined_mask

            # Dilate and smooth for clean edges
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=2)
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

            # Apply the final mask
            result[final_mask == 0] = [0, 0, 0]

        except Exception as e:
            logger.error(f"Error in MOG2 background subtraction, falling back: {e}")
            return self._fast_skin_based_removal(hand_crop)
            
        return result
    
    def _get_skin_detection_mask(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Get skin detection mask using multiple color spaces.
        Enhanced with better thresholds and more robust detection.
        """
        try:
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2YCrCb)
            lab = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2LAB)
            b, g, r = cv2.split(hand_crop)
            
            # Method 1: Enhanced BGR ratio-based detection
            bgr_mask = ((r > 95) & (g > 40) & (b > 20) & 
                       (r > g) & (r > b) & 
                       (abs(r.astype(int) - g.astype(int)) > 15)).astype(np.uint8)
            
            # Method 2: Improved HSV-based detection (better ranges for different lighting)
            h, s, v = cv2.split(hsv)
            # Wider hue range for different skin tones
            hsv_mask1 = ((h >= 0) & (h <= 30) & (s >= 30) & (s <= 170) & (v >= 60) & (v <= 255)).astype(np.uint8)
            hsv_mask2 = ((h >= 150) & (h <= 179) & (s >= 30) & (s <= 170) & (v >= 60) & (v <= 255)).astype(np.uint8)
            hsv_mask = hsv_mask1 | hsv_mask2
            
            # Method 3: Enhanced YCrCb-based detection (most reliable for skin)
            y, cr, cb = cv2.split(ycrcb)
            # Optimized YCrCb ranges for better skin detection
            ycrcb_mask = ((y >= 80) & (y <= 255) & 
                         (cr >= 133) & (cr <= 173) & 
                         (cb >= 77) & (cb <= 127)).astype(np.uint8)
            
            # Method 4: LAB color space (good for lighting invariance)
            l_lab, a_lab, b_lab = cv2.split(lab)
            lab_mask = ((l_lab >= 20) & (l_lab <= 200) & 
                       (a_lab >= 124) & (a_lab <= 150) & 
                       (b_lab >= 114) & (b_lab <= 144)).astype(np.uint8)
            
            # Combine all methods with weights (YCrCb is most reliable)
            # Use weighted combination: YCrCb has highest weight
            combined_mask = ((bgr_mask | hsv_mask | ycrcb_mask | lab_mask)).astype(np.uint8) * 255
            
            # Extra emphasis on YCrCb for skin regions
            ycrcb_weight = ycrcb_mask.astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(combined_mask, ycrcb_weight)
            
            # Enhanced morphological operations
            # 1. Close small gaps in skin regions
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 2. Remove small noise artifacts
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            
            return combined_mask
            
        except Exception as e:
            logger.warning(f"Skin detection failed: {e}")
            return np.zeros(hand_crop.shape[:2], dtype=np.uint8)

    def _fast_skin_based_removal(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Enhanced skin-based removal with improved quality.
        Uses advanced skin detection + morphological operations for better results.
        """
        result = hand_crop.copy()
        try:
            # Get comprehensive skin detection mask
            skin_mask = self._get_skin_detection_mask(hand_crop)
            
            # Apply adaptive thresholding for edge enhancement
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)  # Invert so hand regions are white
            
            # Combine skin detection with adaptive threshold (AND operation for precision)
            combined_mask = cv2.bitwise_and(skin_mask, adaptive_thresh)
            
            # If combined mask is too restrictive, fall back to skin detection only
            skin_pixel_count = np.sum(skin_mask > 0)
            combined_pixel_count = np.sum(combined_mask > 0)
            
            if combined_pixel_count < 0.4 * skin_pixel_count:
                # Use skin detection only if adaptive threshold is too restrictive
                final_mask = skin_mask
            else:
                final_mask = combined_mask
            
            # Find largest connected component (assume it's the hand)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
            
            if num_labels > 1:
                # Get the largest component (excluding background)
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                final_mask = (labels == largest_component).astype(np.uint8) * 255
            
            # Enhanced morphological operations for smoother edges
            # 1. Close small gaps
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 2. Remove small noise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
            
            # 3. Dilate to ensure we capture hand edges
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=2)
            
            # 4. Smooth edges with Gaussian blur
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 1)
            
            # Apply mask with black background
            result[final_mask == 0] = [0, 0, 0]
            
        except Exception as e:
            logger.warning(f"Enhanced skin removal failed, using basic fallback: {e}")
            # Ultimate fallback: simple BGR detection
            b, g, r = cv2.split(hand_crop)
            simple_mask = ((r > 60) & (r > g) & (g > b)).astype(np.uint8) * 255
            result[simple_mask == 0] = [0, 0, 0]
            
        return result

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