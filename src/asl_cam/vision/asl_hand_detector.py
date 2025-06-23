#!/usr/bin/env python3
"""
ASL-Optimized Hand Detector

Integrates background removal and a Kalman Filter tracker to provide a stable,
reliable hand bounding box for ASL gesture recognition.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
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
        self.tracker = HandTracker(patience=15) # Increased patience: allow 15 frames without measurement
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

    def _get_squared_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extracts a squared crop and its coordinates around the bounding box."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0: return None

        center_x, center_y, max_dim = x + w // 2, y + h // 2, max(w, h)
        
        crop_x1 = max(0, center_x - max_dim // 2)
        crop_y1 = max(0, center_y - max_dim // 2)
        crop_x2 = min(frame.shape[1], center_x + max_dim // 2)
        crop_y2 = min(frame.shape[0], center_y + max_dim // 2)
        
        hand_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if hand_crop.size == 0: return None

        crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
        return hand_crop, crop_coords

    def _preprocess_hand(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], target_size: int) -> Optional[np.ndarray]:
        """Crops, squares, removes background, and enhances the hand image for the model."""
        crop_result = self._get_squared_crop(frame, bbox)
        if not crop_result:
            return None
        hand_crop, crop_coords = crop_result

        hand_crop_bg_removed = self._remove_background_from_crop(hand_crop, crop_coords)
        
        # Resize to the target size for the model
        model_input = cv2.resize(hand_crop_bg_removed, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return model_input

    def _get_hand_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Gets the raw hand crop without background removal."""
        crop_result = self._get_squared_crop(frame, bbox)
        return crop_result[0] if crop_result else None

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
            logger.warning(f"Error in fast skin removal: {e}")
            return np.zeros_like(hand_crop)
            
        return result

    def _get_measurement_bbox(self, frame: np.ndarray, predicted_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get measurement bbox around the predicted location for tracker update.
        This validates that there's still a hand-like object at the predicted location.
        """
        try:
            # Extract region around prediction for validation
            x, y, w, h = predicted_bbox
            
            # Expand search area more generously to account for fast movement
            search_margin = max(30, min(w, h) // 3)
            search_x1 = max(0, x - search_margin)
            search_y1 = max(0, y - search_margin)
            search_x2 = min(frame.shape[1], x + w + search_margin)
            search_y2 = min(frame.shape[0], y + h + search_margin)
            
            search_region = frame[search_y1:search_y2, search_x1:search_x2]
            if search_region.size == 0:
                return None
            
            # Use background subtraction to find foreground in search region
            try:
                _, fg_mask = self.bg_remover.remove_background(search_region)
                if fg_mask is None:
                    return None
            except Exception as e:
                logger.debug(f"Background removal failed in measurement: {e}")
                return None
            
            # Check if there's any foreground at all
            if not np.any(fg_mask):
                return None
            
            # Find contours in the search region
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            # Find the largest contour (should be the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Use a more lenient area threshold - adapt based on predicted bbox size
            min_area = max(300, (w * h) * 0.1)  # At least 10% of predicted bbox area
            if contour_area < min_area:
                return None
            
            # Get bounding box of the largest contour
            rel_x, rel_y, rel_w, rel_h = cv2.boundingRect(largest_contour)
            
            # Convert back to absolute coordinates
            abs_x = search_x1 + rel_x
            abs_y = search_y1 + rel_y
            
            # Validate the measurement makes sense (not too far from prediction)
            pred_center_x = x + w // 2
            pred_center_y = y + h // 2
            meas_center_x = abs_x + rel_w // 2
            meas_center_y = abs_y + rel_h // 2
            
            # Allow reasonable distance from prediction
            max_distance = max(w, h)  # Can move up to width or height of bbox
            distance = np.sqrt((pred_center_x - meas_center_x)**2 + (pred_center_y - meas_center_y)**2)
            
            if distance > max_distance:
                # Measurement is too far from prediction, likely noise
                return None
            
            return (abs_x, abs_y, rel_w, rel_h)
            
        except Exception as e:
            logger.debug(f"Failed to get measurement bbox: {e}")
            return None

    def detect_and_process_hand(
        self, frame: np.ndarray, frame_for_bg: np.ndarray, target_size: int
    ) -> Optional[Dict[str, Any]]:
        """
        Main function to detect, track, and process the hand.
        
        Args:
            frame: The current video frame for tracking and processing.
            frame_for_bg: The frame to be used for background subtraction (might be skipped).
            target_size: The size of the output image for the model.

        Returns:
            A dictionary with processed data, or None if no hand is detected.
        """
        status = "UNKNOWN"
        bbox = None
        
        # If the tracker is not tracking, try to re-initialize it
        if not self.tracker.is_tracking:
            # Get foreground mask for hand detection
            try:
                # Use the background remover's remove_background method to get foreground
                _, foreground_mask = self.bg_remover.remove_background(frame_for_bg)
                if foreground_mask is None or not np.any(foreground_mask):
                    return None
            except Exception as e:
                logger.warning(f"Failed to get foreground mask: {e}")
                return None

            # Find the largest contour, which we assume is the hand
            bbox = self._find_best_hand_contour(foreground_mask)
            if bbox:
                self.tracker.initialize(bbox)
                status = "INITIALIZED"
                logger.info(f"HandTracker initialized at position: ({bbox[0] + bbox[2] // 2:.2f}, {bbox[1] + bbox[3] // 2:.2f})")
            else:
                return None
        else:
            # Update the tracker with prediction and measurement
            try:
                # Get prediction from tracker
                predict_result = self.tracker.predict()
                if predict_result is None or predict_result[0] is None:
                    # Tracker has lost tracking due to patience timeout
                    return None
                
                predicted_bbox, predict_status = predict_result
                bbox = predicted_bbox  # Use prediction as default
                status = predict_status
                
                # Try to get a measurement to improve tracking
                measurement_bbox = self._get_measurement_bbox(frame, predicted_bbox)
                if measurement_bbox:
                    # Update with measurement if available
                    self.tracker.update(measurement_bbox)
                    bbox = measurement_bbox  # Use measurement instead
                    status = "TRACKED"
                else:
                    # No measurement available, but don't reset immediately
                    # Let the tracker handle this with its built-in patience system
                    self.tracker.update(None)  # This will increment the lost counter
                    # Continue using the predicted bbox
                    
            except Exception as e:
                logger.warning(f"Tracker update failed: {e}")
                self.tracker.reset()
                return None

        # Validate bbox before processing
        if bbox is None:
            return None
            
        # --- Data Preparation for Model and Visualization ---
        
        # Main hand processing pipeline for the model
        processed_hand = self._preprocess_hand(frame, bbox, target_size)
        
        # For visualization, we need the raw crop and its coordinates
        crop_result = self._get_squared_crop(frame, bbox)
        if not crop_result:
            hand_crop, crop_coords = None, None
        else:
            hand_crop, crop_coords = crop_result

        # Get skin mask for visualization
        skin_mask_crop = self.get_skin_mask_for_crop(hand_crop) if hand_crop is not None else None
        
        return {
            "processed_hand": processed_hand,
            "hand_crop": hand_crop,
            "crop_coords": crop_coords,
            "skin_mask_crop": skin_mask_crop,
            "bbox": bbox,
            "status": status,
            "confidence": 0.9, # Placeholder confidence
            "landmarks": [], # Placeholder
            "tracking": self.tracker.is_tracking,
        }

    def reset(self):
        """Resets the tracker and detection state."""
        self.tracker.reset()
        logger.info("ASL Hand Detector state has been reset.") 