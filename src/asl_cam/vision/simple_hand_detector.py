"""
Simple and effective hand detection using motion + skin detection.

The key insight: hands move, torsos don't. Combine motion detection with skin 
detection to filter out static skin regions like torso.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from .skin import SkinDetector


class SimpleHandDetector(SkinDetector):
    """Simple hand detector using motion + skin detection."""
    
    def __init__(self):
        super().__init__()
        
        # Motion detection - slower learning to maintain hand detection longer
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=30, history=500)  # Slower learning, more sensitive
        
        # Simple filtering parameters
        self.min_area = 2000       # Minimum hand area
        self.max_area = 35000      # Maximum hand area (smaller than before)
        self.min_motion_area = 500  # Reduced threshold for motion
        
        # Motion persistence - keep detecting hands even with less motion
        self.motion_history = []
        self.history_size = 10     # Longer history
        self.learning_rate = 0.005  # Very slow background learning
        
        # Previous frame for frame differencing
        self.prev_gray = None
        
        # Hand persistence tracking
        self.last_hand_regions = []
        self.hand_persistence_frames = 0
        self.max_persistence_frames = 30  # Keep detecting for 30 frames without motion
        
    def detect_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion using background subtraction with slow learning.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Binary motion mask
        """
        # Background subtraction with very slow learning rate
        motion_mask = self.background_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Clean up motion mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to make motion regions more persistent
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        return motion_mask
    
    def detect_frame_diff_motion(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion using frame differencing (backup method).
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Binary motion mask
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)
        
        # Frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        self.prev_gray = gray
        return motion_mask
    
    def combine_skin_and_motion(self, skin_mask: np.ndarray, 
                               motion_mask: np.ndarray) -> np.ndarray:
        """
        Combine skin and motion masks with hand persistence.
        
        Args:
            skin_mask: Binary skin detection mask
            motion_mask: Binary motion detection mask
            
        Returns:
            Combined mask showing moving skin regions with persistence
        """
        # Intersection of skin and motion
        combined = cv2.bitwise_and(skin_mask, motion_mask)
        
        # Dilate motion mask to be more forgiving
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_motion = cv2.dilate(motion_mask, kernel)
        
        # Include skin regions near motion
        skin_near_motion = cv2.bitwise_and(skin_mask, dilated_motion)
        
        # Add persistence for previously detected hand regions
        if self.last_hand_regions and self.hand_persistence_frames < self.max_persistence_frames:
            persistence_mask = np.zeros_like(skin_mask)
            for region in self.last_hand_regions:
                x, y, w, h = region
                # Expand the region slightly
                x = max(0, x - 20)
                y = max(0, y - 20)
                w = min(skin_mask.shape[1] - x, w + 40)
                h = min(skin_mask.shape[0] - y, h + 40)
                persistence_mask[y:y+h, x:x+w] = 255
            
            # Include skin in persistent regions
            persistent_skin = cv2.bitwise_and(skin_mask, persistence_mask)
            combined = cv2.bitwise_or(combined, persistent_skin)
        
        # Combine all approaches
        result = cv2.bitwise_or(combined, skin_near_motion)
        
        return result
    
    def filter_hand_contours(self, contours: List[np.ndarray], 
                           motion_mask: np.ndarray) -> List[np.ndarray]:
        """
        Filter contours to keep only hand-like moving regions.
        
        Args:
            contours: List of skin contours
            motion_mask: Motion detection mask
            
        Returns:
            Filtered list of hand contours
        """
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if area < self.min_area or area > self.max_area:
                continue
            
            # Check if contour has significant motion
            mask = np.zeros(motion_mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Count motion pixels in this region
            motion_in_region = cv2.bitwise_and(motion_mask, mask)
            motion_pixels = np.sum(motion_in_region > 0)
            
            # Require minimum motion
            if motion_pixels < self.min_motion_area:
                continue
            
            # Simple shape filtering
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Hands are not extremely thin or wide
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Check if region is reasonably compact
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.4:  # Very loose requirement
                    continue
            
            valid_contours.append(contour)
        
        # Sort by area (larger first, but we've already limited max size)
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        
        return valid_contours
    
    def detect_hands_simple(self, frame: np.ndarray, 
                           max_hands: int = 1,
                           use_motion: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Simple hand detection combining skin and motion with persistence.
        
        Args:
            frame: BGR input frame
            max_hands: Maximum number of hands to detect
            use_motion: Whether to use motion filtering
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # Get skin mask
        skin_mask = self.detect_skin_mask(frame)
        
        if use_motion:
            # Get motion mask
            motion_mask = self.detect_motion_mask(frame)
            
            # Combine skin and motion with persistence
            combined_mask = self.combine_skin_and_motion(skin_mask, motion_mask)
        else:
            combined_mask = skin_mask
            motion_mask = np.ones_like(skin_mask) * 255  # All motion if disabled
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        if use_motion:
            valid_contours = self.filter_hand_contours(contours, motion_mask)
        else:
            # Fallback to basic filtering
            valid_contours = self.find_hand_contours(combined_mask, min_area=self.min_area)
        
        # Convert to bounding boxes
        hands = []
        for contour in valid_contours[:max_hands]:
            bbox = self.get_hand_bbox(contour, padding=10)
            hands.append(bbox)
        
        # Update hand persistence tracking
        if hands:
            # Found hands - reset persistence counter and update regions
            self.hand_persistence_frames = 0
            self.last_hand_regions = hands.copy()
        else:
            # No hands found - increment persistence counter
            self.hand_persistence_frames += 1
            
            # If within persistence window, try to detect in last known regions
            if (self.last_hand_regions and 
                self.hand_persistence_frames < self.max_persistence_frames):
                
                # Check for skin in last known hand regions
                for region in self.last_hand_regions:
                    x, y, w, h = region
                    # Expand region slightly for tolerance
                    x = max(0, x - 10)
                    y = max(0, y - 10)
                    w = min(skin_mask.shape[1] - x, w + 20)
                    h = min(skin_mask.shape[0] - y, h + 20)
                    
                    # Check if there's still skin in this region
                    region_mask = skin_mask[y:y+h, x:x+w]
                    skin_pixels = np.sum(region_mask > 0)
                    
                    if skin_pixels > self.min_area // 4:  # Quarter of minimum area
                        hands.append((x, y, w, h))
                        break  # Only need one persistent hand
        
        return hands
    
    def visualize_simple_detection(self, frame: np.ndarray, 
                                  show_motion: bool = False,
                                  show_skin: bool = False) -> np.ndarray:
        """
        Visualize simple detection with optional mask overlays.
        
        Args:
            frame: Input frame
            show_motion: Whether to show motion mask
            show_skin: Whether to show skin mask
            
        Returns:
            Visualization frame
        """
        result = frame.copy()
        
        # Detect hands
        hands = self.detect_hands_simple(frame)
        
        # Draw hand detections
        for i, (x, y, w, h) in enumerate(hands):
            # Draw main bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"Hand {i+1}"
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Optional mask overlays
        if show_motion:
            motion_mask = self.detect_motion_mask(frame)
            motion_colored = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
            result = cv2.addWeighted(result, 0.7, motion_colored, 0.3, 0)
        
        if show_skin:
            skin_mask = self.detect_skin_mask(frame)
            skin_colored = cv2.applyColorMap(skin_mask, cv2.COLORMAP_SPRING)
            result = cv2.addWeighted(result, 0.7, skin_colored, 0.3, 0)
        
        # Add info text with persistence info
        persistence_info = f"Persist: {self.hand_persistence_frames}/{self.max_persistence_frames}"
        info_text = f"Hands: {len(hands)} | Motion: ON | {persistence_info}"
        cv2.putText(result, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw persistence regions if active
        if (self.last_hand_regions and 
            self.hand_persistence_frames > 0 and 
            self.hand_persistence_frames < self.max_persistence_frames):
            for region in self.last_hand_regions:
                x, y, w, h = region
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 1)  # Yellow for persistence
                cv2.putText(result, "PERSIST", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return result
    
    def reset_motion_detection(self):
        """Reset motion detection background model and persistence."""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=30, history=500)
        self.prev_gray = None
        self.last_hand_regions = []
        self.hand_persistence_frames = 0
        print("Motion detection and hand persistence reset")
    
    # Override parent method
    def detect_hands(self, img_bgr: np.ndarray, max_hands: int = 1) -> List[Tuple[int, int, int, int]]:
        """Override parent method to use simple motion-based detection."""
        return self.detect_hands_simple(img_bgr, max_hands, use_motion=True) 