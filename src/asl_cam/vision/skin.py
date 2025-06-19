"""
Skin detection and hand segmentation using classical computer vision.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

class SkinDetector:
    """Classical skin detection using color space thresholding."""
    
    def __init__(self):
        # Optimized HSV thresholds (broader range for different skin tones)
        self.hsv_lower = np.array([0, 20, 70])
        self.hsv_upper = np.array([25, 255, 255])
        
        # Optimized YCrCb thresholds (more robust for skin detection)
        self.ycrcb_lower = np.array([0, 135, 85])
        self.ycrcb_upper = np.array([255, 180, 135])
        
        # Larger morphological kernel for better noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def detect_skin_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Create binary mask for skin pixels.
        
        Args:
            img_bgr: BGR image
            
        Returns:
            Binary mask where skin pixels are white (255)
        """
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        
        # Convert to HSV and YCrCb
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        
        # Create masks in both color spaces
        hsv_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        ycrcb_mask = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        
        # Combine masks (intersection for stricter detection)
        combined_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        
        # Enhanced morphological operations
        # Opening to remove noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.small_kernel, iterations=2)
        # Closing to fill holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        # Dilation to make hands more solid
        combined_mask = cv2.dilate(combined_mask, self.kernel, iterations=2)
        
        # Final median blur to smooth edges
        combined_mask = cv2.medianBlur(combined_mask, 7)
        
        return combined_mask
    
    def find_hand_contours(self, mask: np.ndarray, min_area: int = 2000) -> List[np.ndarray]:
        """
        Find hand contours from skin mask.
        
        Args:
            mask: Binary skin mask
            min_area: Minimum contour area to consider (increased for better filtering)
            
        Returns:
            List of contours sorted by area (largest first)
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area, aspect ratio, and convexity for hand-like shapes
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
                
            # Check aspect ratio (hands are roughly rectangular)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Filter out very thin or very wide shapes
                continue
                
            # Check solidity (filled area vs convex hull area)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.6:  # Hands should be reasonably solid
                    continue
            
            valid_contours.append(c)
        
        # Sort by area (largest first)
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        
        return valid_contours
    
    def get_hand_bbox(self, contour: np.ndarray, padding: int = 15) -> Tuple[int, int, int, int]:
        """
        Get bounding box from contour with padding.
        
        Args:
            contour: Hand contour
            padding: Pixels to add around the bounding box (increased for better crops)
            
        Returns:
            (x, y, w, h) bounding box coordinates
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        return (x, y, w, h)
    
    def detect_hands(self, img_bgr: np.ndarray, max_hands: int = 1) -> List[Tuple[int, int, int, int]]:
        """
        Complete hand detection pipeline.
        
        Args:
            img_bgr: BGR input image
            max_hands: Maximum number of hands to detect
            
        Returns:
            List of (x, y, w, h) bounding boxes for detected hands
        """
        # Get skin mask
        mask = self.detect_skin_mask(img_bgr)
        
        # Find contours
        contours = self.find_hand_contours(mask)
        
        # Convert top contours to bounding boxes
        hands = []
        for contour in contours[:max_hands]:
            bbox = self.get_hand_bbox(contour)
            hands.append(bbox)
        
        return hands
    
    def visualize_detection(self, img_bgr: np.ndarray, show_mask: bool = False) -> np.ndarray:
        """
        Visualize hand detection results.
        
        Args:
            img_bgr: Input image
            show_mask: Whether to show skin mask overlay
            
        Returns:
            Image with detection visualizations
        """
        result = img_bgr.copy()
        
        # Detect hands
        hands = self.detect_hands(img_bgr)
        
        # Draw bounding boxes with enhanced visualization
        for i, (x, y, w, h) in enumerate(hands):
            # Draw thick bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add corner markers for better visibility
            corner_size = 10
            corners = [
                (x, y), (x + w, y), (x, y + h), (x + w, y + h)
            ]
            for corner in corners:
                cv2.circle(result, corner, corner_size//2, (0, 255, 0), -1)
            
            # Add label with background
            label = f"Hand {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(result, (x, y - 30), (x + label_size[0] + 10, y), (0, 255, 0), -1)
            cv2.putText(result, label, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Optionally show mask overlay
        if show_mask:
            mask = self.detect_skin_mask(img_bgr)
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            result = cv2.addWeighted(result, 0.75, mask_colored, 0.25, 0)
        
        return result
    
    def tune_thresholds(self, img_bgr: np.ndarray) -> dict:
        """
        Interactive threshold tuning helper.
        
        Args:
            img_bgr: Sample image for tuning
            
        Returns:
            Dictionary with optimal thresholds
        """
        def nothing(x):
            pass
        
        # Create trackbars window
        cv2.namedWindow('Threshold Tuning')
        cv2.createTrackbar('HSV H Min', 'Threshold Tuning', self.hsv_lower[0], 179, nothing)
        cv2.createTrackbar('HSV S Min', 'Threshold Tuning', self.hsv_lower[1], 255, nothing)
        cv2.createTrackbar('HSV V Min', 'Threshold Tuning', self.hsv_lower[2], 255, nothing)
        cv2.createTrackbar('HSV H Max', 'Threshold Tuning', self.hsv_upper[0], 179, nothing)
        cv2.createTrackbar('HSV S Max', 'Threshold Tuning', self.hsv_upper[1], 255, nothing)
        cv2.createTrackbar('HSV V Max', 'Threshold Tuning', self.hsv_upper[2], 255, nothing)
        
        print("Adjust trackbars to tune thresholds. Press 'q' to finish.")
        
        while True:
            # Get current trackbar values
            h_min = cv2.getTrackbarPos('HSV H Min', 'Threshold Tuning')
            s_min = cv2.getTrackbarPos('HSV S Min', 'Threshold Tuning')
            v_min = cv2.getTrackbarPos('HSV V Min', 'Threshold Tuning')
            h_max = cv2.getTrackbarPos('HSV H Max', 'Threshold Tuning')
            s_max = cv2.getTrackbarPos('HSV S Max', 'Threshold Tuning')
            v_max = cv2.getTrackbarPos('HSV V Max', 'Threshold Tuning')
            
            # Update thresholds
            self.hsv_lower = np.array([h_min, s_min, v_min])
            self.hsv_upper = np.array([h_max, s_max, v_max])
            
            # Apply detection
            mask = self.detect_skin_mask(img_bgr)
            result = self.visualize_detection(img_bgr, show_mask=True)
            
            # Show results
            cv2.imshow('Original', img_bgr)
            cv2.imshow('Mask', mask)
            cv2.imshow('Detection', result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        return {
            'hsv_lower': self.hsv_lower.tolist(),
            'hsv_upper': self.hsv_upper.tolist(),
            'ycrcb_lower': self.ycrcb_lower.tolist(),
            'ycrcb_upper': self.ycrcb_upper.tolist()
        } 