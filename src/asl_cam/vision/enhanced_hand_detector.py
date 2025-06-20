"""
Enhanced hand detection with improved geometric and contextual filtering.

This module provides advanced hand detection that can distinguish hands from
other skin regions (like torso) through geometric analysis, position filtering,
and shape characteristics.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from .skin import SkinDetector

@dataclass
class HandCandidate:
    """Container for hand detection candidate with quality metrics."""
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    aspect_ratio: float
    solidity: float
    circularity: float
    position_score: float
    size_score: float
    shape_score: float
    total_score: float

class EnhancedHandDetector(SkinDetector):
    """Enhanced hand detector with improved filtering for complex skin scenarios."""
    
    def __init__(self):
        super().__init__()
        
        # Hand-specific size constraints (in pixels)
        self.min_hand_area = 3000      # Minimum area for a hand
        self.max_hand_area = 50000     # Maximum area (filter out torso)
        self.ideal_hand_width = 120    # Ideal hand width
        self.ideal_hand_height = 160   # Ideal hand height
        
        # Geometric thresholds for hand shapes
        self.min_aspect_ratio = 0.4    # Hands are not too thin
        self.max_aspect_ratio = 2.5    # Hands are not too wide
        self.min_solidity = 0.65       # Hands are reasonably solid
        self.min_circularity = 0.15    # Hands have some roundness but not perfect circles
        self.max_circularity = 0.85    # Filter out very circular shapes (heads, etc.)
        
        # Position-based scoring (hands typically appear in these regions)
        self.hand_regions = {
            'center': (0.2, 0.8, 0.2, 0.8),    # (x_min, x_max, y_min, y_max) as ratios
            'upper': (0.1, 0.9, 0.1, 0.6),     # Upper portion where hands often appear
            'sides': (0.0, 1.0, 0.2, 0.8)      # Side regions for gesture capture
        }
        
        # Motion-based filtering (future enhancement)
        self.enable_motion_filtering = True
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    def calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate circularity of contour.
        
        Circularity = 4π * area / perimeter²
        Perfect circle = 1.0, line = 0.0
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return 4 * np.pi * area / (perimeter ** 2)
    
    def calculate_position_score(self, bbox: Tuple[int, int, int, int], 
                               frame_shape: Tuple[int, int]) -> float:
        """
        Calculate position-based score for hand likelihood.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            frame_shape: (height, width) of frame
            
        Returns:
            Score from 0.0 to 1.0 (higher = more likely hand position)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape
        
        # Calculate center point as ratios
        center_x = (x + w/2) / frame_w
        center_y = (y + h/2) / frame_h
        
        # Score based on typical hand positions
        score = 0.0
        
        # Center region bonus (hands often appear in center)
        center_region = self.hand_regions['center']
        if (center_region[0] <= center_x <= center_region[1] and 
            center_region[2] <= center_y <= center_region[3]):
            score += 0.4
        
        # Upper region bonus (hands raised for gestures)
        upper_region = self.hand_regions['upper']
        if (upper_region[0] <= center_x <= upper_region[1] and 
            upper_region[2] <= center_y <= upper_region[3]):
            score += 0.3
        
        # Avoid bottom region (less likely to be hands)
        if center_y > 0.8:
            score -= 0.2
        
        # Penalize very top region (might be head/face)
        if center_y < 0.15:
            score -= 0.3
        
        # Penalize extreme edges
        if center_x < 0.05 or center_x > 0.95:
            score -= 0.1
        
        return max(0.0, min(1.0, score + 0.3))  # Base score + bonuses/penalties
    
    def calculate_size_score(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate size-based score for hand likelihood.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Score from 0.0 to 1.0 (higher = more hand-like size)
        """
        x, y, w, h = bbox
        area = w * h
        
        # Distance from ideal hand size
        size_diff_w = abs(w - self.ideal_hand_width) / self.ideal_hand_width
        size_diff_h = abs(h - self.ideal_hand_height) / self.ideal_hand_height
        
        # Score based on how close to ideal size
        size_score = 1.0 - min(1.0, (size_diff_w + size_diff_h) / 2)
        
        # Penalty for extremely large areas (torso)
        if area > self.max_hand_area:
            size_score *= 0.1
        
        # Penalty for extremely small areas
        if area < self.min_hand_area:
            size_score *= 0.3
        
        return max(0.0, size_score)
    
    def calculate_shape_score(self, contour: np.ndarray, 
                            aspect_ratio: float, 
                            solidity: float, 
                            circularity: float) -> float:
        """
        Calculate shape-based score for hand likelihood.
        
        Args:
            contour: Hand contour
            aspect_ratio: Width/height ratio
            solidity: Area/convex_hull_area ratio
            circularity: Circularity measure
            
        Returns:
            Score from 0.0 to 1.0 (higher = more hand-like shape)
        """
        score = 0.0
        
        # Aspect ratio scoring (hands have characteristic proportions)
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            # Ideal aspect ratios for hands
            if 0.6 <= aspect_ratio <= 1.4:
                score += 0.3
            else:
                score += 0.2
        else:
            score -= 0.2
        
        # Solidity scoring (hands are reasonably solid but not perfect)
        if self.min_solidity <= solidity <= 0.9:
            score += 0.25
        else:
            score -= 0.1
        
        # Circularity scoring (hands are not circles but have some roundness)
        if self.min_circularity <= circularity <= self.max_circularity:
            score += 0.2
        else:
            score -= 0.15
        
        # Convexity defects analysis (hands have finger gaps)
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None and len(defects) > 0:
                # Hands typically have 2-4 significant defects (finger gaps)
                significant_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 2000:  # Significant depth
                        significant_defects += 1
                
                if 1 <= significant_defects <= 5:  # Reasonable number of finger gaps
                    score += 0.25
                elif significant_defects > 8:  # Too many defects (noisy contour)
                    score -= 0.2
        
        return max(0.0, min(1.0, score + 0.4))  # Base score + shape bonuses/penalties
    
    def analyze_hand_candidates(self, contours: List[np.ndarray], 
                              frame_shape: Tuple[int, int]) -> List[HandCandidate]:
        """
        Analyze contours and score them as hand candidates.
        
        Args:
            contours: List of contours to analyze
            frame_shape: (height, width) of frame
            
        Returns:
            List of HandCandidate objects sorted by score
        """
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if area < self.min_hand_area or area > self.max_hand_area:
                continue
            
            # Calculate geometric properties
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            aspect_ratio = float(w) / h
            
            # Solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Circularity
            circularity = self.calculate_circularity(contour)
            
            # Calculate scores
            position_score = self.calculate_position_score(bbox, frame_shape)
            size_score = self.calculate_size_score(bbox)
            shape_score = self.calculate_shape_score(contour, aspect_ratio, solidity, circularity)
            
            # Combined score with weights
            total_score = (
                0.3 * position_score +
                0.3 * size_score +
                0.4 * shape_score
            )
            
            candidate = HandCandidate(
                contour=contour,
                bbox=bbox,
                area=area,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                circularity=circularity,
                position_score=position_score,
                size_score=size_score,
                shape_score=shape_score,
                total_score=total_score
            )
            
            candidates.append(candidate)
        
        # Sort by total score (highest first)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        
        return candidates
    
    def detect_hands_enhanced(self, img_bgr: np.ndarray, 
                            max_hands: int = 1,
                            min_score: float = 0.4) -> List[Tuple[int, int, int, int]]:
        """
        Enhanced hand detection with improved filtering.
        
        Args:
            img_bgr: BGR input image
            max_hands: Maximum number of hands to detect
            min_score: Minimum score threshold for valid hands
            
        Returns:
            List of (x, y, w, h) bounding boxes for detected hands
        """
        # Get skin mask using parent class method
        mask = self.detect_skin_mask(img_bgr)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze candidates
        candidates = self.analyze_hand_candidates(contours, img_bgr.shape[:2])
        
        # Filter by minimum score and return top candidates
        valid_hands = []
        for candidate in candidates[:max_hands]:
            if candidate.total_score >= min_score:
                valid_hands.append(candidate.bbox)
        
        return valid_hands
    
    def visualize_enhanced_detection(self, img_bgr: np.ndarray, 
                                   show_mask: bool = False,
                                   show_scores: bool = True) -> np.ndarray:
        """
        Visualize enhanced hand detection with scoring information.
        
        Args:
            img_bgr: Input image
            show_mask: Whether to show skin mask overlay
            show_scores: Whether to show candidate scores
            
        Returns:
            Image with enhanced detection visualizations
        """
        result = img_bgr.copy()
        
        # Get skin mask and contours
        mask = self.detect_skin_mask(img_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze all candidates
        candidates = self.analyze_hand_candidates(contours, img_bgr.shape[:2])
        
        # Draw all candidates with color-coded scores
        for i, candidate in enumerate(candidates[:5]):  # Show top 5
            x, y, w, h = candidate.bbox
            
            # Color based on score (red=low, yellow=medium, green=high)
            if candidate.total_score >= 0.6:
                color = (0, 255, 0)  # Green - high confidence
                thickness = 3
            elif candidate.total_score >= 0.4:
                color = (0, 255, 255)  # Yellow - medium confidence
                thickness = 2
            else:
                color = (0, 0, 255)  # Red - low confidence
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Add score information
            if show_scores:
                score_text = f"S:{candidate.total_score:.2f}"
                cv2.putText(result, score_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Detailed scores (smaller text)
                details = f"P:{candidate.position_score:.1f} Sz:{candidate.size_score:.1f} Sh:{candidate.shape_score:.1f}"
                cv2.putText(result, details, (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Optionally show mask overlay
        if show_mask:
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            result = cv2.addWeighted(result, 0.75, mask_colored, 0.25, 0)
        
        return result
    
    # Override parent class method to use enhanced detection
    def detect_hands(self, img_bgr: np.ndarray, max_hands: int = 1) -> List[Tuple[int, int, int, int]]:
        """Override parent method to use enhanced detection."""
        return self.detect_hands_enhanced(img_bgr, max_hands) 