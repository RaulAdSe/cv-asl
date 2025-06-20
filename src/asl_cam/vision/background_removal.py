"""
Simple and effective background removal for detected hand regions.

This module uses a straightforward approach: since hand detection is already working well,
the cropped region is mostly hand/arm. We can use color analysis to remove background
pixels that don't match the dominant skin tones.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from enum import Enum

class SimpleBackgroundMethod(Enum):
    """Simple background removal methods."""
    DOMINANT_COLOR = "dominant"      # Use dominant colors in crop (fastest, most reliable)
    ADAPTIVE_SKIN = "adaptive"       # Adaptive skin detection based on crop
    EDGE_COLOR = "edge"              # Color-based with edge refinement

class SimpleBackgroundRemover:
    """
    Simple, effective background removal system.
    
    Philosophy: Trust the hand detection - the crop is mostly hand/arm.
    Remove pixels that don't match the dominant skin colors.
    """
    
    def __init__(self, method: SimpleBackgroundMethod = SimpleBackgroundMethod.DOMINANT_COLOR):
        """
        Initialize simple background remover.
        
        Args:
            method: Background removal method to use
        """
        self.method = method
        
        # Simple parameters
        self.color_tolerance = 40        # How strict color matching is (lower = stricter)
        self.edge_blur = 3              # Edge smoothing
        self.noise_removal = True       # Remove small noise spots
        
        # Morphological kernel for cleaning
        self.cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def remove_background_simple(self, hand_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple background removal from hand crop.
        
        Args:
            hand_crop: Cropped hand image (already detected as hand)
            
        Returns:
            Tuple of (clean_hand_image, mask)
        """
        if hand_crop.size == 0 or len(hand_crop.shape) != 3:
            # Return black image if invalid input
            h, w = hand_crop.shape[:2] if hand_crop.size > 0 else (100, 100)
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        
        if self.method == SimpleBackgroundMethod.DOMINANT_COLOR:
            return self._dominant_color_removal(hand_crop)
        elif self.method == SimpleBackgroundMethod.ADAPTIVE_SKIN:
            return self._adaptive_skin_removal(hand_crop)
        elif self.method == SimpleBackgroundMethod.EDGE_COLOR:
            return self._edge_color_removal(hand_crop)
        else:
            return self._dominant_color_removal(hand_crop)  # Default fallback
    
    def _dominant_color_removal(self, hand_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove background using dominant color analysis.
        
        Strategy:
        1. Find the dominant colors in the center region (likely hand)
        2. Keep pixels that match these colors within tolerance
        3. Remove everything else as background
        """
        h, w = hand_crop.shape[:2]
        
        # Analyze center region to get dominant hand colors
        center_region = self._get_center_region(hand_crop)
        hand_colors = self._get_dominant_colors(center_region, num_colors=3)
        
        # Create mask for pixels matching hand colors
        mask = self._create_color_mask(hand_crop, hand_colors, self.color_tolerance)
        
        # Clean up the mask
        if self.noise_removal:
            mask = self._clean_mask(mask)
        
        # Apply mask with soft edges
        result = self._apply_mask_with_soft_edges(hand_crop, mask)
        
        return result, mask
    
    def _adaptive_skin_removal(self, hand_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive skin detection based on the specific crop.
        
        Strategy:
        1. Sample skin colors from the crop itself
        2. Create adaptive HSV ranges based on these samples
        3. Apply skin detection with crop-specific parameters
        """
        h, w = hand_crop.shape[:2]
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
        
        # Sample multiple regions to get skin color range
        skin_samples = self._sample_skin_colors(hand_crop)
        
        if len(skin_samples) == 0:
            # Fallback to dominant color method
            return self._dominant_color_removal(hand_crop)
        
        # Create adaptive HSV ranges
        hsv_ranges = self._create_adaptive_hsv_ranges(skin_samples)
        
        # Apply skin detection
        mask = np.zeros((h, w), dtype=np.uint8)
        for hsv_range in hsv_ranges:
            range_mask = cv2.inRange(hsv, hsv_range['lower'], hsv_range['upper'])
            mask = cv2.bitwise_or(mask, range_mask)
        
        # Clean up mask
        if self.noise_removal:
            mask = self._clean_mask(mask)
        
        # Apply mask
        result = self._apply_mask_with_soft_edges(hand_crop, mask)
        
        return result, mask
    
    def _edge_color_removal(self, hand_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Color-based removal with edge refinement.
        
        Strategy:
        1. Use dominant color method as base
        2. Refine edges using gradient information
        3. Smooth transitions for natural look
        """
        # Start with dominant color method
        base_result, base_mask = self._dominant_color_removal(hand_crop)
        
        # Find edges in the original image
        gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges slightly to create transition zones
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
        
        # Create refined mask that respects edges
        refined_mask = base_mask.copy()
        
        # In edge regions, be more conservative (keep more pixels)
        edge_regions = edges_dilated > 0
        refined_mask[edge_regions] = cv2.bitwise_or(
            refined_mask[edge_regions], 
            (base_mask[edge_regions] > 127).astype(np.uint8) * 255
        )
        
        # Apply with extra smoothing
        result = self._apply_mask_with_soft_edges(hand_crop, refined_mask, extra_blur=True)
        
        return result, refined_mask
    
    def _get_center_region(self, image: np.ndarray, center_ratio: float = 0.6) -> np.ndarray:
        """Extract center region of image (likely to be hand)."""
        h, w = image.shape[:2]
        
        center_h = int(h * center_ratio)
        center_w = int(w * center_ratio)
        
        start_y = (h - center_h) // 2
        start_x = (w - center_w) // 2
        
        return image[start_y:start_y + center_h, start_x:start_x + center_w]
    
    def _get_dominant_colors(self, image: np.ndarray, num_colors: int = 3) -> np.ndarray:
        """
        Get dominant colors using K-means clustering.
        
        Returns:
            Array of dominant colors in BGR format
        """
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Apply K-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Return centers as uint8
        return centers.astype(np.uint8)
    
    def _create_color_mask(self, image: np.ndarray, target_colors: np.ndarray, tolerance: int) -> np.ndarray:
        """
        Create mask for pixels matching target colors within tolerance.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for color in target_colors:
            # Create range around each target color
            lower = np.maximum(color - tolerance, 0)
            upper = np.minimum(color + tolerance, 255)
            
            # Create mask for this color range
            color_mask = cv2.inRange(image, lower, upper)
            
            # Add to overall mask
            mask = cv2.bitwise_or(mask, color_mask)
        
        return mask
    
    def _sample_skin_colors(self, image: np.ndarray) -> list:
        """
        Sample skin colors from multiple regions of the image.
        """
        h, w = image.shape[:2]
        samples = []
        
        # Sample from multiple small regions
        sample_regions = [
            (h//4, w//4, h//2, w//2),     # Center
            (h//3, w//3, h//3, w//3),     # Upper center
            (2*h//3, w//3, h//3, w//3),   # Lower center
            (h//3, w//6, h//3, w//3),     # Left center
            (h//3, 2*w//3, h//3, w//6),   # Right center
        ]
        
        for y, x, sh, sw in sample_regions:
            if y + sh < h and x + sw < w:
                region = image[y:y+sh, x:x+sw]
                if region.size > 0:
                    # Get median color from this region
                    median_color = np.median(region.reshape(-1, 3), axis=0)
                    samples.append(median_color.astype(np.uint8))
        
        return samples
    
    def _create_adaptive_hsv_ranges(self, skin_samples: list) -> list:
        """
        Create HSV ranges based on skin samples.
        """
        hsv_ranges = []
        
        for sample in skin_samples:
            # Convert BGR sample to HSV
            sample_bgr = np.uint8([[sample]])
            sample_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)[0][0]
            
            # Create range around this HSV value
            h, s, v = sample_hsv
            
            # Adaptive ranges based on the actual skin color
            h_range = 15 if h > 20 else 10  # Smaller range for typical skin hues
            s_range = 50
            v_range = 60
            
            lower_hsv = np.array([max(0, h - h_range), max(0, s - s_range), max(0, v - v_range)])
            upper_hsv = np.array([min(179, h + h_range), min(255, s + s_range), min(255, v + v_range)])
            
            hsv_ranges.append({'lower': lower_hsv, 'upper': upper_hsv})
        
        return hsv_ranges
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up mask by removing noise and filling holes.
        """
        # Remove small noise
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.cleanup_kernel)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.cleanup_kernel)
        
        # Keep only the largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_component).astype(np.uint8) * 255
        
        return cleaned
    
    def _apply_mask_with_soft_edges(self, image: np.ndarray, mask: np.ndarray, extra_blur: bool = False) -> np.ndarray:
        """
        Apply mask with soft edges for natural look.
        """
        # Create soft mask
        blur_size = self.edge_blur * 2 if extra_blur else self.edge_blur
        if blur_size > 0:
            soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_size*2+1, blur_size*2+1), 0)
            soft_mask = soft_mask / 255.0
        else:
            soft_mask = mask.astype(np.float32) / 255.0
        
        # Apply soft mask to each channel
        result = image.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * soft_mask
        
        return result.astype(np.uint8)


# Compatibility classes for existing code
class BackgroundMethod(Enum):
    """Compatibility enum for existing code."""
    GRABCUT = "dominant"
    CONTOUR_MASK = "adaptive"
    SKIN_MASK = "adaptive"
    MOG2 = "edge"
    WATERSHED = "edge"

class BackgroundRemover:
    """Compatibility class that uses the new simple system."""
    
    def __init__(self, method=None):
        """Initialize with backwards compatibility."""
        if method is None or method == BackgroundMethod.GRABCUT:
            self.method = BackgroundMethod.GRABCUT
            self.remover = SimpleBackgroundRemover(SimpleBackgroundMethod.DOMINANT_COLOR)
        elif method in [BackgroundMethod.CONTOUR_MASK, BackgroundMethod.SKIN_MASK]:
            self.method = method
            self.remover = SimpleBackgroundRemover(SimpleBackgroundMethod.ADAPTIVE_SKIN)
        else:
            self.method = method
            self.remover = SimpleBackgroundRemover(SimpleBackgroundMethod.EDGE_COLOR)
    
    def remove_background_from_crop(self, hand_crop: np.ndarray, 
                                   skin_mask_crop: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced background removal for hand crops (new simple method).
        """
        return self.remover.remove_background_simple(hand_crop)
    
    def remove_background(self, image: np.ndarray, 
                         hand_bbox: Tuple[int, int, int, int],
                         hand_contour: Optional[np.ndarray] = None,
                         skin_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy method - extracts crop and applies simple removal."""
        x, y, w, h = hand_bbox
        if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
            hand_crop = image[y:y+h, x:x+w]
            return self.remover.remove_background_simple(hand_crop)
        else:
            # Return original image if bbox is invalid
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255


# Simple interface functions for easy use
def remove_background_simple(hand_crop: np.ndarray, 
                           method: str = "dominant",
                           tolerance: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple function to remove background from hand crop.
    
    Args:
        hand_crop: Cropped hand image
        method: "dominant", "adaptive", or "edge"
        tolerance: Color tolerance (lower = stricter)
        
    Returns:
        (clean_image, mask)
    """
    method_enum = {
        "dominant": SimpleBackgroundMethod.DOMINANT_COLOR,
        "adaptive": SimpleBackgroundMethod.ADAPTIVE_SKIN,
        "edge": SimpleBackgroundMethod.EDGE_COLOR
    }.get(method, SimpleBackgroundMethod.DOMINANT_COLOR)
    
    remover = SimpleBackgroundRemover(method_enum)
    remover.color_tolerance = tolerance
    
    return remover.remove_background_simple(hand_crop)

def quick_clean_hand(hand_crop: np.ndarray, strict: bool = False) -> np.ndarray:
    """
    Quick one-line function to clean hand crop background.
    
    Args:
        hand_crop: Cropped hand image
        strict: If True, use stricter color matching
        
    Returns:
        Clean hand image with background removed
    """
    tolerance = 25 if strict else 40
    clean_image, _ = remove_background_simple(hand_crop, "dominant", tolerance)
    return clean_image 