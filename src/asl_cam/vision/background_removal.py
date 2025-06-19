"""
Background removal and hand isolation for detected hand regions.

This module provides various techniques to remove background clutter
and isolate hand regions for cleaner machine learning training data.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum

class BackgroundMethod(Enum):
    """Available background removal methods."""
    GRABCUT = "grabcut"           # GrabCut algorithm (best quality)
    CONTOUR_MASK = "contour"      # Use hand contour as mask
    SKIN_MASK = "skin"            # Use skin detection mask
    MOG2 = "mog2"                 # Background subtractor
    WATERSHED = "watershed"       # Watershed segmentation

class BackgroundRemover:
    """Background removal and hand isolation system."""
    
    def __init__(self, method: BackgroundMethod = BackgroundMethod.GRABCUT):
        """
        Initialize background remover.
        
        Args:
            method: Background removal method to use
        """
        self.method = method
        
        # Initialize method-specific components
        if method == BackgroundMethod.MOG2:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False,
                varThreshold=50,
                history=500
            )
        
        # GrabCut parameters
        self.grabcut_iterations = 5
        self.grabcut_margin = 10  # Pixels around hand bbox for probable foreground
        
        # Morphological operations kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def remove_background(self, image: np.ndarray, 
                         hand_bbox: Tuple[int, int, int, int],
                         hand_contour: Optional[np.ndarray] = None,
                         skin_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove background from hand region using selected method.
        
        Args:
            image: Input BGR image
            hand_bbox: Hand bounding box (x, y, w, h)
            hand_contour: Optional hand contour for contour-based methods
            skin_mask: Optional skin detection mask
            
        Returns:
            Tuple of (foreground_image, mask) where:
            - foreground_image: Hand with background removed (transparent or black)
            - mask: Binary mask showing hand region (255=hand, 0=background)
        """
        if self.method == BackgroundMethod.GRABCUT:
            return self._grabcut_removal(image, hand_bbox)
        elif self.method == BackgroundMethod.CONTOUR_MASK:
            return self._contour_mask_removal(image, hand_bbox, hand_contour)
        elif self.method == BackgroundMethod.SKIN_MASK:
            return self._skin_mask_removal(image, hand_bbox, skin_mask)
        elif self.method == BackgroundMethod.MOG2:
            return self._mog2_removal(image, hand_bbox)
        elif self.method == BackgroundMethod.WATERSHED:
            return self._watershed_removal(image, hand_bbox)
        else:
            raise ValueError(f"Unknown background removal method: {self.method}")
    
    def _grabcut_removal(self, image: np.ndarray, 
                        hand_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        GrabCut-based background removal (highest quality).
        
        GrabCut is an iterative algorithm that learns foreground/background
        from user-provided rectangle and iteratively refines the segmentation.
        """
        x, y, w, h = hand_bbox
        
        # Create initial mask for GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Define probable foreground (smaller rect inside hand)
        margin = self.grabcut_margin
        fg_x = max(0, x + margin)
        fg_y = max(0, y + margin)
        fg_w = max(1, w - 2 * margin)
        fg_h = max(1, h - 2 * margin)
        
        # Set probable foreground
        mask[fg_y:fg_y+fg_h, fg_x:fg_x+fg_w] = cv2.GC_PR_FGD
        
        # Set definite background (outside bbox)
        mask[:y, :] = cv2.GC_BGD
        mask[y+h:, :] = cv2.GC_BGD
        mask[:, :x] = cv2.GC_BGD
        mask[:, x+w:] = cv2.GC_BGD
        
        # Set probable background (inside bbox but outside probable foreground)
        mask[y:y+h, x:x+w][mask[y:y+h, x:x+w] == 0] = cv2.GC_PR_BGD
        
        # Initialize GrabCut models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 
                   self.grabcut_iterations, cv2.GC_INIT_WITH_MASK)
        
        # Create final mask (keep only definite and probable foreground)
        final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Clean up mask with morphological operations
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Apply mask to image
        foreground = image.copy()
        foreground[final_mask == 0] = [0, 0, 0]  # Set background to black
        
        return foreground, final_mask
    
    def _contour_mask_removal(self, image: np.ndarray, 
                             hand_bbox: Tuple[int, int, int, int],
                             hand_contour: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Contour-based background removal.
        
        Uses the hand contour detected by skin detection to create a precise mask.
        """
        if hand_contour is None:
            # Fallback to simple rectangle if no contour provided
            return self._simple_rect_removal(image, hand_bbox)
        
        # Create mask from contour
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [hand_contour], 255)
        
        # Clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Apply mask to image
        foreground = image.copy()
        foreground[mask == 0] = [0, 0, 0]
        
        return foreground, mask
    
    def _skin_mask_removal(self, image: np.ndarray, 
                          hand_bbox: Tuple[int, int, int, int],
                          skin_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Skin detection mask-based background removal.
        
        Uses the skin detection mask to isolate hand regions.
        """
        if skin_mask is None:
            # Fallback to simple rectangle if no skin mask provided
            return self._simple_rect_removal(image, hand_bbox)
        
        x, y, w, h = hand_bbox
        
        # Extract skin mask for hand region
        hand_skin_mask = skin_mask[y:y+h, x:x+w]
        
        # Clean up the mask
        hand_skin_mask = cv2.morphologyEx(hand_skin_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        hand_skin_mask = cv2.morphologyEx(hand_skin_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Create full-size mask
        full_mask = np.zeros(image.shape[:2], np.uint8)
        full_mask[y:y+h, x:x+w] = hand_skin_mask
        
        # Apply mask to image
        foreground = image.copy()
        foreground[full_mask == 0] = [0, 0, 0]
        
        return foreground, full_mask
    
    def _mog2_removal(self, image: np.ndarray, 
                     hand_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        MOG2 background subtractor-based removal.
        
        Uses background subtraction to identify moving foreground objects (hands).
        Note: Requires multiple frames to build background model.
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        
        x, y, w, h = hand_bbox
        
        # Extract foreground mask for hand region
        hand_fg_mask = fg_mask[y:y+h, x:x+w]
        
        # Clean up the mask
        hand_fg_mask = cv2.morphologyEx(hand_fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        hand_fg_mask = cv2.morphologyEx(hand_fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Create full-size mask
        full_mask = np.zeros(image.shape[:2], np.uint8)
        full_mask[y:y+h, x:x+w] = hand_fg_mask
        
        # Apply mask to image
        foreground = image.copy()
        foreground[full_mask == 0] = [0, 0, 0]
        
        return foreground, full_mask
    
    def _watershed_removal(self, image: np.ndarray, 
                          hand_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Watershed algorithm-based background removal.
        
        Uses watershed segmentation to separate hand from background.
        """
        x, y, w, h = hand_bbox
        
        # Extract hand region
        hand_region = image[y:y+h, x:x+w].copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(hand_region, markers)
        
        # Create mask (watershed boundaries are marked as -1)
        hand_mask = np.where(markers > 1, 255, 0).astype(np.uint8)
        
        # Clean up mask
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Create full-size mask
        full_mask = np.zeros(image.shape[:2], np.uint8)
        full_mask[y:y+h, x:x+w] = hand_mask
        
        # Apply mask to image
        foreground = image.copy()
        foreground[full_mask == 0] = [0, 0, 0]
        
        return foreground, full_mask
    
    def _simple_rect_removal(self, image: np.ndarray, 
                            hand_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple rectangular mask removal (fallback method).
        """
        x, y, w, h = hand_bbox
        
        # Create rectangular mask
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[y:y+h, x:x+w] = 255
        
        # Apply mask to image
        foreground = image.copy()
        foreground[mask == 0] = [0, 0, 0]
        
        return foreground, mask
    
    def create_transparent_background(self, image: np.ndarray, 
                                    mask: np.ndarray) -> np.ndarray:
        """
        Create image with transparent background instead of black.
        
        Args:
            image: Input BGR image
            mask: Binary mask (255=keep, 0=transparent)
            
        Returns:
            BGRA image with transparent background
        """
        # Convert BGR to BGRA
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on mask
        bgra[:, :, 3] = mask
        
        return bgra
    
    def extract_hand_crop(self, image: np.ndarray, 
                         hand_bbox: Tuple[int, int, int, int],
                         mask: np.ndarray,
                         padding: int = 10) -> np.ndarray:
        """
        Extract clean hand crop with background removed.
        
        Args:
            image: Background-removed image
            hand_bbox: Hand bounding box
            mask: Hand mask
            padding: Extra padding around the hand
            
        Returns:
            Cropped hand image with clean background
        """
        x, y, w, h = hand_bbox
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Extract crop
        hand_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        
        # Ensure clean background in crop
        hand_crop[mask_crop == 0] = [0, 0, 0]
        
        return hand_crop
    
    def visualize_removal_process(self, image: np.ndarray,
                                 hand_bbox: Tuple[int, int, int, int],
                                 foreground: np.ndarray,
                                 mask: np.ndarray) -> np.ndarray:
        """
        Create visualization showing the background removal process.
        
        Args:
            image: Original image
            hand_bbox: Hand bounding box
            foreground: Background-removed image
            mask: Binary mask
            
        Returns:
            Composite visualization image
        """
        x, y, w, h = hand_bbox
        
        # Create visualization panels
        height, width = image.shape[:2]
        vis_height = height
        vis_width = width * 3  # Three panels side by side
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Panel 1: Original with bounding box
        original_with_bbox = image.copy()
        cv2.rectangle(original_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(original_with_bbox, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        visualization[:, 0:width] = original_with_bbox
        
        # Panel 2: Mask visualization
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        cv2.putText(mask_colored, f"Mask ({self.method.value})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        visualization[:, width:2*width] = mask_colored
        
        # Panel 3: Result
        result_with_label = foreground.copy()
        cv2.putText(result_with_label, "Background Removed", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        visualization[:, 2*width:3*width] = result_with_label
        
        return visualization

def compare_methods(image: np.ndarray, 
                   hand_bbox: Tuple[int, int, int, int],
                   hand_contour: Optional[np.ndarray] = None,
                   skin_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compare different background removal methods side by side.
    
    Args:
        image: Input image
        hand_bbox: Hand bounding box
        hand_contour: Optional hand contour
        skin_mask: Optional skin mask
        
    Returns:
        Comparison visualization showing all methods
    """
    methods = [BackgroundMethod.GRABCUT, BackgroundMethod.CONTOUR_MASK, 
               BackgroundMethod.SKIN_MASK, BackgroundMethod.WATERSHED]
    
    results = []
    
    for method in methods:
        try:
            remover = BackgroundRemover(method)
            foreground, mask = remover.remove_background(
                image, hand_bbox, hand_contour, skin_mask
            )
            
            # Add method label
            labeled_result = foreground.copy()
            cv2.putText(labeled_result, method.value.upper(), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            results.append(labeled_result)
            
        except Exception as e:
            # Create error panel
            error_panel = np.zeros_like(image)
            cv2.putText(error_panel, f"{method.value}: ERROR", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            results.append(error_panel)
    
    # Create grid layout (2x2)
    height, width = image.shape[:2]
    
    # Resize results for grid
    results_resized = []
    for result in results:
        resized = cv2.resize(result, (width//2, height//2))
        results_resized.append(resized)
    
    # Create comparison grid
    top_row = np.hstack([results_resized[0], results_resized[1]])
    bottom_row = np.hstack([results_resized[2], results_resized[3]])
    comparison = np.vstack([top_row, bottom_row])
    
    return comparison 