"""
Unit tests for preprocessing utilities.

These tests verify that image preprocessing functions work correctly by:
1. Testing image resizing and normalization
2. Testing lighting enhancement (CLAHE)
3. Testing blur operations for noise reduction
4. Testing adaptive thresholding for binary masks

Think of preprocessing like preparing ingredients before cooking - these functions
clean up and standardize camera images before hand detection can work on them.
"""
import pytest
import numpy as np
import cv2

from src.asl_cam.preprocess import (
    normalize_frame, enhance_lighting, gaussian_blur, adaptive_threshold_mask
)

class TestPreprocessing:
    """Test cases for preprocessing functions."""
    
    @pytest.fixture
    def sample_frame(self):
        """
        Create a sample BGR frame for testing.
        
        This creates a fake camera frame with colored rectangles
        to test preprocessing operations on.
        """
        # Create a simple test image
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add some structure
        frame[50:150, 100:200] = [100, 150, 200]  # Blue-ish rectangle
        frame[100:200, 150:250] = [200, 100, 50]  # Orange-ish rectangle
        
        return frame
    
    def test_normalize_frame_no_resize_needed(self, sample_frame):
        """
        TEST: Does it handle frames that are already the right size?
        
        WHY: Sometimes the camera frame is already the size we want.
        The function should detect this and return the frame unchanged
        to avoid unnecessary processing.
        
        CHECKS: When target size matches input size, output should be
        identical to input (no resize operation).
        """
        target_size = (320, 240)  # Same as sample_frame
        result = normalize_frame(sample_frame, target_size)
        
        # Should be identical
        assert np.array_equal(result, sample_frame)
        assert result.shape == (240, 320, 3)
    
    def test_normalize_frame_resize_needed(self, sample_frame):
        """
        TEST: Can it resize frames to a larger target size?
        
        WHY: Different cameras produce different resolutions. We need
        to standardize frame sizes for consistent processing. This tests
        upscaling (making images bigger).
        
        CHECKS: Output has the correct target dimensions and same data type.
        """
        target_size = (640, 480)
        result = normalize_frame(sample_frame, target_size)
        
        # Should be resized
        assert result.shape == (480, 640, 3)
        assert result.dtype == sample_frame.dtype
    
    def test_normalize_frame_smaller_size(self, sample_frame):
        """
        TEST: Can it resize frames to a smaller target size?
        
        WHY: Sometimes we want to downscale images for faster processing.
        This tests shrinking images while maintaining aspect ratio.
        
        CHECKS: Output has smaller dimensions but correct format.
        """
        target_size = (160, 120)
        result = normalize_frame(sample_frame, target_size)
        
        assert result.shape == (120, 160, 3)
        assert result.dtype == sample_frame.dtype
    
    def test_enhance_lighting(self, sample_frame):
        """
        TEST: Does CLAHE lighting enhancement improve image contrast?
        
        WHY: Poor lighting conditions (too dark, uneven lighting, shadows)
        can hurt hand detection. CLAHE (Contrast Limited Adaptive Histogram
        Equalization) makes lighting more even across the image.
        
        CHECKS: Output has same size/type but different pixel values
        (enhanced contrast).
        """
        enhanced = enhance_lighting(sample_frame, clip_limit=2.0)
        
        # Should have same shape and type
        assert enhanced.shape == sample_frame.shape
        assert enhanced.dtype == sample_frame.dtype
        
        # Should be different (enhanced)
        assert not np.array_equal(enhanced, sample_frame)
    
    def test_enhance_lighting_different_clip_limit(self, sample_frame):
        """
        TEST: Do different clip_limit values produce different results?
        
        WHY: clip_limit controls how aggressive the enhancement is.
        Lower values = subtle enhancement, higher values = stronger
        enhancement. Different settings should produce different outputs.
        
        CHECKS: Different clip_limit parameters create different enhanced images.
        """
        enhanced1 = enhance_lighting(sample_frame, clip_limit=1.0)
        enhanced2 = enhance_lighting(sample_frame, clip_limit=4.0)
        
        # Different clip limits should produce different results
        assert not np.array_equal(enhanced1, enhanced2)
    
    def test_gaussian_blur_odd_kernel(self, sample_frame):
        """
        TEST: Does Gaussian blur work with proper odd kernel sizes?
        
        WHY: Gaussian blur reduces noise and smooths images. OpenCV
        requires odd kernel sizes (3, 5, 7, etc.). This tests the
        normal case with a valid kernel size.
        
        CHECKS: Output is blurred (different from input) but same size/type.
        """
        blurred = gaussian_blur(sample_frame, kernel_size=5)
        
        # Should have same shape and type
        assert blurred.shape == sample_frame.shape
        assert blurred.dtype == sample_frame.dtype
        
        # Should be different (blurred)
        assert not np.array_equal(blurred, sample_frame)
    
    def test_gaussian_blur_even_kernel_auto_correction(self, sample_frame):
        """
        TEST: Does it handle even kernel sizes gracefully?
        
        WHY: Users might accidentally provide even kernel sizes (6, 8, etc.)
        which OpenCV doesn't accept. The function should auto-correct these
        to the next odd number (6→7, 8→9, etc.).
        
        CHECKS: Even kernel size is automatically corrected and blur works.
        """
        blurred = gaussian_blur(sample_frame, kernel_size=6)  # Even number
        
        # Should still work (kernel_size adjusted to 7)
        assert blurred.shape == sample_frame.shape
        assert blurred.dtype == sample_frame.dtype
    
    def test_gaussian_blur_different_kernel_sizes(self, sample_frame):
        """
        TEST: Do different kernel sizes produce different blur levels?
        
        WHY: Larger kernels = more blur, smaller kernels = less blur.
        We need to verify that kernel size actually affects the result.
        
        CHECKS: Heavy blur (kernel=15) should be more different from
        original than light blur (kernel=3).
        """
        blur_light = gaussian_blur(sample_frame, kernel_size=3)
        blur_heavy = gaussian_blur(sample_frame, kernel_size=15)
        
        # Heavier blur should be more different from original
        diff_light = np.sum(np.abs(sample_frame.astype(float) - blur_light.astype(float)))
        diff_heavy = np.sum(np.abs(sample_frame.astype(float) - blur_heavy.astype(float)))
        
        assert diff_heavy > diff_light
    
    def test_adaptive_threshold_mask(self, sample_frame):
        """
        TEST: Does adaptive thresholding create proper binary masks?
        
        WHY: Adaptive thresholding converts color images to black/white
        masks that highlight important features. This is useful for
        finding edges and shapes in varying lighting conditions.
        
        CHECKS: Output is binary (only 0 and 255 values) and correct size.
        """
        mask = adaptive_threshold_mask(sample_frame, block_size=11, C=2)
        
        # Should be binary (0 or 255)
        assert mask.dtype == np.uint8
        assert mask.shape == sample_frame.shape[:2]  # Grayscale
        unique_values = np.unique(mask)
        assert all(val in [0, 255] for val in unique_values)
    
    def test_adaptive_threshold_different_params(self, sample_frame):
        """
        TEST: Do different parameters produce different threshold results?
        
        WHY: block_size and C parameters control how the thresholding
        adapts to local image regions. Different settings should produce
        different binary masks.
        
        CHECKS: Different parameter combinations create different binary masks.
        """
        mask1 = adaptive_threshold_mask(sample_frame, block_size=11, C=2)
        mask2 = adaptive_threshold_mask(sample_frame, block_size=15, C=5)
        
        # Different parameters should produce different results
        assert not np.array_equal(mask1, mask2)
    
    def test_adaptive_threshold_block_size_odd(self, sample_frame):
        """
        TEST: Does adaptive threshold work with odd block sizes?
        
        WHY: Like Gaussian blur, adaptive threshold requires odd block sizes.
        This verifies that valid odd sizes work correctly.
        
        CHECKS: Odd block size produces valid binary mask.
        """
        # Should work without error
        mask = adaptive_threshold_mask(sample_frame, block_size=13, C=2)
        assert mask.shape == sample_frame.shape[:2]
    
    def test_empty_frame_handling(self):
        """
        TEST: Can preprocessing handle very small or minimal frames?
        
        WHY: Edge cases like tiny images or processing failures should
        not crash the system. All preprocessing functions should handle
        minimal inputs gracefully.
        
        CHECKS: All functions work on a 10x10 pixel image without errors.
        """
        tiny_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # All functions should handle tiny frames gracefully
        normalized = normalize_frame(tiny_frame, (20, 20))
        assert normalized.shape == (20, 20, 3)
        
        enhanced = enhance_lighting(tiny_frame)
        assert enhanced.shape == tiny_frame.shape
        
        blurred = gaussian_blur(tiny_frame, kernel_size=3)
        assert blurred.shape == tiny_frame.shape
        
        mask = adaptive_threshold_mask(tiny_frame)
        assert mask.shape == tiny_frame.shape[:2]
    
    def test_single_channel_frame_conversion(self):
        """
        TEST: Do functions work with grayscale images converted to color?
        
        WHY: Sometimes we might receive grayscale images that need to be
        processed as color images. OpenCV conversion from grayscale to
        BGR should work with all our preprocessing functions.
        
        CHECKS: Grayscale→BGR conversion works with all preprocessing functions.
        """
        gray_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Should work without errors
        enhanced = enhance_lighting(bgr_frame)
        blurred = gaussian_blur(bgr_frame)
        mask = adaptive_threshold_mask(bgr_frame)
        
        assert enhanced.shape == bgr_frame.shape
        assert blurred.shape == bgr_frame.shape
        assert mask.shape == bgr_frame.shape[:2] 