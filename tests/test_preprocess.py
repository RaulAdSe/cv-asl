"""
Unit tests for preprocessing utilities.
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
        """Create a sample BGR frame for testing."""
        # Create a simple test image
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add some structure
        frame[50:150, 100:200] = [100, 150, 200]  # Blue-ish rectangle
        frame[100:200, 150:250] = [200, 100, 50]  # Orange-ish rectangle
        
        return frame
    
    def test_normalize_frame_no_resize_needed(self, sample_frame):
        """Test frame normalization when no resize is needed."""
        target_size = (320, 240)  # Same as sample_frame
        result = normalize_frame(sample_frame, target_size)
        
        # Should be identical
        assert np.array_equal(result, sample_frame)
        assert result.shape == (240, 320, 3)
    
    def test_normalize_frame_resize_needed(self, sample_frame):
        """Test frame normalization when resize is needed."""
        target_size = (640, 480)
        result = normalize_frame(sample_frame, target_size)
        
        # Should be resized
        assert result.shape == (480, 640, 3)
        assert result.dtype == sample_frame.dtype
    
    def test_normalize_frame_smaller_size(self, sample_frame):
        """Test normalizing to smaller size."""
        target_size = (160, 120)
        result = normalize_frame(sample_frame, target_size)
        
        assert result.shape == (120, 160, 3)
        assert result.dtype == sample_frame.dtype
    
    def test_enhance_lighting(self, sample_frame):
        """Test lighting enhancement with CLAHE."""
        enhanced = enhance_lighting(sample_frame, clip_limit=2.0)
        
        # Should have same shape and type
        assert enhanced.shape == sample_frame.shape
        assert enhanced.dtype == sample_frame.dtype
        
        # Should be different (enhanced)
        assert not np.array_equal(enhanced, sample_frame)
    
    def test_enhance_lighting_different_clip_limit(self, sample_frame):
        """Test lighting enhancement with different clip limits."""
        enhanced1 = enhance_lighting(sample_frame, clip_limit=1.0)
        enhanced2 = enhance_lighting(sample_frame, clip_limit=4.0)
        
        # Different clip limits should produce different results
        assert not np.array_equal(enhanced1, enhanced2)
    
    def test_gaussian_blur_odd_kernel(self, sample_frame):
        """Test Gaussian blur with odd kernel size."""
        blurred = gaussian_blur(sample_frame, kernel_size=5)
        
        # Should have same shape and type
        assert blurred.shape == sample_frame.shape
        assert blurred.dtype == sample_frame.dtype
        
        # Should be different (blurred)
        assert not np.array_equal(blurred, sample_frame)
    
    def test_gaussian_blur_even_kernel_auto_correction(self, sample_frame):
        """Test that even kernel sizes are automatically corrected."""
        blurred = gaussian_blur(sample_frame, kernel_size=6)  # Even number
        
        # Should still work (kernel_size adjusted to 7)
        assert blurred.shape == sample_frame.shape
        assert blurred.dtype == sample_frame.dtype
    
    def test_gaussian_blur_different_kernel_sizes(self, sample_frame):
        """Test different levels of blur."""
        blur_light = gaussian_blur(sample_frame, kernel_size=3)
        blur_heavy = gaussian_blur(sample_frame, kernel_size=15)
        
        # Heavier blur should be more different from original
        diff_light = np.sum(np.abs(sample_frame.astype(float) - blur_light.astype(float)))
        diff_heavy = np.sum(np.abs(sample_frame.astype(float) - blur_heavy.astype(float)))
        
        assert diff_heavy > diff_light
    
    def test_adaptive_threshold_mask(self, sample_frame):
        """Test adaptive threshold mask creation."""
        mask = adaptive_threshold_mask(sample_frame, block_size=11, C=2)
        
        # Should be binary (0 or 255)
        assert mask.dtype == np.uint8
        assert mask.shape == sample_frame.shape[:2]  # Grayscale
        unique_values = np.unique(mask)
        assert all(val in [0, 255] for val in unique_values)
    
    def test_adaptive_threshold_different_params(self, sample_frame):
        """Test adaptive threshold with different parameters."""
        mask1 = adaptive_threshold_mask(sample_frame, block_size=11, C=2)
        mask2 = adaptive_threshold_mask(sample_frame, block_size=15, C=5)
        
        # Different parameters should produce different results
        assert not np.array_equal(mask1, mask2)
    
    def test_adaptive_threshold_block_size_odd(self, sample_frame):
        """Test that adaptive threshold works with odd block sizes."""
        # Should work without error
        mask = adaptive_threshold_mask(sample_frame, block_size=13, C=2)
        assert mask.shape == sample_frame.shape[:2]
    
    def test_empty_frame_handling(self):
        """Test preprocessing functions with empty/minimal frames."""
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
        """Test that functions work with grayscale input converted to BGR."""
        gray_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Should work without errors
        enhanced = enhance_lighting(bgr_frame)
        blurred = gaussian_blur(bgr_frame)
        mask = adaptive_threshold_mask(bgr_frame)
        
        assert enhanced.shape == bgr_frame.shape
        assert blurred.shape == bgr_frame.shape
        assert mask.shape == bgr_frame.shape[:2] 