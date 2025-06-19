"""
Unit tests for skin detection module.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.asl_cam.vision.skin import SkinDetector

class TestSkinDetector:
    """Test cases for SkinDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a SkinDetector instance for testing."""
        return SkinDetector()
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample BGR frame for testing."""
        # Create a simple test image with skin-colored region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a skin-colored rectangle (approximating hand color)
        # RGB: (220, 180, 140) -> BGR: (140, 180, 220)
        frame[100:300, 200:400] = [140, 180, 220]
        
        return frame
    
    def test_detector_initialization(self, detector):
        """Test that detector initializes with correct default values."""
        assert detector.hsv_lower is not None
        assert detector.hsv_upper is not None
        assert detector.ycrcb_lower is not None
        assert detector.ycrcb_upper is not None
        assert detector.kernel is not None
        
        # Check threshold ranges are sensible
        assert len(detector.hsv_lower) == 3
        assert len(detector.hsv_upper) == 3
        assert all(detector.hsv_lower <= detector.hsv_upper)
    
    def test_detect_skin_mask(self, detector, sample_frame):
        """Test skin mask detection."""
        mask = detector.detect_skin_mask(sample_frame)
        
        # Check output properties
        assert mask.dtype == np.uint8
        assert mask.shape == sample_frame.shape[:2]
        assert np.min(mask) >= 0
        assert np.max(mask) <= 255
        
        # Should detect some skin pixels in our test image
        skin_pixels = np.sum(mask > 0)
        assert skin_pixels > 0
    
    def test_find_hand_contours(self, detector):
        """Test contour finding from mask."""
        # Create a simple binary mask with a rectangle
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:400] = 255
        
        contours = detector.find_hand_contours(mask, min_area=1000)
        
        # Should find at least one contour
        assert len(contours) > 0
        
        # Largest contour should be reasonable size
        largest_area = cv2.contourArea(contours[0])
        assert largest_area >= 1000
    
    def test_get_hand_bbox(self, detector):
        """Test bounding box extraction."""
        # Create a simple contour (rectangle)
        contour = np.array([
            [[200, 100]], [[400, 100]], 
            [[400, 300]], [[200, 300]]
        ])
        
        bbox = detector.get_hand_bbox(contour, padding=10)
        x, y, w, h = bbox
        
        # Check bbox is reasonable
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        
        # Should include padding
        assert x <= 200 - 10  # x should be reduced by padding
        assert y <= 100 - 10  # y should be reduced by padding
    
    def test_detect_hands(self, detector, sample_frame):
        """Test complete hand detection pipeline."""
        hands = detector.detect_hands(sample_frame, max_hands=1)
        
        # Should return a list
        assert isinstance(hands, list)
        assert len(hands) <= 1
        
        # If hands detected, check format
        if hands:
            x, y, w, h = hands[0]
            assert isinstance(x, int) and x >= 0
            assert isinstance(y, int) and y >= 0
            assert isinstance(w, int) and w > 0
            assert isinstance(h, int) and h > 0
    
    def test_visualize_detection(self, detector, sample_frame):
        """Test detection visualization."""
        result = detector.visualize_detection(sample_frame, show_mask=False)
        
        # Output should have same shape as input
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
        
        # Should be a different image (some drawing occurred)
        # Note: this might be the same if no hands detected
        assert isinstance(result, np.ndarray)
    
    def test_visualize_detection_with_mask(self, detector, sample_frame):
        """Test detection visualization with mask overlay."""
        result = detector.visualize_detection(sample_frame, show_mask=True)
        
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
    
    def test_max_hands_limit(self, detector):
        """Test that max_hands parameter limits output."""
        # Create frame with multiple skin regions
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add multiple skin-colored rectangles
        frame[50:150, 100:200] = [140, 180, 220]
        frame[300:400, 400:500] = [140, 180, 220]
        
        hands_1 = detector.detect_hands(frame, max_hands=1)
        hands_2 = detector.detect_hands(frame, max_hands=2)
        
        assert len(hands_1) <= 1
        assert len(hands_2) <= 2
    
    def test_empty_frame(self, detector):
        """Test behavior with empty frame."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        hands = detector.detect_hands(empty_frame)
        mask = detector.detect_skin_mask(empty_frame)
        
        # Should handle gracefully
        assert isinstance(hands, list)
        assert isinstance(mask, np.ndarray)
    
    def test_small_contours_filtered(self, detector):
        """Test that small contours are filtered out."""
        # Create mask with very small regions
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:105, 200:205] = 255  # 5x5 pixel region
        
        contours = detector.find_hand_contours(mask, min_area=1000)
        
        # Should be empty due to min_area filter
        assert len(contours) == 0
    
    @patch('cv2.namedWindow')
    @patch('cv2.createTrackbar')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_tune_thresholds_exit_immediately(self, mock_destroy, mock_waitkey, 
                                            mock_trackbar, mock_window, 
                                            detector, sample_frame):
        """Test threshold tuning interface exits on 'q' key."""
        # Mock waitKey to return 'q' immediately
        mock_waitkey.return_value = ord('q')
        
        result = detector.tune_thresholds(sample_frame)
        
        # Should return threshold dictionary
        assert isinstance(result, dict)
        assert 'hsv_lower' in result
        assert 'hsv_upper' in result
        assert 'ycrcb_lower' in result
        assert 'ycrcb_upper' in result 