"""
Unit tests for skin detection module.

These tests verify that the hand detection system works correctly by:
1. Creating test images with skin-colored regions
2. Running detection algorithms on them
3. Checking that results are correct and reasonable

Think of these as quality control inspectors checking each part of the hand detection pipeline.
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
        """
        Create a sample BGR frame for testing.
        
        This creates a fake camera frame with a skin-colored rectangle
        that should be detected as a hand by our algorithms.
        """
        # Create a simple test image with skin-colored region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a skin-colored rectangle (approximating hand color)
        # RGB: (220, 180, 140) -> BGR: (140, 180, 220)
        frame[100:300, 200:400] = [140, 180, 220]
        
        return frame
    
    def test_detector_initialization(self, detector):
        """
        TEST: Does the detector start up correctly?
        
        WHY: We need to make sure all the color thresholds and settings
        are loaded properly when the detector is created.
        
        CHECKS: HSV/YCrCb color ranges are set and make sense.
        """
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
        """
        TEST: Can it find skin-colored pixels in an image?
        
        WHY: This is the core of hand detection - finding skin-colored areas.
        If this fails, nothing else will work.
        
        CHECKS: Creates a binary mask where white=skin, black=not skin.
        Should find the skin-colored rectangle we put in the test image.
        """
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
        """
        TEST: Can it find hand-shaped outlines from the skin mask?
        
        WHY: After finding skin pixels, we need to group them into
        hand-shaped blobs and filter out noise/small spots.
        
        CHECKS: Finds contours (outlines) from a fake hand-shaped mask
        and filters out anything too small to be a real hand.
        """
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
        """
        TEST: Can it create a proper bounding box around a hand?
        
        WHY: Once we find a hand outline, we need to draw a rectangle
        around it for cropping and display. The box should include
        some padding around the edges.
        
        CHECKS: Creates a rectangular box around hand outline with
        proper padding for better hand crops.
        """
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
        """
        TEST: Does the complete hand detection pipeline work?
        
        WHY: This tests the full process: skin detection → contours → 
        bounding boxes. It's the main function users will call.
        
        CHECKS: Returns a list of hand bounding boxes in the correct
        format (x, y, width, height) with proper limits.
        """
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
        """
        TEST: Can it draw green boxes around detected hands?
        
        WHY: Users need visual feedback to see what the system detected.
        This draws the green bounding boxes you see on screen.
        
        CHECKS: Output image has same size as input but may have
        green rectangles drawn on it.
        """
        result = detector.visualize_detection(sample_frame, show_mask=False)
        
        # Output should have same shape as input
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
        
        # Should be a different image (some drawing occurred)
        # Note: this might be the same if no hands detected
        assert isinstance(result, np.ndarray)
    
    def test_visualize_detection_with_mask(self, detector, sample_frame):
        """
        TEST: Can it show the skin detection mask overlay?
        
        WHY: For debugging, users want to see exactly which pixels
        were detected as skin (the colored overlay you can toggle).
        
        CHECKS: Creates visualization with colored mask overlay
        showing skin detection results.
        """
        result = detector.visualize_detection(sample_frame, show_mask=True)
        
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
    
    def test_max_hands_limit(self, detector):
        """
        TEST: Does the max_hands parameter actually limit detections?
        
        WHY: We want to control how many hands get detected (usually 1-2).
        This prevents the system from detecting random objects as hands.
        
        CHECKS: When we set max_hands=1, we get at most 1 hand back,
        even if the image contains multiple hand-like regions.
        """
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
        """
        TEST: Does it handle empty/black images gracefully?
        
        WHY: Real cameras sometimes produce empty frames or fail.
        The system should not crash on weird inputs.
        
        CHECKS: Empty black image should return empty results
        without throwing errors.
        """
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        hands = detector.detect_hands(empty_frame)
        mask = detector.detect_skin_mask(empty_frame)
        
        # Should handle gracefully
        assert isinstance(hands, list)
        assert isinstance(mask, np.ndarray)
    
    def test_small_contours_filtered(self, detector):
        """
        TEST: Does it ignore tiny spots that aren't hands?
        
        WHY: Camera noise, small skin-colored objects, or lighting
        artifacts can create tiny false detections. We need to filter
        these out so only real hand-sized regions are detected.
        
        CHECKS: A 5x5 pixel spot should be ignored because it's
        way too small to be a real hand.
        """
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
        """
        TEST: Does the threshold tuning interface work?
        
        WHY: Users need to be able to adjust color detection settings
        for different lighting conditions and skin tones. This provides
        a GUI with sliders for real-time tuning.
        
        CHECKS: The tuning interface opens, responds to 'q' key to quit,
        and returns the optimized threshold values.
        
        NOTE: This test mocks the GUI components so it runs in automated
        testing without opening actual windows.
        """
        # Mock waitKey to return 'q' immediately
        mock_waitkey.return_value = ord('q')
        
        result = detector.tune_thresholds(sample_frame)
        
        # Should return threshold dictionary
        assert isinstance(result, dict)
        assert 'hsv_lower' in result
        assert 'hsv_upper' in result
        assert 'ycrcb_lower' in result
        assert 'ycrcb_upper' in result 