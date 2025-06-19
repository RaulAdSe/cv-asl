"""Tests for the capture module."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from asl_cam.capture import setup_camera, process_frame

def test_setup_camera_failure():
    """Test camera setup with invalid device."""
    with pytest.raises(RuntimeError):
        setup_camera(999)  # Non-existent device

def test_process_frame():
    """Test frame processing."""
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[240, 320] = [255, 255, 255]  # White pixel in center
    
    # Process frame (should flip horizontally)
    processed = process_frame(frame)
    
    # Check dimensions unchanged
    assert processed.shape == frame.shape
    
    # Check pixel was flipped horizontally
    assert np.array_equal(processed[240, 319], [255, 255, 255])

@patch('cv2.VideoCapture')
def test_setup_camera_success(mock_cv2_VideoCapture):
    """Test successful camera setup."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2_VideoCapture.return_value = mock_cap
    
    cap = setup_camera(0)
    assert cap.isOpened()
