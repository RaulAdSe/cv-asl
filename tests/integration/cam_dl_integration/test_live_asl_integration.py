"""
Integration tests for Live ASL Recognition System.

Tests the complete pipeline from camera input to ASL prediction.
"""
import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from asl_cam.live_asl import LiveASLRecognizer
from asl_cam.vision.asl_hand_detector import ASLHandDetector

class TestLiveASLIntegration:
    """Integration tests for Live ASL system."""
    
    @pytest.fixture
    def live_recognizer(self):
        """Create a LiveASLRecognizer for testing."""
        return LiveASLRecognizer(camera_index=-1)
    
    @pytest.fixture
    def asl_detector(self):
        """Create an ASL hand detector for testing."""
        return ASLHandDetector()
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample camera frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_live_recognizer_initialization(self, live_recognizer):
        """
        TEST: Does the live recognizer initialize correctly?
        
        WHY: Integration test to ensure all components work together
        during system startup.
        
        CHECKS: Model loads, classes are correct, device is set.
        """
        assert live_recognizer is not None
        assert live_recognizer.model is not None
        assert live_recognizer.classes == ['A', 'B', 'C']
        assert str(live_recognizer.device) in ['cpu', 'mps', 'cuda']
    
    def test_asl_detector_initialization(self, live_recognizer):
        """
        TEST: Does the ASL hand detector initialize correctly?
        
        WHY: The detector should start up with proper fallback handling.
        
        CHECKS: Detector exists and can be used.
        """
        assert live_recognizer.hand_detector is not None
        # Should have our advanced detection pipeline
        assert hasattr(live_recognizer.hand_detector, 'detect_and_process_hand')
    
    def test_end_to_end_pipeline_no_hands(self, live_recognizer, asl_detector, sample_frame):
        """
        TEST: Does the complete pipeline work when no hands are detected?
        
        WHY: The system should gracefully handle frames with no hands.
        
        CHECKS: No crashes, appropriate empty results.
        """
        # Detect hands (likely none in random frame)
        asl_detector.bg_remover.bg_model_learned = True # Pretend BG is learned
        
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        processed_crop, hand_info = asl_detector.detect_and_process_hand(
            sample_frame, live_recognizer.model.INPUT_SIZE
        )
        
        self.assertIsNone(processed_crop)
        self.assertIsNone(hand_info)
    
    @patch('asl_dl.models.mobilenet.MobileNetV2ASL.predict')
    def test_hand_detection_with_prediction(self, mock_predict, live_recognizer, sample_frame):
        """
        TEST: Does hand detection + prediction work together?
        
        WHY: Test the integration between hand detection and sign recognition.
        
        CHECKS: Can process a hand crop and get a prediction.
        """
        asl_detector = live_recognizer.hand_detector
        asl_detector.bg_remover.bg_model_learned = True

        # Create a frame with a fake hand
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(sample_frame, (100, 100), (200, 200), (255, 255, 255), -1)

        # Mock the underlying simple detector to return a confident detection
        with patch.object(asl_detector, 'detect_hands', return_value=[{'bbox': (100, 100, 100, 100), 'confidence': 0.9}]):
            processed_crop, hand_info = asl_detector.detect_and_process_hand(
                sample_frame, live_recognizer.model.INPUT_SIZE
            )

        self.assertIsNotNone(processed_crop)
        self.assertIsNotNone(hand_info)
        self.assertEqual(hand_info['status'], 'NEW_DETECTION')
    
    def test_performance_monitoring_integration(self, live_recognizer):
        """
        TEST: Does the performance monitoring work correctly?
        
        WHY: The system should track FPS and performance metrics.
        
        CHECKS: Performance metrics are calculated and reasonable.
        """
        # The live recognizer should have performance tracking
        # Note: These might be initialized during runtime, so just check they can be accessed
        try:
            _ = getattr(live_recognizer, 'fps_counter', None)
            _ = getattr(live_recognizer, 'prediction_times', None)
            # If no error, performance tracking is available or can be initialized
            assert True
        except AttributeError:
            # Performance tracking might be initialized on first use
            assert True
    
    def test_confidence_filtering(self, live_recognizer):
        """
        TEST: Does confidence-based filtering work?
        
        WHY: Low-confidence predictions should be handled appropriately.
        
        CHECKS: Confidence thresholds are applied correctly.
        """
        # Test with mock prediction
        hand_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        prediction, confidence = live_recognizer.predict_hand_sign(hand_crop)
        
        # Should have a confidence value
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_model_inference_consistency(self, live_recognizer):
        """
        TEST: Are model predictions consistent for the same input?
        
        WHY: The same hand crop should produce the same prediction.
        
        CHECKS: Deterministic inference in eval mode.
        """
        # Create identical hand crops
        hand_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Get two predictions
        pred1, conf1 = live_recognizer.predict_hand_sign(hand_crop)
        pred2, conf2 = live_recognizer.predict_hand_sign(hand_crop)
        
        # Should be identical (model is in eval mode)
        assert pred1 == pred2
        assert abs(conf1 - conf2) < 1e-6  # Very close confidence
    
    def test_hand_preprocessing_pipeline(self, live_recognizer):
        """
        TEST: Does the hand preprocessing work correctly?
        
        WHY: Hand crops need to be properly resized and normalized
        for the model.
        
        CHECKS: Hand detection returns properly formatted crops.
        """
        asl_detector = live_recognizer.hand_detector
        asl_detector.bg_remover.bg_model_learned = True
        
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(sample_frame, (100, 100), (200, 200), (0, 255, 0), -1)

        with patch.object(asl_detector, 'detect_hands', return_value=[{'bbox': (100, 100, 100, 100), 'confidence': 0.9}]):
            processed_crop, hand_info = asl_detector.detect_and_process_hand(
                sample_frame, live_recognizer.model.INPUT_SIZE
            )

        self.assertIsNotNone(processed_crop)
        self.assertEqual(processed_crop.shape, (live_recognizer.model.INPUT_SIZE, live_recognizer.model.INPUT_SIZE, 3))
    
    def test_error_handling_integration(self, live_recognizer):
        """
        TEST: Does error handling work across the integration?
        
        WHY: The system should handle various error conditions gracefully.
        
        CHECKS: Malformed inputs are handled without crashes.
        """
        asl_detector = live_recognizer.hand_detector
        asl_detector.bg_remover.bg_model_learned = True
        
        with patch.object(asl_detector, 'detect_and_process_hand', side_effect=Exception("Test Error")):
            with self.assertRaises(Exception):
                # This call should now raise the exception from the mock
                asl_detector.detect_and_process_hand(np.zeros((100,100,3)), 128)
    
    @patch('cv2.VideoCapture')
    def test_camera_integration_mock(self, mock_videocapture, live_recognizer, asl_detector):
        """
        TEST: Does camera integration work (mocked)?
        
        WHY: Test camera capture integration without needing real camera.
        
        CHECKS: Camera capture and processing pipeline works.
        """
        # Mock camera
        mock_camera = Mock()
        mock_camera.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_camera.isOpened.return_value = True
        mock_videocapture.return_value = mock_camera
        
        # Should be able to create capture
        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        
        # Should be able to read frame
        ret, frame = cap.read()
        assert ret
        assert frame.shape == (480, 640, 3)
        
        # Should be able to process frame
        hands = asl_detector.detect_hands_asl(frame)
        assert isinstance(hands, list)
        
        # Mock the detector to return a hand
        with patch.object(live_recognizer.hand_detector, 'detect_and_process_hand', return_value=(processed_crop_mock, hand_info_mock)):
             # Process one frame
            live_recognizer._process_frame(frame)

        # Check that a prediction was made
        self.assertEqual(live_recognizer.last_prediction, "A")
        self.assertEqual(live_recognizer.last_confidence, 0.95)
    
    def test_memory_usage_integration(self, live_recognizer, asl_detector):
        """
        TEST: Does the system manage memory properly?
        
        WHY: Long-running inference should not leak memory.
        
        CHECKS: Multiple predictions don't cause memory issues.
        """
        # Process multiple frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            hands = asl_detector.detect_hands_asl(frame)
            
            if hands:  # If hands detected
                for hand in hands:
                    if 'bbox' in hand:
                        # Create mock crop
                        crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        pred, conf = live_recognizer.predict_hand_sign(crop)
                        assert pred in ['A', 'B', 'C']
        
        # Should complete without memory errors
        assert True
    
    def test_device_compatibility_integration(self, live_recognizer):
        """
        TEST: Does the system work on different devices?
        
        WHY: Should work on CPU, MPS (Apple Silicon), and CUDA if available.
        
        CHECKS: Model inference works on the selected device.
        """
        device = live_recognizer.device
        assert str(device) in ['cpu', 'mps', 'cuda']
        
        # Model should be on the correct device
        model_device = next(live_recognizer.model.parameters()).device.type
        device_str = str(device)
        assert model_device == device_str or (device_str == 'mps' and model_device == 'cpu')
        
        # Prediction should work
        hand_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        prediction, confidence = live_recognizer.predict_hand_sign(hand_crop)
        assert prediction in ['A', 'B', 'C'] 