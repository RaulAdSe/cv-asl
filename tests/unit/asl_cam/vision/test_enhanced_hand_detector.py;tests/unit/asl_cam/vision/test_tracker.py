import unittest
import numpy as np
import cv2

from asl_cam.vision.enhanced_hand_detector import EnhancedHandDetector

class TestEnhancedHandDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = EnhancedHandDetector()

    def test_detector_initialization(self):
        """Test that the detector initializes without errors."""
        self.assertIsNotNone(self.detector)

    def test_analysis_and_visualization_run_without_error(self):
        """Test that analysis and visualization methods run without crashing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (300, 300), (255, 255, 255), -1)
        
        try:
            candidates = self.detector.analyze_hand_candidates(frame)
            self.assertIsInstance(candidates, list)
            
            result_img = self.detector.visualize_candidates(frame, candidates)
            self.assertIsNotNone(result_img)
            self.assertEqual(result_img.shape, frame.shape)
        except Exception as e:
            self.fail(f"Visualization methods failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import cv2

from asl_cam.vision.tracker import HandTracker

class TestHandTracker(unittest.TestCase):
    """Simplified tests for the Kalman Filter-based HandTracker."""

    def setUp(self):
        self.tracker = HandTracker()
        self.initial_bbox = (100, 100, 50, 50)

    def test_initialization_and_reset(self):
        """Test that the tracker can be initialized and reset."""
        self.assertFalse(self.tracker.initialized)
        self.tracker.initialize(self.initial_bbox)
        self.assertTrue(self.tracker.initialized)
        self.tracker.reset()
        self.assertFalse(self.tracker.initialized)

    def test_update_and_predict_run_without_error(self):
        """Test that update and predict run without crashing."""
        self.tracker.initialize(self.initial_bbox)
        
        try:
            new_bbox = (105, 105, 50, 50)
            smoothed = self.tracker.update(new_bbox)
            self.assertIsNotNone(smoothed)
            
            predicted = self.tracker.predict()
            self.assertIsNotNone(predicted)
        except Exception as e:
            self.fail(f"Tracker update/predict failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main() 