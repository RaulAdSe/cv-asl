import unittest
from asl_dl.training.train import ASLTrainer

class TestCompleteASLPipeline(unittest.TestCase):
    # ... rest of the file

import unittest
from asl_cam.vision.enhanced_hand_detector import EnhancedHandDetector

class TestEnhancedHandDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EnhancedHandDetector()

    def test_analysis_and_visualization_run_without_error(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        candidates = self.detector.analyze_hand_candidates(frame, frame.shape)
        result_img = self.detector.visualize_candidates(frame, candidates)
        self.assertIsNotNone(result_img)

import unittest
from asl_cam.vision.tracker import HandTracker

class TestHandTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = HandTracker()

    def test_predict_after_update(self):
        """Test that predict runs without error after an update."""
        self.tracker.initialize((100, 100, 50, 50))
        self.tracker.update((105, 105, 50, 50))
        try:
            prediction = self.tracker.predict()
            self.assertIsNotNone(prediction)
        except Exception as e:
            self.fail(f"predict() failed after update: {e}") 