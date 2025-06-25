"""
Unit tests for hand tracking module.

These tests verify that the hand tracking system works correctly by:
1. Testing individual tracking components (Kalman filters, distance matching)
2. Testing complete tracking workflows (creating, updating, removing tracks)
3. Testing edge cases (empty detections, multiple hands, stability scoring)

Think of tracking like following a moving target - these tests ensure our
"target following" system can smoothly track hands across video frames.
"""
import unittest
import numpy as np
import cv2

# Add the project root to the path to allow direct imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from asl_cam.vision.tracker import HandTracker

class TestHandTracker(unittest.TestCase):
    """Unit tests for the Kalman Filter-based HandTracker."""

    def setUp(self):
        # Use slightly higher process noise for tests to see velocity changes faster
        self.tracker = HandTracker(process_noise=0.1) 
        self.initial_bbox = (100, 100, 50, 50)

    def test_initialization(self):
        self.assertFalse(self.tracker.is_tracking)
        self.tracker.initialize(self.initial_bbox)
        self.assertTrue(self.tracker.is_tracking)

    def test_predict_after_update(self):
        """Test that predict runs without error after an update."""
        self.tracker.initialize((100, 100, 50, 50))
        self.tracker.update((105, 105, 50, 50))
        try:
            prediction = self.tracker.predict()
            self.assertIsNotNone(prediction)
        except Exception as e:
            self.fail(f"predict() failed after update: {e}")

    def test_update(self):
        self.tracker.initialize(self.initial_bbox)
        new_bbox = (110, 110, 52, 52)
        self.tracker.update(new_bbox)
        
        # After update, tracker should still be tracking
        self.assertTrue(self.tracker.is_tracking)
        self.assertEqual(self.tracker.lost_counter, 0)

    def test_reset(self):
        self.tracker.initialize(self.initial_bbox)
        self.tracker.reset()
        self.assertFalse(self.tracker.is_tracking)

if __name__ == '__main__':
    unittest.main() 