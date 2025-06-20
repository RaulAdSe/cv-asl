"""
Hand Tracker with Kalman Filter

This module provides a Kalman Filter-based tracker to predict and smooth
the bounding box of a detected hand across frames. This helps to reduce
jitter and maintain a stable detection even if the detector misses a frame.
"""
import numpy as np
import logging
import cv2
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class HandTracker:
    """
    A Kalman Filter-based tracker for smoothing a single hand's bounding box.
    """
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.1, error_cov: float = 0.1, max_missed_frames: int = 10):
        """
        Initializes the hand tracker.
        
        Args:
            process_noise: How much we trust the physical model (lower = smoother).
            measurement_noise: How much we trust the measurement (lower = less lag).
            error_cov: Initial estimate of the error.
            max_missed_frames: Maximum frames to "coast" before resetting.
        """
        self.kalman = cv2.KalmanFilter(4, 2)
        # State: [x, y, dx, dy] (center x, center y, velocity x, velocity y)
        # Measurement: [x, y] (center x, center y)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        
        # Noise Covariances
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * error_cov

        self.initialized = False
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames

        logger.info("HandTracker (Kalman Filter) initialized.")

    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculates the center of a bounding box."""
        return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

    def initialize(self, bbox: Tuple[int, int, int, int]):
        """
        Initializes the filter with the first detected bounding box.
        
        Args:
            bbox: The first bounding box (x, y, w, h).
        """
        center_x, center_y = self._get_center(bbox)
        
        self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        self.last_bbox = bbox
        self.initialized = True
        self.missed_frames = 0
        logger.info(f"HandTracker initialized at position: ({center_x:.2f}, {center_y:.2f})")

    def predict(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Predicts the next position of the hand.
        
        Returns:
            The predicted bounding box (x, y, w, h) if tracking, otherwise None.
        """
        if not self.initialized:
            return None

        prediction = self.kalman.predict().flatten()
        self.missed_frames += 1
        
        if self.missed_frames > self.max_missed_frames:
            self.reset()
            return None
        
        center_x, center_y = prediction[0], prediction[1]
        
        if self.last_bbox:
            w, h = self.last_bbox[2], self.last_bbox[3]
            predicted_bbox = (int(center_x - w / 2), int(center_y - h / 2), w, h)
            self.last_bbox = predicted_bbox # Keep track of the predicted state
            return predicted_bbox
        return None

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Updates the filter with a new measurement (a detected bounding box).
        
        Args:
            bbox: The new bounding box (x, y, w, h).
            
        Returns:
            The corrected (smoothed) bounding box (x, y, w, h).
        """
        if not self.initialized:
            self.initialize(bbox)
            return bbox # On the first update, return the bbox itself.
        
        center_x, center_y = self._get_center(bbox)
        measurement = np.array([center_x, center_y], dtype=np.float32)

        self.kalman.correct(measurement)
        self.missed_frames = 0
        
        corrected_state = self.kalman.statePost.flatten()
        smooth_center_x, smooth_center_y = corrected_state[0], corrected_state[1]

        # Exponential moving average for smoothing the size
        last_w, last_h = self.last_bbox[2:]
        new_w, new_h = bbox[2:]
        smooth_w = int(last_w * 0.7 + new_w * 0.3)
        smooth_h = int(last_h * 0.7 + new_h * 0.3)
        
        smooth_bbox = (int(smooth_center_x - smooth_w / 2), 
                       int(smooth_center_y - smooth_h / 2),
                       smooth_w, smooth_h)
        
        self.last_bbox = smooth_bbox
        return smooth_bbox
        
    def reset(self):
        """Resets the tracker to an uninitialized state."""
        self.initialized = False
        self.last_bbox = None
        self.missed_frames = 0
        logger.info("HandTracker has been reset.") 