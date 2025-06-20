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
    Tracks a hand over time using a Kalman Filter to smooth the bounding box.
    """
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1, error_cov=1.0, patience=10):
        self.kf = None
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.error_cov = error_cov
        
        self.tracked_bbox = None
        self.lost_patience = patience # Number of frames to wait before resetting
        self.lost_counter = 0
        self.is_tracking = False

    def initialize(self, initial_bbox: tuple):
        """Initializes the Kalman Filter with the first detected bounding box."""
        self.kf = cv2.KalmanFilter(4, 2)
        # State: [x, y, dx, dy] (center x, center y, velocity x, velocity y)
        # Measurement: [x, y] (center x, center y)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        
        # Noise Covariances
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * self.error_cov

        # Set the initial state
        x, y, w, h = initial_bbox
        self.kf.statePost = np.array([x + w / 2, y + h / 2, 0, 0], dtype=np.float32).reshape((4, 1))
        
        self.tracked_bbox = initial_bbox
        self.is_tracking = True
        self.lost_counter = 0
        logger.info(f"HandTracker initialized at position: ({self.kf.statePost[0][0]:.2f}, {self.kf.statePost[1][0]:.2f})")

    def predict(self) -> tuple:
        """Predicts the next position of the bounding box."""
        if not self.is_tracking:
            return None, "LOST"
            
        prediction = self.kf.predict()
        
        # If we are in a "lost" state, increment the counter
        if self.lost_counter > 0:
            self.lost_counter += 1
            if self.lost_counter > self.lost_patience:
                self.reset()
                return None, "LOST"

        # Extract predicted center and dimensions from the state
        pred_x, pred_y = prediction[0][0], prediction[1][0]
        w, h = self.tracked_bbox[2], self.tracked_bbox[3]
        
        self.tracked_bbox = (int(pred_x - w / 2), int(pred_y - h / 2), w, h)
        
        # If we're actively tracking, it's a stable track. If we are predicting
        # while lost, it's a predicted position.
        status = "PREDICTED" if self.lost_counter > 0 else "TRACKED"
        
        return self.tracked_bbox, status

    def update(self, new_bbox: tuple):
        """Updates the Kalman Filter with a new measurement."""
        if new_bbox is None:
            # If we didn't get a measurement, start the lost counter
            if self.is_tracking:
                self.lost_counter = 1
            return
            
        # If we get a measurement, we are no longer lost
        self.lost_counter = 0
        
        x, y, w, h = new_bbox
        measurement = np.array([x + w / 2, y + h / 2], dtype=np.float32).reshape((2, 1))
        self.kf.correct(measurement)
        self.tracked_bbox = new_bbox

    def reset(self):
        """Resets the tracker to its initial state."""
        self.kf = None
        self.tracked_bbox = None
        self.is_tracking = False
        self.lost_counter = 0
        logger.info("HandTracker has been reset.") 