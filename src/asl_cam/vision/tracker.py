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
    Tracks a hand's position and size over time using a Kalman Filter.
    This more advanced filter tracks width and height in its state, leading
    to a much more stable bounding box.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-4, error_cov=0.1, patience=5):
        self.kf = cv2.KalmanFilter(8, 4) # State: [x, y, w, h, vx, vy, vw, vh]
        
        # Transition Matrix (A)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], np.float32)

        # Measurement Matrix (H)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        # Noise Covariances
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * error_cov

        self.lost_patience = patience
        self.lost_counter = 0
        self.is_tracking = False

    def initialize(self, initial_bbox: tuple):
        """Initializes the Kalman Filter with the first detected bounding box."""
        x, y, w, h = initial_bbox
        # Set the initial state [x_center, y_center, w, h, vx, vy, vw, vh]
        self.kf.statePost = np.array([x + w/2, y + h/2, w, h, 0, 0, 0, 0], dtype=np.float32).reshape((8, 1))
        
        self.is_tracking = True
        self.lost_counter = 0
        logger.info(f"HandTracker initialized at position: ({self.kf.statePost[0][0]:.2f}, {self.kf.statePost[1][0]:.2f})")

    def predict(self) -> tuple:
        """Predicts the next position and size of the bounding box."""
        if not self.is_tracking:
            return None, "LOST"
            
        prediction = self.kf.predict()
        
        if self.lost_counter > 0:
            self.lost_counter += 1
            if self.lost_counter > self.lost_patience:
                self.reset()
                return None, "LOST"
        
        # Extract smoothed state
        x, y, w, h = prediction[0][0], prediction[1][0], prediction[2][0], prediction[3][0]
        
        # Prevent negative or zero dimensions
        w, h = max(10, w), max(10, h)
        
        predicted_bbox = (int(x - w / 2), int(y - h / 2), int(w), int(h))
        status = "PREDICTED" if self.lost_counter > 0 else "TRACKED"
        
        return predicted_bbox, status

    def update(self, new_bbox: tuple):
        """Updates the Kalman Filter with a new measurement."""
        if new_bbox is None:
            if self.is_tracking and self.lost_counter == 0:
                self.lost_counter = 1
            return
            
        self.lost_counter = 0
        x, y, w, h = new_bbox
        measurement = np.array([x + w/2, y + h/2, w, h], dtype=np.float32).reshape((4, 1))
        self.kf.correct(measurement)

    def reset(self):
        """Resets the tracker to its initial state."""
        # Re-initialize the filter's internal state without creating a new object
        self.__init__(patience=self.lost_patience)
        self.is_tracking = False
        logger.info("HandTracker has been reset.") 