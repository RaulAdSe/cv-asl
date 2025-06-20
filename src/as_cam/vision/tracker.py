import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class HandTracker:
    """
    Tracks a hand's position using a Kalman Filter for smoothness, and
    separately smooths the bounding box size using an exponential moving
    average to prevent erratic resizing.
    """
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, error_cov=0.1, patience=5, size_smoothing=0.6):
        # --- Kalman Filter for Position (x, y, vx, vy) ---
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * error_cov

        # --- Tracker State ---
        self.tracked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.lost_patience = patience
        self.lost_counter = 0
        self.is_tracking = False
        self.size_smoothing = size_smoothing # Alpha for EMA on size

    def initialize(self, initial_bbox: tuple):
        """Initializes the tracker with the first detected bounding box."""
        x, y, w, h = initial_bbox
        center_x, center_y = x + w / 2, y + h / 2
        
        # Set the initial state for the Kalman filter
        self.kf.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32).reshape((4, 1))
        
        self.tracked_bbox = initial_bbox
        self.is_tracking = True
        self.lost_counter = 0
        logger.info(f"HandTracker initialized at position: ({center_x:.2f}, {center_y:.2f})")

    def predict(self) -> tuple:
        """Predicts the next position of the bounding box using the last known size."""
        if not self.is_tracking:
            return None, "LOST"
            
        prediction = self.kf.predict()
        
        if self.lost_counter > 0:
            self.lost_counter += 1
            if self.lost_counter > self.lost_patience:
                self.reset()
                return None, "LOST"
        
        # Use the filter's predicted center but the last known stable size
        pred_x, pred_y = prediction[0][0], prediction[1][0]
        last_w, last_h = self.tracked_bbox[2], self.tracked_bbox[3]
        
        predicted_bbox = (int(pred_x - last_w / 2), int(pred_y - last_h / 2), last_w, last_h)
        status = "PREDICTED" if self.lost_counter > 0 else "TRACKED"
        
        return predicted_bbox, status

    def update(self, new_bbox: tuple):
        """Updates the tracker with a new measurement, smoothing position and size separately."""
        if new_bbox is None:
            if self.is_tracking and self.lost_counter == 0:
                self.lost_counter = 1
            return
            
        self.lost_counter = 0
        x, y, w, h = new_bbox
        
        # --- Update Kalman Filter for Position ---
        center_x, center_y = x + w / 2, y + h / 2
        measurement = np.array([center_x, center_y], dtype=np.float32).reshape((2, 1))
        corrected_state = self.kf.correct(measurement)
        smooth_center_x, smooth_center_y = corrected_state[0][0], corrected_state[1][0]
        
        # --- Update Size with Exponential Moving Average ---
        last_w, last_h = self.tracked_bbox[2], self.tracked_bbox[3]
        smooth_w = int(last_w * self.size_smoothing + w * (1 - self.size_smoothing))
        smooth_h = int(last_h * self.size_smoothing + h * (1 - self.size_smoothing))
        
        # Combine smoothed position and smoothed size
        self.tracked_bbox = (
            int(smooth_center_x - smooth_w / 2),
            int(smooth_center_y - smooth_h / 2),
            smooth_w,
            smooth_h
        )

    def reset(self):
        """Resets the tracker to its initial state."""
        self.__init__(
            patience=self.lost_patience, 
            size_smoothing=self.size_smoothing
        )
        self.is_tracking = False
        logger.info("HandTracker has been reset.") 