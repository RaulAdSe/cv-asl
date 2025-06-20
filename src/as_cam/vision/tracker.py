from typing import Tuple
import numpy as np

class Tracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
        self.kalman.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.last_bbox = None
        self.initialized = False
        self.missed_frames = 0

    def initialize(self, bbox: Tuple[int, int, int, int]):
        self.last_bbox = bbox
        self.initialized = True
        self.kalman.statePost = np.array([[bbox[0] + bbox[2] / 2], [bbox[1] + bbox[3] / 2], [bbox[2]], [bbox[3]]], np.float32)
        self.kalman.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self):
        prediction = self.kalman.predict()
        # Use the last known width and height for the predicted bbox
        w, h = self.last_bbox[2:]
        return int(prediction[0] - w / 2), int(prediction[1] - h / 2), w, h

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Updates the tracker with a new bounding box.
        Returns the updated bounding box.
        """
        if not self.initialized:
            self.initialize(bbox)
            return bbox # Return the initial bbox on the first frame
        
        center_x, center_y = self._get_center(bbox)
        measurement = np.array([center_x, center_y], np.float32)

        self.kalman.correct(measurement)
        self.missed_frames = 0
        
        corrected_state = self.kalman.statePost
        smooth_center_x, smooth_center_y = corrected_state[0, 0], corrected_state[1, 0]

        # Smooth the size of the bounding box as well for less jitter
        last_w, last_h = self.last_bbox[2:]
        new_w, new_h = bbox[2:]
        smooth_w = int(last_w * 0.8 + new_w * 0.2)
        smooth_h = int(last_h * 0.8 + new_h * 0.2)
        
        smooth_bbox = (int(smooth_center_x - smooth_w / 2), 
                       int(smooth_center_y - smooth_h / 2),
                       smooth_w, smooth_h)
        
        self.last_bbox = smooth_bbox
        return smooth_bbox

    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2) 