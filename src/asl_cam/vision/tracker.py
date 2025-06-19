"""
Hand tracking using Kalman filters and centroid-based matching.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class TrackedHand:
    """Container for tracked hand information."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]      # (cx, cy)
    age: int                         # Frames since first detection
    hits: int                        # Total number of detections
    time_since_update: int           # Frames since last update
    
class HandTracker:
    """Single-hand tracker using Kalman filter."""
    
    def __init__(self, max_disappeared: int = 30):
        """
        Initialize hand tracker.
        
        Args:
            max_disappeared: Maximum frames to keep track without detection
        """
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracked_hands: Dict[int, TrackedHand] = {}
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}
        
    def _create_kalman_filter(self) -> cv2.KalmanFilter:
        """Create a Kalman filter for tracking hand position."""
        # State: [x, y, vx, vy] (position and velocity)
        kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        return kf
    
    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[TrackedHand]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y, w, h) bounding boxes
            
        Returns:
            List of currently tracked hands
        """
        # If no existing tracks, create new ones
        if len(self.tracked_hands) == 0:
            for detection in detections:
                self._register_hand(detection)
        else:
            # Predict next positions
            predictions = {}
            for hand_id, kf in self.kalman_filters.items():
                pred = kf.predict()
                predictions[hand_id] = (float(pred[0]), float(pred[1]))
            
            # Match detections to existing tracks
            if len(detections) > 0:
                self._match_detections(detections, predictions)
            
            # Update time since last update
            for hand in self.tracked_hands.values():
                hand.time_since_update += 1
        
        # Remove old tracks
        self._cleanup_old_tracks()
        
        return list(self.tracked_hands.values())
    
    def _register_hand(self, bbox: Tuple[int, int, int, int]) -> None:
        """Register a new hand track."""
        hand_id = self.next_id
        self.next_id += 1
        
        center = self._bbox_center(bbox)
        
        # Create tracked hand
        hand = TrackedHand(
            id=hand_id,
            bbox=bbox,
            center=center,
            age=0,
            hits=1,
            time_since_update=0
        )
        
        self.tracked_hands[hand_id] = hand
        
        # Create Kalman filter
        kf = self._create_kalman_filter()
        
        # Initialize state with current position
        kf.statePre = np.array([center[0], center[1], 0, 0], dtype=np.float32)
        kf.statePost = np.array([center[0], center[1], 0, 0], dtype=np.float32)
        
        self.kalman_filters[hand_id] = kf
    
    def _match_detections(self, detections: List[Tuple[int, int, int, int]], 
                         predictions: Dict[int, Tuple[float, float]]) -> None:
        """Match detections to existing tracks using Hungarian algorithm (simplified)."""
        if len(detections) == 0:
            return
        
        # For simplicity, use nearest neighbor matching for single hand
        # In a full implementation, you'd use the Hungarian algorithm
        
        detection_centers = [self._bbox_center(det) for det in detections]
        
        # Find best matches
        matched_detections = set()
        matched_tracks = set()
        
        for hand_id, pred_center in predictions.items():
            if hand_id not in self.tracked_hands:
                continue
                
            best_distance = float('inf')
            best_detection_idx = -1
            
            for i, det_center in enumerate(detection_centers):
                if i in matched_detections:
                    continue
                    
                distance = self._distance(pred_center, det_center)
                
                # Distance threshold (should be tuned based on your setup)
                if distance < 100 and distance < best_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            # Update matched track
            if best_detection_idx >= 0:
                self._update_track(hand_id, detections[best_detection_idx])
                matched_detections.add(best_detection_idx)
                matched_tracks.add(hand_id)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self._register_hand(detection)
    
    def _update_track(self, hand_id: int, bbox: Tuple[int, int, int, int]) -> None:
        """Update existing track with new detection."""
        if hand_id not in self.tracked_hands:
            return
        
        center = self._bbox_center(bbox)
        
        # Update Kalman filter
        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        self.kalman_filters[hand_id].correct(measurement)
        
        # Update tracked hand
        hand = self.tracked_hands[hand_id]
        hand.bbox = bbox
        hand.center = center
        hand.hits += 1
        hand.age += 1
        hand.time_since_update = 0
    
    def _cleanup_old_tracks(self) -> None:
        """Remove tracks that haven't been updated for too long."""
        to_remove = []
        
        for hand_id, hand in self.tracked_hands.items():
            if hand.time_since_update > self.max_disappeared:
                to_remove.append(hand_id)
        
        for hand_id in to_remove:
            del self.tracked_hands[hand_id]
            del self.kalman_filters[hand_id]
    
    def get_primary_hand(self) -> Optional[TrackedHand]:
        """Get the most reliable tracked hand (highest hit count)."""
        if not self.tracked_hands:
            return None
        
        # Return hand with most hits (most stable track)
        return max(self.tracked_hands.values(), key=lambda h: h.hits)
    
    def draw_tracks(self, img: np.ndarray) -> np.ndarray:
        """
        Draw tracked hands on image.
        
        Args:
            img: Input image
            
        Returns:
            Image with tracking visualizations
        """
        result = img.copy()
        
        for hand in self.tracked_hands.values():
            x, y, w, h = hand.bbox
            
            # Choose color based on track stability
            if hand.hits > 10:
                color = (0, 255, 0)  # Green for stable tracks
            elif hand.hits > 5:
                color = (0, 255, 255)  # Yellow for medium stability
            else:
                color = (0, 0, 255)  # Red for new tracks
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center_int = (int(hand.center[0]), int(hand.center[1]))
            cv2.circle(result, center_int, 3, color, -1)
            
            # Add track info
            info_text = f"ID:{hand.id} H:{hand.hits}"
            cv2.putText(result, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result

class MultiHandTracker:
    """Multi-hand tracker using multiple Kalman filters."""
    
    def __init__(self, max_hands: int = 2, max_disappeared: int = 30):
        """
        Initialize multi-hand tracker.
        
        Args:
            max_hands: Maximum number of hands to track
            max_disappeared: Maximum frames to keep track without detection
        """
        self.max_hands = max_hands
        self.tracker = HandTracker(max_disappeared)
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[TrackedHand]:
        """Update tracker with new detections."""
        # Limit detections to max_hands
        limited_detections = detections[:self.max_hands]
        return self.tracker.update(limited_detections)
    
    def get_hands(self) -> List[TrackedHand]:
        """Get all currently tracked hands."""
        return list(self.tracker.tracked_hands.values())
    
    def draw_tracks(self, img: np.ndarray) -> np.ndarray:
        """Draw all tracked hands."""
        return self.tracker.draw_tracks(img) 