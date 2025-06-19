"""
Unit tests for hand tracking module.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.asl_cam.vision.tracker import HandTracker, MultiHandTracker, TrackedHand

class TestTrackedHand:
    """Test cases for TrackedHand dataclass."""
    
    def test_tracked_hand_creation(self):
        """Test TrackedHand creation with valid data."""
        hand = TrackedHand(
            id=1,
            bbox=(100, 150, 80, 120),
            center=(140.0, 210.0),
            age=5,
            hits=10,
            time_since_update=0
        )
        
        assert hand.id == 1
        assert hand.bbox == (100, 150, 80, 120)
        assert hand.center == (140.0, 210.0)
        assert hand.age == 5
        assert hand.hits == 10
        assert hand.time_since_update == 0

class TestHandTracker:
    """Test cases for HandTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create a HandTracker instance for testing."""
        return HandTracker(max_disappeared=30)
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample hand detections."""
        return [
            (100, 150, 80, 120),  # x, y, w, h
            (300, 200, 90, 110)
        ]
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker.max_disappeared == 30
        assert tracker.next_id == 0
        assert len(tracker.tracked_hands) == 0
        assert len(tracker.kalman_filters) == 0
    
    def test_bbox_center_calculation(self, tracker):
        """Test bounding box center calculation."""
        bbox = (100, 150, 80, 120)
        center = tracker._bbox_center(bbox)
        
        expected_x = 100 + 80 / 2.0  # 140.0
        expected_y = 150 + 120 / 2.0  # 210.0
        
        assert center == (expected_x, expected_y)
    
    def test_distance_calculation(self, tracker):
        """Test Euclidean distance calculation."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        
        distance = tracker._distance(p1, p2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_create_kalman_filter(self, tracker):
        """Test Kalman filter creation."""
        kf = tracker._create_kalman_filter()
        
        # Check that it's a valid Kalman filter
        assert isinstance(kf, cv2.KalmanFilter)
        
        # Check dimensions (4 state variables, 2 measurements)
        assert kf.transitionMatrix.shape == (4, 4)
        assert kf.measurementMatrix.shape == (2, 4)
        assert kf.processNoiseCov.shape == (4, 4)
        assert kf.measurementNoiseCov.shape == (2, 2)
    
    def test_update_with_no_existing_tracks(self, tracker, sample_detections):
        """Test updating tracker with no existing tracks."""
        hands = tracker.update(sample_detections)
        
        # Should create new tracks for each detection
        assert len(hands) == len(sample_detections)
        assert len(tracker.tracked_hands) == len(sample_detections)
        assert len(tracker.kalman_filters) == len(sample_detections)
        
        # Check track properties
        for i, hand in enumerate(hands):
            assert hand.id == i
            assert hand.bbox == sample_detections[i]
            assert hand.hits == 1
            assert hand.age == 0
            assert hand.time_since_update == 0
    
    def test_register_hand(self, tracker):
        """Test registering a new hand."""
        bbox = (100, 150, 80, 120)
        tracker._register_hand(bbox)
        
        assert len(tracker.tracked_hands) == 1
        assert len(tracker.kalman_filters) == 1
        assert tracker.next_id == 1
        
        hand = list(tracker.tracked_hands.values())[0]
        assert hand.id == 0
        assert hand.bbox == bbox
        assert hand.hits == 1
    
    def test_update_track(self, tracker):
        """Test updating an existing track."""
        # First register a hand
        initial_bbox = (100, 150, 80, 120)
        tracker._register_hand(initial_bbox)
        
        # Update with new position
        new_bbox = (110, 160, 80, 120)
        tracker._update_track(0, new_bbox)
        
        hand = tracker.tracked_hands[0]
        assert hand.bbox == new_bbox
        assert hand.hits == 2
        assert hand.age == 1
        assert hand.time_since_update == 0
    
    def test_cleanup_old_tracks(self, tracker):
        """Test removal of old tracks."""
        # Register a hand
        bbox = (100, 150, 80, 120)
        tracker._register_hand(bbox)
        
        # Simulate time passing without updates
        hand = tracker.tracked_hands[0]
        hand.time_since_update = tracker.max_disappeared + 1
        
        tracker._cleanup_old_tracks()
        
        # Hand should be removed
        assert len(tracker.tracked_hands) == 0
        assert len(tracker.kalman_filters) == 0
    
    def test_get_primary_hand_empty(self, tracker):
        """Test getting primary hand when no hands tracked."""
        primary = tracker.get_primary_hand()
        assert primary is None
    
    def test_get_primary_hand_single(self, tracker):
        """Test getting primary hand with single tracked hand."""
        bbox = (100, 150, 80, 120)
        tracker._register_hand(bbox)
        
        primary = tracker.get_primary_hand()
        assert primary is not None
        assert primary.id == 0
    
    def test_get_primary_hand_multiple(self, tracker):
        """Test getting primary hand with multiple tracked hands."""
        # Register two hands
        tracker._register_hand((100, 150, 80, 120))
        tracker._register_hand((300, 200, 90, 110))
        
        # Increase hit count for second hand
        tracker.tracked_hands[1].hits = 15
        tracker.tracked_hands[0].hits = 5
        
        primary = tracker.get_primary_hand()
        assert primary is not None
        assert primary.id == 1  # Should be the one with more hits
        assert primary.hits == 15
    
    def test_draw_tracks(self, tracker):
        """Test drawing tracks on image."""
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Register a hand with different hit counts
        tracker._register_hand((100, 150, 80, 120))
        tracker.tracked_hands[0].hits = 12  # Should be green (stable)
        
        result = tracker.draw_tracks(img)
        
        # Should return image of same shape
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        
        # Image should be modified (some pixels should be non-zero)
        assert not np.array_equal(result, img)
    
    def test_match_detections_no_detections(self, tracker):
        """Test matching when no detections provided."""
        # Register a hand first
        tracker._register_hand((100, 150, 80, 120))
        
        # Create predictions dict
        predictions = {0: (140.0, 210.0)}
        
        # Should handle empty detections gracefully
        tracker._match_detections([], predictions)
        
        # Hand should still exist but time_since_update should increase
        assert len(tracker.tracked_hands) == 1
    
    def test_empty_detections_update(self, tracker):
        """Test updating with empty detections list."""
        hands = tracker.update([])
        
        assert len(hands) == 0
        assert len(tracker.tracked_hands) == 0
    
    def test_track_persistence(self, tracker):
        """Test that tracks persist across multiple updates."""
        bbox = (100, 150, 80, 120)
        
        # First update - create track
        hands1 = tracker.update([bbox])
        assert len(hands1) == 1
        original_id = hands1[0].id
        
        # Second update - should maintain same track
        new_bbox = (105, 155, 80, 120)  # Slightly moved
        hands2 = tracker.update([new_bbox])
        assert len(hands2) == 1
        assert hands2[0].id == original_id
        assert hands2[0].hits == 2

class TestMultiHandTracker:
    """Test cases for MultiHandTracker class."""
    
    @pytest.fixture
    def multi_tracker(self):
        """Create a MultiHandTracker instance for testing."""
        return MultiHandTracker(max_hands=2, max_disappeared=30)
    
    def test_initialization(self, multi_tracker):
        """Test multi-hand tracker initialization."""
        assert multi_tracker.max_hands == 2
        assert isinstance(multi_tracker.tracker, HandTracker)
        assert multi_tracker.tracker.max_disappeared == 30
    
    def test_update_with_max_hands_limit(self, multi_tracker):
        """Test that detections are limited to max_hands."""
        # Provide more detections than max_hands
        detections = [
            (100, 150, 80, 120),
            (300, 200, 90, 110),
            (500, 100, 70, 100)  # This should be ignored
        ]
        
        hands = multi_tracker.update(detections)
        
        # Should only track max_hands
        assert len(hands) <= multi_tracker.max_hands
    
    def test_get_hands(self, multi_tracker):
        """Test getting all tracked hands."""
        detections = [(100, 150, 80, 120), (300, 200, 90, 110)]
        multi_tracker.update(detections)
        
        hands = multi_tracker.get_hands()
        assert len(hands) == 2
        assert all(isinstance(hand, TrackedHand) for hand in hands)
    
    def test_draw_tracks(self, multi_tracker):
        """Test drawing all tracks."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [(100, 150, 80, 120)]
        multi_tracker.update(detections)
        
        result = multi_tracker.draw_tracks(img)
        
        assert result.shape == img.shape
        assert result.dtype == img.dtype 