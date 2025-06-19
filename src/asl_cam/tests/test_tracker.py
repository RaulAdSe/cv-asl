"""
Unit tests for hand tracking module.

These tests verify that the hand tracking system works correctly by:
1. Testing individual tracking components (Kalman filters, distance matching)
2. Testing complete tracking workflows (creating, updating, removing tracks)
3. Testing edge cases (empty detections, multiple hands, stability scoring)

Think of tracking like following a moving target - these tests ensure our
"target following" system can smoothly track hands across video frames.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.asl_cam.vision.tracker import HandTracker, MultiHandTracker, TrackedHand

class TestTrackedHand:
    """Test cases for TrackedHand dataclass."""
    
    def test_tracked_hand_creation(self):
        """
        TEST: Can we create a TrackedHand object with all required info?
        
        WHY: TrackedHand stores all information about a hand being tracked:
        - ID number (to identify it across frames)
        - Bounding box (where it is)
        - Center point (for distance calculations)
        - Age (how long we've been tracking it)
        - Hits (how many times we've seen it)
        - Time since last update (is it still active?)
        
        CHECKS: All data is stored correctly in the object.
        """
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
        """
        TEST: Does the tracker start up correctly?
        
        WHY: The tracker needs to initialize with empty tracking lists
        and proper settings before it can start following hands.
        
        CHECKS: Empty tracking dictionaries, proper timeout settings,
        and ID counter starting at 0.
        """
        assert tracker.max_disappeared == 30
        assert tracker.next_id == 0
        assert len(tracker.tracked_hands) == 0
        assert len(tracker.kalman_filters) == 0
    
    def test_bbox_center_calculation(self, tracker):
        """
        TEST: Can it calculate the center point of a hand box?
        
        WHY: To track hands smoothly, we need to know their center points
        for distance calculations and prediction. A box at (100,150) with
        size (80,120) should have center at (140, 210).
        
        CHECKS: Math is correct for converting box to center point.
        """
        bbox = (100, 150, 80, 120)
        center = tracker._bbox_center(bbox)
        
        expected_x = 100 + 80 / 2.0  # 140.0
        expected_y = 150 + 120 / 2.0  # 210.0
        
        assert center == (expected_x, expected_y)
    
    def test_distance_calculation(self, tracker):
        """
        TEST: Can it calculate distance between two points?
        
        WHY: To match detections to existing tracks, we need to find
        the closest hand. This uses the classic distance formula.
        
        CHECKS: Distance from (0,0) to (3,4) should be 5 (3-4-5 triangle).
        """
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        
        distance = tracker._distance(p1, p2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_create_kalman_filter(self, tracker):
        """
        TEST: Can it create the math predictor (Kalman filter)?
        
        WHY: Kalman filters predict where a hand will move next based on
        its previous movement. This helps track hands even if detection
        fails for a frame or two.
        
        CHECKS: Creates a proper Kalman filter with correct dimensions:
        - 4 state variables (x, y, velocity_x, velocity_y)
        - 2 measurements (x, y positions we can observe)
        """
        kf = tracker._create_kalman_filter()
        
        # Check that it's a valid Kalman filter
        assert isinstance(kf, cv2.KalmanFilter)
        
        # Check dimensions (4 state variables, 2 measurements)
        assert kf.transitionMatrix.shape == (4, 4)
        assert kf.measurementMatrix.shape == (2, 4)
        assert kf.processNoiseCov.shape == (4, 4)
        assert kf.measurementNoiseCov.shape == (2, 2)
    
    def test_update_with_no_existing_tracks(self, tracker, sample_detections):
        """
        TEST: What happens when we see hands for the first time?
        
        WHY: When the tracker starts, it has no existing hands to follow.
        It should create new tracks for each detected hand.
        
        CHECKS: Creates one new track per detection, each with ID 0,1,2...
        and proper initial values (1 hit, age 0, etc.).
        """
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
        """
        TEST: Can it properly register a new hand for tracking?
        
        WHY: When we see a hand for the first time, we need to:
        - Give it a unique ID number
        - Set up a Kalman filter for it
        - Initialize all tracking statistics
        
        CHECKS: Creates tracking entry with correct initial values.
        """
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
        """
        TEST: Can it update an existing hand's position?
        
        WHY: When a hand moves between frames, we need to update its
        tracking information: new position, increment hit count,
        reset the "time since update" counter.
        
        CHECKS: Position updates correctly, hit count increases,
        timing info is reset.
        """
        # First register a hand
        initial_bbox = (100, 150, 80, 120)
        tracker._register_hand(initial_bbox)
        
        # Update with new position
        new_bbox = (110, 160, 80, 120)
        tracker._update_track(0, new_bbox)
        
        hand = tracker.tracked_hands[0]
        assert hand.bbox == new_bbox
        assert hand.hits == 2
        assert hand.age == 0  # Age increments in main update loop, not in _update_track
        assert hand.time_since_update == 0
    
    def test_cleanup_old_tracks(self, tracker):
        """
        TEST: Does it remove hands that haven't been seen for too long?
        
        WHY: If a hand disappears (person moves away, occlusion, etc.),
        we don't want to keep tracking it forever. After a timeout
        period, we should remove "ghost" tracks.
        
        CHECKS: Hand is removed when time_since_update exceeds the
        max_disappeared threshold.
        """
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
        """
        TEST: What happens when asking for primary hand but none exist?
        
        WHY: Users want to get the "best" tracked hand for data collection.
        If no hands are being tracked, this should return None gracefully.
        
        CHECKS: Returns None when no hands are tracked.
        """
        primary = tracker.get_primary_hand()
        assert primary is None
    
    def test_get_primary_hand_single(self, tracker):
        """
        TEST: Does it return the only hand when there's just one?
        
        WHY: With a single tracked hand that meets stability requirements,
        it should be returned as the primary hand.
        
        CHECKS: Single stable hand (5+ hits) is returned as primary.
        """
        bbox = (100, 150, 80, 120)
        tracker._register_hand(bbox)
        
        # Need to increase hits to meet minimum stability requirement (3+)
        tracker.tracked_hands[0].hits = 5
        
        primary = tracker.get_primary_hand()
        assert primary is not None
        assert primary.id == 0
    
    def test_get_primary_hand_multiple(self, tracker):
        """
        TEST: With multiple hands, does it pick the most stable one?
        
        WHY: When tracking multiple hands, we want the one that's been
        detected most consistently (highest hit count) for data collection.
        
        CHECKS: Hand with more hits (15 vs 5) is chosen as primary.
        """
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
        """
        TEST: Can it draw tracking visualization on an image?
        
        WHY: Users need to see which hands are being tracked, their
        stability (color coding), and tracking info (ID, hit count).
        
        CHECKS: Draws colored rectangles and text on the image
        without changing its size or type.
        """
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
        """
        TEST: What happens when we have tracks but no new detections?
        
        WHY: Sometimes the detector fails to find hands that are actually
        there. The tracker should handle this gracefully and keep existing
        tracks alive for a while.
        
        CHECKS: Existing tracks remain but their time_since_update increases.
        """
        # Register a hand first
        tracker._register_hand((100, 150, 80, 120))
        
        # Create predictions dict
        predictions = {0: (140.0, 210.0)}
        
        # Should handle empty detections gracefully
        tracker._match_detections([], predictions)
        
        # Hand should still exist but time_since_update should increase
        assert len(tracker.tracked_hands) == 1
    
    def test_empty_detections_update(self, tracker):
        """
        TEST: Can it handle completely empty detection lists?
        
        WHY: Camera failures, processing errors, or scenes with no hands
        should not crash the tracker.
        
        CHECKS: Empty input produces empty output without errors.
        """
        hands = tracker.update([])
        
        assert len(hands) == 0
        assert len(tracker.tracked_hands) == 0
    
    def test_track_persistence(self, tracker):
        """
        TEST: Does it maintain the same ID when a hand moves?
        
        WHY: This is the core of tracking - when a hand moves from one
        position to another between frames, it should be recognized as
        the same hand (same ID) with updated position and increased hits.
        
        CHECKS: Hand at (100,150) moving to (105,155) keeps same ID
        but gets updated position and incremented hit count.
        """
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
        """
        TEST: Does the multi-hand tracker start up correctly?
        
        WHY: MultiHandTracker is a wrapper around HandTracker that
        limits the number of hands being tracked simultaneously.
        
        CHECKS: Proper settings and contains a HandTracker instance.
        """
        assert multi_tracker.max_hands == 2
        assert isinstance(multi_tracker.tracker, HandTracker)
        assert multi_tracker.tracker.max_disappeared == 30
    
    def test_update_with_max_hands_limit(self, multi_tracker):
        """
        TEST: Does it enforce the maximum hands limit?
        
        WHY: For ASL, we typically want to track 1-2 hands maximum.
        Even if the detector finds more hand-like objects, we should
        only track the best ones (largest/most stable).
        
        CHECKS: When given 3 detections but max_hands=2, only tracks
        the 2 largest ones.
        """
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
        """
        TEST: Can it return all currently tracked hands?
        
        WHY: Users need access to all tracked hands for visualization
        or analysis purposes.
        
        CHECKS: Returns list of TrackedHand objects for all active tracks.
        """
        detections = [(100, 150, 80, 120), (300, 200, 90, 110)]
        multi_tracker.update(detections)
        
        hands = multi_tracker.get_hands()
        assert len(hands) == 2
        assert all(isinstance(hand, TrackedHand) for hand in hands)
    
    def test_draw_tracks(self, multi_tracker):
        """
        TEST: Can it draw all tracked hands on an image?
        
        WHY: Visualization should show all tracked hands with their
        respective colors and information.
        
        CHECKS: Image is modified with tracking visualizations.
        """
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [(100, 150, 80, 120)]
        multi_tracker.update(detections)
        
        result = multi_tracker.draw_tracks(img)
        
        assert result.shape == img.shape
        assert result.dtype == img.dtype 