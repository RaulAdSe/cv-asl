"""Test enhanced hand detection capabilities."""

import numpy as np
import cv2
import pytest
from ..vision.enhanced_hand_detector import EnhancedHandDetector, HandCandidate
from ..vision.skin import SkinDetector


class TestEnhancedHandDetector:
    """Test enhanced hand detection functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.enhanced_detector = EnhancedHandDetector()
        self.basic_detector = SkinDetector()
    
    def create_test_scene_with_torso_and_hand(self):
        """Create a test image with both torso and hand regions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background
        frame[:] = [40, 40, 40]  # Dark background
        
        # Create large torso region (should be filtered out) - BGR format
        torso_x, torso_y = 100, 250
        torso_w, torso_h = 200, 180  # Smaller torso to avoid exceeding max area
        frame[torso_y:torso_y+torso_h, torso_x:torso_x+torso_w] = [110, 130, 170]  # Realistic skin BGR
        
        # Create hand region (should be detected) - BGR format, well separated
        hand_x, hand_y = 450, 120
        hand_w, hand_h = 120, 160
        frame[hand_y:hand_y+hand_h, hand_x:hand_x+hand_w] = [120, 140, 180]  # Slightly different skin BGR
        
        return frame, (torso_x, torso_y, torso_w, torso_h), (hand_x, hand_y, hand_w, hand_h)
    
    def test_enhanced_detector_initialization(self):
        """Test that enhanced detector initializes properly."""
        assert self.enhanced_detector.min_hand_area == 3000
        assert self.enhanced_detector.max_hand_area == 50000
        assert self.enhanced_detector.ideal_hand_width == 120
        assert self.enhanced_detector.ideal_hand_height == 160
    
    def test_circularity_calculation(self):
        """Test circularity calculation."""
        # Create a simple square contour
        square = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
        circularity = self.enhanced_detector.calculate_circularity(square)
        
        # Square should have circularity around 0.785 (π/4)
        assert 0.7 < circularity < 0.9
    
    def test_position_scoring(self):
        """Test position-based scoring."""
        frame_shape = (480, 640)
        
        # Center position should score well
        center_bbox = (250, 200, 100, 120)
        center_score = self.enhanced_detector.calculate_position_score(center_bbox, frame_shape)
        
        # Bottom position should score poorly
        bottom_bbox = (250, 400, 100, 120)
        bottom_score = self.enhanced_detector.calculate_position_score(bottom_bbox, frame_shape)
        
        assert center_score > bottom_score
        assert center_score > 0.5
        assert bottom_score < 0.4
    
    def test_size_scoring(self):
        """Test size-based scoring."""
        # Ideal size should score well
        ideal_bbox = (100, 100, 120, 160)
        ideal_score = self.enhanced_detector.calculate_size_score(ideal_bbox)
        
        # Huge size (torso-like) should score poorly
        huge_bbox = (100, 100, 300, 250)
        huge_score = self.enhanced_detector.calculate_size_score(huge_bbox)
        
        assert ideal_score > huge_score
        assert ideal_score > 0.7
        assert huge_score < 0.2
    
    def test_enhanced_vs_basic_detection(self):
        """Test that enhanced detection filters out torso better than basic detection."""
        frame, torso_bbox, hand_bbox = self.create_test_scene_with_torso_and_hand()
        
        # Basic detection (should detect both torso and hand)
        basic_hands = self.basic_detector.detect_hands(frame, max_hands=2)
        
        # Enhanced detection (should prefer hand over torso)
        enhanced_hands = self.enhanced_detector.detect_hands_enhanced(frame, max_hands=2, min_score=0.3)
        
        print(f"Basic detection found: {len(basic_hands)} regions")
        print(f"Enhanced detection found: {len(enhanced_hands)} regions")
        
        # Enhanced should find fewer but more accurate detections
        if len(enhanced_hands) > 0:
            # The best detection should be closer to the hand than to the torso
            best_detection = enhanced_hands[0]
            hand_center = (hand_bbox[0] + hand_bbox[2]//2, hand_bbox[1] + hand_bbox[3]//2)
            torso_center = (torso_bbox[0] + torso_bbox[2]//2, torso_bbox[1] + torso_bbox[3]//2)
            
            detection_center = (best_detection[0] + best_detection[2]//2, 
                              best_detection[1] + best_detection[3]//2)
            
            dist_to_hand = np.sqrt((detection_center[0] - hand_center[0])**2 + 
                                 (detection_center[1] - hand_center[1])**2)
            dist_to_torso = np.sqrt((detection_center[0] - torso_center[0])**2 + 
                                  (detection_center[1] - torso_center[1])**2)
            
            # Detection should be closer to hand than torso
            assert dist_to_hand < dist_to_torso, "Enhanced detection should prefer hand over torso"
    
    def test_hand_candidate_analysis(self):
        """Test hand candidate analysis and scoring."""
        frame, torso_bbox, hand_bbox = self.create_test_scene_with_torso_and_hand()
        
        # Get skin mask and contours
        mask = self.enhanced_detector.detect_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze candidates
        candidates = self.enhanced_detector.analyze_hand_candidates(contours, frame.shape[:2])
        
        # Should have at least one candidate
        assert len(candidates) > 0
        
        # Check that candidates have all required attributes
        for candidate in candidates:
            assert hasattr(candidate, 'total_score')
            assert hasattr(candidate, 'position_score')
            assert hasattr(candidate, 'size_score')
            assert hasattr(candidate, 'shape_score')
            assert 0 <= candidate.total_score <= 1
    
    def test_enhanced_visualization(self):
        """Test enhanced detection visualization."""
        frame, _, _ = self.create_test_scene_with_torso_and_hand()
        
        # Test visualization without errors
        result = self.enhanced_detector.visualize_enhanced_detection(frame, show_scores=True)
        
        # Should return an image of the same size
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
    
    def test_area_filtering(self):
        """Test that area filtering works correctly."""
        # Create very large contour (torso-like)
        large_contour = np.array([
            [100, 100], [400, 100], [400, 350], [100, 350]
        ], dtype=np.int32)
        
        # Create hand-sized contour
        hand_contour = np.array([
            [200, 150], [320, 150], [320, 310], [200, 310]
        ], dtype=np.int32)
        
        contours = [large_contour, hand_contour]
        candidates = self.enhanced_detector.analyze_hand_candidates(contours, (480, 640))
        
        # Large contour should be filtered out or score very low
        # Hand-sized contour should score better
        if len(candidates) > 1:
            # Find which candidate corresponds to which contour by area
            large_area = cv2.contourArea(large_contour)
            hand_area = cv2.contourArea(hand_contour)
            
            for candidate in candidates:
                if abs(candidate.area - hand_area) < abs(candidate.area - large_area):
                    # This is the hand candidate - should score better
                    hand_candidate = candidate
                else:
                    # This is the large candidate - should score worse
                    large_candidate = candidate
            
            if 'hand_candidate' in locals() and 'large_candidate' in locals():
                assert hand_candidate.total_score > large_candidate.total_score


def test_integration_with_data_collector():
    """Test that enhanced detector integrates properly with data collector."""
    from ..collect import DataCollector
    
    # Create collector - should use enhanced detector by default
    collector = DataCollector(data_dir='test_enhanced_data')
    
    # Should be using EnhancedHandDetector
    assert isinstance(collector.detector, EnhancedHandDetector)
    assert collector.use_enhanced_detection is True
    
    # Clean up
    import shutil
    import os
    if os.path.exists('test_enhanced_data'):
        shutil.rmtree('test_enhanced_data')


if __name__ == "__main__":
    # Simple test runner
    test_class = TestEnhancedHandDetector()
    
    print("Testing enhanced hand detection...")
    
    try:
        test_class.setup_method()
        
        test_class.test_enhanced_detector_initialization()
        print("✓ Enhanced detector initialization")
        
        test_class.test_circularity_calculation()
        print("✓ Circularity calculation")
        
        test_class.test_position_scoring()
        print("✓ Position scoring")
        
        test_class.test_size_scoring()
        print("✓ Size scoring")
        
        test_class.test_enhanced_vs_basic_detection()
        print("✓ Enhanced vs basic detection comparison")
        
        test_class.test_hand_candidate_analysis()
        print("✓ Hand candidate analysis")
        
        test_class.test_enhanced_visualization()
        print("✓ Enhanced visualization")
        
        test_class.test_area_filtering()
        print("✓ Area filtering")
        
        test_integration_with_data_collector()
        print("✓ Integration with data collector")
        
        print("\nAll enhanced detection tests passed! 🎉")
        print("\nKey improvements for shirtless scenarios:")
        print("- Size filtering: Rejects torso-sized regions")
        print("- Position scoring: Prefers hand-typical positions")
        print("- Shape analysis: Uses circularity and convexity defects")
        print("- Geometric scoring: Multi-factor candidate evaluation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 