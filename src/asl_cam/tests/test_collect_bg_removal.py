"""Test background removal integration in data collection."""

import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import pytest

from ..collect import DataCollector
from ..vision.background_removal import BackgroundMethod


class TestDataCollectorBackgroundRemoval:
    """Test background removal integration in data collector."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = DataCollector(data_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_frame_with_hand(self):
        """Create a test frame with a simple hand-like region."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background noise
        frame[:] = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        
        # Create a hand-like rectangular region
        hand_x, hand_y = 200, 150
        hand_w, hand_h = 120, 160
        
        # Fill hand region with skin-like color
        frame[hand_y:hand_y+hand_h, hand_x:hand_x+hand_w] = [180, 120, 100]  # Skin-ish color
        
        return frame, (hand_x, hand_y, hand_w, hand_h)
    
    def test_background_removal_enabled_by_default(self):
        """Test that background removal is enabled by default."""
        assert self.collector.use_background_removal is True
        assert self.collector.save_both_versions is True
    
    def test_background_removal_toggle(self):
        """Test toggling background removal setting."""
        original_state = self.collector.use_background_removal
        self.collector.use_background_removal = not original_state
        assert self.collector.use_background_removal != original_state
    
    def test_save_sample_with_background_removal(self):
        """Test saving sample with background removal enabled."""
        frame, bbox = self.create_test_frame_with_hand()
        
        # Enable background removal
        self.collector.use_background_removal = True
        self.collector.save_both_versions = True
        
        # Save sample
        image_path = self.collector._save_sample(frame, bbox, confidence=0.8)
        
        # Check files were created
        assert os.path.exists(image_path)
        
        crop_path = image_path.replace('.jpg', '_crop.jpg')
        assert os.path.exists(crop_path)
        
        original_crop_path = image_path.replace('.jpg', '_crop_original.jpg')
        assert os.path.exists(original_crop_path)
        
        # Check sample metadata
        sample = self.collector.samples[-1]
        assert sample.metadata['background_removed'] is True
        assert sample.metadata['original_crop_path'] == original_crop_path
        assert sample.metadata['bg_removal_method'] == BackgroundMethod.GRABCUT.value
    
    def test_save_sample_without_background_removal(self):
        """Test saving sample with background removal disabled."""
        frame, bbox = self.create_test_frame_with_hand()
        
        # Disable background removal
        self.collector.use_background_removal = False
        
        # Save sample
        image_path = self.collector._save_sample(frame, bbox, confidence=0.8)
        
        # Check files were created
        assert os.path.exists(image_path)
        
        crop_path = image_path.replace('.jpg', '_crop.jpg')
        assert os.path.exists(crop_path)
        
        # Check that original crop was not saved separately
        original_crop_path = image_path.replace('.jpg', '_crop_original.jpg')
        assert not os.path.exists(original_crop_path)
        
        # Check sample metadata
        sample = self.collector.samples[-1]
        assert sample.metadata['background_removed'] is False
        assert sample.metadata['original_crop_path'] is None
        assert sample.metadata['bg_removal_method'] is None
    
    def test_manifest_includes_background_removal_settings(self):
        """Test that manifest includes background removal settings."""
        # Save manifest
        self.collector._save_manifest()
        
        # Check manifest file
        manifest_file = Path(self.temp_dir) / "manifest.json"
        assert manifest_file.exists()
        
        import json
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        settings = manifest_data['collection_settings']
        assert 'use_background_removal' in settings
        assert 'save_both_versions' in settings
        assert 'bg_removal_method' in settings
        assert settings['bg_removal_method'] == BackgroundMethod.GRABCUT.value
    
    def test_background_remover_initialization(self):
        """Test that background remover is properly initialized."""
        assert self.collector.bg_remover is not None
        assert self.collector.bg_remover.method == BackgroundMethod.GRABCUT
    
    def test_sample_with_failed_background_removal(self):
        """Test behavior when background removal fails."""
        frame, bbox = self.create_test_frame_with_hand()
        
        # Create a mock background remover that returns None
        class MockBGRemover:
            def __init__(self):
                self.method = BackgroundMethod.GRABCUT
            
            def remove_background(self, image, hand_bbox):
                return None  # Simulate failure
        
        self.collector.bg_remover = MockBGRemover()
        self.collector.use_background_removal = True
        
        # Save sample
        image_path = self.collector._save_sample(frame, bbox, confidence=0.8)
        
        # Check that sample was still saved (using original)
        assert os.path.exists(image_path)
        crop_path = image_path.replace('.jpg', '_crop.jpg')
        assert os.path.exists(crop_path)
        
        # Check metadata indicates background removal failed
        sample = self.collector.samples[-1]
        assert sample.metadata['background_removed'] is False
        assert sample.metadata['original_crop_path'] is None


if __name__ == "__main__":
    # Simple test runner
    test_class = TestDataCollectorBackgroundRemoval()
    
    print("Testing background removal integration...")
    
    try:
        test_class.setup_method()
        test_class.test_background_removal_enabled_by_default()
        print("✓ Background removal enabled by default")
        
        test_class.test_background_removal_toggle()
        print("✓ Background removal toggle works")
        
        test_class.test_save_sample_with_background_removal()
        print("✓ Save sample with background removal")
        
        test_class.test_save_sample_without_background_removal()
        print("✓ Save sample without background removal")
        
        test_class.test_manifest_includes_background_removal_settings()
        print("✓ Manifest includes background removal settings")
        
        test_class.test_background_remover_initialization()
        print("✓ Background remover initialization")
        
        test_class.test_sample_with_failed_background_removal()
        print("✓ Handle failed background removal")
        
        print("\nAll tests passed! 🎉")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_class.teardown_method() 