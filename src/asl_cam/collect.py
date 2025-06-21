"""
Dataset builder CLI for collecting hand images and labels.

This module provides interactive data collection with integrated background removal
to create clean training datasets that match the target model format. The collector
can optionally apply advanced background removal techniques (GrabCut, contour-based,
skin detection, etc.) to create clean hand images while preserving original versions
for comparison.

Features:
- Real-time hand detection and tracking
- Quality-controlled sample collection
- Multiple background removal methods
- Comprehensive metadata tracking
- YOLO format export capability
"""
import cv2
import numpy as np
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import logging

from .vision.skin import SkinDetector
from .vision.enhanced_hand_detector import EnhancedHandDetector
from .vision.simple_hand_detector import SimpleHandDetector
from .vision.background_removal import BackgroundRemover
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)

@dataclass
class Sample:
    """Container for a collected sample."""
    image_path: str
    bbox: Tuple[int, int, int, int]
    label: str
    timestamp: float
    confidence: float
    metadata: Dict

class DataCollector:
    """Collects and manages hand detection training data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data collector.
        
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "images").mkdir(exist_ok=True)
        (self.data_dir / "annotations").mkdir(exist_ok=True)
        
        # Initialize components with optimized settings
        self.detector = SimpleHandDetector()  # Use simple motion-based detection
        self.tracker = MultiHandTracker(max_hands=1, max_disappeared=10)  # Single hand, faster cleanup
        self.bg_remover = BackgroundRemover(BackgroundMethod.GRABCUT)  # High-quality background removal
        
        # Simple background removal method cycling
        self.bg_methods = [BackgroundMethod.GRABCUT, BackgroundMethod.CONTOUR_MASK, BackgroundMethod.SKIN_MASK]
        self.bg_method_names = ["dominant", "adaptive", "edge"]
        self.current_bg_method_idx = 0
        self.bg_tolerance = 40  # Color tolerance for background removal
        
        # Collection state
        self.samples: List[Sample] = []
        self.current_label = "hand"
        self.sample_count = 0
        self.batch_size = 5  # Save more frequently for safety
        
        # Quality control parameters
        self.min_stability_hits = 8  # Require more stability before allowing collection
        self.min_hand_size = 50      # Minimum hand size (width or height)
        self.max_hand_size = 400     # Maximum hand size to filter out false positives
        
        # Background removal settings
        self.use_background_removal = True  # Enable background removal for training data
        self.save_both_versions = True      # Save both original and background-removed versions
        
        # Detection mode settings
        self.use_motion_detection = True     # Use motion-based filtering by default
        self.show_motion_mask = False       # Show motion mask overlay
        self.show_skin_mask = False         # Show skin mask overlay
        
        # Load existing data
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """Load existing samples from disk."""
        manifest_file = self.data_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                data = json.load(f)
                self.samples = [Sample(**item) for item in data.get('samples', [])]
                self.sample_count = len(self.samples)
                print(f"Loaded {self.sample_count} existing samples")
    
    def _save_manifest(self) -> None:
        """Save sample manifest to disk."""
        manifest_file = self.data_dir / "manifest.json"
        manifest_data = {
            'samples': [asdict(sample) for sample in self.samples],
            'created': time.time(),
            'total_samples': len(self.samples),
            'current_label': self.current_label,
            'collection_settings': {
                'min_stability_hits': self.min_stability_hits,
                'min_hand_size': self.min_hand_size,
                'max_hand_size': self.max_hand_size,
                'use_background_removal': self.use_background_removal,
                'save_both_versions': self.save_both_versions,
                'bg_removal_method': self.bg_remover.method.value
            }
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
    def _is_hand_valid_for_collection(self, hand: TrackedHand) -> Tuple[bool, str]:
        """
        Check if a tracked hand is valid for data collection.
        
        Args:
            hand: Tracked hand object
            
        Returns:
            (is_valid, reason) - validation result and reason
        """
        # Check stability
        if hand.hits < self.min_stability_hits:
            return False, f"Not stable enough (hits: {hand.hits}, need: {self.min_stability_hits})"
        
        # Check recent updates
        if hand.time_since_update > 2:
            return False, f"Hand not recently detected (last update: {hand.time_since_update} frames ago)"
        
        # Check hand size
        x, y, w, h = hand.bbox
        if w < self.min_hand_size or h < self.min_hand_size:
            return False, f"Hand too small ({w}x{h}, min: {self.min_hand_size})"
            
        if w > self.max_hand_size or h > self.max_hand_size:
            return False, f"Hand too large ({w}x{h}, max: {self.max_hand_size})"
        
        # Check aspect ratio (hands shouldn't be too elongated)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.5:
            return False, f"Hand aspect ratio too extreme ({aspect_ratio:.1f})"
        
        return True, "Valid for collection"
    
    def _save_sample(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                    confidence: float) -> str:
        """
        Save a single sample to disk.
        
        Args:
            frame: Full frame image
            bbox: Hand bounding box
            confidence: Detection confidence
            
        Returns:
            Path to saved image
        """
        # Generate filename with more info
        timestamp = time.time()
        filename = f"{self.current_label}_{self.sample_count:06d}_{int(timestamp)}.jpg"
        image_path = self.data_dir / "images" / filename
        
        # Extract hand crop with optimized padding
        x, y, w, h = bbox
        padding = max(20, min(w, h) // 4)  # Adaptive padding based on hand size
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        hand_crop = frame[y1:y2, x1:x2]
        
        # Apply background removal if enabled
        processed_crop = hand_crop
        bg_removed_crop = None
        if self.use_background_removal:
            # Get skin mask for the hand region to improve background removal
            skin_mask = self.detector.detect_skin_mask(frame)
            skin_mask_crop = skin_mask[y:y+h, x:x+w] if skin_mask is not None else None
            
            # Apply enhanced background removal designed for hand crops
            result = self.bg_remover.remove_background_from_crop(hand_crop, skin_mask_crop)
            if result is not None:
                bg_removed_crop, mask = result  # Unpack the tuple
                if bg_removed_crop is not None:
                    processed_crop = bg_removed_crop
        
        # Save full frame and crop
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_path = str(image_path).replace('.jpg', '_crop.jpg')
        cv2.imwrite(crop_path, processed_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Optionally save original crop version as well
        if self.save_both_versions and bg_removed_crop is not None:
            original_crop_path = str(image_path).replace('.jpg', '_crop_original.jpg')
            cv2.imwrite(original_crop_path, hand_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create comprehensive sample record
        sample = Sample(
            image_path=str(image_path),
            bbox=bbox,
            label=self.current_label,
            timestamp=timestamp,
            confidence=confidence,
            metadata={
                'crop_path': crop_path,
                'frame_shape': frame.shape,
                'hand_crop_shape': hand_crop.shape,
                'original_bbox': bbox,
                'padded_bbox': (x1, y1, x2-x1, y2-y1),
                'hand_area': w * h,
                'padding_used': padding,
                'aspect_ratio': w / h,
                'background_removed': self.use_background_removal and bg_removed_crop is not None,
                'original_crop_path': str(image_path).replace('.jpg', '_crop_original.jpg') if self.save_both_versions and bg_removed_crop is not None else None,
                'bg_removal_method': self.bg_remover.method.value if self.use_background_removal else None
            }
        )
        
        self.samples.append(sample)
        self.sample_count += 1
        
        # Save manifest periodically
        if self.sample_count % self.batch_size == 0:
            self._save_manifest()
            print(f"Saved manifest with {self.sample_count} samples")
        
        return str(image_path)
    
    def _capture_and_visualize_hand_data(self, frame: np.ndarray, hand: 'TrackedHand') -> None:
        """
        Capture and visualize detailed hand data in a separate window.
        
        Args:
            frame: Current frame
            hand: Tracked hand object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract hand region
        x, y, w, h = hand.bbox
        hand_crop = frame[y:y+h, x:x+w].copy()
        
        # Get detection masks
        skin_mask = self.detector.detect_skin_mask(frame)
        motion_mask = self.detector.detect_motion_mask(frame)
        
        # Extract mask regions
        skin_crop = skin_mask[y:y+h, x:x+w]
        motion_crop = motion_mask[y:y+h, x:x+w]
        
        # Apply background removal if enabled
        processed_crop = hand_crop.copy()
        if self.use_background_removal:
            # Get skin mask for the hand region to improve background removal
            skin_mask = self.detector.detect_skin_mask(frame)
            skin_mask_crop = skin_mask[y:y+h, x:x+w] if skin_mask is not None else None
            
            # Apply enhanced background removal designed for hand crops
            result = self.bg_remover.remove_background_from_crop(hand_crop, skin_mask_crop)
            if result is not None:
                bg_removed_crop, mask = result
                if bg_removed_crop is not None:
                    processed_crop = bg_removed_crop
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Hand Data Capture - Label: {self.current_label}', fontsize=16, fontweight='bold')
        
        # Original hand crop
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Hand Crop\n{w}x{h} pixels')
        plt.axis('off')
        
        # Processed hand crop (with background removal if enabled)
        plt.subplot(2, 4, 2)
        if self.use_background_removal and processed_crop is not hand_crop:
            plt.imshow(cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB))
            plt.title('Background Removed')
        else:
            plt.imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            plt.title('Processed Crop\n(No BG removal)')
        plt.axis('off')
        
        # Skin mask
        plt.subplot(2, 4, 3)
        plt.imshow(skin_crop, cmap='Reds')
        plt.title('Skin Detection Mask')
        plt.axis('off')
        
        # Motion mask
        plt.subplot(2, 4, 4)
        plt.imshow(motion_crop, cmap='Blues')
        plt.title('Motion Detection Mask')
        plt.axis('off')
        
        # Hand data info
        plt.subplot(2, 4, 5)
        plt.axis('off')
        
        # Calculate some statistics
        skin_pixels = np.sum(skin_crop > 0)
        motion_pixels = np.sum(motion_crop > 0)
        total_pixels = w * h
        
        info_text = f"""Hand Information:
        
Size: {w} × {h} pixels
Area: {total_pixels:,} pixels
Aspect Ratio: {w/h:.2f}

Tracking:
Stability: {hand.hits} hits
Confidence: {min(1.0, hand.hits / 20.0):.2f}

Detection:
Skin pixels: {skin_pixels:,} ({100*skin_pixels/total_pixels:.1f}%)
Motion pixels: {motion_pixels:,} ({100*motion_pixels/total_pixels:.1f}%)

Settings:
Label: {self.current_label}
Motion detection: {'ON' if self.use_motion_detection else 'OFF'}
Background removal: {'ON' if self.use_background_removal else 'OFF'}
Persistence: {self.detector.max_persistence_frames}f
"""
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Full frame context
        plt.subplot(2, 4, 6)
        frame_display = frame.copy()
        # Draw hand bbox
        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_display, f'Hand: {w}x{h}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
        plt.title('Full Frame Context')
        plt.axis('off')
        
        # Combined masks visualization
        plt.subplot(2, 4, 7)
        combined = np.zeros_like(hand_crop)
        if len(hand_crop.shape) == 3:
            combined[:, :, 0] = motion_crop  # Red channel for motion
            combined[:, :, 1] = skin_crop   # Green channel for skin
            overlap = np.logical_and(skin_crop > 0, motion_crop > 0)
            combined[overlap, 2] = 255      # Blue for overlap
        
        plt.imshow(combined)
        plt.title('Combined Masks\nRed=Motion, Green=Skin, Blue=Both')
        plt.axis('off')
        
        # Histogram of hand crop
        plt.subplot(2, 4, 8)
        if len(hand_crop.shape) == 3:
            for i, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([hand_crop], [i], None, [256], [0, 256])
                plt.plot(hist, color=color, alpha=0.7, label=f'Channel {i}')
        plt.title('Color Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking to allow continued collection
        
        # Save capture data if requested
        timestamp = time.time()
        capture_dir = Path(self.data_dir) / "captures"
        capture_dir.mkdir(exist_ok=True)
        
        capture_base = capture_dir / f"capture_{self.current_label}_{timestamp:.0f}"
        
        # Save images
        cv2.imwrite(f"{capture_base}_original.jpg", hand_crop)
        cv2.imwrite(f"{capture_base}_processed.jpg", processed_crop)
        cv2.imwrite(f"{capture_base}_skin_mask.jpg", skin_crop)
        cv2.imwrite(f"{capture_base}_motion_mask.jpg", motion_crop)
        
        # Save metadata
        capture_metadata = {
            'timestamp': timestamp,
            'label': self.current_label,
            'bbox': hand.bbox,
            'hand_size': [w, h],
            'hand_area': total_pixels,
            'aspect_ratio': w / h,
            'tracking_hits': hand.hits,
            'confidence': min(1.0, hand.hits / 20.0),
            'skin_pixels': int(skin_pixels),
            'motion_pixels': int(motion_pixels),
            'skin_percentage': float(100 * skin_pixels / total_pixels),
            'motion_percentage': float(100 * motion_pixels / total_pixels),
            'settings': {
                'motion_detection': self.use_motion_detection,
                'background_removal': self.use_background_removal,
                'persistence_frames': self.detector.max_persistence_frames
            },
            'files': {
                'original': f"{capture_base.name}_original.jpg",
                'processed': f"{capture_base.name}_processed.jpg", 
                'skin_mask': f"{capture_base.name}_skin_mask.jpg",
                'motion_mask': f"{capture_base.name}_motion_mask.jpg"
            }
        }
        
        with open(f"{capture_base}_metadata.json", 'w') as f:
            json.dump(capture_metadata, f, indent=2)
        
        print(f"\n✓ Hand data captured and visualized!")
        print(f"  Size: {w}×{h} pixels ({total_pixels:,} total)")
        print(f"  Tracking: {hand.hits} hits (confidence: {min(1.0, hand.hits / 20.0):.2f})")
        print(f"  Skin coverage: {100*skin_pixels/total_pixels:.1f}%")
        print(f"  Motion coverage: {100*motion_pixels/total_pixels:.1f}%")
        print(f"  Files saved to: {capture_dir}")
        print(f"  Close the visualization window when done viewing.")

    def collect_interactive(self, camera_id: int = 0, show_mask: bool = False) -> None:
        """
        Interactive data collection from camera.
        
        Args:
            camera_id: Camera device ID
            show_mask: Whether to show skin detection mask
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Optimized camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("\n=== Optimized Data Collection Mode ===")
        print(f"Current label: {self.current_label}")
        print(f"Samples collected: {self.sample_count}")
        print(f"Quality requirements:")
        print(f"  - Minimum stability: {self.min_stability_hits} hits")
        print(f"  - Hand size range: {self.min_hand_size}-{self.max_hand_size} pixels")
        print(f"  - Maximum aspect ratio: 2.5")
        print("\nControls:")
        print("  S - Save current hand detection (if valid)")
        print("  L - Change label")
        print("  M - Toggle motion mask overlay")
        print("  K - Toggle skin mask overlay")
        print("  B - Toggle background removal")
        print("  N - Cycle background removal method (dominant/adaptive/edge)")
        print("  Z - Adjust background removal tolerance (strictness)")
        print("  X - Toggle motion detection (filters out static torso)")
        print("  P - Adjust hand persistence (how long to keep tracking still hands)")
        print("  C - Capture and visualize current hand data")
        print("  R - Reset motion detection & tracker")
        print("  T - Tune detection thresholds")
        print("  Q - Quit")
        print("\nPosition your hand in the frame and wait for GREEN 'READY' status...")
        
        frames_processed = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Detect hands using motion + skin detection
            hands = self.detector.detect_hands_simple(frame, max_hands=1, use_motion=self.use_motion_detection)
            
            # Update tracker
            tracked_hands = self.tracker.update(hands)
            
            # Prepare visualization
            if self.show_motion_mask or self.show_skin_mask:
                display_frame = self.detector.visualize_simple_detection(
                    frame, show_motion=self.show_motion_mask, show_skin=self.show_skin_mask)
            else:
                display_frame = frame.copy()
            
            # Draw tracking information
            display_frame = self.tracker.draw_tracks(display_frame)
            
            # Check collection readiness
            collection_ready = False
            primary_hand = self.tracker.get_primary_hand()
            status_message = "No stable hand detected"
            status_color = (0, 0, 255)  # Red
            
            if primary_hand:
                is_valid, reason = self._is_hand_valid_for_collection(primary_hand)
                if is_valid:
                    collection_ready = True
                    status_message = "READY TO COLLECT"
                    status_color = (0, 255, 0)  # Green
                    
                    # Add collection indicator
                    x, y, w, h = primary_hand.bbox
                    cv2.rectangle(display_frame, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)
                else:
                    status_message = f"Not ready: {reason}"
                    status_color = (0, 255, 255)  # Yellow
            
            # Add comprehensive UI info
            info_y = 25
            line_height = 25
            
            # Status
            cv2.putText(display_frame, status_message, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            info_y += line_height
            
            # Collection info
            if self.use_background_removal:
                bg_method = self.bg_method_names[self.current_bg_method_idx]
                bg_status = f"BG: {bg_method}({self.bg_tolerance})"
            else:
                bg_status = "BG: OFF"
            motion_status = "Motion: ON" if self.use_motion_detection else "Motion: OFF"
            persist_frames = self.detector.max_persistence_frames
            persist_status = f"Persist: {persist_frames}f"
            cv2.putText(display_frame, f"Label: {self.current_label} | Samples: {self.sample_count} | {bg_status} | {motion_status} | {persist_status}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            info_y += line_height
            
            # Performance info
            if frames_processed % 30 == 0:  # Update every 30 frames
                fps = frames_processed / (time.time() - start_time)
                cv2.putText(display_frame, f"FPS: {fps:.1f} | Hands: {len(tracked_hands)}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('ASL Data Collection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save sample
                if collection_ready and primary_hand:
                    confidence = min(1.0, primary_hand.hits / 20.0)
                    image_path = self._save_sample(frame, primary_hand.bbox, confidence)
                    print(f"✓ Saved sample {self.sample_count}: {image_path}")
                    print(f"  Hand size: {primary_hand.bbox[2]}x{primary_hand.bbox[3]}")
                    print(f"  Stability: {primary_hand.hits} hits")
                else:
                    print(f"✗ Cannot save: {status_message}")
            elif key == ord('l'):
                # Change label
                new_label = input("\nEnter new label: ").strip()
                if new_label:
                    self.current_label = new_label
                    print(f"Label changed to: {self.current_label}")
            elif key == ord('m'):
                # Toggle motion mask overlay
                self.show_motion_mask = not self.show_motion_mask
                print(f"Motion mask overlay: {'ON' if self.show_motion_mask else 'OFF'}")
            elif key == ord('k'):
                # Toggle skin mask overlay
                self.show_skin_mask = not self.show_skin_mask
                print(f"Skin mask overlay: {'ON' if self.show_skin_mask else 'OFF'}")
            elif key == ord('b'):
                # Toggle background removal
                self.use_background_removal = not self.use_background_removal
                print(f"Background removal: {'ON' if self.use_background_removal else 'OFF'}")
            elif key == ord('n'):
                # Cycle background removal method
                self.current_bg_method_idx = (self.current_bg_method_idx + 1) % len(self.bg_methods)
                new_method = self.bg_methods[self.current_bg_method_idx]
                method_name = self.bg_method_names[self.current_bg_method_idx]
                self.bg_remover = BackgroundRemover(new_method)
                self.bg_remover.remover.color_tolerance = self.bg_tolerance
                print(f"Background removal method: {method_name} (simple color-based)")
                if not self.use_background_removal:
                    print("  (Enable with 'B' key to see effect)")
            elif key == ord('z'):
                # Adjust background removal tolerance
                print(f"\nCurrent tolerance: {self.bg_tolerance} (lower = stricter)")
                print("Options: 1=Very strict (20), 2=Strict (30), 3=Normal (40), 4=Loose (50), 5=Very loose (60)")
                choice = input("Choose tolerance level (1-5): ")
                tolerance_map = {'1': 20, '2': 30, '3': 40, '4': 50, '5': 60}
                if choice in tolerance_map:
                    self.bg_tolerance = tolerance_map[choice]
                    self.bg_remover.remover.color_tolerance = self.bg_tolerance
                    strictness = ["very strict", "strict", "normal", "loose", "very loose"][int(choice)-1]
                    print(f"Background removal tolerance set to: {self.bg_tolerance} ({strictness})")
            elif key == ord('x'):
                # Toggle motion detection
                self.use_motion_detection = not self.use_motion_detection
                mode_desc = "Motion filtering (filters out static torso)" if self.use_motion_detection else "All skin detection"
                print(f"Detection mode: {mode_desc}")
            elif key == ord('p'):
                # Adjust hand persistence
                current = self.detector.max_persistence_frames
                print(f"\nCurrent hand persistence: {current} frames (~{current/30:.1f} seconds)")
                print("Options: 1=Short (15 frames), 2=Medium (30 frames), 3=Long (60 frames)")
                choice = input("Choose persistence level (1-3): ")
                if choice == '1':
                    self.detector.max_persistence_frames = 15
                elif choice == '2':
                    self.detector.max_persistence_frames = 30
                elif choice == '3':
                    self.detector.max_persistence_frames = 60
                print(f"Hand persistence set to: {self.detector.max_persistence_frames} frames")
            elif key == ord('c'):
                # Capture and visualize current hand data
                if collection_ready and primary_hand:
                    self._capture_and_visualize_hand_data(frame, primary_hand)
                else:
                    print(f"✗ Cannot capture: {status_message}")
            elif key == ord('t'):
                # Tune thresholds
                print("Opening threshold tuning...")
                self.detector.tune_thresholds(frame)
            elif key == ord('r'):
                # Reset motion detection and tracker
                self.detector.reset_motion_detection()
                self.tracker = MultiHandTracker(max_hands=1, max_disappeared=10)
                print("Motion detection and tracker reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final manifest
        self._save_manifest()
        print(f"\nCollection finished!")
        print(f"Total samples collected: {self.sample_count}")
        print(f"Average FPS: {frames_processed / (time.time() - start_time):.1f}")
    
    def export_yolo_format(self, output_dir: str) -> None:
        """
        Export collected data in YOLO format.
        
        Args:
            output_dir: Directory to save YOLO format files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create label mapping
        unique_labels = list(set(sample.label for sample in self.samples))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Save class names
        with open(output_path / "classes.txt", 'w') as f:
            for label in unique_labels:
                f.write(f"{label}\n")
        
        # Save dataset info
        dataset_info = {
            'total_samples': len(self.samples),
            'classes': unique_labels,
            'label_distribution': {label: sum(1 for s in self.samples if s.label == label) 
                                 for label in unique_labels},
            'export_timestamp': time.time()
        }
        
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Convert samples
        for sample in self.samples:
            # Load image to get dimensions
            img = cv2.imread(sample.image_path)
            if img is None:
                print(f"Warning: Could not load image {sample.image_path}")
                continue
                
            h, w = img.shape[:2]
            
            # Convert bbox to YOLO format (normalized center coordinates)
            x, y, bw, bh = sample.bbox
            center_x = (x + bw / 2) / w
            center_y = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            
            # Create annotation file
            base_name = Path(sample.image_path).stem
            ann_file = output_path / f"{base_name}.txt"
            
            class_id = label_map[sample.label]
            with open(ann_file, 'w') as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Exported {len(self.samples)} samples to YOLO format in {output_path}")
        print(f"Classes: {unique_labels}")
        print(f"Dataset info saved to: {output_path}/dataset_info.json")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive collection statistics."""
        if not self.samples:
            return {"total_samples": 0}
        
        from collections import Counter
        
        label_counts = Counter(sample.label for sample in self.samples)
        confidence_avg = np.mean([sample.confidence for sample in self.samples])
        
        # Analyze hand sizes
        areas = [sample.metadata.get('hand_area', 0) for sample in self.samples]
        aspect_ratios = [sample.metadata.get('aspect_ratio', 1.0) for sample in self.samples]
        
        return {
            "total_samples": len(self.samples),
            "labels": dict(label_counts),
            "average_confidence": confidence_avg,
            "hand_size_stats": {
                "min_area": min(areas) if areas else 0,
                "max_area": max(areas) if areas else 0,
                "avg_area": np.mean(areas) if areas else 0
            },
            "aspect_ratio_stats": {
                "min": min(aspect_ratios) if aspect_ratios else 0,
                "max": max(aspect_ratios) if aspect_ratios else 0,
                "avg": np.mean(aspect_ratios) if aspect_ratios else 0
            },
            "quality_settings": {
                "min_stability_hits": self.min_stability_hits,
                "min_hand_size": self.min_hand_size,
                "max_hand_size": self.max_hand_size
            },
            "data_directory": str(self.data_dir)
        }

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Optimized ASL Hand Detection Data Collector")
    parser.add_argument("--data-dir", default="data/raw", 
                       help="Directory to save collected data")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID")
    parser.add_argument("--show-mask", action="store_true",
                       help="Show skin detection mask")
    parser.add_argument("--export-yolo", 
                       help="Export to YOLO format in specified directory")
    parser.add_argument("--stats", action="store_true",
                       help="Show collection statistics")
    
    args = parser.parse_args()
    
    collector = DataCollector(args.data_dir)
    
    if args.stats:
        stats = collector.get_statistics()
        print("\n=== Collection Statistics ===")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
    elif args.export_yolo:
        collector.export_yolo_format(args.export_yolo)
    else:
        collector.collect_interactive(args.camera, args.show_mask)

if __name__ == "__main__":
    main() 