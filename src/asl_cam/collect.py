"""
Dataset builder CLI for collecting hand images and labels.
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

from .vision.skin import SkinDetector
from .vision.tracker import HandTracker, TrackedHand

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
        
        # Initialize components
        self.detector = SkinDetector()
        self.tracker = HandTracker()
        
        # Collection state
        self.samples: List[Sample] = []
        self.current_label = "hand"
        self.sample_count = 0
        self.batch_size = 10  # Save every N samples
        
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
            'total_samples': len(self.samples)
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
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
        # Generate filename
        timestamp = time.time()
        filename = f"{self.current_label}_{self.sample_count:06d}_{int(timestamp)}.jpg"
        image_path = self.data_dir / "images" / filename
        
        # Extract hand crop with padding
        x, y, w, h = bbox
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        hand_crop = frame[y1:y2, x1:x2]
        
        # Save full frame and crop
        cv2.imwrite(str(image_path), frame)
        crop_path = str(image_path).replace('.jpg', '_crop.jpg')
        cv2.imwrite(crop_path, hand_crop)
        
        # Create sample record
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
                'padded_bbox': (x1, y1, x2-x1, y2-y1)
            }
        )
        
        self.samples.append(sample)
        self.sample_count += 1
        
        # Save manifest periodically
        if self.sample_count % self.batch_size == 0:
            self._save_manifest()
            print(f"Saved manifest with {self.sample_count} samples")
        
        return str(image_path)
    
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
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n=== Data Collection Mode ===")
        print(f"Current label: {self.current_label}")
        print(f"Samples collected: {self.sample_count}")
        print("\nControls:")
        print("  S - Save current hand detection")
        print("  L - Change label")
        print("  M - Toggle mask view")
        print("  T - Tune detection thresholds")
        print("  Q - Quit")
        print("\nPosition your hand in the frame and press 'S' to save samples...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect hands
            hands = self.detector.detect_hands(frame, max_hands=1)
            
            # Update tracker
            tracked_hands = self.tracker.update(hands)
            
            # Prepare visualization
            if show_mask:
                display_frame = self.detector.visualize_detection(frame, show_mask=True)
            else:
                display_frame = frame.copy()
            
            # Draw tracking information
            if tracked_hands:
                display_frame = self.tracker.draw_tracks(display_frame)
                
                # Show collection status
                primary_hand = self.tracker.get_primary_hand()
                if primary_hand and primary_hand.hits > 5:
                    x, y, w, h = primary_hand.bbox
                    cv2.putText(display_frame, "READY TO COLLECT", (x, y + h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add UI info
            info_y = 30
            cv2.putText(display_frame, f"Label: {self.current_label}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Samples: {self.sample_count}", (10, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Hands: {len(tracked_hands)}", (10, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Data Collection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save sample
                if tracked_hands:
                    primary_hand = self.tracker.get_primary_hand()
                    if primary_hand and primary_hand.hits > 5:
                        confidence = min(1.0, primary_hand.hits / 20.0)
                        image_path = self._save_sample(frame, primary_hand.bbox, confidence)
                        print(f"Saved sample {self.sample_count}: {image_path}")
                    else:
                        print("Hand not stable enough for collection")
                else:
                    print("No hands detected")
            elif key == ord('l'):
                # Change label
                new_label = input("Enter new label: ").strip()
                if new_label:
                    self.current_label = new_label
                    print(f"Label changed to: {self.current_label}")
            elif key == ord('m'):
                # Toggle mask view
                show_mask = not show_mask
                print(f"Mask view: {'ON' if show_mask else 'OFF'}")
            elif key == ord('t'):
                # Tune thresholds
                print("Opening threshold tuning...")
                self.detector.tune_thresholds(frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final manifest
        self._save_manifest()
        print(f"Collection finished. Total samples: {self.sample_count}")
    
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
        
        # Convert samples
        for sample in self.samples:
            # Load image to get dimensions
            img = cv2.imread(sample.image_path)
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
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        if not self.samples:
            return {"total_samples": 0}
        
        from collections import Counter
        
        label_counts = Counter(sample.label for sample in self.samples)
        confidence_avg = np.mean([sample.confidence for sample in self.samples])
        
        return {
            "total_samples": len(self.samples),
            "labels": dict(label_counts),
            "average_confidence": confidence_avg,
            "data_directory": str(self.data_dir)
        }

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ASL Hand Detection Data Collector")
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
            print(f"{key}: {value}")
    elif args.export_yolo:
        collector.export_yolo_format(args.export_yolo)
    else:
        collector.collect_interactive(args.camera, args.show_mask)

if __name__ == "__main__":
    main() 