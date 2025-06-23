#!/usr/bin/env python3
"""
Live Training Data Collection Script

This script captures ASL signs using the SAME background removal pipeline
as the live system, ensuring perfect match between training and inference data.

Usage:
    python collect_live_training_data.py

Controls:
    A, B, C: Capture current hand sign for that letter
    SPACE: Preview current background removal without saving
    R: Reset background learning
    Q: Quit
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
import json
from typing import Dict, Any

# Import our existing components
from src.asl_cam.vision.asl_hand_detector import ASLHandDetector
from src.asl_cam.vision.background_removal import AdvancedBackgroundRemover

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTrainingDataCollector:
    """
    Collects training data using the exact same pipeline as live ASL recognition.
    This ensures perfect compatibility between training and inference.
    """
    
    def __init__(self):
        self.hand_detector = ASLHandDetector()
        self.background_remover = AdvancedBackgroundRemover()
        
        # Data collection settings
        self.target_size = (224, 224)
        self.data_dir = Path("data/raw/live_training_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each class
        for class_name in ['A', 'B', 'C']:
            (self.data_dir / class_name).mkdir(exist_ok=True)
        
        # Counters
        self.counts = {'A': 0, 'B': 0, 'C': 0}
        self.session_id = int(time.time())
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual camera settings
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"📹 Camera: {width}x{height} @ {fps} FPS")
        
        # Session metadata
        self.session_metadata = {
            'session_id': self.session_id,
            'timestamp': time.time(),
            'camera_settings': {
                'width': width,
                'height': height,
                'fps': fps
            },
            'pipeline_info': {
                'background_removal': 'multi_colorspace_skin_detection',
                'target_size': self.target_size,
                'normalization': 'imagenet_standard'
            },
            'collected_samples': {}
        }
        
        logger.info("🎯 Live Training Data Collector initialized")
        logger.info(f"📂 Data will be saved to: {self.data_dir}")
        
    def _preprocess_hand_crop(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Apply the EXACT same preprocessing as live ASL recognition.
        This ensures training/inference compatibility.
        """
        # Step 1: Background removal (same as live system)
        bg_removed = self.hand_detector._fast_skin_based_removal(hand_crop)
        
        # Step 2: Resize to target size
        resized = cv2.resize(bg_removed, self.target_size)
        
        return resized
    
    def _calculate_data_quality(self, processed_crop: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for the processed hand crop."""
        
        # Calculate background percentage (should be high for good training data)
        black_pixels = np.sum(np.all(processed_crop == [0, 0, 0], axis=2))
        total_pixels = processed_crop.shape[0] * processed_crop.shape[1]
        black_percentage = (black_pixels / total_pixels) * 100
        
        # Calculate skin area percentage
        skin_mask = np.any(processed_crop > 5, axis=2)
        skin_percentage = np.sum(skin_mask) / skin_mask.size * 100
        
        # Calculate color consistency (lower std = better)
        if np.any(skin_mask):
            skin_pixels = processed_crop[skin_mask]
            color_variance = np.mean(np.std(skin_pixels, axis=0))
        else:
            color_variance = 100  # High variance if no skin detected
        
        # Overall quality score
        quality_score = min(100, black_percentage * 0.6 + skin_percentage * 0.4 - color_variance * 0.1)
        
        return {
            'black_percentage': black_percentage,
            'skin_percentage': skin_percentage,
            'color_variance': color_variance,
            'quality_score': max(0, quality_score)
        }
    
    def _save_sample(self, class_name: str, original_crop: np.ndarray, 
                    processed_crop: np.ndarray, quality_metrics: Dict[str, float]):
        """Save a training sample with metadata."""
        
        # Update counter
        self.counts[class_name] += 1
        count = self.counts[class_name]
        
        # Create filename with session and count
        filename_base = f"{class_name}_{self.session_id}_{count:04d}"
        
        # Save processed image (this is what the model will train on)
        processed_path = self.data_dir / class_name / f"{filename_base}_processed.jpg"
        cv2.imwrite(str(processed_path), processed_crop)
        
        # Save original for comparison
        original_path = self.data_dir / class_name / f"{filename_base}_original.jpg"
        cv2.imwrite(str(original_path), original_crop)
        
        # Save metadata
        metadata = {
            'class': class_name,
            'session_id': self.session_id,
            'sample_count': count,
            'timestamp': time.time(),
            'quality_metrics': quality_metrics,
            'file_paths': {
                'processed': str(processed_path),
                'original': str(original_path)
            },
            'pipeline_steps': [
                'hand_detection',
                'roi_extraction', 
                'multi_colorspace_background_removal',
                'resize_224x224',
                'ready_for_imagenet_normalization'
            ]
        }
        
        metadata_path = self.data_dir / class_name / f"{filename_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update session metadata
        if class_name not in self.session_metadata['collected_samples']:
            self.session_metadata['collected_samples'][class_name] = []
        
        self.session_metadata['collected_samples'][class_name].append({
            'count': count,
            'quality_score': quality_metrics['quality_score'],
            'timestamp': metadata['timestamp']
        })
        
        logger.info(f"✅ Saved {class_name} sample #{count} (Quality: {quality_metrics['quality_score']:.1f}/100)")
        
    def _draw_interface(self, frame: np.ndarray, hand_crop: np.ndarray = None, 
                       processed_crop: np.ndarray = None, quality_metrics: Dict[str, float] = None):
        """Draw the collection interface."""
        
        # Draw title
        cv2.putText(frame, "Live Training Data Collection", (30, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Draw controls
        controls = [
            "A, B, C: Capture sign for that letter",
            "SPACE: Preview background removal", 
            "R: Reset background learning",
            "Q: Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (30, 80 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw collection counts
        y_start = 220
        cv2.putText(frame, "Collected Samples:", (30, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        for i, (class_name, count) in enumerate(self.counts.items()):
            color = (0, 255, 0) if count > 0 else (0, 0, 255)
            cv2.putText(frame, f"{class_name}: {count}", (30, y_start + 35 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show current hand preview if available
        if hand_crop is not None and processed_crop is not None:
            # Resize for display
            display_original = cv2.resize(hand_crop, (150, 150))
            display_processed = cv2.resize(processed_crop, (150, 150))
            
            # Position on the right side of the frame
            x_pos = frame.shape[1] - 320
            y_pos = 50
            
            # Place original crop
            frame[y_pos:y_pos+150, x_pos:x_pos+150] = display_original
            cv2.putText(frame, "Original Crop", (x_pos, y_pos - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Place processed crop
            frame[y_pos:y_pos+150, x_pos+160:x_pos+310] = display_processed
            cv2.putText(frame, "Processed (Training)", (x_pos+160, y_pos - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show quality metrics
            if quality_metrics:
                metrics_text = [
                    f"Quality: {quality_metrics['quality_score']:.1f}/100",
                    f"Black BG: {quality_metrics['black_percentage']:.1f}%",
                    f"Skin: {quality_metrics['skin_percentage']:.1f}%"
                ]
                
                for i, text in enumerate(metrics_text):
                    color = (0, 255, 0) if quality_metrics['quality_score'] > 60 else (0, 255, 255) if quality_metrics['quality_score'] > 30 else (0, 0, 255)
                    cv2.putText(frame, text, (x_pos, y_pos + 170 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run(self):
        """Main collection loop."""
        
        logger.info("🟢 Starting Live Training Data Collection...")
        logger.info("📋 Controls: A/B/C=Capture | SPACE=Preview | R=Reset | Q=Quit")
        
        preview_mode = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Update background model
                self.background_remover.learn_background(frame)
                
                # Detect hands
                hand_detections = self.hand_detector.detect_hands(frame)
                current_hand_crop = None
                current_processed_crop = None
                current_quality_metrics = None
                
                if hand_detections:
                    # Use the first detected hand
                    hand_info = hand_detections[0]
                    bbox = hand_info['bbox']
                    x, y, w, h = bbox
                    
                    # Extract hand crop
                    current_hand_crop = frame[y:y+h, x:x+w].copy()
                    
                    # Apply same preprocessing as live system
                    current_processed_crop = self._preprocess_hand_crop(current_hand_crop)
                    
                    # Calculate quality metrics
                    current_quality_metrics = self._calculate_data_quality(current_processed_crop)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Show quality indicator
                    quality_score = current_quality_metrics['quality_score']
                    quality_color = (0, 255, 0) if quality_score > 60 else (0, 255, 255) if quality_score > 30 else (0, 0, 255)
                    cv2.putText(frame, f"Quality: {quality_score:.0f}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
                
                # Draw interface
                frame = self._draw_interface(frame, current_hand_crop, current_processed_crop, current_quality_metrics)
                
                # Show background learning status
                if self.background_remover.learned:
                    cv2.putText(frame, "Background: Learned", (30, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Background: Learning...", (30, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                cv2.imshow("Live Training Data Collection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.background_remover.reset()
                    logger.info("🔄 Background learning reset")
                elif key == ord(' '):
                    preview_mode = not preview_mode
                    logger.info(f"👁️ Preview mode: {'ON' if preview_mode else 'OFF'}")
                elif key in [ord('a'), ord('b'), ord('c')]:
                    class_name = chr(key).upper()
                    
                    if current_hand_crop is not None and current_processed_crop is not None:
                        if current_quality_metrics['quality_score'] > 30:  # Minimum quality threshold
                            self._save_sample(class_name, current_hand_crop, current_processed_crop, current_quality_metrics)
                        else:
                            logger.warning(f"❌ Sample quality too low ({current_quality_metrics['quality_score']:.1f}). Try better lighting or hand position.")
                    else:
                        logger.warning(f"❌ No hand detected. Cannot save {class_name} sample.")
        
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        
        finally:
            # Save session metadata
            self.session_metadata['end_timestamp'] = time.time()
            self.session_metadata['total_samples'] = sum(self.counts.values())
            
            session_file = self.data_dir / f"session_{self.session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(self.session_metadata, f, indent=2)
            
            logger.info(f"📊 Session metadata saved: {session_file}")
            logger.info(f"📈 Total samples collected: {sum(self.counts.values())}")
            for class_name, count in self.counts.items():
                logger.info(f"  {class_name}: {count} samples")
            
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = LiveTrainingDataCollector()
    collector.run() 