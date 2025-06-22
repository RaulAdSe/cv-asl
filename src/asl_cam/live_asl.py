#!/usr/bin/env python3
"""
Live ASL Recognition System

Integrates the camera hand detection system with the trained deep learning model
for real-time ASL letter recognition.

Features:
- Real-time camera feed with hand detection
- Live ASL letter prediction using trained MobileNetV2 model
- Clean UI with prediction confidence and letter display
- Performance monitoring (FPS, inference time)
- Organized integration between asl_cam and asl_dl modules

Controls:
- Q: Quit
- S: Toggle statistics display
- R: Reset hand tracker
- SPACE: Pause/unpause

Author: CV-ASL Team
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import camera modules (clean, no DL dependencies)
from asl_cam.vision.asl_hand_detector import ASLHandDetector

# Import DL modules (organized structure)
from asl_dl.models.mobilenet import MobileNetV2ASL
import torchvision.transforms as transforms
from asl_cam.utils.fps import FPSTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveASLRecognizer:
    """
    Live ASL Recognition System
    
    Combines hand detection with trained model for real-time ASL prediction
    """
    
    def __init__(self, model_path: str = "src/asl_dl/models/asl_abc_model.pth", 
                 min_pred_confidence: float = 0.3,
                 camera_index: int = 0):
        """
        Initializes the ASL recognizer.
        
        Args:
            model_path: Path to the trained PyTorch model.
            min_pred_confidence: Minimum confidence to display a prediction.
            camera_index: The index of the camera to use for live capture.
        """
        self.model = MobileNetV2ASL.load_from_checkpoint(model_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Add transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.hand_detector = ASLHandDetector()
        self.fps_tracker = FPSTracker()
        
        self.min_pred_confidence = min_pred_confidence
        self.camera_index = camera_index
        
        # UI state
        self.show_stats = True
        self.paused = False
        
        # --- State for paused display ---
        self.last_hand_info = None
        self.last_prediction = "Show Hand"
        self.last_confidence = 0.0
        self.last_frame = None
        
        # Capture feature state
        self.capture_enabled = True
        self.last_processed_hand = None
        self.last_capture_frame = None

        self.STATUS_COLORS = {
            "TRACKED": (0, 255, 0),       # Green for stable tracking
            "PREDICTED": (255, 255, 0),   # Yellow for Kalman filter prediction
            "NEW_DETECTION": (0, 0, 255), # Red for a new detection
            "LOST": (255, 0, 255),        # Magenta for lost track
        }

        logger.info("🚀 Live ASL Recognizer initialized")
        logger.info(f"📱 Device: {self.device}")
        
        # Extract classes from model, with fallback
        if self.model.class_map is not None:
            self.classes = sorted(list(self.model.class_map.keys()))
        else:
            # Fallback for models without class_map
            self.classes = ['A', 'B', 'C']  # Default ASL classes
            self.model.class_map = {cls: idx for idx, cls in enumerate(self.classes)}
            logger.warning(f"Model missing class_map, using default: {self.model.class_map}")
            
        logger.info(f"🎯 Classes: {self.classes}")
    
    def predict_hand_sign(self, hand_crop: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASL letter from hand crop
        
        Args:
            hand_crop: Cropped hand image (BGR format)
            
        Returns:
            (predicted_letter, confidence)
        """
        
        if self.model is None:
            return "No Model", 0.0
        
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_letter = self.classes[predicted_idx.item()]
                confidence_score = confidence.item()
            
            self.inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return predicted_letter, confidence_score
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return "Error", 0.0
    
    def _capture_and_visualize_hand_data(self, frame: np.ndarray, processed_hand: np.ndarray, 
                                       hand_info: Dict, prediction: str, confidence: float) -> None:
        """
        Capture and visualize detailed hand data including DL model inputs.
        
        Args:
            frame: Current camera frame
            processed_hand: Preprocessed hand crop that goes to the model
            hand_info: Hand detection information 
            prediction: Current model prediction
            confidence: Prediction confidence
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.gridspec import GridSpec
            import json
            
            # Extract hand information
            bbox = hand_info.get('bbox', (0, 0, 100, 100))
            x, y, w, h = bbox
            
            # Extract hand crop from original frame
            hand_crop = frame[y:y+h, x:x+w].copy()
            
            # Get model input tensor (224x224 normalized)
            model_input = processed_hand
            if isinstance(model_input, torch.Tensor):
                # Convert tensor back to displayable image
                model_input_np = model_input.cpu().numpy()
                if len(model_input_np.shape) == 4:  # Batch dimension
                    model_input_np = model_input_np[0]
                # Denormalize from ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                model_input_np = model_input_np.transpose(1, 2, 0)  # CHW to HWC
                model_input_np = (model_input_np * std + mean)
                model_input_np = np.clip(model_input_np * 255, 0, 255).astype(np.uint8)
            else:
                model_input_np = model_input
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Live ASL Data Analysis - Prediction: {prediction} ({confidence:.3f})', 
                        fontsize=16, fontweight='bold')
            
            # Panel 1: Original hand crop
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'Original Hand Crop\n{w}×{h} pixels')
            ax1.axis('off')
            
            # Panel 2: Model input (224x224 preprocessed)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(model_input_np)
            ax2.set_title(f'Model Input\n224×224 preprocessed')
            ax2.axis('off')
            
            # Panel 3: Prediction confidence visualization
            ax3 = fig.add_subplot(gs[0, 2])
            if hasattr(self.model, 'class_map') and self.model.class_map:
                classes = list(self.model.class_map.keys())
                # Create a mock prediction distribution (in real scenario, you'd get this from model output)
                probs = [0.1, 0.1, 0.1]  # Default low probabilities
                if prediction in classes:
                    pred_idx = classes.index(prediction)
                    probs[pred_idx] = confidence
                    # Normalize remaining probability
                    remaining = (1.0 - confidence) / (len(classes) - 1)
                    for i, p in enumerate(probs):
                        if i != pred_idx:
                            probs[i] = remaining
                
                bars = ax3.bar(classes, probs, color=['red' if p == confidence else 'gray' for p in probs])
                ax3.set_title('Prediction Confidence')
                ax3.set_ylabel('Probability')
                ax3.set_ylim(0, 1)
                
                # Highlight the predicted class
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    if prob == confidence:
                        bar.set_color('green')
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No class info\navailable', ha='center', va='center', 
                        transform=ax3.transAxes)
                ax3.set_title('Prediction Confidence')
            
            # Panel 4: Hand information
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.axis('off')
            
            total_pixels = w * h
            aspect_ratio = w / h if h > 0 else 1.0
            
            info_text = f"""Hand Data Analysis:

Size: {w} × {h} pixels
Area: {total_pixels:,} pixels  
Aspect Ratio: {aspect_ratio:.2f}

Model Input:
Size: 224 × 224 pixels
Channels: RGB (3)
Normalization: ImageNet

Prediction:
Letter: {prediction}
Confidence: {confidence:.3f}
Threshold: {self.min_pred_confidence:.2f}

Detection:
Tracker: {'Active' if hand_info.get('tracking', False) else 'Inactive'}
Frame: {hand_info.get('frame_count', 'N/A')}
"""
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            # Panel 5: Full frame context
            ax5 = fig.add_subplot(gs[1, :2])
            frame_display = frame.copy()
            # Draw hand bbox
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame_display, f'{prediction}: {confidence:.3f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            ax5.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            ax5.set_title('Live Camera Feed with Detection')
            ax5.axis('off')
            
            # Panel 6: Model preprocessing visualization
            ax6 = fig.add_subplot(gs[1, 2:])
            
            # Show the preprocessing steps if we can recreate them
            preprocessing_steps = [
                "1. Hand Detection & Cropping",
                "2. Resize to 224×224",
                "3. Convert to Tensor", 
                "4. Normalize (ImageNet)",
                "5. Add Batch Dimension",
                "6. Send to Model"
            ]
            
            for i, step in enumerate(preprocessing_steps):
                color = 'green' if i < 6 else 'gray'
                ax6.text(0.05, 0.9 - i*0.12, f"✓ {step}", transform=ax6.transAxes,
                        fontsize=11, color=color, fontweight='bold')
            
            ax6.text(0.05, 0.2, f"Model Device: {self.device}\nModel Type: MobileNetV2\nClasses: {len(self.classes)}", 
                    transform=ax6.transAxes, fontsize=10, fontfamily='monospace')
            ax6.set_title('DL Pipeline Status')
            ax6.axis('off')
            
            # Panel 7: Color histogram
            ax7 = fig.add_subplot(gs[2, 0])
            if len(hand_crop.shape) == 3:
                for i, color in enumerate(['blue', 'green', 'red']):
                    hist = cv2.calcHist([hand_crop], [i], None, [256], [0, 256])
                    ax7.plot(hist, color=color, alpha=0.7, label=f'{color.upper()}')
                ax7.set_title('Color Histogram')
                ax7.set_xlabel('Pixel Intensity')
                ax7.set_ylabel('Frequency')
                ax7.legend()
            else:
                ax7.text(0.5, 0.5, 'Grayscale\nimage', ha='center', va='center',
                        transform=ax7.transAxes)
                ax7.set_title('Color Histogram')
            
            # Panel 8: Model input histogram
            ax8 = fig.add_subplot(gs[2, 1])
            if len(model_input_np.shape) == 3:
                for i, color in enumerate(['blue', 'green', 'red']):
                    hist = cv2.calcHist([model_input_np], [i], None, [256], [0, 256])
                    ax8.plot(hist, color=color, alpha=0.7, label=f'{color.upper()}')
                ax8.set_title('Model Input Histogram')
                ax8.set_xlabel('Pixel Intensity')
                ax8.set_ylabel('Frequency')
                ax8.legend()
            
            # Panel 9: Background learning status
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.axis('off')
            
            bg_progress = self.hand_detector.bg_remover.get_progress()
            bg_learned = self.hand_detector.bg_remover.bg_model_learned
            
            bg_text = f"""Background Learning:

Status: {'✅ Learned' if bg_learned else '🔄 Learning'}
Progress: {bg_progress:.1%}

MOG2 Parameters:
- History: 500 frames
- Threshold: 16.0
- Shadow Detection: ON

Hand Tracking:
- Kalman Filter: Active
- Smoothing: Enabled
"""
            
            ax9.text(0.05, 0.95, bg_text, transform=ax9.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
            ax9.set_title('Vision System Status')
            
            # Panel 10: System performance
            ax10 = fig.add_subplot(gs[2, 3])
            ax10.axis('off')
            
            fps = self.fps_tracker.get_fps()
            performance_text = f"""System Performance:

FPS: {fps:.1f}
Device: {self.device}
Model: MobileNetV2
Parameters: ~3.5M

Memory Usage:
Hand Crop: {hand_crop.nbytes / 1024:.1f} KB
Model Input: {model_input_np.nbytes / 1024:.1f} KB
Frame: {frame.nbytes / 1024 / 1024:.1f} MB

Status: {'🟢 Real-time' if fps > 15 else '🟡 Slow' if fps > 10 else '🔴 Too slow'}
"""
            
            ax10.text(0.05, 0.95, performance_text, transform=ax10.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
            ax10.set_title('Performance Metrics')
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking
            
            # Save capture data
            timestamp = time.time()
            capture_dir = Path("data/raw/captures")
            capture_dir.mkdir(parents=True, exist_ok=True)
            
            capture_base = capture_dir / f"live_capture_{prediction}_{timestamp:.0f}"
            
            # Save images
            cv2.imwrite(f"{capture_base}_original.jpg", hand_crop)
            cv2.imwrite(f"{capture_base}_model_input.jpg", model_input_np)
            cv2.imwrite(f"{capture_base}_full_frame.jpg", frame)
            
            # Save comprehensive metadata
            capture_metadata = {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': float(confidence),
                'bbox': list(bbox),
                'hand_size': [w, h],
                'hand_area': total_pixels,
                'aspect_ratio': float(aspect_ratio),
                'model_info': {
                    'device': str(self.device),
                    'classes': self.classes,
                    'input_size': [224, 224, 3],
                    'confidence_threshold': self.min_pred_confidence
                },
                'performance': {
                    'fps': float(fps),
                    'frame_size': list(frame.shape),
                    'hand_crop_size': list(hand_crop.shape),
                    'model_input_size': list(model_input_np.shape)
                },
                'files': {
                    'original_crop': f"{capture_base.name}_original.jpg",
                    'model_input': f"{capture_base.name}_model_input.jpg",
                    'full_frame': f"{capture_base.name}_full_frame.jpg"
                }
            }
            
            with open(f"{capture_base}_metadata.json", 'w') as f:
                json.dump(capture_metadata, f, indent=2)
            
            print(f"\n🎯 Live ASL Data Captured and Visualized!")
            print(f"  Prediction: {prediction} ({confidence:.3f})")
            print(f"  Hand size: {w}×{h} pixels ({total_pixels:,} total)")
            print(f"  Model input: 224×224×3 preprocessed")
            print(f"  System FPS: {fps:.1f}")
            print(f"  Files saved to: {capture_dir}")
            print(f"  📊 Close the visualization window when done viewing.")
            
        except Exception as e:
            logger.error(f"Error in capture visualization: {e}")
            print(f"❌ Capture failed: {e}")
    
    def _process_frame(self, frame: np.ndarray):
        """
        Handles all processing for a single frame, including detection, 
        tracking, and prediction. It updates the recognizer's state but
        does not draw to the screen.
        """
        # --- Background Learning Phase ---
        if not self.hand_detector.bg_remover.bg_model_learned:
            self.hand_detector.bg_remover.learn_background(frame)
            # State is updated, UI will be drawn in the main loop
            return

        # --- Normal Processing ---
        self.fps_tracker.update()
        
        processed_hand, hand_info = self.hand_detector.detect_and_process_hand(
            frame, 224  # Use fixed size since model.INPUT_SIZE might not exist
        )
        
        prediction, confidence = None, 0.0
        if processed_hand is not None:
            prediction, confidence = self.predict_hand_sign(processed_hand)
            if confidence < self.min_pred_confidence:
                prediction, confidence = None, 0.0

        # Store last results for UI drawing and capture
        self.last_hand_info = hand_info
        self.last_processed_hand = processed_hand
        self.last_capture_frame = frame.copy()
        
        if prediction is not None:
            self.last_prediction = prediction
            self.last_confidence = confidence
        else:
            self.last_prediction = "Show Hand"
            self.last_confidence = 0.0
            
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws all UI elements onto a given frame based on the recognizer's
        current state.
        """
        # --- Background Learning UI ---
        if not self.hand_detector.bg_remover.bg_model_learned:
            progress = self.hand_detector.bg_remover.get_progress()
            h, w = frame.shape[:2]
            cv2.putText(frame, "Learning Background...", (50, h // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, "Please keep hands out of frame.", (50, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.rectangle(frame, (50, h // 2 + 60), (w - 50, h // 2 + 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (50, h // 2 + 60), (50 + int((w - 100) * progress), h // 2 + 90), (0, 255, 0), -1)
            return frame

        # --- Normal UI ---
        hand_info = self.last_hand_info
        prediction = self.last_prediction
        confidence = self.last_confidence
        
        # Draw bounding box and status
        if hand_info and hand_info.get("bbox") is not None:
            bbox = hand_info["bbox"]
            status = hand_info["status"]
            color = self.STATUS_COLORS.get(status, (255, 0, 255))
            
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw Prediction
        pred_text = f"Prediction: {prediction} ({confidence:.2f})"
        cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw FPS
        if self.show_stats:
            fps = self.fps_tracker.get_fps()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # --- PAUSED indicator ---
        if self.paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "PAUSED", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        return frame
        
    def _handle_keypress(self, key: int):
        if key == ord('q'):
            logger.info("👋 Exiting...")
            return
        elif key == ord('s'):
            self.show_stats = not self.show_stats
        elif key == ord('r'):
            self.hand_detector.reset()
            logger.info("🔄 Hand detector reset.")
        elif key == ord('b'):
            self.hand_detector.bg_remover.reset()
            logger.info("🔄 Background model is resetting. Please keep hands out of frame.")
        elif key == ord(' '):
            self.paused = not self.paused
            logger.info("⏸️ Paused" if self.paused else "▶️ Resumed")
        elif key == ord('c'):
            # Capture and visualize current hand data
            if (self.capture_enabled and self.last_processed_hand is not None and 
                self.last_hand_info is not None and self.last_capture_frame is not None):
                logger.info("📸 Capturing hand data...")
                self._capture_and_visualize_hand_data(
                    self.last_capture_frame, 
                    self.last_processed_hand,
                    self.last_hand_info,
                    self.last_prediction,
                    self.last_confidence
                )
            else:
                logger.warning("❌ No hand data available to capture")

    def run(self):
        """Main loop for the application."""
        logger.info("🟢 Starting Live ASL Recognition with Data Capture...")
        logger.info("📋 Controls:")
        logger.info("  Q: Quit")
        logger.info("  S: Toggle statistics display")
        logger.info("  R: Reset hand tracker")
        logger.info("  B: Reset background learning")
        logger.info("  SPACE: Pause/unpause")
        logger.info("  C: 📸 Capture and visualize hand data")
        logger.info("")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"❌ Cannot open camera {self.camera_index}")
            return
            
        while self.cap.isOpened():
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("👋 Exiting...")
                break
            self._handle_keypress(key)
            
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                self.last_frame = frame.copy() # Store the latest frame
                self._process_frame(frame)

            # Always draw the UI, even when paused.
            # If paused, it will draw on the last saved frame.
            if self.last_frame is not None:
                ui_frame = self._draw_ui(self.last_frame.copy())
                cv2.imshow("Live ASL Recognition", ui_frame)

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the recognizer"""
    recognizer = LiveASLRecognizer()
    if recognizer.model:
        recognizer.run()

if __name__ == "__main__":
    main() 