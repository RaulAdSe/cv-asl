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
    
    def __init__(self, model_path: str = "models/asl_model.pth", 
                 min_pred_confidence: float = 0.7,
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
        
        self.hand_detector = ASLHandDetector(min_detection_confidence=0.6)
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

        logger.info("🚀 Live ASL Recognizer initialized")
        logger.info(f"📱 Device: {self.device}")
        self.classes = sorted(list(self.model.class_map.keys()))
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
    
    def _process_frame(self, frame: np.ndarray):
        """
        Handles all processing for a single frame, including detection, 
        tracking, prediction, and UI drawing.
        """
        # --- Background Learning Phase ---
        if not self.hand_detector.bg_remover.bg_model_learned:
            self.hand_detector.bg_remover.learn_background(frame)
            # The drawing for this phase will be handled in _draw_ui
            return

        # --- Main Processing (only if not paused) ---
        if not self.paused:
            self.fps_tracker.update()
            
            processed_hand, hand_info = self.hand_detector.detect_and_process_hand(
                frame, self.model.INPUT_SIZE
            )
            
            if processed_hand is not None:
                prediction, confidence = self.model.predict(processed_hand)
                if confidence < self.min_pred_confidence:
                    self.last_prediction, self.last_confidence = None, 0.0
                else:
                    self.last_prediction, self.last_confidence = prediction, confidence
            else:
                self.last_prediction, self.last_confidence = None, 0.0
                
            self.last_hand_info = hand_info

    def _draw_ui(self, frame: np.ndarray):
        """Draws the complete UI onto the frame."""
        height, width, _ = frame.shape

        # --- Draw Background Learning UI (if applicable) ---
        if not self.hand_detector.bg_remover.bg_model_learned:
            progress = self.hand_detector.bg_remover.get_progress()
            cv2.putText(frame, "Learning Background...", (50, height // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, "Please keep hands out of frame.", (50, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # Progress bar
            cv2.rectangle(frame, (50, height // 2 + 60), (width - 50, height // 2 + 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (50, height // 2 + 60), (50 + int((width - 100) * progress), height // 2 + 90), (0, 255, 0), -1)
            return frame # Return early

        # --- Draw Normal UI ---
        # Draw stats if enabled
        if self.show_stats:
            fps_text = f"FPS: {self.fps_tracker.get_fps():.1f}"
            cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw hand info if available
        if self.last_hand_info:
            x, y, w, h = self.last_hand_info['bbox']
            status = self.last_hand_info.get('status', 'DETECTION')
            color = {'TRACKED': (0, 255, 0), 'PREDICTED': (0, 255, 255), 'NEW_DETECTION': (255, 0, 0)}.get(status, (255, 0, 0))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Status: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw prediction text
        pred_text = "Show Hand"
        if self.last_prediction and self.last_confidence > 0:
            pred_text = f"{self.last_prediction} ({self.last_confidence:.2f})"
        
        text_size, _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_x = (width - text_size[0]) // 2
        text_y = height - 40
        cv2.putText(frame, pred_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw controls help
        help_text = "Q: Quit | S: Stats | R: Reset | B: Reset BG | Space: Pause"
        cv2.putText(frame, help_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    def run(self):
        """Main loop for the live recognition system."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)

            # Process the frame (handles learning, detection, prediction)
            self._process_frame(frame)

            # Draw the UI on a copy of the frame
            display_frame = self._draw_ui(frame.copy())
            
            # Display the final result
            cv2.imshow("Live ASL Recognition", display_frame)

            # --- User Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("👋 Exiting...")
                break
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

        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the recognizer"""
    recognizer = LiveASLRecognizer()
    if recognizer.model:
        recognizer.run()

if __name__ == "__main__":
    main() 