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

        self.STATUS_COLORS = {
            "TRACKED": (0, 255, 0),       # Green for stable tracking
            "PREDICTED": (255, 255, 0),   # Yellow for Kalman filter prediction
            "NEW_DETECTION": (0, 0, 255), # Red for a new detection
            "LOST": (255, 0, 255),        # Magenta for lost track
        }

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
            frame, self.model.INPUT_SIZE
        )
        
        prediction, confidence = None, 0.0
        if processed_hand is not None:
            prediction, confidence = self.predict_hand_sign(processed_hand)
            if confidence < self.min_pred_confidence:
                prediction, confidence = None, 0.0

        # Store last results for UI drawing
        self.last_hand_info = hand_info
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

    def run(self):
        """Main loop for the application."""
        logger.info("🟢 Starting camera feed...")
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