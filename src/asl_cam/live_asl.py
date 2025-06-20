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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveASLRecognizer:
    """
    Live ASL Recognition System
    
    Combines hand detection with trained model for real-time ASL prediction
    """
    
    def __init__(self, model_path: str = "src/asl_dl/models/asl_abc_model.pth"):
        """Initialize the live ASL recognition system"""
        
        self.model_path = Path(model_path)
        
        # Set device to MPS (Apple Silicon GPU) if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Initialize ASL-optimized hand detector
        self.hand_detector = ASLHandDetector(
            min_detection_confidence=0.6 # Only process high-confidence hands
        )
        
        # Load trained model
        self.model = None
        self.classes = []
        self.load_model()
        
        # Setup image preprocessing for the model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.inference_time = 0.0
        
        # UI state
        self.show_stats = True
        self.paused = False
        
        logger.info("🚀 Live ASL Recognizer initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🎯 Classes: {self.classes}")
    
    def load_model(self) -> bool:
        """Load the trained ASL model"""
        
        if not self.model_path.exists():
            logger.error(f"❌ Model not found: {self.model_path}")
            logger.info("💡 Train a model first with: python src/asl_dl/scripts/train_abc.py")
            return False
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model info
            self.classes = checkpoint['classes']
            num_classes = checkpoint['num_classes']
            
            # Create model
            self.model = MobileNetV2ASL(num_classes=num_classes, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model loaded: {num_classes} classes")
            logger.info(f"📊 Model accuracy: {checkpoint.get('final_val_acc', 'N/A'):.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
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
    
    def draw_ui(self, frame: np.ndarray, prediction: str, confidence: float, 
                hand_info: Optional[Dict] = None) -> np.ndarray:
        """Draw the user interface on the frame"""
        
        height, width = frame.shape[:2]
        
        # Draw hand bounding box if detected
        if hand_info:
            x, y, w, h = hand_info['bbox']
            is_stable = hand_info.get('is_stable', False)
            hand_confidence = hand_info.get('confidence', 0.0)
            
            # Color based on stability
            box_color = (0, 255, 0) if is_stable else (0, 165, 255)
            label = f"Hand Stable ({hand_confidence:.1%})" if is_stable else f"Hand Detected ({hand_confidence:.1%})"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        else:
            hand_confidence = None
        
        # Draw prediction panel
        panel_height = 120
        panel_color = (50, 50, 50)
        cv2.rectangle(frame, (0, height - panel_height), (width, height), panel_color, -1)
        
        # Main prediction display
        if prediction != "No Hand":
            # Large letter display
            letter_size = 3.0
            letter_thickness = 4
            letter_color = (0, 255, 255) if confidence > 0.7 else (0, 165, 255)
            
            text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, letter_size, letter_thickness)[0]
            letter_x = (width - text_size[0]) // 2
            letter_y = height - 70
            
            cv2.putText(frame, prediction, (letter_x, letter_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, letter_size, letter_color, letter_thickness)
            
            # Confidence bar
            bar_width = 250
            bar_height = 25
            bar_x = (width - bar_width) // 2
            bar_y = height - 40
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Confidence fill
            fill_width = int(bar_width * confidence)
            fill_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
            
            # Confidence text
            conf_text = f"Sign Confidence: {confidence:.1%}"
            cv2.putText(frame, conf_text, (bar_x + bar_width + 10, bar_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display hand detection confidence if available
            if hand_confidence is not None:
                hand_conf_text = f"Hand Confidence: {hand_confidence:.1%}"
                cv2.putText(frame, hand_conf_text, (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No hand detected
            no_hand_text = "Show your hand to camera"
            text_size = cv2.getTextSize(no_hand_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 60
            cv2.putText(frame, no_hand_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw statistics if enabled
        if self.show_stats:
            stats_y = 30
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {self.inference_time:.1f}ms", (10, stats_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Device: {self.device}", (10, stats_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw controls help
        help_text = "Q: Quit | S: Stats | R: Reset"
        cv2.putText(frame, help_text, (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Main loop for the live recognition system"""
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
            
        prediction = "Show Hand"
        confidence = 0.0
        hand_info = None

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to grab frame")
                    break
                
                # --- Core Processing Pipeline (OPTIMIZED) ---
                
                # 1. Detect hands (single call per frame)
                detected_hands = self.hand_detector.detect_hands_asl(frame)
                
                # 2. If a confident hand is found, process it
                if detected_hands:
                    # Get the most confident hand
                    hand_info = detected_hands[0]
                    
                    # 3. Predict sign from the enhanced crop
                    hand_crop = hand_info.get('enhanced_crop')
                    if hand_crop is not None and hand_crop.size > 0:
                        prediction, confidence = self.predict_hand_sign(hand_crop)
                    else:
                        prediction = "Processing..."
                        confidence = 0.0
                else:
                    # No confident hand detected
                    prediction = "Show Hand"
                    confidence = 0.0
                    hand_info = None

                # Update performance metrics
                self.update_fps()

            # --- UI and Display ---
            display_frame = frame.copy()
            ui_frame = self.draw_ui(display_frame, prediction, confidence, hand_info)
            cv2.imshow("Live ASL Recognition", ui_frame)

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