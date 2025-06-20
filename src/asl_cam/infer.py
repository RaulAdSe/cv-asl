"""
Real-time ASL Inference Module

This module integrates trained ASL models with the existing hand detection system
for real-time ASL sign recognition.

Author: CV-ASL Team
Date: 2024
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import mediapipe as mp

# Import our existing modules
from .vision.simple_hand_detector import SimpleHandDetector
from .vision.tracker import HandTracker
from .train import EfficientNetLSTM, MediaPipeClassifier
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLInferenceEngine:
    """Real-time ASL inference engine"""
    
    def __init__(self, model_path: str, model_type: str = "efficientnet_lstm", 
                 confidence_threshold: float = 0.7):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ("efficientnet_lstm", "mediapipe", "mobilenet")
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and metadata
        self.model, self.classes = self._load_model()
        self.num_classes = len(self.classes)
        
        # Initialize hand detection
        self.hand_detector = SimpleHandDetector()
        self.hand_tracker = HandTracker()
        
        # Initialize MediaPipe if needed
        self.use_mediapipe = (model_type == "mediapipe")
        if self.use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        self.sequence_buffer = deque(maxlen=16)  # For temporal models
        
        # Statistics
        self.frame_count = 0
        self.prediction_count = 0
        self.total_inference_time = 0
        
        logger.info(f"ASL Inference Engine initialized")
        logger.info(f"Model: {model_type}")
        logger.info(f"Classes: {self.num_classes}")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self) -> Tuple[torch.nn.Module, List[str]]:
        """Load trained model and class information"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        classes = checkpoint['classes']
        num_classes = len(classes)
        
        # Create model
        if self.model_type == "efficientnet_lstm":
            model = EfficientNetLSTM(num_classes=num_classes)
        elif self.model_type == "mediapipe":
            model = MediaPipeClassifier(num_classes=num_classes)
        elif self.model_type == "mobilenet":
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small()
            model.classifier[3] = torch.nn.Linear(
                model.classifier[3].in_features, num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully with accuracy: {checkpoint.get('accuracy', 'N/A')}")
        
        return model, classes
    
    def _extract_mediapipe_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract MediaPipe hand landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        features = np.zeros((21 * 2 * 2))  # 21 landmarks × 2 coords × 2 hands
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                base_idx = hand_idx * 42 + landmark_idx * 2
                features[base_idx] = landmark.x
                features[base_idx + 1] = landmark.y
        
        return features
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        if self.use_mediapipe:
            # Extract MediaPipe features
            features = self._extract_mediapipe_features(image)
            if features is None:
                return None
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)
        else:
            # Standard image preprocessing
            if len(image.shape) == 3:
                tensor = self.transform(image)
                return tensor.unsqueeze(0).to(self.device)
            return None
    
    def _smooth_predictions(self, prediction: torch.Tensor) -> Tuple[str, float]:
        """Apply temporal smoothing to predictions"""
        probabilities = F.softmax(prediction, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.classes[predicted_idx.item()]
        confidence_value = confidence.item()
        
        # Add to history
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence_value,
            'probabilities': probabilities.cpu().numpy()
        })
        
        # Smooth predictions using recent history
        if len(self.prediction_history) >= 5:
            # Average probabilities over recent predictions
            avg_probs = np.mean([p['probabilities'] for p in list(self.prediction_history)[-5:]], axis=0)
            smoothed_confidence = np.max(avg_probs)
            smoothed_class_idx = np.argmax(avg_probs)
            smoothed_class = self.classes[smoothed_class_idx]
            
            return smoothed_class, smoothed_confidence
        
        return predicted_class, confidence_value
    
    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """Predict ASL sign from image"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self._preprocess_image(image)
        if input_tensor is None:
            return None
        
        # Inference
        with torch.no_grad():
            if self.model_type == "efficientnet_lstm":
                # For LSTM models, we need sequence input
                self.sequence_buffer.append(input_tensor.squeeze(0))
                
                if len(self.sequence_buffer) < 8:  # Wait for enough frames
                    return None
                
                # Create sequence tensor
                sequence = torch.stack(list(self.sequence_buffer)).unsqueeze(0)
                output = self.model(sequence)
            else:
                # Single frame models
                output = self.model(input_tensor)
            
            # Get prediction with smoothing
            predicted_class, confidence = self._smooth_predictions(output)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.prediction_count += 1
        
        # Return result if confidence is high enough
        if confidence >= self.confidence_threshold:
            return {
                'class': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time
            }
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        avg_inference_time = (self.total_inference_time / self.prediction_count 
                             if self.prediction_count > 0 else 0)
        
        return {
            'frames_processed': self.frame_count,
            'predictions_made': self.prediction_count,
            'avg_inference_time': avg_inference_time,
            'fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        }

class RealTimeASL:
    """Main class for real-time ASL recognition"""
    
    def __init__(self, model_path: str, model_type: str = "efficientnet_lstm"):
        self.inference_engine = ASLInferenceEngine(model_path, model_type)
        self.hand_detector = SimpleHandDetector()
        
        # UI settings
        self.show_debug = True
        self.show_stats = True
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=30)
        
    def run(self, camera_id: int = 0):
        """Run real-time ASL recognition"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting real-time ASL recognition...")
        logger.info("Controls:")
        logger.info("  SPACE: Toggle debug info")
        logger.info("  S: Toggle statistics")
        logger.info("  Q/ESC: Quit")
        
        last_prediction = None
        last_prediction_time = 0
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # Detect hands
            hands = self.hand_detector.detect_hands(frame)
            
            # Process each detected hand
            current_prediction = None
            
            for hand in hands:
                x, y, w, h = hand['bbox']
                
                # Draw hand detection box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract hand region for classification
                hand_roi = original_frame[y:y+h, x:x+w]
                
                if hand_roi.size > 0:
                    # Predict ASL sign
                    prediction = self.inference_engine.predict(hand_roi)
                    
                    if prediction:
                        current_prediction = prediction
                        
                        # Draw prediction
                        text = f"{prediction['class']}: {prediction['confidence']:.2f}"
                        
                        # Background for text
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), (0, 255, 0), -1)
                        
                        # Text
                        cv2.putText(frame, text, (x + 5, y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        last_prediction = prediction
                        last_prediction_time = time.time()
            
            # Update FPS tracking
            frame_time = time.time() - start_time
            self.fps_tracker.append(1.0 / frame_time if frame_time > 0 else 0)
            
            # Draw UI elements
            self._draw_ui(frame, current_prediction or last_prediction)
            
            # Show frame
            cv2.imshow('ASL Real-time Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                self.show_debug = not self.show_debug
            elif key == ord('s'):  # S
                self.show_stats = not self.show_stats
            
            self.inference_engine.frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.inference_engine.get_statistics()
        logger.info("\nFinal Statistics:")
        logger.info(f"Frames processed: {stats['frames_processed']}")
        logger.info(f"Predictions made: {stats['predictions_made']}")
        logger.info(f"Average inference time: {stats['avg_inference_time']:.3f}s")
        logger.info(f"Average FPS: {stats['fps']:.1f}")
    
    def _draw_ui(self, frame: np.ndarray, prediction: Optional[Dict]):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        if self.show_stats:
            # FPS
            fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Model info
            cv2.putText(frame, f"Model: {self.inference_engine.model_type}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Inference stats
            stats = self.inference_engine.get_statistics()
            cv2.putText(frame, f"Predictions: {stats['predictions_made']}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current prediction (large display)
        if prediction:
            pred_text = f"Sign: {prediction['class']}"
            conf_text = f"Confidence: {prediction['confidence']:.2f}"
            
            # Position at bottom center
            (pred_w, pred_h), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            pred_x = (w - pred_w) // 2
            conf_x = (w - conf_w) // 2
            pred_y = h - 60
            conf_y = h - 30
            
            # Background
            cv2.rectangle(frame, (pred_x - 10, pred_y - pred_h - 5), 
                         (pred_x + pred_w + 10, conf_y + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, pred_text, (pred_x, pred_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, conf_text, (conf_x, conf_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        if self.show_debug:
            instructions = [
                "SPACE: Toggle debug",
                "S: Toggle stats",
                "Q/ESC: Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (w - 200, 30 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    """Main function for real-time ASL recognition"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time ASL Recognition")
    parser.add_argument("--model", "-m", required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--type", "-t", default="efficientnet_lstm",
                       choices=["efficientnet_lstm", "mediapipe", "mobilenet"],
                       help="Model type")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera ID")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run real-time ASL recognition
        asl_system = RealTimeASL(args.model, args.type)
        asl_system.run(args.camera)
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
