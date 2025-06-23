"""
Real-time ASL Inference Module

This module integrates MobileNetV2-based ASL models with the existing hand detection system
for 30 FPS real-time ASL sign recognition.

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
from .train import MobileNetV2ASL, MobileNetV2Lite, MediaPipeClassifier
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLInferenceEngine:
    """Real-time ASL inference engine optimized for 30 FPS"""
    
    def __init__(self, model_path: str, model_type: str = "mobilenetv2", 
                 confidence_threshold: float = 0.7, target_fps: int = 30):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ("mobilenetv2", "mobilenetv2_lite", "mediapipe")
            confidence_threshold: Minimum confidence for predictions
            target_fps: Target FPS for optimization
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.target_fps = target_fps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and metadata
        self.model, self.classes, self.config = self._load_model()
        self.num_classes = len(self.classes)
        
        # Initialize hand detection
        self.hand_detector = SimpleHandDetector()
        
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
        
        # Image preprocessing optimized for speed
        input_size = self.config.get('input_size', 224)
        self.input_size = input_size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Prediction smoothing (reduced for speed)
        self.prediction_history = deque(maxlen=5)  # Shorter history for responsiveness
        
        # Statistics and benchmarking
        self.frame_count = 0
        self.prediction_count = 0
        self.total_inference_time = 0
        self.fps_tracker = deque(maxlen=30)
        
        # Warmup the model
        self._warmup_model()
        
        logger.info(f"ASL Inference Engine initialized for {target_fps} FPS")
        logger.info(f"Model: {model_type}")
        logger.info(f"Classes: {self.num_classes}")
        logger.info(f"Input size: {input_size}x{input_size}")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self) -> Tuple[torch.nn.Module, List[str], Dict]:
        """Load trained model and metadata"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        classes = checkpoint['classes']
        config = checkpoint.get('config', {})
        num_classes = len(classes)
        
        # Create model
        if self.model_type == "mobilenetv2":
            model = MobileNetV2ASL(
                num_classes=num_classes,
                input_size=config.get('input_size', 224),
                width_mult=config.get('width_mult', 1.0)
            )
        elif self.model_type == "mobilenetv2_lite":
            model = MobileNetV2Lite(num_classes=num_classes)
        elif self.model_type == "mediapipe":
            model = MediaPipeClassifier(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        benchmark_info = checkpoint.get('benchmark', {})
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Training accuracy: {checkpoint.get('accuracy', 'N/A')}")
        logger.info(f"Parameters: {total_params:,}")
        if benchmark_info:
            logger.info(f"Benchmark FPS: {benchmark_info.get('fps', 'N/A'):.1f}")
        
        return model, classes, config
    
    def _warmup_model(self):
        """Warmup model for consistent performance"""
        logger.info("Warming up model...")
        
        if self.use_mediapipe:
            dummy_input = torch.randn(1, 84).to(self.device)
        else:
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        logger.info("Model warmed up")
    
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
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess image for model input"""
        if self.use_mediapipe:
            # Extract MediaPipe features
            features = self._extract_mediapipe_features(image)
            if features is None:
                return None
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)
        else:
            # Standard image preprocessing
            if len(image.shape) == 3 and image.size > 0:
                try:
                    tensor = self.transform(image)
                    return tensor.unsqueeze(0).to(self.device)
                except Exception as e:
                    logger.warning(f"Error preprocessing image: {e}")
                    return None
            return None
    
    def _smooth_predictions(self, prediction: torch.Tensor) -> Tuple[str, float]:
        """Apply lightweight temporal smoothing to predictions"""
        probabilities = F.softmax(prediction, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.classes[predicted_idx.item()]
        confidence_value = confidence.item()
        
        # Add to history (shorter for responsiveness)
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence_value,
            'probabilities': probabilities.cpu().numpy()
        })
        
        # Simple smoothing using recent history
        if len(self.prediction_history) >= 3:
            # Get most common class in recent predictions
            recent_classes = [p['class'] for p in list(self.prediction_history)[-3:]]
            most_common = max(set(recent_classes), key=recent_classes.count)
            
            # Average confidence for the most common class
            avg_confidence = np.mean([p['confidence'] for p in list(self.prediction_history)[-3:] 
                                    if p['class'] == most_common])
            
            return most_common, avg_confidence
        
        return predicted_class, confidence_value
    
    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """Predict ASL sign from image - optimized for 30 FPS"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self._preprocess_image(image)
        if input_tensor is None:
            return None
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Get prediction with smoothing
            predicted_class, confidence = self._smooth_predictions(output)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.prediction_count += 1
        
        # Update FPS tracking
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_tracker.append(fps)
        
        # Return result if confidence is high enough
        if confidence >= self.confidence_threshold:
            return {
                'class': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time,
                'fps': fps
            }
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        avg_inference_time = (self.total_inference_time / self.prediction_count 
                             if self.prediction_count > 0 else 0)
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        
        return {
            'frames_processed': self.frame_count,
            'predictions_made': self.prediction_count,
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'target_fps': self.target_fps,
            'meets_target': avg_fps >= self.target_fps
        }

class RealTimeASL:
    """Main class for 30 FPS real-time ASL recognition"""
    
    def __init__(self, model_path: str, model_type: str = "mobilenetv2", target_fps: int = 30):
        self.inference_engine = ASLInferenceEngine(model_path, model_type, target_fps=target_fps)
        self.hand_detector = SimpleHandDetector()
        self.target_fps = target_fps
        
        # UI settings
        self.show_debug = True
        self.show_stats = True
        self.show_fps_warning = True
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)
        
    def run(self, camera_id: int = 0):
        """Run 30 FPS real-time ASL recognition"""
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera to 30 FPS
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Verify camera FPS
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera FPS: {actual_fps}")
        
        logger.info(f"Starting {self.target_fps} FPS real-time ASL recognition...")
        logger.info("Controls:")
        logger.info("  SPACE: Toggle debug info")
        logger.info("  S: Toggle statistics")
        logger.info("  F: Toggle FPS warning")
        logger.info("  Q/ESC: Quit")
        
        last_prediction = None
        frame_count = 0
        
        while True:
            loop_start = time.time()
            
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
            prediction_time = 0
            
            for hand in hands:
                x, y, w, h = hand['bbox']
                
                # Draw hand detection box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract hand region for classification
                hand_roi = original_frame[y:y+h, x:x+w]
                
                if hand_roi.size > 0:
                    # Predict ASL sign
                    pred_start = time.time()
                    prediction = self.inference_engine.predict(hand_roi)
                    prediction_time = time.time() - pred_start
                    
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
            
            # Update FPS tracking
            loop_time = time.time() - loop_start
            frame_fps = 1.0 / loop_time if loop_time > 0 else 0
            self.fps_tracker.append(frame_fps)
            self.frame_times.append(loop_time)
            
            # Draw UI elements
            self._draw_ui(frame, current_prediction or last_prediction, prediction_time)
            
            # Show frame
            cv2.imshow('ASL Real-time Recognition (30 FPS)', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                self.show_debug = not self.show_debug
            elif key == ord('s'):  # S
                self.show_stats = not self.show_stats
            elif key == ord('f'):  # F
                self.show_fps_warning = not self.show_fps_warning
            
            self.inference_engine.frame_count += 1
            frame_count += 1
            
            # FPS limiting to target (if needed)
            target_time = 1.0 / self.target_fps
            if loop_time < target_time:
                time.sleep(target_time - loop_time)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.inference_engine.get_statistics()
        avg_frame_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        
        logger.info(f"\nFinal Statistics:")
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Predictions made: {stats['predictions_made']}")
        logger.info(f"Average inference time: {stats['avg_inference_time']:.3f}s")
        logger.info(f"Average inference FPS: {stats['avg_fps']:.1f}")
        logger.info(f"Average frame FPS: {avg_frame_fps:.1f}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"Meets target: {'✅ YES' if avg_frame_fps >= self.target_fps * 0.9 else '❌ NO'}")
    
    def _draw_ui(self, frame: np.ndarray, prediction: Optional[Dict], prediction_time: float):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        if self.show_stats:
            # Current FPS
            current_fps = self.fps_tracker[-1] if self.fps_tracker else 0
            avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
            
            # Color based on FPS performance
            fps_color = (0, 255, 0) if avg_fps >= self.target_fps * 0.9 else (0, 165, 255) if avg_fps >= self.target_fps * 0.7 else (0, 0, 255)
            
            cv2.putText(frame, f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
            
            # Target FPS
            cv2.putText(frame, f"Target: {self.target_fps} FPS", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Model info
            cv2.putText(frame, f"Model: {self.inference_engine.model_type}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Inference time
            if prediction_time > 0:
                inf_fps = 1.0 / prediction_time
                cv2.putText(frame, f"Inference: {prediction_time*1000:.1f}ms ({inf_fps:.0f} FPS)", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS Warning
        if self.show_fps_warning:
            avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
            if avg_fps < self.target_fps * 0.8:
                warning_text = f"⚠️ LOW FPS: {avg_fps:.1f} < {self.target_fps}"
                (text_w, text_h), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (w//2 - text_w//2 - 10, 10), (w//2 + text_w//2 + 10, 40), (0, 0, 255), -1)
                cv2.putText(frame, warning_text, (w//2 - text_w//2, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
                "F: Toggle FPS warning",
                "Q/ESC: Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (w - 220, 30 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    """Main function for 30 FPS real-time ASL recognition"""
    import argparse
    
    parser = argparse.ArgumentParser(description="30 FPS Real-time ASL Recognition")
    parser.add_argument("--model", "-m", required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--type", "-t", default="mobilenetv2",
                       choices=["mobilenetv2", "mobilenetv2_lite", "mediapipe"],
                       help="Model type")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera ID")
    parser.add_argument("--fps", "-f", type=int, default=30,
                       help="Target FPS (default: 30)")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run real-time ASL recognition
        asl_system = RealTimeASL(args.model, args.type, args.fps)
        asl_system.run(args.camera)
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
