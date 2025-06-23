"""
Real-time ASL Inference Module

This module integrates MobileNetV2-based ASL models with the existing hand detection system
for 30 FPS real-time ASL sign recognition.

Author: CV-ASL Team
Date: 2024
"""

import time
from collections import deque
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

# Import only the models we actually use
from .training.train import MobileNetV2ASL, MobileNetV2Lite

class ASLInference:
    """
    Real-time ASL inference engine with performance monitoring.
    Supports MobileNetV2 models for efficient inference.
    """
    
    def __init__(self, 
                 model_path: str, 
                 model_type: str = "mobilenetv2",
                 confidence_threshold: float = 0.5,
                 enable_smoothing: bool = True,
                 smoothing_window: int = 5,
                 performance_mode: bool = False):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model file
            model_type: Type of model ("mobilenetv2", "mobilenetv2_lite")  
            confidence_threshold: Minimum confidence for valid predictions
            enable_smoothing: Whether to apply temporal smoothing
            smoothing_window: Number of frames for smoothing
            performance_mode: Enable performance optimizations
        """
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.enable_smoothing = enable_smoothing
        self.performance_mode = performance_mode
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.prediction_history = deque(maxlen=smoothing_window) if enable_smoothing else None
        
        # Device configuration
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model and preprocessing
        self.model, self.classes, self.preprocess = self._load_model()
        
        logger = logging.getLogger(__name__)
        logger.info(f"✅ Model loaded from {model_path} with {len(self.classes)} classes.")
        
    def _load_model(self) -> Tuple[torch.nn.Module, List[str], transforms.Compose]:
        """Load the trained model and get class names."""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration  
        num_classes = checkpoint.get('num_classes', 3)  # Default fallback
        model_config = checkpoint.get('model_config', {})
        
        # Initialize model based on type
        if self.model_type == "mobilenetv2":
            model = MobileNetV2ASL(num_classes=num_classes, **model_config)
        elif self.model_type == "mobilenetv2_lite": 
            model = MobileNetV2Lite(num_classes=num_classes, **model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Get class names
        classes = checkpoint.get('classes', [f'class_{i}' for i in range(num_classes)])
        
        # Setup preprocessing
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return model, classes, preprocess
        
    def predict(self, image: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Make prediction on input image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        start_time = time.time()
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Preprocess image
        input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Apply smoothing if enabled
        if self.enable_smoothing and self.prediction_history is not None:
            self.prediction_history.append((predicted_class, confidence))
            predicted_class, confidence = self._apply_smoothing()
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return predicted_class, confidence, probabilities.tolist()
    
    def _apply_smoothing(self) -> Tuple[str, float]:
        """Apply temporal smoothing to predictions."""
        if not self.prediction_history:
            return "Unknown", 0.0
            
        # Get recent predictions
        recent_predictions = list(self.prediction_history)
        
        # Count occurrences of each class
        class_counts = {}
        confidence_sums = {}
        
        for pred_class, conf in recent_predictions:
            if pred_class not in class_counts:
                class_counts[pred_class] = 0
                confidence_sums[pred_class] = 0.0
            class_counts[pred_class] += 1
            confidence_sums[pred_class] += conf
        
        # Find most frequent class
        most_frequent_class = max(class_counts, key=class_counts.get)
        avg_confidence = confidence_sums[most_frequent_class] / class_counts[most_frequent_class]
        
        return most_frequent_class, avg_confidence
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {"avg_inference_time": 0.0, "fps": 0.0}
            
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "min_time": np.min(self.inference_times),
            "max_time": np.max(self.inference_times)
        }
    
    def reset_history(self):
        """Reset prediction history for smoothing."""
        if self.prediction_history is not None:
            self.prediction_history.clear()
        self.inference_times.clear()

def main():
    """CLI interface for testing inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASL Inference")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--type", default="mobilenetv2", 
                       choices=["mobilenetv2", "mobilenetv2_lite"],
                       help="Model type")
    parser.add_argument("--image", help="Test image path")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = ASLInference(
        model_path=args.model,
        model_type=args.type,
        confidence_threshold=args.confidence
    )
    
    if args.image:
        # Test single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
            
        prediction, confidence, probabilities = engine.predict(image)
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
        print(f"All probabilities: {probabilities}")
        
        # Performance stats
        stats = engine.get_performance_stats()
        print(f"Inference time: {stats['avg_inference_time']:.3f}s")
    
if __name__ == "__main__":
    main()
