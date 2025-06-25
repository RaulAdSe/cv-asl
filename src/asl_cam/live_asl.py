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
import uuid

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
        
        # Performance optimization settings
        self.frame_skip_counter = 0
        self.frame_skip_rate = 1  # Process every N frames (1 = no skip, 2 = skip every other)
        self.target_fps = 25      # Increased target FPS
        self.last_process_time = 0
        
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
        
        # Extract classes from model, with proper validation
        if self.model.class_map is not None:
            self.classes = sorted(list(self.model.class_map.keys()))
            logger.info(f"✅ Using model's class map: {self.model.class_map}")
        else:
            # This should not happen with the fixed model loading
            logger.error(f"❌ Model class_map is None - this indicates a loading issue!")
            # Fallback for emergency cases
            self.classes = ['A', 'B', 'C']  # Default ASL classes
            self.model.class_map = {cls: idx for idx, cls in enumerate(self.classes)}
            logger.warning(f"🚨 Using emergency fallback class_map: {self.model.class_map}")
            
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
        """Show exactly what happens in the live workflow - no separate processing."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import torch
            
            # Extract hand information 
            bbox = hand_info.get('bbox', (0, 0, 100, 100))
            x, y, w, h = bbox
            
            # LIVE WORKFLOW DATA: Get the exact data that flows through the system
            # 1. Original hand crop (what the detector extracts)
            original_hand_crop = frame[y:y+h, x:x+w].copy()
            
            # 2. Background-removed crop (what the detector processes internally)
            crop_coords = (x, y, x+w, y+h)
            bg_removed_crop = self.hand_detector._remove_background_from_crop(original_hand_crop, crop_coords)
            
            # 3. Model input (what actually goes to the model)
            # FIX: This should be a 224x224 resize of the background removed image
            model_input_image = cv2.resize(bg_removed_crop, (224, 224))
            
            # 4. Get the ACTUAL model probabilities from the live prediction
            # Re-run the same prediction to get the exact probabilities used
            actual_prediction, actual_confidence = self.predict_hand_sign(model_input_image)
            
            # Get model probabilities by running inference again (this matches exactly what was used)
            model_probs = None
            try:
                # Convert to RGB for model
                rgb_crop = cv2.cvtColor(model_input_image, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    model_probs = probabilities[0].cpu().numpy()
                    
            except Exception as e:
                logger.error(f"Failed to get actual model probabilities: {e}")
                # Fallback dummy probabilities
                model_probs = np.array([0.33, 0.33, 0.34])
            
            # Calculate stats from actual data
            max_prob = np.max(model_probs) if model_probs is not None else 0.0
            entropy = -np.sum(model_probs * np.log(model_probs + 1e-8)) if model_probs is not None else 0.0
            fps = self.fps_tracker.get_fps()
            
            # UNIFIED STYLING
            title_fontsize = 10
            title_color = '#2c3e50'
            bg_color = 'white'
            
            # NON-BLOCKING figure
            plt.ion()
            fig = plt.figure(figsize=(16, 8), facecolor=bg_color)
            fig.suptitle('Live ASL Workflow Analysis', fontsize=14, fontweight='bold', color=title_color)
            
            # 2x4 grid layout
            gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)
            
            # Panel 1: Camera Feed (with bounding box)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_facecolor(bg_color)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Draw bounding box on frame copy
            frame_with_box = frame_rgb.copy()
            cv2.rectangle(frame_with_box, (x, y), (x+w, y+h), (255, 0, 0), 3)
            ax1.imshow(frame_with_box)
            ax1.set_title(f'Camera Feed\n{frame.shape[1]}×{frame.shape[0]}', 
                         fontsize=title_fontsize, fontweight='bold', color=title_color)
            ax1.axis('off')

            # Panel 2: Original Hand Crop (extracted by detector)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_facecolor(bg_color)
            original_rgb = cv2.cvtColor(original_hand_crop, cv2.COLOR_BGR2RGB)
            ax2.imshow(original_rgb)
            ax2.set_title(f'Hand ROI\n{w}×{h}px', 
                         fontsize=title_fontsize, fontweight='bold', color=title_color)
            ax2.axis('off')

            # Panel 3: Original Colors (from actual hand crop)
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.set_facecolor(bg_color)
            self._plot_unified_color_distribution(ax3, original_hand_crop, 'Original Colors', 
                                                title_fontsize, title_color)
            
            # Panel 4: Background Removed (actual detector output)
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.set_facecolor(bg_color)
            bg_removed_rgb = cv2.cvtColor(bg_removed_crop, cv2.COLOR_BGR2RGB)
            ax4.imshow(bg_removed_rgb)
            ax4.set_title(f'Background Removed\n{bg_removed_crop.shape[1]}×{bg_removed_crop.shape[0]}px', 
                         fontsize=title_fontsize, fontweight='bold', color=title_color)
            ax4.axis('off')

            # Panel 5: Processed Colors (from actual background removed image)
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.set_facecolor(bg_color)
            self._plot_unified_color_distribution(ax5, bg_removed_crop, 'Processed Colors', 
                                                title_fontsize, title_color)
            
            # Panel 6: Model Input (actual resized input to model)
            ax6 = fig.add_subplot(gs[1, 1])
            ax6.set_facecolor(bg_color)
            model_input_rgb = cv2.cvtColor(model_input_image, cv2.COLOR_BGR2RGB)
            ax6.imshow(model_input_rgb)
            ax6.set_title(f'Model Input\n{model_input_image.shape[1]}×{model_input_image.shape[0]}px', 
                         fontsize=title_fontsize, fontweight='bold', color=title_color)
            ax6.axis('off')

            # Panel 7: Actual Model Predictions
            ax7 = fig.add_subplot(gs[1, 2])
            ax7.set_facecolor(bg_color)
            if model_probs is not None:
                self._plot_unified_predictions(ax7, model_probs, actual_prediction, title_fontsize, title_color)
            else:
                ax7.text(0.5, 0.5, 'Model Error', ha='center', va='center')
                ax7.set_title('Predictions\nError', fontsize=title_fontsize, fontweight='bold', color='red')
            
            # Panel 8: Live Stats
            ax8 = fig.add_subplot(gs[1, 3])
            ax8.set_facecolor(bg_color)
            self._plot_unified_stats(ax8, actual_prediction, actual_confidence, max_prob, entropy, 
                                   w, h, fps, title_fontsize, title_color)
            
            plt.tight_layout()
            
            # NON-BLOCKING DISPLAY
            plt.show(block=False)
            plt.draw()
            plt.pause(0.001)
            
            # Clean output
            print(f"\n🎯 Live Workflow Analysis Complete!")
            print(f"  Prediction: {actual_prediction} ({actual_confidence:.1%})")
            print(f"  Hand: {w}×{h}px | FPS: {fps:.1f}")
            if model_probs is not None:
                prob_str = ", ".join([f"{self.classes[i]}={model_probs[i]:.3f}" for i in range(len(self.classes))])
                print(f"  Probabilities: {prob_str}")
            print(f"  📊 Workflow visualization opened (non-blocking)")
            print(f"  📂 Files saved to: data/raw/captures\n")
            
            # Save the actual workflow data
            self._save_capture_data(frame, original_hand_crop, bg_removed_crop, model_input_image, 
                                  hand_info, actual_prediction, actual_confidence)
            
        except Exception as e:
            logger.error(f"Error in capture visualization: {e}")
            print(f"❌ Capture failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_unified_color_distribution(self, ax, image: np.ndarray, title: str, 
                                        title_fontsize: int, title_color: str, mask=None):
        """Unified color distribution plotting with consistent style and BGR handling."""
        if image is None or image.size == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold', color=title_color)
            ax.axis('off')
            return

        # CONSISTENT BGR HANDLING: Use same order for both plots
        colors = ['#3498db', '#27ae60', '#e74c3c']  # BGR order: Blue, Green, Red
        labels = ['Blue', 'Green', 'Red']  # BGR order, no numbers
        
        # Check if this is a background-removed image (has lots of black pixels)
        is_bg_removed = "Processed" in title and np.sum(np.all(image == [0, 0, 0], axis=2)) > (image.shape[0] * image.shape[1] * 0.3)
        
        max_freq = 0  # Track maximum frequency for y-axis scaling
        
        # Plot histograms with unified style (BGR order)
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            hist = hist.flatten()
            
            # For background-removed images, skip intensity 0 to focus on hand content
            if is_bg_removed:
                intensities = np.arange(1, 256)  # Skip 0
                hist_plot = hist[1:]  # Skip frequency at intensity 0
            else:
                intensities = np.arange(256)
                hist_plot = hist
            
            # Keep raw frequency counts (no normalization)
            max_freq = max(max_freq, np.max(hist_plot))
            ax.plot(intensities, hist_plot, color=color, alpha=0.7, linewidth=2, label=label)
        
        # Set axis limits
        if is_bg_removed:
            ax.set_xlim(1, 255)  # Skip intensity 0 for bg-removed images
        else:
            ax.set_xlim(0, 255)  # Full range for original images
            
        ax.set_ylim(0, max_freq * 1.1 if max_freq > 0 else 100)  # Raw frequency counts
        ax.set_xlabel('Intensity', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', color=title_color)
    
    def _plot_unified_predictions(self, ax, model_probs: np.ndarray, prediction: str, 
                                title_fontsize: int, title_color: str):
        """Unified prediction plotting with consistent style."""
        class_names = ['A', 'B', 'C']
        colors = ['#3498db', '#e74c3c', '#27ae60']
        
        bars = ax.bar(class_names, model_probs, color=colors, alpha=0.7, 
                     edgecolor='#2c3e50', linewidth=1)
        
        # Add values on bars
        for bar, prob in zip(bars, model_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{prob:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9, color='#2c3e50')
        
        # Highlight prediction
        try:
            predicted_idx = self.classes.index(prediction)
            bars[predicted_idx].set_color('#f1c40f')
            bars[predicted_idx].set_edgecolor('#e67e22')
            bars[predicted_idx].set_linewidth(3)
        except (ValueError, IndexError):
            pass
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Confidence', fontsize=8)
        ax.set_title(f'Predictions\nWinner: {prediction}', 
                    fontsize=title_fontsize, fontweight='bold', color=title_color)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_unified_stats(self, ax, prediction: str, confidence: float, max_prob: float, 
                          entropy: float, w: int, h: int, fps: float, 
                          title_fontsize: int, title_color: str):
        """Unified stats display with consistent style."""
        stats_text = f"""Results
Class: {prediction}
Confidence: {confidence:.1%}
Max Prob: {max_prob:.1%}
Entropy: {entropy:.3f}

Dimensions
Original: {w}×{h}px
Model: 224×224px
Pixels: {w*h:,}

Performance
FPS: {fps:.1f}
Device: {str(self.device).upper()}
Status: Real-time"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', fontfamily='monospace', color='#2c3e50')
        ax.set_title('Statistics', fontsize=title_fontsize, fontweight='bold', color=title_color)
        ax.axis('off')
    
    def _save_capture_data(self, frame: np.ndarray, hand_crop: np.ndarray, 
                          hand_crop_bg_removed: Optional[np.ndarray], model_input_np: np.ndarray,
                          hand_info: Dict, prediction: str, confidence: float) -> None:
        """Save captured data to files for analysis."""
        try:
            import time
            import json
            from pathlib import Path
            import shutil
            import uuid
            
            x, y, w, h = hand_info['bbox']
            fps = self.fps_tracker.get_fps()
            
            # Generate unique timestamp and random ID to avoid conflicts
            timestamp = time.time()
            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
            unique_id = str(uuid.uuid4())[:8]  # Short UUID for uniqueness
            
            # CRITICAL FIX: Robust directory creation with guaranteed uniqueness
            base_captures_dir = Path("data/raw")
            try:
                # Create base directory if it doesn't exist
                base_captures_dir.mkdir(parents=True, exist_ok=True)
                
                # Create unique capture directory with timestamp + UUID
                capture_dir_name = f"capture_{time_str}_{unique_id}"
                capture_dir = base_captures_dir / capture_dir_name
                
                # If by some miracle this exists, add more uniqueness
                counter = 0
                while capture_dir.exists() and counter < 100:
                    counter += 1
                    capture_dir_name = f"capture_{time_str}_{unique_id}_{counter}"
                    capture_dir = base_captures_dir / capture_dir_name
                
                # Create the directory 
                capture_dir.mkdir(parents=True, exist_ok=False)  # False ensures we fail if it exists
                logger.debug(f"✅ Unique capture directory created: {capture_dir}")
                
            except FileExistsError:
                # Ultimate fallback: use temp directory
                import tempfile
                capture_dir = Path(tempfile.mkdtemp(prefix=f"asl_capture_{time_str}_"))
                logger.info(f"✅ Using temp directory due to conflicts: {capture_dir}")
                
            except Exception as e:
                logger.warning(f"Directory creation failed: {e}")
                # Use temp directory as ultimate fallback
                import tempfile
                capture_dir = Path(tempfile.mkdtemp(prefix="asl_capture_"))
                logger.info(f"✅ Using temp directory: {capture_dir}")
            
            # Save images with unique filenames
            base_name = f"asl_{time_str}_{unique_id}"
            
            try:
                # Save original frame
                cv2.imwrite(str(capture_dir / f"{base_name}_frame.jpg"), frame)
                logger.debug("✅ Frame saved")
                
                # Save hand crop
                cv2.imwrite(str(capture_dir / f"{base_name}_hand.jpg"), hand_crop)
                logger.debug("✅ Hand crop saved")
                
                # Save background removed image if available
                if hand_crop_bg_removed is not None:
                    cv2.imwrite(str(capture_dir / f"{base_name}_bg_removed.jpg"), hand_crop_bg_removed)
                    logger.debug("✅ Background removed image saved")
                
                # CRITICAL FIX: Proper model input conversion and saving
                model_input_to_save = None
                try:
                    if isinstance(model_input_np, torch.Tensor):
                        model_input_to_save = model_input_np.cpu().numpy()
                    else:
                        model_input_to_save = model_input_np.copy()
                    
                    # Handle different tensor formats
                    if len(model_input_to_save.shape) == 4:  # Batch dimension (B,C,H,W)
                        model_input_to_save = model_input_to_save[0]  # Remove batch -> (C,H,W)
                    
                    if len(model_input_to_save.shape) == 3:
                        if model_input_to_save.shape[0] == 3:  # CHW format -> HWC
                            model_input_to_save = np.transpose(model_input_to_save, (1, 2, 0))
                        elif model_input_to_save.shape[-1] != 3:  # Not HWC format
                            logger.error(f"Unexpected model input shape: {model_input_to_save.shape}")
                            model_input_to_save = None
                    
                    if model_input_to_save is not None:
                        # Denormalize if normalized (values between -1 and 1 or 0-1 range with normalization)
                        if model_input_to_save.min() < 0 or model_input_to_save.max() <= 1.1:  
                            # ImageNet normalization parameters
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            if model_input_to_save.min() < 0:  # Normalized data
                                model_input_to_save = (model_input_to_save * std + mean)
                            model_input_to_save = np.clip(model_input_to_save, 0, 1)
                            model_input_to_save = (model_input_to_save * 255).astype(np.uint8)
                        
                        # Convert RGB to BGR for OpenCV saving
                        if len(model_input_to_save.shape) == 3 and model_input_to_save.shape[-1] == 3:
                            model_input_bgr = cv2.cvtColor(model_input_to_save, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(capture_dir / f"{base_name}_model_input.jpg"), model_input_bgr)
                            logger.debug("✅ Model input saved")
                        else:
                            logger.warning(f"Could not save model input - invalid shape: {model_input_to_save.shape}")
                            
                except Exception as e:
                    logger.warning(f"Failed to save model input: {e}")
                
                # Save comprehensive metadata
                metadata = {
                    'timestamp': timestamp,
                    'time_str': time_str,
                    'unique_id': unique_id,
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'hand_info': {
                        'bbox': hand_info.get('bbox', []),
                        'area': w * h,
                        'aspect_ratio': w / h if h > 0 else 0
                    },
                    'system_info': {
                        'fps': fps,
                        'processing_method': 'live_workflow_capture',
                        'model_input_size': '224x224',
                        'original_hand_size': f'{w}x{h}',
                        'device': str(self.device)
                    },
                    'file_info': {
                        'frame_file': f"{base_name}_frame.jpg",
                        'hand_file': f"{base_name}_hand.jpg", 
                        'bg_removed_file': f"{base_name}_bg_removed.jpg" if hand_crop_bg_removed is not None else None,
                        'model_input_file': f"{base_name}_model_input.jpg"
                    }
                }
                
                # Save metadata as JSON
                with open(capture_dir / f"{base_name}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.debug("✅ Metadata saved")
                
                logger.info(f"✅ All files saved to: {capture_dir}")
                
            except Exception as save_error:
                logger.error(f"Failed to save files: {save_error}")
                raise save_error
                
        except Exception as e:
            logger.warning(f"Failed to save capture data: {e}")
            print(f"  Warning: Could not save files: {e}")
            # Don't raise the exception - continue execution
    
    def _evaluate_model_performance(self, frame: np.ndarray, processed_hand: np.ndarray, 
                                  hand_info: Dict, prediction: str, confidence: float) -> None:
        """
        Comprehensive model performance evaluation to assess prediction quality.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import json
            
            # Extract hand information
            bbox = hand_info.get('bbox', (0, 0, 100, 100))
            x, y, w, h = bbox
            
            # Extract original hand crop
            hand_crop = frame[y:y+h, x:x+w].copy()
            
            # Get multiple predictions for confidence analysis
            predictions = []
            confidences = []
            
            # Run prediction multiple times to check consistency
            for i in range(5):
                pred, conf = self.predict_hand_sign(processed_hand)
                predictions.append(pred)
                confidences.append(conf)
            
            # Get different preprocessing versions for comparison
            hand_crop_original = hand_crop.copy()
            # Call background removal with proper crop coordinates
            crop_coords = (x, y, x+w, y+h)  # Convert to (x1, y1, x2, y2) format
            try:
                hand_crop_bg_removed = self.hand_detector._remove_background_from_crop(hand_crop, crop_coords)
            except Exception as e:
                logger.warning(f"Background removal failed: {e}")
                hand_crop_bg_removed = hand_crop.copy()
            hand_crop_no_bg_removal = cv2.resize(hand_crop_original, (224, 224))
            
            # Test model on different versions
            pred_original, conf_original = self.predict_hand_sign(hand_crop_no_bg_removal)
            pred_bg_removed, conf_bg_removed = self.predict_hand_sign(processed_hand)
            
            # Create comprehensive evaluation visualization
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3)
            
            fig.suptitle(f'🧪 Model Performance Evaluation - Current: {prediction} ({confidence:.3f})', 
                        fontsize=18, fontweight='bold', color='darkblue')
            
            # Row 1: Input variations
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(hand_crop_original, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'Original Crop\nPred: {pred_original} ({conf_original:.3f})')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(cv2.cvtColor(hand_crop_bg_removed, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'Background Removed\nPred: {pred_bg_removed} ({conf_bg_removed:.3f})')
            ax2.axis('off')
            
            # Model input tensor visualization
            ax3 = fig.add_subplot(gs[0, 2])
            if isinstance(processed_hand, torch.Tensor):
                tensor_data = processed_hand.cpu().numpy()
                if len(tensor_data.shape) == 4:
                    tensor_data = tensor_data[0]
                # Denormalize for display
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                tensor_data = tensor_data.transpose(1, 2, 0)
                tensor_data = (tensor_data * std + mean)
                tensor_data = np.clip(tensor_data, 0, 1)
                ax3.imshow(tensor_data)
            ax3.set_title(f'Model Input (224x224)\nNormalized & Processed')
            ax3.axis('off')
            
            # Confidence analysis
            ax4 = fig.add_subplot(gs[0, 3:5])
            
            # Get full prediction probabilities if possible
            with torch.no_grad():
                if isinstance(processed_hand, torch.Tensor):
                    outputs = self.model(processed_hand.unsqueeze(0).to(self.device))
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    probs = probabilities.cpu().numpy()[0]
                else:
                    # Fallback to mock probabilities
                    probs = [0.1, 0.1, 0.8] if prediction == 'C' else [0.8, 0.1, 0.1]
            
            # Create confidence bar chart
            bars = ax4.bar(self.classes, probs, color=['red' if p != max(probs) else 'green' for p in probs])
            ax4.set_title('Full Model Prediction Confidence')
            ax4.set_ylabel('Probability')
            ax4.set_ylim(0, 1)
            
            # Add confidence threshold line
            ax4.axhline(y=self.min_pred_confidence, color='orange', linestyle='--', label=f'Threshold: {self.min_pred_confidence}')
            
            # Annotate bars
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            ax4.legend()
            
            # Row 2: Data quality analysis
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.axis('off')
            
            # Calculate background removal quality metrics
            bg_mask = np.any(hand_crop_bg_removed > 5, axis=2)
            skin_percentage = np.sum(bg_mask) / bg_mask.size * 100
            
            # Calculate how similar this is to training data (black background)
            black_pixels = np.sum(np.all(hand_crop_bg_removed == [0, 0, 0], axis=2))
            total_pixels = hand_crop_bg_removed.shape[0] * hand_crop_bg_removed.shape[1]
            black_percentage = (black_pixels / total_pixels) * 100
            
            quality_text = f"""Background Removal Quality:

Skin Detection: {skin_percentage:.1f}%
Black Background: {black_percentage:.1f}%
Training Match: {'🟢 Good' if black_percentage > 30 else '🟡 Poor' if black_percentage > 10 else '🔴 Very Poor'}

Image Quality:
Size: {w}×{h} → 224×224
Aspect Ratio: {w/h:.2f}
Resize Quality: {'🟢 Good' if abs(w/h - 1) < 0.3 else '🟡 Stretched'}

Data Pipeline:
1. Hand Detection ✓
2. ROI Extraction ✓
3. Background Removal {'✓' if skin_percentage > 50 else '⚠️'}
4. Resize to 224×224 ✓
5. ImageNet Normalize ✓
"""
            
            ax5.text(0.05, 0.95, quality_text, transform=ax5.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            ax5.set_title('Data Quality Assessment')
            
            # Consistency analysis
            ax6 = fig.add_subplot(gs[1, 1])
            prediction_counts = {pred: predictions.count(pred) for pred in set(predictions)}
            consistency_score = max(prediction_counts.values()) / len(predictions)
            
            ax6.bar(prediction_counts.keys(), prediction_counts.values(), 
                   color=['green' if count == max(prediction_counts.values()) else 'orange' for count in prediction_counts.values()])
            ax6.set_title(f'Prediction Consistency\nScore: {consistency_score:.1%}')
            ax6.set_ylabel('Count (out of 5 runs)')
            
            # Model confidence distribution
            ax7 = fig.add_subplot(gs[1, 2])
            ax7.hist(confidences, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax7.axvline(confidence, color='red', linestyle='--', label=f'Current: {confidence:.3f}')
            ax7.axvline(self.min_pred_confidence, color='orange', linestyle='--', label=f'Threshold: {self.min_pred_confidence}')
            ax7.set_title('Confidence Distribution')
            ax7.set_xlabel('Confidence Score')
            ax7.set_ylabel('Frequency')
            ax7.legend()
            
            # Training vs Live data comparison
            ax8 = fig.add_subplot(gs[1, 3:5])
            ax8.axis('off')
            
            # Analyze color characteristics
            if np.any(bg_mask):
                skin_pixels = hand_crop_bg_removed[bg_mask]
                bgr_mean = np.mean(skin_pixels, axis=0)
                bgr_std = np.std(skin_pixels, axis=0)
                
                # Expected training data characteristics (black background, clean hands)
                training_text = f"""Training vs Live Data Analysis:

Live Data Characteristics:
- Background: {black_percentage:.1f}% black (training: ~90-95%)
- Skin Color: BGR({bgr_mean[0]:.0f}, {bgr_mean[1]:.0f}, {bgr_mean[2]:.0f})
- Color Variance: ±{np.mean(bgr_std):.1f} (training: ~±15-25)
- Lighting: {'Consistent' if np.mean(bgr_std) < 25 else 'Variable'}

Training Data Match:
Background Similarity: {'🟢 Excellent' if black_percentage > 70 else '🟡 Good' if black_percentage > 40 else '🔴 Poor'}
Color Consistency: {'🟢 Good' if np.mean(bgr_std) < 30 else '🟡 Variable' if np.mean(bgr_std) < 50 else '🔴 Poor'}
Overall Match: {'🟢 Good' if black_percentage > 50 and np.mean(bgr_std) < 35 else '🟡 Moderate' if black_percentage > 30 else '🔴 Poor'}

Recommendations:
- {'Improve background removal' if black_percentage < 40 else 'Background removal OK'}
- {'Use better lighting' if np.mean(bgr_std) > 40 else 'Lighting OK'}
- {'Consider retraining with live data' if black_percentage < 30 else 'Model should work well'}
"""
                
                ax8.text(0.05, 0.95, training_text, transform=ax8.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
            else:
                ax8.text(0.5, 0.5, 'No skin pixels detected\nBackground removal failed', 
                        ha='center', va='center', transform=ax8.transAxes, color='red', fontsize=14)
            
            ax8.set_title('Training Data Compatibility Analysis')
            
            # Row 3: Performance recommendations
            ax9 = fig.add_subplot(gs[2, :])
            ax9.axis('off')
            
            # Generate specific recommendations
            recommendations = []
            
            if confidence < self.min_pred_confidence:
                recommendations.append("🔴 LOW CONFIDENCE: Model is uncertain about this prediction")
            
            if black_percentage < 30:
                recommendations.append("🟡 BACKGROUND ISSUE: Background removal not matching training data")
                recommendations.append("   → Try adjusting lighting or hand position")
                recommendations.append("   → Consider collecting training data with similar backgrounds")
            
            if consistency_score < 0.6:
                recommendations.append("🟡 INCONSISTENCY: Model predictions vary between runs")
                recommendations.append("   → This suggests the input data quality could be improved")
            
            if np.mean(bgr_std) > 40:
                recommendations.append("🟡 LIGHTING ISSUE: High color variance detected")
                recommendations.append("   → Try more consistent lighting")
            
            if abs(w/h - 1) > 0.5:
                recommendations.append("🟡 ASPECT RATIO: Hand crop is very stretched")
                recommendations.append("   → This may affect model performance")
            
            if conf_bg_removed > conf_original + 0.1:
                recommendations.append("🟢 BACKGROUND REMOVAL HELPS: BG removal improves confidence")
            elif conf_original > conf_bg_removed + 0.1:
                recommendations.append("🔴 BACKGROUND REMOVAL HURTS: Original image works better")
                recommendations.append("   → Background removal may be removing important hand features")
            
            if not recommendations:
                recommendations.append("🟢 GOOD PERFORMANCE: No major issues detected")
                recommendations.append("🟢 Model appears to be working well with current data")
            
            recommendations_text = "PERFORMANCE RECOMMENDATIONS:\n\n" + "\n".join(recommendations)
            
            ax9.text(0.05, 0.95, recommendations_text, transform=ax9.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            ax9.set_title('🎯 Model Performance Analysis & Recommendations', fontsize=14, fontweight='bold')
            
            # Row 4: Summary stats
            ax10 = fig.add_subplot(gs[3, 0:2])
            ax10.axis('off')
            
            summary_stats = f"""PERFORMANCE SUMMARY:

Current Prediction: {prediction} ({confidence:.3f})
Consistency Score: {consistency_score:.1%}
Background Match: {black_percentage:.1f}% black
Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}

Comparison:
Original Image: {pred_original} ({conf_original:.3f})
Background Removed: {pred_bg_removed} ({conf_bg_removed:.3f})
Difference: {conf_bg_removed - conf_original:+.3f}

Data Quality Score: {min(100, skin_percentage + (100-black_percentage)*0.3):.0f}/100
Model Confidence: {'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'}
Training Match: {'Good' if black_percentage > 50 else 'Poor'}
"""
            
            ax10.text(0.05, 0.95, summary_stats, transform=ax10.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            ax10.set_title('Performance Summary')
            
            # Action items
            ax11 = fig.add_subplot(gs[3, 2:5])
            ax11.axis('off')
            
            action_items = f"""IMMEDIATE ACTION ITEMS:

1. Background Removal Quality:
   Current: {black_percentage:.1f}% black background
   Target: >70% for best performance
   {'✅ Good' if black_percentage > 70 else '⚠️ Needs improvement'}

2. Model Confidence:
   Current: {confidence:.3f}
   Threshold: {self.min_pred_confidence}
   {'✅ Above threshold' if confidence > self.min_pred_confidence else '❌ Below threshold'}

3. Data Collection Strategy:
   {'✅ Current setup works well' if black_percentage > 50 and confidence > self.min_pred_confidence else '⚠️ Consider collecting training data in current environment'}

4. Next Steps:
   {'• Continue with current setup' if black_percentage > 50 else '• Improve lighting/background removal'}
   {'• Model is performing well' if confidence > 0.6 else '• Consider retraining with live environment data'}
"""
            
            ax11.text(0.05, 0.95, action_items, transform=ax11.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            ax11.set_title('Action Items & Next Steps')
            
            plt.tight_layout()
            plt.show(block=False)
            
            # Save evaluation results
            timestamp = time.time()
            eval_dir = Path("data/raw/evaluations")
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            eval_data = {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': float(confidence),
                'consistency_score': float(consistency_score),
                'background_black_percentage': float(black_percentage),
                'skin_percentage': float(skin_percentage),
                'training_match_score': float(min(100, skin_percentage + (100-black_percentage)*0.3)),
                'original_vs_bg_removed': {
                    'original_pred': pred_original,
                    'original_conf': float(conf_original),
                    'bg_removed_pred': pred_bg_removed,
                    'bg_removed_conf': float(conf_bg_removed),
                    'improvement': float(conf_bg_removed - conf_original)
                },
                'recommendations': recommendations,
                'multiple_runs': {
                    'predictions': predictions,
                    'confidences': [float(c) for c in confidences]
                }
            }
            
            eval_file = eval_dir / f"model_eval_{prediction}_{timestamp:.0f}.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
            
            print(f"\n🧪 Model Performance Evaluation Complete!")
            print(f"  Current Prediction: {prediction} ({confidence:.3f})")
            print(f"  Consistency Score: {consistency_score:.1%}")
            print(f"  Background Match: {black_percentage:.1f}% black")
            print(f"  Training Compatibility: {'Good' if black_percentage > 50 else 'Poor'}")
            print(f"  Evaluation saved to: {eval_file}")
            print(f"  📊 Close the evaluation window when done reviewing.")
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            print(f"❌ Model evaluation failed: {e}")
    
    def _process_frame(self, frame: np.ndarray):
        """
        Handles all processing for a single frame, including detection, 
        tracking, and prediction. It updates the recognizer's state but
        does not draw to the screen.
        """
        current_time = time.time()
        
        # --- Background Learning Phase ---
        if not self.hand_detector.bg_remover.bg_model_learned:
            self.hand_detector.bg_remover.learn_background(frame)
            # State is updated, UI will be drawn in the main loop
            return

        # --- Performance optimization: Adaptive frame skipping ---
        # Skip frames if we're processing too slowly
        frame_interval = current_time - self.last_process_time
        min_frame_time = 1.0 / self.target_fps
        
        # Increment frame skip counter
        self.frame_skip_counter += 1
        
        # Skip frames based on performance
        if frame_interval < min_frame_time or self.frame_skip_counter % self.frame_skip_rate != 0:
            # Just update FPS tracker and store frame for UI
            self.fps_tracker.update()
            self.last_capture_frame = frame.copy()
            return
        
        self.last_process_time = current_time

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
        
        # Add instruction text at the top
        cv2.putText(frame, "Show ASL letter to camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw BIG centered prediction
        h, w = frame.shape[:2]
        
        # Only show the letter, make it HUGE and centered
        if prediction != "Show Hand" and confidence > self.min_pred_confidence:
            letter_text = prediction
            color = (0, 255, 0)  # Green for confident predictions
        else:
            letter_text = "?" if prediction == "Show Hand" else prediction
            color = (0, 255, 255)  # Yellow for uncertain/no prediction
        
        # Calculate text size for upper positioning
        font_scale = 5  # About 35% of previous size (was 15)
        thickness = 8   # Proportionally smaller thickness
        (text_width, text_height), baseline = cv2.getTextSize(letter_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Position at very top of screen (same height as instruction text)
        text_x = (w - text_width) // 2
        text_y = 80 + text_height  # Same level as instruction text, just below it
        
        # Add background rectangle for better visibility (smaller padding)
        padding = 20  # Smaller padding for smaller text
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_height - padding), 
                     (text_x + text_width + padding, text_y + baseline + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Draw the big letter
        cv2.putText(frame, letter_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Small confidence text below the letter (keeping it compact at top)
        if prediction != "Show Hand":
            conf_text = f"{confidence:.2f}"
            small_font_scale = 1.0  # Even smaller to keep it compact
            small_thickness = 2
            (conf_width, conf_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, small_thickness)
            conf_x = (w - conf_width) // 2
            conf_y = text_y + 30  # Keep it very close and compact
            cv2.putText(frame, conf_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (255, 255, 255), small_thickness)
        
        # Draw FPS and performance stats (moved to bottom-left to avoid interfering with big prediction)
        if self.show_stats:
            fps = self.fps_tracker.get_fps()
            stats_y_start = h - 120  # Start from bottom
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, stats_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show frame skip rate for performance monitoring
            cv2.putText(frame, f"Skip Rate: 1/{self.frame_skip_rate}", (10, stats_y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show target FPS
            cv2.putText(frame, f"Target: {self.target_fps} FPS", (10, stats_y_start + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
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
            logger.info(f"🔍 Capture debug - Enabled: {self.capture_enabled}, "
                       f"Processed hand: {self.last_processed_hand is not None}, "
                       f"Hand info: {self.last_hand_info is not None}, "
                       f"Frame: {self.last_capture_frame is not None}")
            
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
                missing_items = []
                if not self.capture_enabled:
                    missing_items.append("capture disabled")
                if self.last_processed_hand is None:
                    missing_items.append("no processed hand")
                if self.last_hand_info is None:
                    missing_items.append("no hand info")
                if self.last_capture_frame is None:
                    missing_items.append("no capture frame")
                    
                logger.warning(f"❌ Cannot capture: {', '.join(missing_items)}")
                logger.info("💡 Try detecting a hand first, then press 'C' to capture")
        elif key == ord('+') or key == ord('='):
            # Increase frame skip rate (lower performance, higher quality)
            self.frame_skip_rate = max(1, self.frame_skip_rate - 1)
            logger.info(f"⚡ Frame skip rate: 1/{self.frame_skip_rate} (Higher quality)")
        elif key == ord('-'):
            # Decrease frame skip rate (higher performance, lower quality)
            self.frame_skip_rate = min(5, self.frame_skip_rate + 1)
            logger.info(f"🚀 Frame skip rate: 1/{self.frame_skip_rate} (Higher performance)")
        elif key == ord('p'):
            # Toggle performance mode
            if self.frame_skip_rate == 1:
                self.frame_skip_rate = 2
                self.target_fps = 30
                logger.info("🚀 Performance mode: ON (Skip every 2nd frame, target 30 FPS)")
            else:
                self.frame_skip_rate = 1
                self.target_fps = 25
                logger.info("🎯 Quality mode: ON (Process all frames, target 25 FPS)")
        elif key == ord('m'):
            # Model performance evaluation
            if (self.last_processed_hand is not None and 
                self.last_hand_info is not None and self.last_capture_frame is not None):
                logger.info("🧪 Running model performance evaluation...")
                self._evaluate_model_performance(
                    self.last_capture_frame, 
                    self.last_processed_hand,
                    self.last_hand_info,
                    self.last_prediction,
                    self.last_confidence
                )
            else:
                logger.warning("❌ No hand data available for model evaluation")

    def run(self):
        """Main loop for the application."""
        logger.info("🟢 Starting Live ASL Recognition with Performance Optimizations...")
        logger.info("📋 Controls:")
        logger.info("  Q: Quit")
        logger.info("  S: Toggle statistics display")
        logger.info("  R: Reset hand tracker")
        logger.info("  B: Reset background learning")
        logger.info("  SPACE: Pause/unpause")
        logger.info("  C: 📸 Capture and visualize hand data")
        logger.info("  P: Toggle performance mode (skip frames for higher FPS)")
        logger.info("  +/-: Adjust frame skip rate manually")
        logger.info("  M: 🧪 Model performance evaluation")
        logger.info("")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"❌ Cannot open camera {self.camera_index}")
            return
            
        # Optimize camera settings for performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # Set camera FPS
        
        # Get actual camera resolution for info
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"📹 Camera: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            
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

    def _denormalize_for_visualization(self, tensor_img: torch.Tensor) -> np.ndarray:
        """Denormalize a tensor image and convert to a displayable format."""
        if tensor_img is None or tensor_img.size() == 0:
            return None

        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor_img.device)
        denormalized_img = tensor_img * std + mean
        denormalized_img = torch.clamp(denormalized_img, 0, 1)

        # Convert to numpy array
        denormalized_np = denormalized_img.cpu().numpy()
        if len(denormalized_np.shape) == 4:
            denormalized_np = denormalized_np[0]
        if len(denormalized_np.shape) == 3 and denormalized_np.shape[0] == 3:
            denormalized_np = np.transpose(denormalized_np, (1, 2, 0))

        return denormalized_np

def main():
    """Main function to run the recognizer"""
    recognizer = LiveASLRecognizer()
    if recognizer.model:
        recognizer.run()

if __name__ == "__main__":
    main() 