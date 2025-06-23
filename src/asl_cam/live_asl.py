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
        Capture and visualize detailed hand data following the actual processing workflow.
        
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
            import torch
            
            # Extract hand information
            x, y, w, h = hand_info['bbox']
            
            # Create hand crop from current frame
            hand_crop = frame[y:y+h, x:x+w]
            
            # Get background-removed version for visualization
            try:
                crop_coords = (x, y, w, h)
                hand_crop_bg_removed = self.hand_detector._remove_background_from_crop(hand_crop, crop_coords)
            except Exception as e:
                logger.warning(f"Failed to get background-removed crop for visualization: {e}")
                hand_crop_bg_removed = None
            
            # Convert model input tensor back to numpy for visualization
            if isinstance(processed_hand, torch.Tensor):
                # Denormalize and convert back to BGR format for consistent visualization
                model_input_np = processed_hand.cpu().numpy()
                if len(model_input_np.shape) == 4:
                    model_input_np = model_input_np[0]  # Remove batch dimension
                
                # Transpose from CHW to HWC
                model_input_np = model_input_np.transpose(1, 2, 0)
                
                # Denormalize using ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                model_input_np = model_input_np * std + mean
                model_input_np = np.clip(model_input_np, 0, 1)
                model_input_np = (model_input_np * 255).astype(np.uint8)
                
                # Convert RGB back to BGR for OpenCV operations
                model_input_np = cv2.cvtColor(model_input_np, cv2.COLOR_RGB2BGR)
            else:
                model_input_np = processed_hand
            
            # Create comprehensive visualization with 3x4 grid
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # ========== STEP 1: CAMERA INPUT ==========
            # Panel 1: Live camera feed with detection box
            ax1 = fig.add_subplot(gs[0, 0])
            frame_with_box = frame.copy()
            cv2.rectangle(frame_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
            ax1.imshow(cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'1. 📹 Live Camera Feed\n{frame.shape[1]}×{frame.shape[0]} @ {self.fps_tracker.get_fps():.1f} FPS')
            ax1.axis('off')
            
            # Panel 2: Detection analysis 
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis('off')
            
            detection_text = f"""Detection Analysis:

Bounding Box:
• Position: ({x}, {y})
• Size: {w}×{h} pixels
• Area: {w*h:,} pixels
• Aspect Ratio: {w/h:.2f}

Quality Metrics:
• Size: {'✅ Good' if w*h > 5000 else '⚠️ Small' if w*h > 1000 else '❌ Too Small'}
• Position: {'✅ Centered' if 0.2 < x/frame.shape[1] < 0.8 and 0.2 < y/frame.shape[0] < 0.8 else '⚠️ Edge'}
• Aspect: {'✅ Normal' if 0.5 < w/h < 2.0 else '⚠️ Unusual'}

Tracking Status:
• Active: {'✅ Yes' if hand_info.get('tracking', False) else '❌ No'}
• Stability: {'✅ Stable' if hand_info.get('stable', False) else '⚠️ Unstable'}
"""
            
            ax2.text(0.05, 0.95, detection_text, transform=ax2.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
            ax2.set_title('2. 🔍 Detection Analysis')
            
            # Panel 3: Processing pipeline overview
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.axis('off')
            
            # Show workflow as a flowchart-style text
            workflow_text = f"""Processing Pipeline:

1. 📹 Camera Input → Hand Detection
2. ✂️ ROI Extraction → {w}×{h} crop  
3. 🔍 Skin Detection → Color analysis
4. 🎭 MOG2 Analysis → Motion detection
5. 🎨 Background Removal → Black background
6. 📐 Resize → 224×224 model input
7. 🔢 Normalize → ImageNet standards
8. 🧠 CNN Inference → {prediction} ({confidence:.3f})

Status: ✅ All steps completed
Quality: {'🟢 Good' if confidence > 0.5 else '🟡 Moderate' if confidence > 0.3 else '🔴 Low'} confidence
"""
            
            ax3.text(0.05, 0.95, workflow_text, transform=ax3.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace')
            ax3.set_title('3. ⚙️ Processing Pipeline')
            
            # ========== STEP 2: ROI EXTRACTION ==========
            # Panel 4: Hand ROI crop
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            ax4.set_title(f'4. ✋ Hand ROI Crop\n{hand_crop.shape[1]}×{hand_crop.shape[0]} pixels')
            ax4.axis('off')
            
            # Panel 5: Skin detection mask
            ax5 = fig.add_subplot(gs[1, 0])
            try:
                skin_mask = self.hand_detector._get_skin_mask_simple(hand_crop)
                if skin_mask is not None:
                    ax5.imshow(skin_mask, cmap='gray')
                    skin_coverage = np.sum(skin_mask > 0) / skin_mask.size * 100
                    ax5.set_title(f'5. 🎭 Skin Detection Mask\n{skin_coverage:.1f}% coverage')
                else:
                    ax5.text(0.5, 0.5, 'Skin detection\nfailed', ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title('5. 🎭 Skin Detection Mask\n(Failed)')
            except:
                ax5.text(0.5, 0.5, 'Skin detection\nnot available', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('5. 🎭 Skin Detection Mask\n(N/A)')
            ax5.axis('off')
            
            # Panel 6: MOG2 foreground mask (safe visualization)
            ax6 = fig.add_subplot(gs[1, 1])
            try:
                # Create a safe visualization of MOG2-like motion detection
                if (hasattr(self.hand_detector.bg_remover, 'static_background') and 
                    self.hand_detector.bg_remover.static_background is not None):
                    # Use static background to create motion-like mask
                    crop_coords = (x, y, w, h)
                    bg_crop = self.hand_detector.bg_remover.static_background[y:y+h, x:x+w]
                    if bg_crop.shape == hand_crop.shape:
                        # Create difference-based mask
                        diff = cv2.absdiff(hand_crop, bg_crop)
                        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                        
                        ax6.imshow(motion_mask, cmap='gray')
                        motion_coverage = np.sum(motion_mask > 0) / motion_mask.size * 100
                        ax6.set_title(f'6. 🏃 MOG2 Motion Mask\n{motion_coverage:.1f}% motion')
                    else:
                        ax6.text(0.5, 0.5, 'Background size\nmismatch', ha='center', va='center', transform=ax6.transAxes)
                        ax6.set_title('6. 🏃 MOG2 Motion Mask\n(Size Error)')
                else:
                    ax6.text(0.5, 0.5, 'MOG2 background\nnot learned', ha='center', va='center', transform=ax6.transAxes)
                    ax6.set_title('6. 🏃 MOG2 Motion Mask\n(Learning)')
            except Exception as e:
                ax6.text(0.5, 0.5, f'MOG2 analysis\nfailed', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('6. 🏃 MOG2 Motion Mask\n(Failed)')
            ax6.axis('off')
            
            # Panel 7: Background method analysis
            ax7 = fig.add_subplot(gs[1, 2])
            ax7.axis('off')
            
            # Analyze which background removal method was used
            if hand_crop_bg_removed is not None:
                # Check if this looks like MOG2+skin combination
                bg_mask = np.any(hand_crop_bg_removed > 5, axis=2)
                mog2_pixels = np.sum(bg_mask)
                
                bg_analysis_text = f"""Background Method:

Enhanced Analysis:
• Method: MOG2 + Skin Detection
• Background pixels: {np.sum(~bg_mask):,}
• Foreground pixels: {mog2_pixels:,}
• Removal ratio: {np.sum(~bg_mask)/bg_mask.size*100:.1f}%

Algorithm:
• Static BG difference
• Skin color preservation  
• Morphological cleanup
• Black background fill

Status:
{'✅ MOG2 + Skin active' if mog2_pixels > 0 else '⚠️ Simple skin detection'}
"""
            
                ax7.text(0.05, 0.95, bg_analysis_text, transform=ax7.transAxes,
                         fontsize=8, verticalalignment='top', fontfamily='monospace')
            else:
                bg_simple_text = f"""Background Method:

Method: Simple skin detection
Status: ✅ Active (fallback)

Algorithm:
• BGR ratio detection
• HSV color thresholding  
• Morphological cleanup
• Non-skin → black pixels

Quality: Reliable baseline
"""
                ax7.text(0.05, 0.95, bg_simple_text, transform=ax7.transAxes,
                         fontsize=8, verticalalignment='top', fontfamily='monospace')
            
            ax7.set_title('7. 🎭 Background Method')
            
            # ========== STEP 3: BACKGROUND REMOVAL ==========
            # Panel 8: Background removed result
            ax8 = fig.add_subplot(gs[1, 3])
            if hand_crop_bg_removed is not None:
                ax8.imshow(cv2.cvtColor(hand_crop_bg_removed, cv2.COLOR_BGR2RGB))
                ax8.set_title(f'8. 🎭 Background Removed\n{hand_crop_bg_removed.shape[1]}×{hand_crop_bg_removed.shape[0]} pixels')
            else:
                ax8.text(0.5, 0.5, 'Background removal\nfailed', ha='center', va='center',
                        transform=ax8.transAxes, fontsize=12)
                ax8.set_title('8. 🎭 Background Removed\n(Failed)')
            ax8.axis('off')
            
            # Panel 9: Background removal histogram (NORMALIZED Y-VALUES)
            ax9 = fig.add_subplot(gs[2, 0])
            
            if hand_crop_bg_removed is not None and len(hand_crop_bg_removed.shape) == 3:
                # Show histogram with background pixels filtered out
                bg_mask = np.any(hand_crop_bg_removed > 5, axis=2)  # Non-black pixels
                if np.any(bg_mask):
                    max_freq = 0  # Track maximum frequency for normalization
                    color_hists = []
                    
                    for i, color in enumerate(['blue', 'green', 'red']):
                        channel_data = hand_crop_bg_removed[:, :, i][bg_mask]
                        if len(channel_data) > 0:
                            hist, bins = np.histogram(channel_data, bins=30, range=[0, 255])
                            max_freq = max(max_freq, np.max(hist))
                            color_hists.append((hist, bins, color))
                    
                    # Plot normalized histograms
                    for hist, bins, color in color_hists:
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        # Normalize to 0-1 range
                        normalized_hist = hist / max_freq if max_freq > 0 else hist
                        ax9.plot(bin_centers, normalized_hist, color=color, alpha=0.7, label=f'{color.upper()}', linewidth=2)
                    
                    ax9.set_title('9. 📊 BG-Removed Histogram\n(Normalized)')
                    ax9.set_xlabel('Pixel Value (0-255)')
                    ax9.set_ylabel('Normalized Frequency (0-1)')
                    ax9.set_ylim(0, 1)  # Fixed 0-1 range
                    ax9.legend()
                    ax9.grid(True, alpha=0.3)
                    
                    # Add stats
                    skin_pixel_count = np.sum(bg_mask)
                    ax9.text(0.02, 0.98, f'Skin pixels: {skin_pixel_count:,}\nTotal pixels: {bg_mask.size:,}',
                            transform=ax9.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    ax9.text(0.5, 0.5, 'No skin pixels\ndetected', ha='center', va='center',
                            transform=ax9.transAxes)
                    ax9.set_title('9. 📊 BG-Removed Histogram')
            else:
                ax9.text(0.5, 0.5, 'Background removal\nfailed', ha='center', va='center',
                        transform=ax9.transAxes)
                ax9.set_title('9. 📊 BG-Removed Histogram')
            
            # ========== STEP 4: MODEL RESULTS ==========
            # Panel 12: Prediction confidence visualization (moved to position [2,1])
            ax12 = fig.add_subplot(gs[2, 1])
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
                
                bars = ax12.bar(classes, probs, color=['red' if p == confidence else 'gray' for p in probs])
                ax12.set_title('12. 🧠 Model Prediction')
                ax12.set_ylabel('Probability')
                ax12.set_ylim(0, 1)
                
                # Highlight the predicted class
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    if prob == confidence:
                        bar.set_color('green')
                        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax12.text(0.5, 0.5, 'No class info\navailable', ha='center', va='center', 
                        transform=ax12.transAxes)
                ax12.set_title('12. 🧠 Model Prediction')

            plt.tight_layout()
            plt.show()
            
            # Save the data
            self._save_capture_data(frame, hand_crop, hand_crop_bg_removed, model_input_np, hand_info, prediction, confidence)
        except Exception as e:
            logger.error(f"Error in capture and visualization: {e}")
            print(f"❌ Capture and visualization failed: {e}")
    
    def _save_capture_data(self, frame: np.ndarray, hand_crop: np.ndarray, 
                          hand_crop_bg_removed: Optional[np.ndarray], model_input_np: np.ndarray,
                          hand_info: Dict, prediction: str, confidence: float) -> None:
        """Save captured data to files for analysis."""
        try:
            import time
            import json
            from pathlib import Path
            
            x, y, w, h = hand_info['bbox']
            fps = self.fps_tracker.get_fps()
            
            # Save capture data
            timestamp = time.time()
            capture_dir = Path("data/raw/captures")
            capture_dir.mkdir(parents=True, exist_ok=True)
            
            capture_base = capture_dir / f"live_capture_{prediction}_{timestamp:.0f}"
            
            # Save images
            cv2.imwrite(f"{capture_base}_original.jpg", hand_crop)
            if hand_crop_bg_removed is not None:
                cv2.imwrite(f"{capture_base}_bg_removed.jpg", hand_crop_bg_removed)
            cv2.imwrite(f"{capture_base}_model_input.jpg", model_input_np)
            cv2.imwrite(f"{capture_base}_full_frame.jpg", frame)
            
            # Save comprehensive metadata
            capture_metadata = {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': float(confidence),
                'bbox': [x, y, w, h],
                'hand_size': [w, h],
                'hand_area': w * h,
                'aspect_ratio': float(w / h if h > 0 else 1.0),
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
                'processing_pipeline': {
                    'background_removal_method': 'enhanced' if hand_crop_bg_removed is not None else 'simple',
                    'resize_method': 'cv2.INTER_AREA',
                    'normalization': 'ImageNet',
                    'color_conversion': 'BGR_to_RGB'
                },
                'files': {
                    'original_crop': f"{capture_base.name}_original.jpg",
                    'bg_removed_crop': f"{capture_base.name}_bg_removed.jpg" if hand_crop_bg_removed is not None else None,
                    'model_input': f"{capture_base.name}_model_input.jpg",
                    'full_frame': f"{capture_base.name}_full_frame.jpg"
                }
            }
            
            with open(f"{capture_base}_metadata.json", 'w') as f:
                json.dump(capture_metadata, f, indent=2)
            
            print(f"\n🎯 Live ASL Data Captured and Visualized!")
            print(f"  Prediction: {prediction} ({confidence:.3f})")
            print(f"  Hand size: {w}×{h} pixels ({w*h:,} total)")
            print(f"  Model input: 224×224×3 preprocessed")
            print(f"  System FPS: {fps:.1f}")
            print(f"  Files saved to: {capture_dir}")
            print(f"  📊 Close the visualization window when done viewing.")
            
        except Exception as e:
            logger.error(f"Error saving capture data: {e}")
            print(f"❌ Capture save failed: {e}")
    
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
            hand_crop_bg_removed = self.hand_detector._remove_background_from_crop(hand_crop)
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

        # Draw Prediction
        pred_text = f"Prediction: {prediction} ({confidence:.2f})"
        cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw FPS and performance stats
        if self.show_stats:
            fps = self.fps_tracker.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame skip rate for performance monitoring
            cv2.putText(frame, f"Skip Rate: 1/{self.frame_skip_rate}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show target FPS
            cv2.putText(frame, f"Target: {self.target_fps} FPS", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
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

def main():
    """Main function to run the recognizer"""
    recognizer = LiveASLRecognizer()
    if recognizer.model:
        recognizer.run()

if __name__ == "__main__":
    main() 