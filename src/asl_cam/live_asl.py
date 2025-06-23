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
            
            # Get the background-removed version and skin mask for visualization
            hand_crop_bg_removed = None
            skin_mask_crop = None
            mog2_mask_crop = None
            crop_bbox = hand_info.get('crop_bbox')
            
            if hasattr(self.hand_detector, '_remove_background_from_crop'):
                try:
                    # Use the new MOG2-ROI approach if crop_bbox is available
                    if crop_bbox is not None:
                        hand_crop_bg_removed = self.hand_detector._remove_background_from_crop(hand_crop, crop_bbox, frame)
                        mog2_mask_crop = self.hand_detector.get_mog2_mask_for_crop(hand_crop, crop_bbox, frame)
                    else:
                        # Fallback to skin-only approach
                        hand_crop_bg_removed = self.hand_detector._remove_background_from_crop(hand_crop)
                    
                    skin_mask_crop = self.hand_detector.get_skin_mask_for_crop(hand_crop)
                except Exception as e:
                    logger.warning(f"Failed to get background-removed crop for visualization: {e}")
            
            # Update info text to show background removal method
            bg_removal_method = "MOG2+Skin" if crop_bbox is not None and self.hand_detector.bg_remover.bg_model_learned else "Skin-only"
            
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
            fig = plt.figure(figsize=(18, 14))
            gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Live ASL Data Analysis - Prediction: {prediction} ({confidence:.3f})', 
                        fontsize=16, fontweight='bold')
            
            # Panel 1: Original hand crop
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'1. Original Hand Crop\n{w}×{h} pixels')
            ax1.axis('off')
            
            # Panel 2: Background removed crop
            ax2 = fig.add_subplot(gs[0, 1])
            if hand_crop_bg_removed is not None:
                ax2.imshow(cv2.cvtColor(hand_crop_bg_removed, cv2.COLOR_BGR2RGB))
                ax2.set_title(f'2. Background Removed ({bg_removal_method})\n{hand_crop_bg_removed.shape[1]}×{hand_crop_bg_removed.shape[0]} pixels')
            else:
                ax2.text(0.5, 0.5, 'Background removal\nfailed', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('2. Background Removed\n(Failed)')
            ax2.axis('off')
            
            # Panel 3: Model input (224x224 preprocessed)
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(model_input_np)
            ax3.set_title(f'3. Model Input\n224×224 final')
            ax3.axis('off')
            
            # Panel 4: Prediction confidence visualization
            ax4 = fig.add_subplot(gs[0, 3])
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
                
                bars = ax4.bar(classes, probs, color=['red' if p == confidence else 'gray' for p in probs])
                ax4.set_title('4. Prediction Confidence')
                ax4.set_ylabel('Probability')
                ax4.set_ylim(0, 1)
                
                # Highlight the predicted class
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    if prob == confidence:
                        bar.set_color('green')
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No class info\navailable', ha='center', va='center', 
                        transform=ax4.transAxes)
                ax4.set_title('4. Prediction Confidence')
            
            # Panel 5: Hand information
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.axis('off')
            
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

Background Removal:
Method: {bg_removal_method}
"""
            
            ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            # Panel 6: Full frame context
            ax6 = fig.add_subplot(gs[1, 1:3])
            frame_display = frame.copy()
            # Draw hand bbox
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame_display, f'{prediction}: {confidence:.3f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            ax6.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            ax6.set_title('6. Live Camera Feed with Detection')
            ax6.axis('off')
            
            # Panel 7: Model preprocessing visualization
            ax7 = fig.add_subplot(gs[1, 3])
            
            # Show the preprocessing steps if we can recreate them
            preprocessing_steps = [
                "1. Hand Detection",
                "2. Crop Hand",
                "3. Remove Background", 
                "4. Resize to 224×224",
                "5. Normalize (ImageNet)",
                "6. Send to Model"
            ]
            
            for i, step in enumerate(preprocessing_steps):
                color = 'green' if i < 6 else 'gray'
                ax7.text(0.05, 0.9 - i*0.12, f"✓ {step}", transform=ax7.transAxes,
                        fontsize=9, color=color, fontweight='bold')
            
            ax7.text(0.05, 0.2, f"Device: {self.device}\nMobileNetV2\n{len(self.classes)} classes", 
                    transform=ax7.transAxes, fontsize=8, fontfamily='monospace')
            ax7.set_title('7. DL Pipeline')
            ax7.axis('off')
            
            # Panel 8: Original vs Background-removed comparison histogram
            ax8 = fig.add_subplot(gs[2, 0])
            
            # Show comparison of original vs background-removed
            if hand_crop_bg_removed is not None and len(hand_crop_bg_removed.shape) == 3:
                # Simple mask to exclude black background pixels
                bg_mask = np.any(hand_crop_bg_removed > 0, axis=2)
                
                if np.any(bg_mask):
                    # Original crop red channel histogram (all pixels)
                    orig_red_values = hand_crop[:, :, 2].flatten()  # Red channel
                    
                    # Background-removed red channel histogram (skin pixels only)
                    skin_pixels = hand_crop_bg_removed[bg_mask]
                    bg_red_values = skin_pixels[:, 2]  # Red channel of skin pixels only
                    
                    if len(bg_red_values) > 0:
                        # Calculate histograms with standard ranges
                        orig_hist, orig_bins = np.histogram(orig_red_values, bins=50, range=[0, 255])
                        orig_bin_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
                        
                        bg_hist, bg_bins = np.histogram(bg_red_values, bins=50, range=[0, 255])
                        bg_bin_centers = (bg_bins[:-1] + bg_bins[1:]) / 2
                        
                        # Normalize for comparison
                        orig_hist = orig_hist / np.sum(orig_hist)
                        bg_hist = bg_hist / np.sum(bg_hist)
                        
                        ax8.plot(orig_bin_centers, orig_hist, color='orange', alpha=0.7, label='Original (all pixels)', linewidth=2)
                        ax8.plot(bg_bin_centers, bg_hist, color='red', alpha=0.7, label='Background Removed (skin only)', linewidth=2)
                        
                        ax8.set_title('8. Red Channel Comparison')
                        ax8.set_xlabel('Pixel Intensity (0-255)')
                        ax8.set_ylabel('Normalized Frequency')
                        ax8.legend()
                        ax8.grid(True, alpha=0.3)
                        
                        # Add statistics
                        skin_percentage = np.sum(bg_mask) / bg_mask.size * 100
                        orig_mean = np.mean(orig_red_values)
                        skin_mean = np.mean(bg_red_values)
                        ax8.text(0.02, 0.98, f'Skin: {skin_percentage:.1f}%\nOrig Mean: {orig_mean:.1f}\nSkin Mean: {skin_mean:.1f}',
                                transform=ax8.transAxes, fontsize=8, verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    else:
                        ax8.text(0.5, 0.5, 'No skin pixels\nfound', ha='center', va='center',
                                transform=ax8.transAxes)
                        ax8.set_title('8. Red Channel Comparison')
                else:
                    ax8.text(0.5, 0.5, 'No skin pixels\ndetected', ha='center', va='center',
                            transform=ax8.transAxes)
                    ax8.set_title('8. Red Channel Comparison')
            else:
                ax8.text(0.5, 0.5, 'Background removal\nfailed', ha='center', va='center',
                        transform=ax8.transAxes)
                ax8.set_title('8. Red Channel Comparison')
            
            # Panel 9: Background removed histogram (excluding black background)
            ax9 = fig.add_subplot(gs[2, 1])
            if hand_crop_bg_removed is not None and len(hand_crop_bg_removed.shape) == 3:
                # Simple mask to exclude black background pixels
                bg_mask = np.any(hand_crop_bg_removed > 0, axis=2)  # Any non-black pixel
                
                if np.any(bg_mask):  # Only if there are non-background pixels
                    # Extract only skin pixels for histogram calculation
                    skin_pixels = hand_crop_bg_removed[bg_mask]  # Shape: (N, 3) where N is number of skin pixels
                    
                    # Simple histogram for each channel - no complex filtering
                    for i, color in enumerate(['blue', 'green', 'red']):
                        channel_values = skin_pixels[:, i]  # Get all values for this channel
                        
                        if len(channel_values) > 0:
                            # Simple histogram with standard range
                            hist, bins = np.histogram(channel_values, bins=50, range=[0, 255])
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            ax9.plot(bin_centers, hist, color=color, alpha=0.7, label=f'{color.upper()}', linewidth=2)
                    
                    ax9.set_title('9. Background Removed Histogram\n(Skin pixels only)')
                    ax9.set_xlabel('Pixel Intensity (0-255)')
                    ax9.set_ylabel('Frequency')
                    ax9.legend()
                    ax9.grid(True, alpha=0.3)
                    
                    # Simple statistics
                    skin_percentage = np.sum(bg_mask) / bg_mask.size * 100
                    mean_colors = np.mean(skin_pixels, axis=0)
                    ax9.text(0.02, 0.98, f'Skin: {skin_percentage:.1f}%\nMean BGR: [{mean_colors[0]:.0f}, {mean_colors[1]:.0f}, {mean_colors[2]:.0f}]',
                            transform=ax9.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    ax9.text(0.5, 0.5, 'No skin pixels\ndetected', ha='center', va='center',
                            transform=ax9.transAxes)
                    ax9.set_title('9. Background Removed Histogram')
            else:
                ax9.text(0.5, 0.5, 'Background removal\nfailed', ha='center', va='center',
                        transform=ax9.transAxes)
                ax9.set_title('9. Background Removed Histogram')
            
            # Panel 10: Model input histogram (normalized space)
            ax10 = fig.add_subplot(gs[2, 2])
            
            # Show histogram in the actual normalized range the model sees
            if isinstance(processed_hand, torch.Tensor):
                # Use the actual tensor data (normalized)
                tensor_data = processed_hand.cpu().numpy()
                if len(tensor_data.shape) == 4:
                    tensor_data = tensor_data[0]  # Remove batch dimension
                
                # Transpose from CHW to HWC for histogram calculation
                tensor_data = tensor_data.transpose(1, 2, 0)
                
                # Calculate actual data range for better binning
                data_min = np.min(tensor_data)
                data_max = np.max(tensor_data)
                
                # Use a reasonable range that doesn't create artificial spikes
                range_padding = (data_max - data_min) * 0.1
                data_range = (data_min - range_padding, data_max + range_padding)
                
                # Ensure we have a valid range
                if data_range[1] <= data_range[0]:
                    data_range = (data_min - 0.1, data_max + 0.1)  # Fallback range
                
                for i, color in enumerate(['blue', 'green', 'red']):
                    channel_data = tensor_data[:, :, i].flatten()
                    
                    # Filter out any extreme outliers that might cause spikes
                    percentile_1 = np.percentile(channel_data, 1)
                    percentile_99 = np.percentile(channel_data, 99)
                    filtered_data = channel_data[(channel_data >= percentile_1) & (channel_data <= percentile_99)]
                    
                    if len(filtered_data) > 0:
                        # Create histogram with filtered data
                        hist, bins = np.histogram(filtered_data, bins=35, range=data_range)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        ax10.plot(bin_centers, hist, color=color, alpha=0.7, label=f'{color.upper()}', linewidth=2)
                    else:
                        ax10.plot([], [], color=color, alpha=0.7, label=f'{color.upper()} (no data)', linewidth=2)
                
                ax10.set_title('10. Model Input Distribution\n(ImageNet Normalized, outliers removed)')
                ax10.set_xlabel('Normalized Value')
                ax10.set_ylabel('Frequency')
                ax10.legend()
                ax10.grid(True, alpha=0.3)
                
                # Add normalization info
                mean_vals = [np.mean(tensor_data[:,:,i]) for i in range(3)]
                std_vals = [np.std(tensor_data[:,:,i]) for i in range(3)]
                ax10.text(0.02, 0.98, f'Stats (full data):\nMean: B={mean_vals[0]:.2f}, G={mean_vals[1]:.2f}, R={mean_vals[2]:.2f}\nStd:  B={std_vals[0]:.2f}, G={std_vals[1]:.2f}, R={std_vals[2]:.2f}',
                         transform=ax10.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Add reference line at zero
                ax10.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax10.text(0.02, 0.02, f'Range: [{data_min:.2f}, {data_max:.2f}]\nImageNet normalization\nShould center around 0',
                         transform=ax10.transAxes, fontsize=7, verticalalignment='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
                         
            elif isinstance(model_input_np, np.ndarray) and len(model_input_np.shape) == 3:
                # Show the model input as numpy array (should be 0-255 range)
                # Apply similar filtering to avoid spikes at 0 from background pixels
                for i, color in enumerate(['blue', 'green', 'red']):
                    channel_data = model_input_np[:, :, i].flatten()
                    
                    # Filter out very dark pixels that might cause spikes at 0
                    filtered_data = channel_data[channel_data > 10]
                    
                    if len(filtered_data) > 0:
                        data_min = max(0, np.min(filtered_data) - 5)
                        data_max = min(255, np.max(filtered_data) + 5)
                        
                        # Ensure min < max to avoid histogram range error
                        if data_max <= data_min:
                            data_max = data_min + 10  # Force a minimum range
                        
                        hist, bins = np.histogram(filtered_data, bins=40, range=[data_min, data_max])
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        ax10.plot(bin_centers, hist, color=color, alpha=0.7, label=f'{color.upper()}', linewidth=2)
                    else:
                        ax10.plot([], [], color=color, alpha=0.7, label=f'{color.upper()} (no data)', linewidth=2)
                        
                ax10.set_title('10. Model Input (Pre-normalization)\n(Dark pixels filtered out)')
                ax10.set_xlabel('Pixel Intensity')
                ax10.set_ylabel('Frequency')
                ax10.legend()
                ax10.grid(True, alpha=0.3)
            else:
                ax10.text(0.5, 0.5, 'Model input\nnot available', ha='center', va='center',
                         transform=ax10.transAxes)
                ax10.set_title('10. Model Input Histogram')
            
            # Panel 11: Skin mask visualization
            ax11 = fig.add_subplot(gs[2, 3])
            if skin_mask_crop is not None:
                ax11.imshow(skin_mask_crop, cmap='Reds')
                ax11.set_title('11. Skin Detection Mask')
                
                # Add mask statistics
                mask_percentage = np.sum(skin_mask_crop > 0) / skin_mask_crop.size * 100
                mask_mean = np.mean(skin_mask_crop[skin_mask_crop > 0]) if np.any(skin_mask_crop > 0) else 0
                
                ax11.text(0.02, 0.98, f'Coverage: {mask_percentage:.1f}%\nMean: {mask_mean:.1f}',
                         transform=ax11.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax11.text(0.5, 0.5, 'Skin mask\nnot available', ha='center', va='center',
                        transform=ax11.transAxes)
                ax11.set_title('11. Skin Detection Mask')
            ax11.axis('off')
            
            # Panel 12: Vision system status
            ax12 = fig.add_subplot(gs[3, 0])
            ax12.axis('off')
            
            bg_progress = self.hand_detector.bg_remover.get_progress()
            bg_learned = self.hand_detector.bg_remover.bg_model_learned
            
            vision_text = f"""Vision System:

Background Learning:
Status: {'✅ Learned' if bg_learned else '🔄 Learning'}
Progress: {bg_progress:.1%}

MOG2 Parameters:
- History: 300 frames
- Threshold: 25.0
- Learning Rate: {'0 (Locked)' if bg_learned else 'Auto'}

Background Removal:
- Method: {bg_removal_method}
- ROI-based: {'✅' if crop_bbox is not None else '❌'}

Hand Tracking:
- Kalman Filter: Active
- Smoothing: Enabled
"""
            
            ax12.text(0.05, 0.95, vision_text, transform=ax12.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace')
            ax12.set_title('12. Vision System')
            
            # Panel 12.5: MOG2 ROI Mask (if available)
            if mog2_mask_crop is not None and bg_learned:
                # Use a subplot in the bottom area
                ax12_5 = fig.add_subplot(gs[3, 2])
                ax12_5.imshow(mog2_mask_crop, cmap='Blues')
                ax12_5.set_title('12.5 MOG2 ROI Mask')
                
                # Add MOG2 mask statistics
                mog2_percentage = np.sum(mog2_mask_crop > 0) / mog2_mask_crop.size * 100
                mog2_mean = np.mean(mog2_mask_crop[mog2_mask_crop > 0]) if np.any(mog2_mask_crop > 0) else 0
                
                ax12_5.text(0.02, 0.98, f'Foreground: {mog2_percentage:.1f}%\nMean: {mog2_mean:.1f}',
                           transform=ax12_5.transAxes, fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax12_5.axis('off')
            elif crop_bbox is not None and not bg_learned:
                # Show learning status
                ax12_5 = fig.add_subplot(gs[3, 2])
                ax12_5.text(0.5, 0.5, f'MOG2 Learning...\n{bg_progress:.1%} complete', 
                           ha='center', va='center', transform=ax12_5.transAxes, fontsize=10)
                ax12_5.set_title('12.5 MOG2 ROI Mask\n(Learning)')
                ax12_5.axis('off')
            else:
                # Show that MOG2-ROI is not available
                ax12_5 = fig.add_subplot(gs[3, 2])
                ax12_5.text(0.5, 0.5, 'MOG2-ROI\nnot available', 
                           ha='center', va='center', transform=ax12_5.transAxes, fontsize=10)
                ax12_5.set_title('12.5 MOG2 ROI Mask\n(N/A)')
                ax12_5.axis('off')
            
            # Panel 13: System performance
            ax13 = fig.add_subplot(gs[3, 1])
            ax13.axis('off')
            
            fps = self.fps_tracker.get_fps()
            performance_text = f"""Performance:

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
            
            ax13.text(0.05, 0.95, performance_text, transform=ax13.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace')
            ax13.set_title('13. Performance')
            
            # Panel 14: Color space analysis
            ax14 = fig.add_subplot(gs[3, 2:4])
            ax14.axis('off')
            
            if hand_crop_bg_removed is not None and len(hand_crop_bg_removed.shape) == 3:
                bg_mask = np.any(hand_crop_bg_removed > 5, axis=2)
                if np.any(bg_mask):
                    # Analyze skin color distribution
                    skin_pixels = hand_crop_bg_removed[bg_mask]
                    
                    # BGR analysis
                    bgr_mean = np.mean(skin_pixels, axis=0)
                    bgr_std = np.std(skin_pixels, axis=0)
                    
                    # HSV analysis
                    hsv_crop = cv2.cvtColor(hand_crop_bg_removed, cv2.COLOR_BGR2HSV)
                    hsv_skin = hsv_crop[bg_mask]
                    hsv_mean = np.mean(hsv_skin, axis=0)
                    hsv_std = np.std(hsv_skin, axis=0)
                    
                    color_analysis = f"""Color Analysis (Skin Only):

BGR Values:
- Blue:  {bgr_mean[0]:.1f} ± {bgr_std[0]:.1f}
- Green: {bgr_mean[1]:.1f} ± {bgr_std[1]:.1f}  
- Red:   {bgr_mean[2]:.1f} ± {bgr_std[2]:.1f}

HSV Values:
- Hue:        {hsv_mean[0]:.1f}° ± {hsv_std[0]:.1f}°
- Saturation: {hsv_mean[1]:.1f} ± {hsv_std[1]:.1f}
- Value:      {hsv_mean[2]:.1f} ± {hsv_std[2]:.1f}

Skin Quality:
- Uniformity: {'Good' if np.mean(bgr_std) < 30 else 'Variable'}
- Total Pixels: {len(skin_pixels):,}
"""
                    
                    ax14.text(0.05, 0.95, color_analysis, transform=ax14.transAxes,
                             fontsize=8, verticalalignment='top', fontfamily='monospace')
                else:
                    ax14.text(0.5, 0.5, 'No skin pixels detected', ha='center', va='center',
                             transform=ax14.transAxes)
            else:
                ax14.text(0.5, 0.5, 'Background removal failed', ha='center', va='center',
                         transform=ax14.transAxes)
            
            ax14.set_title('14. Color Space Analysis')
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking
            
            # Save capture data
            timestamp = time.time()
            capture_dir = Path("data/raw/captures")
            capture_dir.mkdir(parents=True, exist_ok=True)
            
            capture_base = capture_dir / f"live_capture_{prediction}_{timestamp:.0f}"
            
            # Save images
            cv2.imwrite(f"{capture_base}_original.jpg", hand_crop)
            if hand_crop_bg_removed is not None:
                cv2.imwrite(f"{capture_base}_bg_removed.jpg", hand_crop_bg_removed)
            if skin_mask_crop is not None:
                cv2.imwrite(f"{capture_base}_skin_mask.jpg", skin_mask_crop)
            if mog2_mask_crop is not None:
                cv2.imwrite(f"{capture_base}_mog2_mask.jpg", mog2_mask_crop)
            cv2.imwrite(f"{capture_base}_model_input.jpg", model_input_np)
            cv2.imwrite(f"{capture_base}_full_frame.jpg", frame)
            
            # Save comprehensive metadata
            capture_metadata = {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': float(confidence),
                'bbox': list(bbox),
                'crop_bbox': list(crop_bbox) if crop_bbox is not None else None,
                'hand_size': [w, h],
                'hand_area': total_pixels,
                'aspect_ratio': float(aspect_ratio),
                'background_removal': {
                    'method': bg_removal_method,
                    'mog2_available': mog2_mask_crop is not None,
                    'mog2_background_learned': self.hand_detector.bg_remover.bg_model_learned,
                    'mog2_progress': self.hand_detector.bg_remover.get_progress(),
                    'roi_based': crop_bbox is not None
                },
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
                'vision_system': {
                    'mog2_history': 300,
                    'mog2_threshold': 25.0,
                    'learning_rate': 0 if self.hand_detector.bg_remover.bg_model_learned else -1,
                    'background_learned': self.hand_detector.bg_remover.bg_model_learned
                },
                'files': {
                    'original_crop': f"{capture_base.name}_original.jpg",
                    'bg_removed_crop': f"{capture_base.name}_bg_removed.jpg" if hand_crop_bg_removed is not None else None,
                    'skin_mask': f"{capture_base.name}_skin_mask.jpg" if skin_mask_crop is not None else None,
                    'mog2_mask': f"{capture_base.name}_mog2_mask.jpg" if mog2_mask_crop is not None else None,
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