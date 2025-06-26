# ASL Real-Time Recognition System

A complete real-time American Sign Language recognition system featuring advanced computer vision algorithms, MobileNetV2 deep learning, and comprehensive evaluation tools for accurate hand sign classification.

## Overview

This system provides end-to-end ASL recognition capabilities from dataset training to real-time inference. It combines custom computer vision algorithms with modern deep learning to achieve high accuracy and real-time performance.

### Key Features

- **Real-time Recognition**: 30+ FPS ASL letter classification (A, B, C)
- **Advanced Hand Detection**: Multi-factor scoring with MOG2 background subtraction
- **GPU Accelerated Training**: Automatic MPS (Apple Silicon) and CUDA support
- **Comprehensive Evaluation**: Professional-grade metrics and visualizations
- **Production Ready**: Optimized MobileNetV2 for deployment scenarios

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone <repository-url>
cd CV-asl
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model
```bash
# Download dataset and train on proper Kaggle ASL data
python -m src.asl_dl.training.train --mode kaggle_abc
```

### 3. Run Live Recognition
```bash
python run_live_asl.py
```

### 4. Generate Evaluation Reports
```bash
python -m src.asl_dl.scripts.comprehensive_evaluation
```

## 📊 System Architecture

### Computer Vision Pipeline
```
Raw Camera Feed → Hand Detection → Background Removal → Preprocessing → Neural Network
      ↓               ↓                   ↓               ↓              ↓
   640x480 RGB   Skin+Motion Filter   MOG2 Algorithm   224x224 Crop   MobileNetV2
```

### Deep Learning Pipeline
```
Kaggle Dataset → Data Augmentation → MobileNetV2 Training → Model Export → Real-time Inference
     ↓                ↓                     ↓                  ↓             ↓
210 ASL Images   Rotation+Color Jitter  Transfer Learning   .pth Format   30+ FPS
```

## 🔧 Core Components

### Computer Vision (`src/asl_cam/`)

#### Advanced Hand Detection
- **Multi-factor scoring**: Position, size, and shape analysis
- **MOG2 background subtraction**: Robust motion detection
- **Skin detection**: HSV and YCrCb color space filtering
- **Kalman filter tracking**: Smooth bounding box stabilization
- **Geometric validation**: Circularity, solidity, aspect ratio filtering

#### Key Algorithms
```python
# Hand detection pipeline
skin_mask = detect_skin_hsv_ycrcb(frame)
motion_mask = mog2_background_subtractor.apply(frame)
candidates = find_contours(skin_mask & motion_mask)
best_hand = score_candidates(candidates)
```

### Deep Learning (`src/asl_dl/`)

#### MobileNetV2 Architecture
- **Transfer learning**: ImageNet pretrained weights
- **Real-time optimization**: 30+ FPS inference capability
- **Compact model**: ~36MB for deployment efficiency
- **High accuracy**: 92.9% overall accuracy on evaluation set

#### Training Features
- **GPU acceleration**: Automatic MPS/CUDA detection
- **Data augmentation**: Rotation, color jitter, spatial transforms
- **Learning rate scheduling**: Adaptive optimization
- **Comprehensive logging**: TensorBoard integration

## 📈 Performance Metrics

### Model Performance
```
Overall Accuracy: 92.9%
Macro F1-Score: 92.7%
Mean Confidence: 84.3%

Per-Class Results:
  A: Precision=100%, Recall=84.6%, F1=91.7%
  B: Precision=85.7%, Recall=100%, F1=92.3%  
  C: Precision=94.1%, Recall=94.1%, F1=94.1%
```

### System Performance
| Metric | Value | Target |
|--------|-------|--------|
| Inference FPS | 30+ | 30 |
| Model Size | 36MB | <50MB |
| Accuracy | 92.9% | >90% |
| GPU Training Time | 10-15 min | <30 min |

## 🎮 Usage Controls

### Live Recognition Interface
- **Q**: Quit application
- **S**: Toggle performance statistics
- **R**: Reset hand tracking system
- **B**: Reset background learning model
- **SPACE**: Pause/resume processing
- **C**: Capture hand data for analysis
- **X/Z**: Adjust prediction smoothing

### Visual Display
- **Large colored letters**: Real-time predictions with confidence-based coloring
- **Confidence scores**: Numerical confidence values
- **Performance monitoring**: FPS and processing statistics
- **Smoothing indicator**: Prediction stability metrics

## 📊 Evaluation and Visualization

### Comprehensive Analysis
The system generates four types of professional visualizations:

#### 1. Confusion Matrix
- Prediction accuracy breakdown by class
- Percentage and count displays
- Color-coded performance indicators

#### 2. Performance Metrics (4-Panel Analysis)
- **Panel 1**: Precision/Recall/F1-Score comparison
- **Panel 2**: Overall confidence distribution
- **Panel 3**: Accuracy vs confidence threshold
- **Panel 4**: Class-wise confidence analysis

#### 3. Error Analysis
- Error rate breakdown by true class
- Confidence comparison for correct vs incorrect predictions

#### 4. Detailed Report (JSON)
- Machine-readable metrics export
- Complete statistical breakdown
- Programmatic analysis support

### Generate Evaluations
```bash
# Standard evaluation on test set
python -m src.asl_dl.scripts.comprehensive_evaluation

# Custom model/data evaluation
python -m src.asl_dl.scripts.comprehensive_evaluation --model path/to/model.pth --data path/to/data
```

## 🗂️ Project Structure

```
CV-asl/
├── src/
│   ├── asl_cam/                    # Computer Vision Module
│   │   ├── live_asl.py            # Real-time recognition system
│   │   ├── vision/                # CV algorithms
│   │   │   ├── asl_hand_detector.py      # Main detection pipeline
│   │   │   ├── background_removal.py     # MOG2 implementation
│   │   │   └── tracker.py         # Kalman filter tracking
│   │   └── utils/                 # Utility functions
│   └── asl_dl/                    # Deep Learning Module
│       ├── training/              # Model training pipeline
│       │   └── train.py          # Main training script
│   │   ├── scripts/               # Evaluation scripts
│   │   │   └── comprehensive_evaluation.py
│   │   ├── models/                # Model architectures
│   │   │   └── mobilenet.py      # MobileNetV2 implementation
│   │   └── visualization/         # Training monitoring
│   │       └── plots/            # Generated visualizations
│   ├── models/                        # Trained model storage
│   │   └── best_mobilenetv2_model.pth
│   ├── data/                         # Dataset storage
│   │   └── raw/kaggle_asl/train/    # Training images (A, B, C)
│   ├── tests/                        # Test suite
│   ├── documentation/                # Technical documentation
│   └── run_live_asl.py              # Main application entry point
```

## 🔧 Hardware Requirements

### Recommended
- **GPU**: Apple Silicon (M1/M2/M3) with MPS or NVIDIA GPU with CUDA
- **RAM**: 8GB+ for training, 4GB+ for inference
- **Storage**: 2GB for model and dataset
- **Camera**: USB or built-in camera for live recognition

### Minimum
- **CPU**: Multi-core processor (training will be slower)
- **RAM**: 4GB minimum
- **Storage**: 1GB for essential components

## 🛠️ Advanced Configuration

### Training Modes
```bash
# Kaggle ABC training (recommended)
python -m src.asl_dl.training.train --mode kaggle_abc

# Custom dataset training
python -m src.asl_dl.training.train --mode custom --data-dir path/to/data

# Model architecture comparison
python -m src.asl_dl.training.train --mode compare
```

### Custom Training Parameters
```bash
# Extended training with custom parameters
python -m src.asl_dl.training.train --mode kaggle_abc --epochs 50 --batch-size 64 --lr 0.002
```

### GPU Optimization
```bash
# Verify GPU acceleration
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"
```

## 🔍 Technical Deep Dive

### Computer Vision Innovations
- **Adaptive background learning**: MOG2 with 500-frame history
- **Multi-modal detection**: Skin detection combined with motion analysis
- **Geometric filtering**: Advanced shape validation algorithms
- **Temporal tracking**: Kalman filter for stable hand tracking

### Deep Learning Optimizations
- **Transfer learning**: Leverages ImageNet pretrained features
- **Data augmentation**: Aggressive augmentation for robustness
- **Architecture choice**: MobileNetV2 for efficiency-accuracy balance
- **Real-time inference**: Optimized for deployment scenarios

### Performance Engineering
- **Memory optimization**: Efficient data loading and processing
- **CPU/GPU utilization**: Automatic device selection and optimization
- **Real-time constraints**: Sub-33ms inference for 30+ FPS

## 🚀 Deployment Considerations

### Model Export
The trained model (`best_mobilenetv2_model.pth`) is ready for:
- **Mobile deployment**: Core ML (iOS) or TensorFlow Lite conversion
- **Edge devices**: ONNX format export capability
- **Cloud services**: Direct PyTorch model serving

### Integration Points
```python
# Load trained model for integration
from src.asl_dl.models.mobilenet import MobileNetV2ASL
model = MobileNetV2ASL.load_from_checkpoint('models/best_mobilenetv2_model.pth')
prediction, confidence = model.predict(hand_image)
```

## 📝 Development Workflow

### Complete Development Cycle
```bash
# 1. Environment setup
source venv/bin/activate

# 2. Model training
python -m src.asl_dl.training.train --mode kaggle_abc

# 3. Evaluation generation
python -m src.asl_dl.scripts.comprehensive_evaluation

# 4. Live testing
python run_live_asl.py

# 5. Results analysis
open src/asl_dl/visualization/plots/performance_metrics.png
```

This system provides a complete, production-ready ASL recognition solution with professional evaluation capabilities and real-time performance suitable for practical applications.
