# ASL Computer Vision & Deep Learning System

A complete real-time American Sign Language recognition system built from scratch to learn computer vision fundamentals, deep learning training, and efficient real-time inference pipelines.

## 🎯 Project Vision & Learning Goals

This project was designed as a comprehensive learning journey covering:

- **Computer Vision from Scratch**: Understanding OpenCV fundamentals, image processing, and hand detection without relying on pre-built solutions like MediaPipe
- **MOG2 Background Subtraction**: Deep dive into theory and practical implementation for robust background removal
- **Deep Learning Pipeline**: End-to-end model training, from dataset preparation to real-time inference
- **Real-time Performance**: Building efficient pipelines that work at 30+ FPS
- **Data Pipeline Optimization**: Making live camera data match training data characteristics

## 🚀 System Architecture

### Computer Vision Pipeline (`src/asl_cam/`)
```
Camera Input → Hand Detection → Background Removal → Preprocessing → Model Input
    ↓              ↓                    ↓                ↓             ↓
 OpenCV      Skin + Motion         MOG2 Algorithm    Crop & Resize   224x224 RGB
```

### Deep Learning Pipeline (`src/asl_dl/`)
```
Kaggle Dataset → Data Preprocessing → MobileNetV2 Training → Model Export → Live Inference
     ↓                ↓                     ↓                  ↓             ↓
ASL Images      Augmentation         Transfer Learning     .pth Format    Real-time
```

## 🔬 Technical Achievements

### Advanced Hand Detection
- **Multi-factor scoring system** combining position, size, and shape analysis
- **MOG2 background subtraction** for robust motion detection
- **Skin detection** using HSV and YCrCb color spaces
- **Kalman filter tracking** for smooth, stable bounding boxes
- **Geometric validation** with circularity, solidity, and aspect ratio filtering

### Deep Learning Training
- **MobileNetV2 architecture** optimized for real-time inference
- **Transfer learning** with ImageNet pretrained weights
- **Data augmentation** pipeline matching live camera characteristics
- **Learning rate scheduling** with ReduceLROnPlateau
- **Comprehensive metrics** tracking with visualization

### Real-time Inference
- **30+ FPS performance** with confidence scoring
- **Entropy calculation** for prediction uncertainty quantification
- **Live workflow visualization** showing complete processing pipeline
- **Non-blocking data capture** for continuous training data collection

## 🔧 Key Technical Learnings

### Computer Vision Fundamentals
- **OpenCV mastery**: Image processing, contour analysis, morphological operations
- **Color space theory**: RGB, HSV, YCrCb for robust skin detection
- **Background subtraction**: MOG2 theory and practical implementation
- **Geometric analysis**: Shape descriptors, convexity defects, solidity

### Deep Learning Pipeline
- **Dataset preparation**: Kaggle ASL dataset integration and preprocessing
- **Model architecture**: MobileNetV2 for efficient mobile-ready inference
- **Training optimization**: Batch normalization, dropout, data augmentation
- **Performance monitoring**: Loss curves, accuracy metrics, learning rate scheduling

### Real-time Systems
- **FPS optimization**: Efficient memory management and processing pipelines
- **Quality control**: Entropy-based confidence assessment
- **Data consistency**: Making live camera match training data characteristics
- **System integration**: Seamless camera → vision → DL → UI pipeline

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <repo-url>
cd CV-asl
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Live ASL Recognition
```bash
python run_live_asl.py
```

### 3. Collect Training Data
```bash
# Data collection is built into the live system - use SPACE key during live recognition
python run_live_asl.py  # Press SPACE to capture training data
```

### 4. Train Custom Model
```bash
python -m src.asl_dl.scripts.train_abc --epochs 25
```

## 🎮 System Controls

### Live Recognition
- **Space** - Toggle capture mode for workflow analysis
- **Q** - Quit application
- **R** - Reset hand tracker

### Data Collection
- **S** - Save current hand detection
- **L** - Change ASL letter label (A/B/C)
- **C** - Capture and visualize data pipeline
- **B** - Toggle background removal
- **Q** - Quit

## 📊 Performance Metrics

### Real-time Performance
- **30+ FPS** on modern hardware
- **< 33ms** inference latency
- **95%+ accuracy** on clear hand gestures
- **Real-time entropy calculation** for confidence assessment

### Model Specifications
| Component | Specification |
|-----------|---------------|
| Architecture | MobileNetV2 |
| Input Size | 224×224×3 |
| Parameters | ~2.3M |
| Model Size | ~9MB |
| Classes | A, B, C (expandable) |

## 🔍 Technical Deep Dives

### Background Removal (MOG2)
- **Adaptive learning** with configurable history (500 frames)
- **Shadow detection disabled** for performance
- **Variance threshold tuning** for motion sensitivity
- **Learning rate optimization** for stability

### Hand Detection Pipeline
```python
# Multi-stage detection process
skin_mask = detect_skin_hsv_ycrcb(frame)
motion_mask = mog2_background_subtractor.apply(frame)
hand_candidates = find_contours(skin_mask & motion_mask)
best_hand = score_candidates(hand_candidates)  # Position + Size + Shape
```

### Entropy-based Confidence
```python
# Shannon entropy for prediction uncertainty
probabilities = softmax(model_output)
entropy = -sum(p * log(p) for p in probabilities)
# Low entropy = high confidence, High entropy = uncertain
```

## 📁 Project Structure

```
CV-asl/
├── src/
│   ├── asl_cam/              # Computer Vision Module
│   │   ├── live_asl.py       # Real-time recognition system
│   │   ├── vision/           # CV algorithms
│   │   │   ├── asl_hand_detector.py    # Main detection pipeline
│   │   │   ├── background_removal.py   # MOG2 implementation
│   │   │   ├── enhanced_hand_detector.py  # Advanced filtering
│   │   │   └── tracker.py     # Kalman filter tracking
│   │   └── utils/
│   └── asl_dl/               # Deep Learning Module
│       ├── training/         # Model training pipeline
│       ├── data/            # Dataset handling
│       ├── scripts/         # Training scripts
│       └── visualization/   # Training monitoring
├── tests/                   # Comprehensive test suite
├── documentation/           # Technical documentation
└── run_live_asl.py         # Main application entry (includes data capture)
```

## 🎯 Key Insights & Learnings

### Computer Vision
- **Motion + Color is powerful**: Combining skin detection with background subtraction solved the "torso detection" problem elegantly
- **Geometric validation matters**: Simple area and aspect ratio filtering eliminates most false positives
- **Kalman filters are magic**: Smooth tracking dramatically improves user experience

### Deep Learning
- **Data consistency is crucial**: Making live camera data match training data characteristics was the key breakthrough
- **Transfer learning works**: MobileNetV2 pretrained weights provided excellent starting point
- **Real-time optimization**: Model architecture choice (MobileNetV2) enabled 30+ FPS performance

### System Design
- **Modular architecture pays off**: Clear separation between vision and DL modules enabled independent development
- **Quality metrics are essential**: Entropy calculation provides valuable confidence assessment
- **Non-blocking UI**: Keeping main loop responsive while capturing data improved usability

## 🚀 Future Enhancements

- **Expand alphabet**: Add more ASL letters beyond A, B, C
- **Mobile deployment**: Optimize for iOS/Android using Core ML/TensorFlow Lite
- **Real-time training**: Online learning from user corrections
- **3D hand pose**: Integrate depth information for better accuracy

---

**Built with ❤️ to learn computer vision and deep learning fundamentals**
