# ASL Hand Detection - Build It Yourself Approach

A classical computer vision approach to American Sign Language (ASL) hand detection and tracking, built from scratch using OpenCV and PyTorch.

## 🎯 Project Philosophy

This project takes a "build-it-yourself" approach to hand detection, avoiding ready-made solutions like MediaPipe. Instead, we build our own pipeline using classical computer vision techniques and custom neural networks.

## 🗺️ Development Roadmap

### Stage A: Skin-mask prototype (Current)
- ✅ Classical skin detection using HSV/YCrCb color spaces
- ✅ Contour-based hand segmentation
- ✅ Real-time hand bounding box detection
- ✅ Kalman filter-based tracking

### Stage B: Tracking loop (Next)
- Improved hand tracking across frames
- Temporal stability and motion prediction
- Multi-hand tracking support

### Stage C: Heuristic robustness
- Adaptive lighting compensation
- Background subtraction (MOG2)
- Motion-based fallback detection

### Stage D: Dataset builder
- ✅ Interactive data collection tool
- ✅ Auto-cropping and labeling
- ✅ Advanced background removal integration
- ✅ YOLO format export

### Stage E: Train custom detector
- Tiny-YOLO or custom CNN training
- EgoHands + Oxford Hands datasets
- Real-time CPU inference

### Stage F: Integration
- Replace heuristic detection with trained model
- End-to-end pipeline optimization

### Stage G: Sign classifier
- ResNet-based sign classification
- Temporal sequence modeling

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Stage A: Try the Skin Detection

```bash
# Interactive data collection with skin detection
python -m asl_cam.collect --show-mask

# Controls:
# S - Save hand detection
# M - Toggle mask view
# B - Toggle background removal  
# T - Tune thresholds
# Q - Quit
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_skin.py
pytest tests/test_tracker.py
```

## 📁 Project Structure

```
src/asl_cam/
├── vision/                 # Core computer vision modules
│   ├── skin.py            # Skin detection and segmentation
│   ├── tracker.py         # Kalman filter-based tracking
│   ├── background_removal.py # Advanced background removal methods
│   └── detector/          # Future: Custom hand detector models
├── collect.py             # Interactive data collection CLI
├── preprocess.py          # Image preprocessing utilities
└── ...                    # Future: train_detector.py, infer.py

tests/                     # Comprehensive test suite
├── test_skin.py          # Skin detection tests
├── test_tracker.py       # Tracking algorithm tests
└── test_preprocess.py    # Preprocessing tests
```

## 🔧 Current Features

### Skin Detection (`vision/skin.py`)
- HSV and YCrCb color space thresholding
- Morphological operations for noise reduction
- Contour-based hand detection
- Interactive threshold tuning
- Visualization with mask overlays

### Hand Tracking (`vision/tracker.py`)
- Kalman filter-based position prediction
- Centroid matching for track association
- Track stability scoring (hit counts)
- Multi-hand tracking support
- Temporal track persistence

### Data Collection (`collect.py`)
- Real-time hand detection and tracking
- Interactive sample collection
- Automatic cropping and metadata saving
- Advanced background removal (GrabCut, contour-based, skin masks)
- Dual-version saving (original + background-removed)
- YOLO format export for training
- Progress tracking and statistics

### Background Removal (`vision/background_removal.py`)
- Multiple removal methods: GrabCut, contour masks, skin detection, MOG2, watershed
- Configurable algorithms with quality vs. speed trade-offs
- Transparent background generation
- Visualization tools for method comparison
- Integration with data collection pipeline

## 🎛️ Configuration

### Skin Detection Thresholds
```python
# HSV thresholds (adjust for different lighting)
hsv_lower = [0, 30, 60]
hsv_upper = [20, 150, 255]

# YCrCb thresholds (more robust)
ycrcb_lower = [0, 133, 77] 
ycrcb_upper = [255, 173, 127]
```

### Tracking Parameters
```python
# Kalman filter tracking
max_disappeared = 30      # Frames to keep tracks without detection
distance_threshold = 100  # Max pixels for track association
```

## 📊 Performance

Current performance on standard hardware:
- **Frame Rate**: 30+ FPS (640x480)
- **Detection Latency**: <10ms per frame
- **Memory Usage**: <100MB
- **CPU Usage**: <20% (single core)

## 🧪 Development Workflow

1. **Stage A** (Current): Perfect skin detection + tracking
2. **Stage B**: Add robustness (lighting, backgrounds)  
3. **Stage C**: Collect training data (500+ samples/label)
4. **Stage D**: Train custom detector (Tiny-YOLO)
5. **Stage E**: Integrate trained model
6. **Stage F**: Add sign classification

## 🤝 Contributing

This is a learning-focused project. Key principles:

- **Build from scratch** - Understand every component
- **Test everything** - Comprehensive unit tests
- **Document thoroughly** - Clear code and comments
- **Incremental progress** - Small, verifiable steps

## 📚 Learning Resources

- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [Kalman Filter Explained](https://www.kalmanfilter.net/)
- [EgoHands Dataset](http://vision.soic.indiana.edu/projects/egohands/)
- [YOLO Object Detection](https://github.com/ultralytics/yolov5)

## 🔮 Future Enhancements

- Background subtraction for motion detection
- Gesture sequence recognition
- Real-time ASL translation
- Mobile deployment optimization
- Custom CNN architectures
