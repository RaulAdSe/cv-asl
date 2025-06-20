# ASL Deep Learning Training Guide

## 🚀 Quick Start - Train Your ASL Classifier

This guide walks you through training lightweight ASL classification models for real-time hand sign recognition.

## 📋 Prerequisites

### 1. Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Install deep learning dependencies
pip install -r requirements.txt

# For Kaggle dataset download (optional)
pip install kaggle
```

### 2. Kaggle Setup (Optional - for automatic download)
```bash
# 1. Go to https://www.kaggle.com/account
# 2. Create API token (downloads kaggle.json)
# 3. Place in ~/.kaggle/kaggle.json
# 4. Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

## 🗂️ Dataset Setup

### Option A: Automatic Download (Recommended)
```bash
# Setup ASL dataset automatically
python -m src.asl_cam.utils.dataset_setup

# View dataset information
python -m src.asl_cam.utils.dataset_setup --info
```

### Option B: Manual Download
1. Go to: https://www.kaggle.com/datasets/ayuraj/asl-dataset
2. Download dataset ZIP
3. Extract to: `data/raw/asl_dataset/`
4. Run setup script: `python -m src.asl_cam.utils.dataset_setup`

### Expected Directory Structure
```
data/raw/asl_dataset/
├── Train/
│   ├── A/
│   ├── B/
│   └── ... (26 letters + numbers)
├── Test/
└── unified/
    ├── train_images/
    └── test_images/
```

## 🧠 Model Training

### 1. Compare All Models
Train and compare all model architectures:

```bash
# Train all models (EfficientNet+LSTM, MediaPipe, MobileNet)
python -m src.asl_cam.train
```

### 2. Train Specific Model
```bash
# EfficientNet + LSTM (Best accuracy, medium speed)
python -c "
from src.asl_cam.train import ASLTrainer
config = {'batch_size': 32, 'num_epochs': 30, 'learning_rate': 0.001}
trainer = ASLTrainer(config)
result = trainer.train('data/raw/asl_dataset/unified/train_images', 'efficientnet_lstm')
print(f'Best accuracy: {result[\"best_accuracy\"]:.2f}%')
"

# MediaPipe Features (Fastest inference)
python -c "
from src.asl_cam.train import ASLTrainer
config = {'batch_size': 64, 'num_epochs': 20, 'learning_rate': 0.002}
trainer = ASLTrainer(config)
result = trainer.train('data/raw/asl_dataset/unified/train_images', 'mediapipe')
print(f'Best accuracy: {result[\"best_accuracy\"]:.2f}%')
"

# MobileNet (Good balance)
python -c "
from src.asl_cam.train import ASLTrainer
config = {'batch_size': 32, 'num_epochs': 25, 'learning_rate': 0.001}
trainer = ASLTrainer(config)
result = trainer.train('data/raw/asl_dataset/unified/train_images', 'mobilenet')
print(f'Best accuracy: {result[\"best_accuracy\"]:.2f}%')
"
```

## 🔄 Real-time Inference

### 1. Test Trained Model
```bash
# Use EfficientNet+LSTM model
python -m src.asl_cam.infer --model models/best_efficientnet_lstm_model.pth --type efficientnet_lstm

# Use MediaPipe model (fastest)
python -m src.asl_cam.infer --model models/best_mediapipe_model.pth --type mediapipe

# Use MobileNet model
python -m src.asl_cam.infer --model models/best_mobilenet_model.pth --type mobilenet
```

### 2. Real-time Controls
- **SPACE**: Toggle debug info
- **S**: Toggle statistics display
- **Q/ESC**: Quit application

## 📊 Model Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|---------|----------|
| **MediaPipe** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time apps |
| **MobileNet** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Mobile/Edge |
| **EfficientNet+LSTM** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | High accuracy |

## ⚙️ Configuration Options

### Training Configuration
```python
config = {
    'batch_size': 32,           # Batch size for training
    'learning_rate': 0.001,     # Learning rate
    'num_epochs': 30,           # Number of training epochs
    'weight_decay': 1e-4,       # L2 regularization
    'scheduler_step': 10,       # LR scheduler step
    'sequence_length': 16       # For LSTM models
}
```

### Model Parameters
```python
# EfficientNet+LSTM
EfficientNetLSTM(
    num_classes=26,         # Number of ASL classes
    sequence_length=16,     # Input sequence length
    hidden_size=128,        # LSTM hidden size
    num_layers=2           # LSTM layers
)

# MediaPipe Classifier
MediaPipeClassifier(
    num_classes=26,         # Number of classes
    input_dim=84           # MediaPipe features (21*2*2)
)
```

## 🚀 Performance Optimization Tips

### 1. For Maximum Speed (Real-time)
- Use **MediaPipe** model
- Lower image resolution (224x224 → 128x128)
- Reduce confidence threshold (0.7 → 0.5)
- Use CPU inference for lightweight models

### 2. For Maximum Accuracy
- Use **EfficientNet+LSTM** model
- Higher image resolution (224x224 → 256x256)
- More training epochs (30 → 50+)
- Ensemble multiple models

### 3. For Mobile/Edge Deployment
- Use **MobileNet** model
- Quantize model weights (FP32 → INT8)
- Use TensorRT/ONNX optimization
- Batch size = 1 for inference

## 📈 Expected Results

### Dataset Statistics
- **Classes**: 26 (A-Z) + 10 (0-9) = 36 total
- **Training samples**: ~87,000 images
- **Test samples**: ~29,000 images
- **Image size**: 200x200 pixels (resized to 224x224)

### Performance Benchmarks
| Model | Accuracy | Inference Time | FPS |
|-------|----------|----------------|-----|
| MediaPipe | 85-90% | 2-5ms | 200+ |
| MobileNet | 90-95% | 8-15ms | 60-120 |
| EfficientNet+LSTM | 95-98% | 20-40ms | 25-50 |

## 🔧 Troubleshooting

### Common Issues

#### 1. "Command not found: python"
```bash
# Activate virtual environment
source venv/bin/activate
```

#### 2. GPU/CUDA Issues
```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# For CPU-only inference, models will automatically use CPU
```

#### 3. Low Accuracy
- **Check dataset**: Ensure proper directory structure
- **Increase epochs**: Try 50+ epochs for better convergence
- **Data augmentation**: Enable rotation, flip, color jitter
- **Learning rate**: Try 0.0005 for more stable training

#### 4. Slow Inference
- **Use MediaPipe**: Fastest model for real-time use
- **Reduce resolution**: 224→128 pixels
- **Batch size 1**: For real-time inference
- **CPU inference**: For lightweight models

### Memory Issues
```bash
# Reduce batch size
config['batch_size'] = 16  # Instead of 32

# Use gradient accumulation
# Effective batch size = batch_size * accumulation_steps
```

## 🎯 Integration with Existing System

### Use with Hand Detection
```python
from src.asl_cam.infer import RealTimeASL
from src.asl_cam.vision.simple_hand_detector import SimpleHandDetector

# Initialize ASL system with your trained model
asl_system = RealTimeASL('models/best_efficientnet_lstm_model.pth', 'efficientnet_lstm')

# Run with hand detection
asl_system.run(camera_id=0)
```

### Custom Integration
```python
from src.asl_cam.infer import ASLInferenceEngine

# Initialize inference engine
engine = ASLInferenceEngine('models/best_mobilenet_model.pth', 'mobilenet')

# Predict on hand crop
prediction = engine.predict(hand_crop_image)
if prediction:
    print(f"Sign: {prediction['class']} (confidence: {prediction['confidence']:.2f})")
```

## 📁 File Organization

```
CV-asl/
├── models/                     # Trained models
│   ├── best_efficientnet_lstm_model.pth
│   ├── best_mediapipe_model.pth
│   └── best_mobilenet_model.pth
├── data/raw/asl_dataset/       # Training data
├── src/asl_cam/
│   ├── train.py               # Training script
│   ├── infer.py               # Inference script
│   └── utils/dataset_setup.py # Dataset management
└── model_comparison_results.json # Training results
```

## 🎯 Next Steps

1. **Train your first model**: Start with MediaPipe for quick results
2. **Test real-time inference**: Use the inference script
3. **Optimize for your use case**: Adjust parameters for speed vs accuracy
4. **Integrate with your app**: Use the inference engine in your application
5. **Collect more data**: Use the existing data collection system to improve models

## 📚 Additional Resources

- **Hand Detection Guide**: See `SIMPLE_HAND_DETECTION.md`
- **Data Collection**: See `CAPTURE_FEATURE_GUIDE.md`
- **Full Documentation**: See `ASL_PROJECT_DOCUMENTATION.md`
- **Kaggle Dataset**: https://www.kaggle.com/datasets/ayuraj/asl-dataset

---

¡Happy training! 🚀 Your ASL classification system is ready to learn from your hand signs. 