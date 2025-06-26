# ASL Training Guide - MobileNetV2 Real-Time Recognition

## Overview

This guide walks you through training MobileNetV2 models for real-time ASL hand sign recognition using the proper Kaggle ASL dataset, generating comprehensive evaluations, and running live inference.

## 🚀 Quick Start

### 1. Download Dataset and Train Model
```bash
# Train on proper Kaggle ASL dataset (A, B, C letters)
python -m src.asl_dl.training.train --mode kaggle_abc
```

This single command will:
- Automatically download the correct Kaggle ASL dataset
- Train MobileNetV2 optimized for real-time performance
- Save the best model to `models/best_mobilenetv2_model.pth`
- Use GPU acceleration (Apple Silicon MPS or CUDA)

### 2. Run Live Recognition
```bash
python run_live_asl.py
```

### 3. Generate Comprehensive Evaluation
```bash
python -m src.asl_dl.scripts.comprehensive_evaluation
```

## 📊 Training Modes

The training script supports multiple modes:

### Kaggle ABC Mode (Recommended)
```bash
python -m src.asl_dl.training.train --mode kaggle_abc
```
- Downloads and trains on proper Kaggle ASL dataset
- Focuses on A, B, C letters with 70 images each
- Optimized for high accuracy and real-time performance

### Custom Data Mode
```bash
python -m src.asl_dl.training.train --mode custom --data-dir path/to/your/data
```
- Train on your own collected data
- Data should be organized as: `data_dir/class_name/images.jpg`

### Model Comparison Mode
```bash
python -m src.asl_dl.training.train --mode compare
```
- Trains and compares multiple architectures
- Generates performance benchmarks

## ⚙️ Training Configuration

### Default Configuration (Optimized)
The training uses these optimized settings:

```python
{
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 30,           # Increased for better convergence
    'input_size': 224,
    'device': 'mps',            # Apple Silicon GPU support
    'data_augmentation': True   # Aggressive augmentation enabled
}
```

### Data Augmentation Pipeline
Enhanced augmentation for better A vs C distinction:
- **Random Crop**: Improves spatial invariance
- **Random Horizontal Flip**: 50% probability
- **Random Rotation**: ±20 degrees
- **Color Jitter**: Brightness, contrast, saturation variation
- **Normalization**: ImageNet statistics

### Custom Configuration
You can override default settings:

```bash
python -m src.asl_dl.training.train --mode kaggle_abc --epochs 50 --batch-size 64 --lr 0.002
```

## 📈 Evaluation and Visualization

### Comprehensive Evaluation
Generate detailed analysis of your trained model:

```bash
python -m src.asl_dl.scripts.comprehensive_evaluation
```

This creates four types of visualizations in `src/asl_dl/visualization/plots/`:

#### 1. Confusion Matrix (`confusion_matrix.png`)
- Shows exact prediction vs true label relationships
- Includes both counts and percentages
- Identifies which letters are confused for others

#### 2. Performance Metrics (`performance_metrics.png`)
Four-panel analysis:
- **Precision/Recall/F1-Score**: By class comparison
- **Confidence Distribution**: Overall prediction confidence
- **Accuracy vs Threshold**: How accuracy changes with confidence
- **Class-wise Confidence**: Confidence distribution per letter

#### 3. Error Analysis (`error_analysis.png`)
Two-panel error breakdown:
- **Error Rate by Class**: Which letters are hardest to predict
- **Confidence Comparison**: Correct vs incorrect prediction confidence

#### 4. Detailed Report (`evaluation_report.json`)
Machine-readable metrics including:
- Overall accuracy, precision, recall, F1-scores
- Per-class performance statistics
- Confidence distribution statistics

### Custom Evaluation
Evaluate specific model or data:

```bash
python -m src.asl_dl.scripts.comprehensive_evaluation --model path/to/model.pth --data path/to/test/data
```

## 🔧 Hardware Optimization

### Apple Silicon (M1/M2/M3)
The training automatically detects and uses MPS (Metal Performance Shaders):
- **Significantly faster training** (3-5x speedup vs CPU)
- **Automatic device selection** in training script
- **Memory optimization** for Apple Silicon

### NVIDIA GPUs
CUDA support is automatically detected:
```bash
# Training will use CUDA if available
python -m src.asl_dl.training.train --mode kaggle_abc
```

### CPU Fallback
If no GPU is available, training falls back to CPU:
- **Longer training times** but still functional
- **Same accuracy results**

## 📁 File Organization

### Training Output Structure
```
models/
└── best_mobilenetv2_model.pth    # Best trained model

src/asl_dl/visualization/plots/
├── confusion_matrix.png          # Confusion matrix
├── performance_metrics.png       # 4-panel performance analysis
├── error_analysis.png           # Error breakdown
└── evaluation_report.json       # Detailed metrics

logs/
└── asl_training_YYYYMMDD_HHMMSS/ # TensorBoard logs

data/raw/kaggle_asl/
└── train/
    ├── A/                        # 70 ASL 'A' images
    ├── B/                        # 70 ASL 'B' images
    └── C/                        # 70 ASL 'C' images
```

## 🎯 Expected Results

### Training Performance
With the optimized configuration:
- **Training Time**: ~10-15 minutes on Apple Silicon M3
- **Final Accuracy**: 95-100% validation accuracy
- **Model Size**: ~36MB
- **Inference Speed**: 30+ FPS real-time

### Recent Training Results
```
Overall Accuracy: 92.9%
Macro F1-Score: 92.7%
Mean Confidence: 84.3%

Per-Class Performance:
  A: Precision=100%, Recall=84.6%, F1=91.7%
  B: Precision=85.7%, Recall=100%, F1=92.3%  
  C: Precision=94.1%, Recall=94.1%, F1=94.1%
```

Key improvement: **Perfect precision for letter 'A'** - when the model predicts 'A', it's never wrong.

## 🔄 Live Recognition Usage

### Basic Usage
```bash
python run_live_asl.py
```

### Controls
- **Q**: Quit application
- **S**: Toggle statistics display
- **R**: Reset hand tracker
- **B**: Reset background learning
- **SPACE**: Pause/unpause
- **C**: Capture hand data for analysis
- **X/Z**: Adjust prediction smoothing

### Display Features
- **Large colored letters**: A, B, C with confidence-based colors
- **Confidence scores**: Numerical confidence display
- **FPS monitoring**: Real-time performance stats
- **Smoothing control**: Adjustable prediction stability

## 🛠️ Troubleshooting

### Common Issues

#### Training Speed Issues
```bash
# Verify GPU usage
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Use smaller batch size if memory issues
python -m src.asl_dl.training.train --mode kaggle_abc --batch-size 16
```

#### Dataset Download Issues
```bash
# Manual dataset setup
mkdir -p data/raw/kaggle_asl/train
# Place A, B, C folders with images in the train directory
```

#### Model Loading Issues
```bash
# Check model file exists
ls -la models/best_mobilenetv2_model.pth

# Retrain if corrupted
python -m src.asl_dl.training.train --mode kaggle_abc
```

## 📝 Complete Workflow Example

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train model on Kaggle data
python -m src.asl_dl.training.train --mode kaggle_abc

# 3. Generate comprehensive evaluation
python -m src.asl_dl.scripts.comprehensive_evaluation

# 4. Run live recognition
python run_live_asl.py

# 5. View results
open src/asl_dl/visualization/plots/confusion_matrix.png
open src/asl_dl/visualization/plots/performance_metrics.png
```

This workflow provides a complete end-to-end ASL recognition system with professional-grade evaluation and real-time performance. 