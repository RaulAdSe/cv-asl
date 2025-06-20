#!/usr/bin/env python3
"""
ASL Training Visualization Demo

Demonstrates the beautiful training visualization tools using our trained model data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asl_dl.visualization.training_plots import TrainingVisualizer
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_visualization():
    """Demo the training visualization with sample data from our ABC model"""
    
    logger.info("🎨 ASL Training Visualization Demo")
    logger.info("=" * 50)
    
    # Sample data from our actual ABC training (10 epochs)
    # These are realistic values from the model we just trained
    sample_metrics = {
        'train_losses': [1.0788, 0.7958, 0.6915, 0.5836, 0.4830, 0.3805, 0.4495, 0.2789, 0.1785, 0.1102],
        'val_losses': [1.0971, 1.1888, 1.2760, 1.1494, 2.3696, 0.9540, 0.7609, 0.4752, 0.2759, 0.2443],
        'train_accs': [38.69, 62.80, 66.07, 71.13, 76.19, 84.23, 81.55, 90.18, 94.35, 96.13],
        'val_accs': [39.29, 39.29, 39.29, 39.29, 29.76, 54.76, 77.38, 82.14, 89.29, 92.86],
        'epoch_times': [113, 111, 112, 111, 112, 110, 108, 111, 109, 109]  # seconds per epoch
    }
    
    sample_model_info = {
        'model_type': 'MobileNetV2ASL',
        'total_params': 3011843,
        'num_classes': 3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dataset': 'Kaggle ASL A-B-C'
    }
    
    # Create visualizer
    visualizer = TrainingVisualizer()
    
    # Generate all visualizations
    logger.info("📊 Creating training visualizations...")
    report_paths = visualizer.create_training_report(sample_metrics, sample_model_info)
    
    logger.info("\n✅ Visualization demo completed!")
    logger.info("📁 Generated files:")
    for name, path in report_paths.items():
        logger.info(f"   {name}: {path}")
    
    # Also show individual plots
    logger.info("\n🎯 Individual plot demonstrations:")
    
    # 1. Just training curves
    curves_path = visualizer.plot_training_curves(
        sample_metrics['train_losses'],
        sample_metrics['val_losses'], 
        sample_metrics['train_accs'],
        sample_metrics['val_accs'],
        "ASL A-B-C Model Training Results"
    )
    
    # 2. Model comparison demo (comparing same model with different configurations)
    from asl_dl.visualization.training_plots import plot_model_comparison
    
    # Simulate comparison data
    comparison_data = [
        {
            'name': 'MobileNetV2 (Current)',
            'train_losses': sample_metrics['train_losses'],
            'val_losses': sample_metrics['val_losses'],
            'train_accs': sample_metrics['train_accs'],
            'val_accs': sample_metrics['val_accs']
        },
        {
            'name': 'MobileNetV2 Lite (Simulated)',
            'train_losses': [1.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2],
            'val_losses': [1.3, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.3],
            'train_accs': [35, 55, 60, 65, 70, 75, 80, 85, 88, 90],
            'val_accs': [30, 50, 58, 62, 68, 72, 76, 80, 85, 87]
        }
    ]
    
    comparison_path = plot_model_comparison(comparison_data)
    logger.info(f"   Model comparison: {comparison_path}")
    
    logger.info(f"\n🎉 All visualizations saved to: src/asl_dl/visualization/plots/")
    logger.info("💡 You can now use these tools with your own training data!")

if __name__ == "__main__":
    demo_visualization() 