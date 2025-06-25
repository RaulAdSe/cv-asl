"""
Training Visualization Tools

Beautiful plots for monitoring ASL model training progress, metrics evolution,
and performance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
import datetime

# Set beautiful plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Beautiful training visualization for ASL models
    
    Features:
    - Loss and accuracy curves
    - Learning rate scheduling
    - Training time analysis
    - Model comparison plots
    - Confusion matrices
    """
    
    def __init__(self, save_dir: str = "src/asl_dl/visualization/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up beautiful plotting parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def plot_training_curves(self, 
                           train_losses: List[float], 
                           val_losses: List[float],
                           train_accs: List[float], 
                           val_accs: List[float],
                           title: str = "ASL Model Training Progress") -> str:
        """
        Create beautiful training curves showing loss and accuracy evolution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'o-', label='Training Loss', 
                linewidth=2.5, markersize=6, alpha=0.8)
        ax1.plot(epochs, val_losses, 's-', label='Validation Loss', 
                linewidth=2.5, markersize=6, alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('📉 Loss Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Accuracy curves
        ax2.plot(epochs, train_accs, 'o-', label='Training Accuracy', 
                linewidth=2.5, markersize=6, alpha=0.8)
        ax2.plot(epochs, val_accs, 's-', label='Validation Accuracy', 
                linewidth=2.5, markersize=6, alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('📈 Accuracy Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"training_curves_{timestamp}.png"
        plt.savefig(save_path)
        plt.show()
        
        logger.info(f"📊 Training curves saved to: {save_path}")
        return str(save_path)
    
    def plot_training_summary(self, 
                            metrics: Dict,
                            model_info: Dict) -> str:
        """
        Create comprehensive training summary with multiple subplots
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(metrics['train_losses']) + 1)
        ax1.plot(epochs, metrics['train_losses'], 'o-', label='Train', linewidth=2)
        ax1.plot(epochs, metrics['val_losses'], 's-', label='Validation', linewidth=2)
        ax1.set_title('📉 Loss Evolution', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(epochs, metrics['train_accs'], 'o-', label='Train', linewidth=2)
        ax2.plot(epochs, metrics['val_accs'], 's-', label='Validation', linewidth=2)
        ax2.set_title('📈 Accuracy Evolution', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 3. Model info
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        info_text = f"""
        🧠 Model Information
        
        Architecture: {model_info.get('model_type', 'MobileNetV2')}
        Parameters: {model_info.get('total_params', 'N/A'):,}
        Classes: {model_info.get('num_classes', 'N/A')}
        
        📊 Training Details
        
        Epochs: {len(epochs)}
        Batch Size: {model_info.get('batch_size', 'N/A')}
        Learning Rate: {model_info.get('learning_rate', 'N/A')}
        
        ⏱️ Performance
        
        Best Val Acc: {max(metrics['val_accs']):.2f}%
        Final Loss: {metrics['val_losses'][-1]:.4f}
        """
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 4. Training time per epoch
        if 'epoch_times' in metrics:
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.bar(epochs, metrics['epoch_times'], alpha=0.7)
            ax4.set_title('⏱️ Time per Epoch', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Time (seconds)')
            ax4.grid(True, alpha=0.3)
        
        # 5. Final metrics comparison
        ax5 = fig.add_subplot(gs[2, :])
        final_metrics = ['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss']
        values = [
            metrics['train_accs'][-1],
            metrics['val_accs'][-1], 
            metrics['train_losses'][-1],
            metrics['val_losses'][-1]
        ]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax5.bar(final_metrics, values, color=colors, alpha=0.8)
        ax5.set_title('📋 Final Training Metrics', fontweight='bold')
        ax5.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('🎯 ASL Model Training Summary', fontsize=20, fontweight='bold')
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"training_summary_{timestamp}.png"
        plt.savefig(save_path)
        plt.show()
        
        logger.info(f"📊 Training summary saved to: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, 
                            y_true: List, 
                            y_pred: List, 
                            class_names: List[str],
                            title: str = "Confusion Matrix") -> str:
        """
        Create beautiful confusion matrix plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title(f'🎯 {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(save_path)
        plt.show()
        
        logger.info(f"📊 Confusion matrix saved to: {save_path}")
        return str(save_path)
    
    def create_training_report(self, 
                             metrics: Dict, 
                             model_info: Dict,
                             save_plots: bool = True) -> Dict[str, str]:
        """
        Create complete training report with all visualizations
        """
        report_paths = {}
        
        logger.info("📊 Creating comprehensive training report...")
        
        # 1. Training curves
        curves_path = self.plot_training_curves(
            metrics['train_losses'], metrics['val_losses'],
            metrics['train_accs'], metrics['val_accs'],
            f"ASL {model_info.get('model_type', 'Model')} Training"
        )
        report_paths['training_curves'] = curves_path
        
        # 2. Training summary
        summary_path = self.plot_training_summary(metrics, model_info)
        report_paths['training_summary'] = summary_path
        
        # 3. Save metrics as JSON for later analysis
        metrics_path = self.save_dir / f"training_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'model_info': model_info,
                'timestamp': datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        report_paths['metrics_json'] = str(metrics_path)
        
        logger.info("✅ Training report completed!")
        logger.info(f"📁 All files saved to: {self.save_dir}")
        
        return report_paths

def plot_model_comparison(models_data: List[Dict], save_dir: str = None) -> str:
    """
    Compare multiple model training results
    """
    if save_dir is None:
        save_dir = Path("src/asl_dl/visualization/plots")
    else:
        save_dir = Path(save_dir)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = sns.color_palette("husl", len(models_data))
    
    for i, model_data in enumerate(models_data):
        label = model_data.get('name', f'Model {i+1}')
        color = colors[i]
        epochs = range(1, len(model_data['train_losses']) + 1)
        
        # Training loss
        ax1.plot(epochs, model_data['train_losses'], 'o-', 
                label=label, color=color, linewidth=2)
        
        # Validation loss  
        ax2.plot(epochs, model_data['val_losses'], 's-',
                label=label, color=color, linewidth=2)
        
        # Training accuracy
        ax3.plot(epochs, model_data['train_accs'], 'o-',
                label=label, color=color, linewidth=2)
        
        # Validation accuracy
        ax4.plot(epochs, model_data['val_accs'], 's-', 
                label=label, color=color, linewidth=2)
    
    # Configure subplots
    ax1.set_title('📉 Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('📉 Validation Loss Comparison')
    ax2.set_xlabel('Epoch') 
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('📈 Training Accuracy Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('📈 Validation Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('🏆 Model Performance Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"model_comparison_{timestamp}.png"
    plt.savefig(save_path)
    plt.show()
    
    return str(save_path) 