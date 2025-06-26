#!/usr/bin/env python3
"""
Comprehensive ASL Model Evaluation and Visualization

This script evaluates a trained ASL model and generates comprehensive
visualizations including confusion matrices, performance metrics,
confidence distributions, and prediction analysis.
"""
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import logging
from datetime import datetime
import json

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asl_dl.models.mobilenet import MobileNetV2ASL
from asl_dl.training.train import ASLDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive model evaluation with multiple visualization types"""
    
    def __init__(self, model_path: str, data_path: str, output_dir: str):
        self.model_path = model_path
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"🚀 Loading model from: {model_path}")
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = MobileNetV2ASL.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"✅ Model loaded successfully on device: {self.device}")
        
        # Add validation transforms
        from torchvision import transforms
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        self.dataset = ASLDataset(self.data_path, transform=self.val_transforms)
        
        # Create test split manually (20% of data)
        dataset_size = len(self.dataset)
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size
        
        # Split dataset
        from torch.utils.data import random_split
        _, test_dataset = random_split(self.dataset, [train_size, test_size])
        
        # Create test dataloader
        from torch.utils.data import DataLoader
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        logger.info(f"📊 Evaluation dataset loaded with {len(test_dataset)} samples.")
        
        # Collect predictions
        self.y_true = []
        self.y_pred = []
        self.confidences = []
        self.class_names = list(self.model.class_map.keys())
        
    def collect_predictions(self):
        """Collect all predictions for analysis"""
        logger.info("🔄 Collecting predictions...")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                
                self.y_true.extend(labels.cpu().numpy())
                self.y_pred.extend(predicted.cpu().numpy())
                self.confidences.extend(max_probs.cpu().numpy())
        
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        self.confidences = np.array(self.confidences)
        
        logger.info(f"✅ Collected {len(self.y_true)} predictions")
    
    def generate_confusion_matrix(self):
        """Generate and save confusion matrix"""
        logger.info("📊 Generating confusion matrix...")
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create labels with counts and percentages
        labels = np.array([f'{count}\n({percent:.1f}%)' 
                          for count, percent in zip(cm.flatten(), cm_percent.flatten())])
        labels = labels.reshape(cm.shape)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('Confusion Matrix\n(Counts and Percentages)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Confusion matrix saved to: {output_path}")
        plt.close()
    
    def generate_performance_metrics(self):
        """Generate detailed performance metrics visualization"""
        logger.info("📈 Generating performance metrics...")
        
        # Calculate metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Create metrics DataFrame for visualization
        metrics_data = {
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        }
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot for precision, recall, f1
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', color='skyblue', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', color='lightgreen', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', color='salmon', alpha=0.8)
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax1.text(i-width, p+0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=10)
            ax1.text(i, r+0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=10)
            ax1.text(i+width, f+0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Confidence distribution
        ax2.hist(self.confidences, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(self.confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.confidences):.3f}')
        ax2.legend()
        
        # Accuracy by confidence threshold
        thresholds = np.arange(0.1, 1.0, 0.05)
        accuracies = []
        sample_counts = []
        
        for threshold in thresholds:
            mask = self.confidences >= threshold
            if np.sum(mask) > 0:
                acc = accuracy_score(self.y_true[mask], self.y_pred[mask])
                accuracies.append(acc)
                sample_counts.append(np.sum(mask))
            else:
                accuracies.append(0)
                sample_counts.append(0)
        
        ax3.plot(thresholds, accuracies, 'o-', color='blue', linewidth=2, markersize=6)
        ax3.set_xlabel('Confidence Threshold')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Confidence Threshold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Sample count vs threshold (secondary y-axis)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(thresholds, sample_counts, 's-', color='orange', alpha=0.7, linewidth=2)
        ax3_twin.set_ylabel('Sample Count', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        # Class-wise confidence distribution
        for i, class_name in enumerate(self.class_names):
            class_mask = self.y_true == i
            class_confidences = self.confidences[class_mask]
            ax4.hist(class_confidences, bins=15, alpha=0.6, label=f'Class {class_name}')
        
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution by Class')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'performance_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Performance metrics saved to: {output_path}")
        plt.close()
        
        return metrics_data
    
    def generate_error_analysis(self):
        """Generate error analysis visualization"""
        logger.info("🔍 Generating error analysis...")
        
        # Find misclassified samples
        errors = self.y_true != self.y_pred
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            logger.info("🎉 Perfect classification! No errors to analyze.")
            return
        
        # Create error analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error distribution by true class
        error_by_class = []
        total_by_class = []
        for i, class_name in enumerate(self.class_names):
            class_mask = self.y_true == i
            class_errors = np.sum(errors[class_mask])
            class_total = np.sum(class_mask)
            error_by_class.append(class_errors)
            total_by_class.append(class_total)
        
        error_rates = [e/t if t > 0 else 0 for e, t in zip(error_by_class, total_by_class)]
        
        colors = ['red' if rate > 0.2 else 'orange' if rate > 0.1 else 'green' for rate in error_rates]
        bars = ax1.bar(self.class_names, error_rates, color=colors, alpha=0.7)
        ax1.set_xlabel('True Class')
        ax1.set_ylabel('Error Rate')
        ax1.set_title('Error Rate by True Class')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, error_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # Confidence of misclassified samples
        error_confidences = self.confidences[errors]
        correct_confidences = self.confidences[~errors]
        
        ax2.hist([correct_confidences, error_confidences], bins=20, 
                label=['Correct', 'Incorrect'], color=['green', 'red'], alpha=0.7)
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution: Correct vs Incorrect')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'error_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Error analysis saved to: {output_path}")
        plt.close()
    
    def save_detailed_report(self, metrics_data):
        """Save detailed evaluation report as JSON"""
        logger.info("📝 Generating detailed report...")
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Calculate macro and weighted averages
        macro_precision = np.mean(metrics_data['Precision'])
        macro_recall = np.mean(metrics_data['Recall'])
        macro_f1 = np.mean(metrics_data['F1-Score'])
        
        weighted_precision = np.average(metrics_data['Precision'], weights=metrics_data['Support'])
        weighted_recall = np.average(metrics_data['Recall'], weights=metrics_data['Support'])
        weighted_f1 = np.average(metrics_data['F1-Score'], weights=metrics_data['Support'])
        
        report = {
            'model_path': str(self.model_path),
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.y_true),
                'classes': self.class_names,
                'class_distribution': {class_name: int(np.sum(self.y_true == i)) 
                                     for i, class_name in enumerate(self.class_names)}
            },
            'overall_metrics': {
                'accuracy': float(overall_accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'weighted_precision': float(weighted_precision),
                'weighted_recall': float(weighted_recall),
                'weighted_f1': float(weighted_f1)
            },
            'per_class_metrics': {
                class_name: {
                    'precision': float(metrics_data['Precision'][i]),
                    'recall': float(metrics_data['Recall'][i]),
                    'f1_score': float(metrics_data['F1-Score'][i]),
                    'support': int(metrics_data['Support'][i])
                }
                for i, class_name in enumerate(self.class_names)
            },
            'confidence_stats': {
                'mean_confidence': float(np.mean(self.confidences)),
                'std_confidence': float(np.std(self.confidences)),
                'min_confidence': float(np.min(self.confidences)),
                'max_confidence': float(np.max(self.confidences))
            }
        }
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Detailed report saved to: {report_path}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"📋 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Macro F1-Score: {macro_f1:.3f}")
        print(f"Mean Confidence: {np.mean(self.confidences):.3f}")
        print(f"\nPer-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: P={metrics_data['Precision'][i]:.3f}, "
                  f"R={metrics_data['Recall'][i]:.3f}, "
                  f"F1={metrics_data['F1-Score'][i]:.3f}")
        print(f"{'='*60}")
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("🚀 Starting comprehensive evaluation...")
        
        # Collect predictions
        self.collect_predictions()
        
        # Generate all visualizations
        self.generate_confusion_matrix()
        metrics_data = self.generate_performance_metrics()
        self.generate_error_analysis()
        self.save_detailed_report(metrics_data)
        
        logger.info(f"✅ Comprehensive evaluation complete! Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive ASL Model Evaluation")
    parser.add_argument("--model", default="models/best_mobilenetv2_model.pth",
                       help="Path to trained model")
    parser.add_argument("--data", default="data/raw/kaggle_asl",
                       help="Path to evaluation data")
    parser.add_argument("--output", default="src/asl_dl/visualization/plots",
                       help="Output directory for plots and reports")
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(args.model, args.data, args.output)
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main() 