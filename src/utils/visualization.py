import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# TensorBoard import (optional, fails gracefully if not installed or Python 3.14 protobuf issue)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, TypeError):
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TensorBoardLogger:
    """TensorBoard logging utility for federated MAML training"""
    
    def __init__(self, log_dir: str = "results/tensorboard"):
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
        else:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"TensorBoard logging to: {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under one main tag"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values (e.g., weights, gradients)"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_round_metrics(self, round_num: int, train_loss: float, train_acc: float,
                          client_metrics: Optional[List[Dict]] = None):
        """Log metrics for a federated round"""
        if not self.writer:
            return
        
        self.log_scalar("Train/Loss", train_loss, round_num)
        self.log_scalar("Train/Accuracy", train_acc, round_num)
        
        if client_metrics:
            for m in client_metrics:
                cid = m.get('client_id', 0)
                self.log_scalar(f"Client_{cid}/Loss", m.get('loss', 0), round_num)
                self.log_scalar(f"Client_{cid}/Accuracy", m.get('accuracy', 0), round_num)
    
    def log_privacy_metrics(self, step: int, epsilon: float, delta: float):
        """Log differential privacy budget consumption"""
        if self.writer:
            self.log_scalar("Privacy/Epsilon", epsilon, step)
            self.log_scalar("Privacy/Delta", delta, step)
    
    def log_model_params(self, model, step: int):
        """Log model parameter histograms"""
        if not self.writer:
            return
        for name, param in model.named_parameters():
            self.log_histogram(f"Parameters/{name}", param.data.cpu().numpy(), step)
    
    def close(self):
        """Close the TensorBoard writer"""
        if self.writer:
            self.writer.close()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training loss and accuracy over rounds"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(rounds, history['train_loss'], marker='o', label='Train Loss')
    if 'test_loss' in history:
        axes[0].plot(rounds, history['test_loss'], marker='s', label='Test Loss')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(rounds, history['train_acc'], marker='o', label='Train Accuracy')
    if 'test_acc' in history:
        axes[1].plot(rounds, history['test_acc'], marker='s', label='Test Accuracy')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training History - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_client_performance(client_metrics: Dict, save_path: Optional[str] = None):
    """Plot performance comparison across clients"""
    client_ids = list(client_metrics.keys())
    accuracies = [client_metrics[cid]['accuracy'] for cid in client_ids]
    
    plt.figure(figsize=(10, 6))
    plt.bar(client_ids, accuracies, color='steelblue', edgecolor='black')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy (%)')
    plt.title('Client Performance Comparison')
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix [num_classes x num_classes]
        class_names: Optional list of class names
        save_path: Optional path to save figure
        title: Plot title
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(
    roc_data: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        roc_data: Dictionary mapping class index to (fpr, tpr, auc_score)
        class_names: Optional list of class names
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for cls, (fpr, tpr, roc_auc) in roc_data.items():
        label = class_names[cls] if class_names else f"Class {cls}"
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_privacy_utility_tradeoff(
    epsilons: List[float],
    accuracies: List[float],
    save_path: Optional[str] = None
):
    """
    Plot privacy-utility tradeoff curve.
    
    Args:
        epsilons: List of epsilon values
        accuracies: Corresponding accuracy values
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Accuracy (%)')
    plt.title('Privacy-Utility Tradeoff')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_baseline_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    save_path: Optional[str] = None
):
    """
    Plot comparison of different algorithms/baselines.
    
    Args:
        results: Dictionary mapping algorithm name to metrics dict
        metric: Which metric to compare
        save_path: Optional path to save figure
    """
    algorithms = list(results.keys())
    values = [results[alg].get(metric, 0) for alg in algorithms]
    
    colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'gold']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, values, color=colors[:len(algorithms)], edgecolor='black')
    plt.xlabel('Algorithm')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
