import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(history, save_path=None):
    """Plot training loss and accuracy over rounds"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(rounds, history['train_loss'], marker='o', label='Train Loss')
    axes[0].plot(rounds, history['test_loss'], marker='s', label='Test Loss')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(rounds, history['train_acc'], marker='o', label='Train Accuracy')
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


def plot_client_performance(client_metrics, save_path=None):
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
