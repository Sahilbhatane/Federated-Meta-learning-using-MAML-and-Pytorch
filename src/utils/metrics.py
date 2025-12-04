import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)


def calculate_metrics(predictions, targets) -> Dict[str, float]:
    """Calculate basic classification metrics"""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    correct = (predictions == targets).sum()
    total = len(targets)
    accuracy = 100 * correct / total
    return {'accuracy': accuracy}


def calculate_comprehensive_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    num_classes: int = 4
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for evaluation.
    
    Args:
        predictions: Model predictions (class indices)
        targets: Ground truth labels
        probabilities: Optional softmax probabilities for ROC/AUC
        num_classes: Number of classes
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, and per-class metrics
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions) * 100,
        'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0) * 100,
        'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0) * 100,
        'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0) * 100,
        'precision_weighted': precision_score(targets, predictions, average='weighted', zero_division=0) * 100,
        'recall_weighted': recall_score(targets, predictions, average='weighted', zero_division=0) * 100,
        'f1_weighted': f1_score(targets, predictions, average='weighted', zero_division=0) * 100,
    }
    
    # Per-class metrics
    for cls in range(num_classes):
        cls_mask = targets == cls
        if cls_mask.sum() > 0:
            cls_acc = (predictions[cls_mask] == targets[cls_mask]).mean() * 100
            metrics[f'class_{cls}_accuracy'] = cls_acc
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4
) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.
    
    Returns:
        Confusion matrix as numpy array [num_classes x num_classes]
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def compute_roc_curves(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute ROC curves and AUC for each class (one-vs-rest).
    
    Args:
        probabilities: Softmax probabilities [N x num_classes]
        targets: Ground truth labels [N]
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping class index to (fpr, tpr, auc_score)
    """
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    
    roc_data = {}
    for cls in range(num_classes):
        binary_targets = (targets == cls).astype(int)
        if binary_targets.sum() == 0 or binary_targets.sum() == len(binary_targets):
            continue  # Skip if only one class present
        
        fpr, tpr, _ = roc_curve(binary_targets, probabilities[:, cls])
        roc_auc = auc(fpr, tpr)
        roc_data[cls] = (fpr, tpr, roc_auc)
    
    return roc_data


def aggregate_metrics(client_metrics: List[Dict], weights: List[float] = None) -> Dict[str, float]:
    """Aggregate metrics from multiple clients with optional weighting"""
    if not client_metrics:
        return {}
    
    if weights is None:
        weights = [1.0 / len(client_metrics)] * len(client_metrics)
    
    aggregated = {}
    for key in client_metrics[0].keys():
        if isinstance(client_metrics[0][key], (int, float)):
            aggregated[key] = sum(m.get(key, 0) * w for m, w in zip(client_metrics, weights))
    
    return aggregated


def calculate_privacy_metrics(epsilon: float, delta: float, num_steps: int) -> Dict[str, float]:
    """
    Calculate privacy budget metrics for differential privacy.
    
    Args:
        epsilon: Privacy budget epsilon
        delta: Privacy budget delta
        num_steps: Number of training steps taken
    
    Returns:
        Dictionary with privacy-related metrics
    """
    return {
        'epsilon': epsilon,
        'delta': delta,
        'num_steps': num_steps,
        'privacy_budget_used': epsilon  # simplified; real accounting uses Opacus
    }
