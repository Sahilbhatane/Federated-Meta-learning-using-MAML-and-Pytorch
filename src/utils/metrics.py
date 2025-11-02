import numpy as np
from typing import Dict, List


def calculate_metrics(predictions, targets):
    """Calculate classification metrics"""
    correct = (predictions == targets).sum()
    total = len(targets)
    accuracy = 100 * correct / total
    return {'accuracy': accuracy}


def aggregate_metrics(client_metrics: List[Dict], weights: List[float] = None):
    """Aggregate metrics from multiple clients"""
    if weights is None:
        weights = [1.0 / len(client_metrics)] * len(client_metrics)
    
    aggregated = {}
    for key in client_metrics[0].keys():
        aggregated[key] = sum(m[key] * w for m, w in zip(client_metrics, weights))
    
    return aggregated
