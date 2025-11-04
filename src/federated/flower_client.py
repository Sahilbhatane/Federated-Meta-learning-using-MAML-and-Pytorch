"""
Flower Federated Learning Client for MAML

Integrates MAML meta-learning with Flower federated framework.
Based on Phase 2 insights: 4 clients with 30-42 samples each, high heterogeneity.
"""

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np


class MAMLFlowerClient(fl.client.NumPyClient):
    """
    Flower client that performs MAML meta-learning
    
    Each client:
    1. Receives global meta-learned model initialization
    2. Performs inner loop adaptation on local support set
    3. Evaluates on local query set
    4. Sends gradients/parameters back to server
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        support_loader: DataLoader,
        query_loader: DataLoader,
        inner_lr: float = 0.01,
        inner_steps: int = 3,
        device: str = 'cpu'
    ):
        """
        Args:
            client_id: Client identifier
            model: Neural network model
            support_loader: Support set for adaptation (k-shot)
            query_loader: Query set for evaluation
            inner_lr: Inner loop learning rate
            inner_steps: Number of adaptation steps
            device: Computation device
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays
        
        Returns current model state to server for aggregation.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters from server
        
        Receives meta-learned initialization from federated aggregation.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def adapt_local(self) -> nn.Module:
        """
        Inner loop: Adapt model to local support set
        
        Performs MAML adaptation on client's personalization data.
        
        Returns:
            Adapted model
        """
        from copy import deepcopy
        
        # Clone model for adaptation
        learner = deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(learner.parameters(), lr=self.inner_lr)
        
        # Adaptation steps
        learner.train()
        for _ in range(self.inner_steps):
            for features, labels in self.support_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                inner_optimizer.zero_grad()
                outputs = learner(features)
                loss = self.criterion(outputs, labels)
                loss.backward(create_graph=True)  # Important for MAML!
                inner_optimizer.step()
        
        return learner
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train client model (MAML adaptation + evaluation)
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
        
        Returns:
            (updated_parameters, num_examples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Adapt to local data (inner loop)
        self.model.train()
        adapted_model = self.adapt_local()
        
        # Evaluate on query set
        adapted_model.eval()
        query_loss = 0.0
        query_correct = 0
        query_total = 0
        
        with torch.no_grad():
            for features, labels in self.query_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = adapted_model(features)
                loss = self.criterion(outputs, labels)
                
                query_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                query_total += labels.size(0)
                query_correct += (predicted == labels).sum().item()
        
        query_acc = 100 * query_correct / query_total if query_total > 0 else 0
        query_loss /= len(self.query_loader) if len(self.query_loader) > 0 else 1
        
        # Return adapted parameters
        adapted_params = [val.cpu().numpy() for _, val in adapted_model.state_dict().items()]
        
        metrics = {
            "client_id": self.client_id,
            "query_loss": query_loss,
            "query_accuracy": query_acc,
            "num_examples": query_total
        }
        
        print(f"Client {self.client_id}: Loss={query_loss:.4f}, Acc={query_acc:.2f}%")
        
        return adapted_params, query_total, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate client model
        
        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration
        
        Returns:
            (loss, num_examples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Adapt to local support set
        self.model.train()
        adapted_model = self.adapt_local()
        
        # Evaluate on query set
        adapted_model.eval()
        query_loss = 0.0
        query_correct = 0
        query_total = 0
        
        with torch.no_grad():
            for features, labels in self.query_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = adapted_model(features)
                loss = self.criterion(outputs, labels)
                
                query_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                query_total += labels.size(0)
                query_correct += (predicted == labels).sum().item()
        
        query_acc = 100 * query_correct / query_total if query_total > 0 else 0
        query_loss /= len(self.query_loader) if len(self.query_loader) > 0 else 1
        
        metrics = {
            "client_id": self.client_id,
            "accuracy": query_acc,
            "num_examples": query_total
        }
        
        return query_loss, query_total, metrics


def create_flower_client(
    client_id: int,
    model: nn.Module,
    support_loader: DataLoader,
    query_loader: DataLoader,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    device: str = 'cpu'
) -> fl.client.NumPyClient:
    """
    Factory function to create Flower client
    
    Args:
        client_id: Client identifier
        model: Neural network model
        support_loader: Support set DataLoader
        query_loader: Query set DataLoader
        inner_lr: Inner loop learning rate
        inner_steps: Number of adaptation steps
        device: Computation device
    
    Returns:
        Configured Flower client
    """
    return MAMLFlowerClient(
        client_id=client_id,
        model=model,
        support_loader=support_loader,
        query_loader=query_loader,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        device=device
    )
