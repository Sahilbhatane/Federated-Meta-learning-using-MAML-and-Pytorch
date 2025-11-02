"""
MAML Trainer for Federated Meta-Learning

Based on Phase 2 insights:
- 4 clients with 30-42 samples each (small dataset regime)
- High heterogeneity (label variance: 83.84) → personalization critical
- Few-shot learning setup for fast adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import learn2learn as l2l
from copy import deepcopy


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning trainer for federated health monitoring
    
    Implements:
    - Inner loop: Fast adaptation to user-specific data (1-5 gradient steps)
    - Outer loop: Meta-learning across clients for good initialization
    - Support for both first-order and second-order MAML
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 3,
        first_order: bool = False,
        device: str = 'cpu'
    ):
        """
        Args:
            model: Base neural network (HealthMonitorNet)
            inner_lr: Learning rate for inner loop adaptation (default: 0.01)
            meta_lr: Learning rate for meta-update (default: 0.001)
            inner_steps: Number of gradient steps for adaptation (default: 3)
            first_order: Use first-order MAML approximation (faster, less accurate)
            device: Device for training ('cpu' or 'cuda')
        """
        self.device = device
        self.model = model.to(device)
        
        # Wrap model with learn2learn MAML
        self.maml = l2l.algorithms.MAML(
            self.model,
            lr=inner_lr,
            first_order=first_order,
            allow_unused=True
        )
        
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(self.maml.parameters(), lr=meta_lr)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def adapt(self, 
              learner, 
              support_loader: DataLoader, 
              criterion: nn.Module) -> nn.Module:
        """
        Inner loop: Adapt model to user-specific support set
        
        Args:
            learner: Cloned MAML model
            support_loader: Support set DataLoader (k-shot samples)
            criterion: Loss function
        
        Returns:
            Adapted model
        """
        for _ in range(self.inner_steps):
            for features, labels in support_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = learner(features)
                loss = criterion(outputs, labels)
                
                # Adaptation step (inner loop gradient)
                learner.adapt(loss)
        
        return learner
    
    def meta_train_step(
        self,
        client_loaders: Dict[int, Tuple[DataLoader, DataLoader]],
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Outer loop: Meta-training step across all clients
        
        Args:
            client_loaders: Dictionary of (support_loader, query_loader) per client
            criterion: Loss function
        
        Returns:
            (meta_loss, meta_accuracy)
        """
        self.meta_optimizer.zero_grad()
        meta_train_loss = 0.0
        meta_train_acc = 0.0
        num_clients = len(client_loaders)
        
        for client_id, (support_loader, query_loader) in client_loaders.items():
            # Clone model for this client
            learner = self.maml.clone()
            
            # Inner loop: Adapt to client's support set
            learner = self.adapt(learner, support_loader, criterion)
            
            # Evaluate on client's query set
            for features, labels in query_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = learner(features)
                loss = criterion(outputs, labels)
                
                # Accumulate loss for meta-update
                meta_train_loss += loss / num_clients
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                meta_train_acc += acc.item() / num_clients
        
        # Outer loop: Meta-update
        meta_train_loss.backward()
        self.meta_optimizer.step()
        
        return meta_train_loss.item(), meta_train_acc
    
    def meta_evaluate(
        self,
        client_loaders: Dict[int, Tuple[DataLoader, DataLoader]],
        criterion: nn.Module,
        return_per_client: bool = False
    ) -> Tuple[float, float]:
        """
        Meta-evaluation on validation clients
        
        Args:
            client_loaders: Dictionary of (support_loader, query_loader) per client
            criterion: Loss function
            return_per_client: If True, return per-client metrics
        
        Returns:
            (meta_loss, meta_accuracy) or dict of per-client metrics
        """
        meta_val_loss = 0.0
        meta_val_acc = 0.0
        num_clients = len(client_loaders)
        
        per_client_metrics = {}
        
        with torch.no_grad():
            for client_id, (support_loader, query_loader) in client_loaders.items():
                # Clone model for this client
                learner = self.maml.clone()
                
                # Adapt to client's support set (with gradient for adaptation)
                with torch.enable_grad():
                    learner = self.adapt(learner, support_loader, criterion)
                
                # Evaluate on client's query set
                client_loss = 0.0
                client_correct = 0
                client_total = 0
                
                for features, labels in query_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    outputs = learner(features)
                    loss = criterion(outputs, labels)
                    
                    client_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    client_total += labels.size(0)
                    client_correct += (predicted == labels).sum().item()
                
                client_acc = 100 * client_correct / client_total if client_total > 0 else 0
                client_loss /= len(query_loader)
                
                meta_val_loss += client_loss / num_clients
                meta_val_acc += client_acc / num_clients
                
                per_client_metrics[client_id] = {
                    'loss': client_loss,
                    'accuracy': client_acc,
                    'samples': client_total
                }
        
        if return_per_client:
            return per_client_metrics
        
        return meta_val_loss, meta_val_acc
    
    def train_epoch(
        self,
        train_client_loaders: Dict[int, Tuple[DataLoader, DataLoader]],
        val_client_loaders: Dict[int, Tuple[DataLoader, DataLoader]],
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Full meta-training epoch
        
        Args:
            train_client_loaders: Training client data
            val_client_loaders: Validation client data
            criterion: Loss function
        
        Returns:
            Dictionary with epoch metrics
        """
        # Meta-training
        self.model.train()
        train_loss, train_acc = self.meta_train_step(train_client_loaders, criterion)
        
        # Meta-validation
        self.model.eval()
        val_loss, val_acc = self.meta_evaluate(val_client_loaders, criterion)
        
        # Store metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        print(f"Checkpoint loaded from {path}")
        return checkpoint


def compare_global_vs_personalized(
    global_model: nn.Module,
    maml_model: nn.Module,
    test_loader: DataLoader,
    support_loader: DataLoader,
    criterion: nn.Module,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compare global model vs MAML-personalized model on new user
    
    Demonstrates the value of fast adaptation for personalization.
    
    Args:
        global_model: Baseline global model (no adaptation)
        maml_model: MAML meta-learned model
        test_loader: Test set for evaluation
        support_loader: Support set for adaptation
        criterion: Loss function
        inner_lr: Adaptation learning rate
        inner_steps: Number of adaptation steps
        device: Device for computation
    
    Returns:
        Dictionary with global and personalized metrics
    """
    # Evaluate global model (no adaptation)
    global_model.eval()
    global_loss = 0.0
    global_correct = 0
    global_total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = global_model(features)
            loss = criterion(outputs, labels)
            
            global_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            global_total += labels.size(0)
            global_correct += (predicted == labels).sum().item()
    
    global_acc = 100 * global_correct / global_total if global_total > 0 else 0
    global_loss /= len(test_loader)
    
    # Adapt MAML model and evaluate
    maml_wrapper = l2l.algorithms.MAML(maml_model, lr=inner_lr, first_order=False)
    learner = maml_wrapper.clone()
    
    # Adaptation on support set
    for _ in range(inner_steps):
        for features, labels in support_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = learner(features)
            loss = criterion(outputs, labels)
            learner.adapt(loss)
    
    # Evaluate adapted model
    learner.eval()
    personalized_loss = 0.0
    personalized_correct = 0
    personalized_total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = learner(features)
            loss = criterion(outputs, labels)
            
            personalized_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            personalized_total += labels.size(0)
            personalized_correct += (predicted == labels).sum().item()
    
    personalized_acc = 100 * personalized_correct / personalized_total if personalized_total > 0 else 0
    personalized_loss /= len(test_loader)
    
    return {
        'global_loss': global_loss,
        'global_accuracy': global_acc,
        'personalized_loss': personalized_loss,
        'personalized_accuracy': personalized_acc,
        'improvement': personalized_acc - global_acc
    }
