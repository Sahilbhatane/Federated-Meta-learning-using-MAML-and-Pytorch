# MAML Trainer for Federated Meta-Learning
# Simple MAML implementation using PyTorch deepcopy
# Phase 4: Added differential privacy (Opacus) and TensorBoard support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from copy import deepcopy

# Optional Opacus import for differential privacy
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None

# Optional TensorBoard import (may fail on Python 3.14 due to protobuf issues)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, TypeError):
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class MAMLTrainer:
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 3,
        first_order: bool = False,
        device: str = 'cpu',
        # Differential Privacy parameters
        dp_enabled: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 1.0,
        # TensorBoard parameters
        tensorboard_dir: Optional[str] = None
    ):
        self.device = device
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Differential Privacy setup
        self.dp_enabled = dp_enabled and OPACUS_AVAILABLE
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_max_grad_norm = dp_max_grad_norm
        self.privacy_engine = None
        self.privacy_spent = {'epsilon': 0.0, 'delta': dp_delta}
        
        if dp_enabled and not OPACUS_AVAILABLE:
            print("Warning: Opacus not installed. Differential privacy disabled.")
            print("Install with: pip install opacus")
        
        # TensorBoard setup
        self.tb_writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard logging enabled: {tensorboard_dir}")
        elif tensorboard_dir and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not installed. Logging disabled.")
    
    def _setup_dp_for_loader(self, model: nn.Module, optimizer, data_loader: DataLoader) -> Tuple:
        """Attach Opacus PrivacyEngine to model/optimizer for a specific loader"""
        if not self.dp_enabled or not OPACUS_AVAILABLE:
            return model, optimizer, data_loader
        
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,  # Controls noise added to gradients
            max_grad_norm=self.dp_max_grad_norm,
        )
        return model, optimizer, data_loader

    def adapt(self, model: nn.Module, support_loader: DataLoader, criterion: nn.Module) -> nn.Module:
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=self.inner_lr)
        model.train()
        for _ in range(self.inner_steps):
            for features, labels in support_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                inner_optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward(create_graph=not self.first_order)
                inner_optimizer.step()
        return model
    
    def meta_train_step(self, client_loaders: Dict[int, Tuple[DataLoader, DataLoader]], criterion: nn.Module) -> Tuple[float, float]:
        self.meta_optimizer.zero_grad()
        meta_train_loss = 0.0
        meta_train_acc = 0.0
        num_clients = len(client_loaders)
        
        for client_id, (support_loader, query_loader) in client_loaders.items():
            learner = deepcopy(self.model)
            learner = self.adapt(learner, support_loader, criterion)
            learner.eval()
            
            for features, labels in query_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = learner(features)
                loss = criterion(outputs, labels)
                meta_train_loss += loss / num_clients
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                meta_train_acc += acc.item() / num_clients
        
        meta_train_loss.backward()
        
        # Apply gradient clipping for DP (even without full PrivacyEngine)
        if self.dp_enabled:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_max_grad_norm)
        
        self.meta_optimizer.step()
        return meta_train_loss.item(), meta_train_acc
    
    def meta_evaluate(self, client_loaders: Dict[int, Tuple[DataLoader, DataLoader]], criterion: nn.Module, return_per_client: bool = False):
        meta_val_loss = 0.0
        meta_val_acc = 0.0
        num_clients = len(client_loaders)
        per_client_metrics = {}
        
        for client_id, (support_loader, query_loader) in client_loaders.items():
            learner = deepcopy(self.model)
            with torch.enable_grad():
                learner = self.adapt(learner, support_loader, criterion)
            learner.eval()
            
            client_loss = 0.0
            client_correct = 0
            client_total = 0
            
            with torch.no_grad():
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
            per_client_metrics[client_id] = {'loss': client_loss, 'accuracy': client_acc, 'samples': client_total}
        
        return per_client_metrics if return_per_client else (meta_val_loss, meta_val_acc)
    
    def train_epoch(self, train_client_loaders: Dict[int, Tuple[DataLoader, DataLoader]], val_client_loaders: Dict[int, Tuple[DataLoader, DataLoader]], criterion: nn.Module, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        train_loss, train_acc = self.meta_train_step(train_client_loaders, criterion)
        self.model.eval()
        val_loss, val_acc = self.meta_evaluate(val_client_loaders, criterion)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('Train/Loss', train_loss, epoch)
            self.tb_writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.tb_writer.add_scalar('Val/Loss', val_loss, epoch)
            self.tb_writer.add_scalar('Val/Accuracy', val_acc, epoch)
            if self.dp_enabled:
                self.tb_writer.add_scalar('Privacy/Epsilon', self.privacy_spent['epsilon'], epoch)
        
        return {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Return the privacy budget spent so far"""
        return self.privacy_spent
    
    def close(self):
        """Clean up resources"""
        if self.tb_writer:
            self.tb_writer.close()
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'dp_enabled': self.dp_enabled,
            'privacy_spent': self.privacy_spent
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.privacy_spent = checkpoint.get('privacy_spent', {'epsilon': 0.0, 'delta': self.dp_delta})
        print(f"Checkpoint loaded from {path}")
        return checkpoint


def compare_global_vs_personalized(global_model: nn.Module, maml_model: nn.Module, test_loader: DataLoader, support_loader: DataLoader, criterion: nn.Module, inner_lr: float = 0.01, inner_steps: int = 3, device: str = 'cpu') -> Dict[str, float]:
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
    
    learner = deepcopy(maml_model)
    inner_optimizer = torch.optim.SGD(learner.parameters(), lr=inner_lr)
    learner.train()
    
    for _ in range(inner_steps):
        for features, labels in support_loader:
            features, labels = features.to(device), labels.to(device)
            inner_optimizer.zero_grad()
            outputs = learner(features)
            loss = criterion(outputs, labels)
            loss.backward()
            inner_optimizer.step()
    
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
    
    return {'global_loss': global_loss, 'global_accuracy': global_acc, 'personalized_loss': personalized_loss, 'personalized_accuracy': personalized_acc, 'improvement': personalized_acc - global_acc}
