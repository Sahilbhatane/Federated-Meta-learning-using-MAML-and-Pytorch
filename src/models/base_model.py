import torch
import torch.nn as nn
import torch.nn.functional as F


class HealthMonitorNet(nn.Module):
    """
    Neural network for wearable health data classification
    
    Designed based on Phase 2 insights:
    - Input: 9 features (Temperature, BP, Heart Rate, Battery levels)
    - Small architecture for limited data (30-42 samples per user)
    - Optimized for fast MAML adaptation
    - Handles non-IID heterogeneous data (variance 83.84)
    """
    
    def __init__(self, input_dim=9, hidden_dims=[32, 16], num_classes=2, dropout=0.2):
        """
        Args:
            input_dim (int): Number of input features (default: 9 from Phase 2)
            hidden_dims (list): Hidden layer dimensions (smaller for limited data)
            num_classes (int): Output classes (2 for Healthy/Unhealthy, 4 for BP levels)
            dropout (float): Dropout rate for regularization (0.2 for small datasets)
        """
        super(HealthMonitorNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with smaller capacity for few-shot learning
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Stabilize training on small batches
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for faster MAML adaptation
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better gradient flow in meta-learning"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy
