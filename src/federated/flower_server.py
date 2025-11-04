"""
Flower Federated Learning Server for MAML

Coordinates federated meta-learning across clients.
Based on Phase 2 insights: 4 clients with heterogeneous data distribution.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class MAMLFederatedStrategy(FedAvg):
    """
    Custom Flower strategy for MAML federated learning
    
    Extends FedAvg to:
    1. Aggregate adapted model parameters from clients
    2. Maintain meta-learned initialization
    3. Track heterogeneous client performance
    4. Save checkpoints and metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "results/federated",
        min_fit_clients: int = 4,
        min_evaluate_clients: int = 4,
        min_available_clients: int = 4,
        **kwargs
    ):
        """
        Args:
            model: Base model for parameter initialization
            save_dir: Directory for saving checkpoints
            min_fit_clients: Minimum clients for training round
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum clients available
        """
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.round_metrics = []
        self.best_accuracy = 0.0
    
    def initialize_parameters(
        self,
        client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """
        Initialize global model parameters
        
        Returns meta-learned initialization for all clients.
        """
        ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return fl.common.ndarrays_to_parameters(ndarrays)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate adapted parameters from clients
        
        Performs FedAvg aggregation on MAML-adapted models.
        """
        # Call parent aggregation (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Extract client metrics
        client_metrics = []
        for _, fit_res in results:
            client_metrics.append(fit_res.metrics)
        
        # Calculate average metrics across clients
        avg_loss = np.mean([m.get("query_loss", 0) for m in client_metrics])
        avg_acc = np.mean([m.get("query_accuracy", 0) for m in client_metrics])
        
        metrics = {
            "round": server_round,
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            "num_clients": len(results)
        }
        
        print(f"\n[Round {server_round}] Aggregated - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
        
        # Save best model
        if avg_acc > self.best_accuracy:
            self.best_accuracy = avg_acc
            self.save_model(aggregated_parameters, server_round, metrics)
        
        self.round_metrics.append(metrics)
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients
        """
        if not results:
            return None, {}
        
        # Extract client metrics
        client_metrics = []
        for _, evaluate_res in results:
            client_metrics.append(evaluate_res.metrics)
        
        # Calculate average metrics
        avg_loss = np.mean([evaluate_res.loss for _, evaluate_res in results])
        avg_acc = np.mean([m.get("accuracy", 0) for m in client_metrics])
        
        metrics = {
            "round": server_round,
            "eval_loss": avg_loss,
            "eval_accuracy": avg_acc,
            "num_clients": len(results)
        }
        
        print(f"[Round {server_round}] Evaluation - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
        
        return avg_loss, metrics
    
    def save_model(self, parameters: Parameters, round_num: int, metrics: Dict):
        """Save model checkpoint"""
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        
        # Reconstruct state dict
        state_dict = {}
        for i, key in enumerate(self.model.state_dict().keys()):
            state_dict[key] = torch.tensor(ndarrays[i])
        
        checkpoint = {
            'round': round_num,
            'state_dict': state_dict,
            'metrics': metrics,
            'best_accuracy': self.best_accuracy
        }
        
        checkpoint_path = self.save_dir / f"best_model_round_{round_num}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Best model saved to {checkpoint_path}")
    
    def save_metrics(self, filename: str = "federated_metrics.npy"):
        """Save training metrics"""
        metrics_path = self.save_dir / filename
        np.save(metrics_path, self.round_metrics)
        print(f"Metrics saved to {metrics_path}")


def start_federated_server(
    model: nn.Module,
    num_rounds: int = 50,
    num_clients: int = 4,
    save_dir: str = "results/federated",
    server_address: str = "localhost:8080"
) -> None:
    """
    Start Flower federated learning server
    
    Args:
        model: Base model for initialization
        num_rounds: Number of federated rounds
        num_clients: Expected number of clients (default: 4 from Phase 2)
        save_dir: Directory for saving results
        server_address: Server address for client connections
    """
    # Create strategy
    strategy = MAMLFederatedStrategy(
        model=model,
        save_dir=save_dir,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0
    )
    
    print(f"Starting Flower server at {server_address}")
    print(f"Expected {num_clients} clients for {num_rounds} rounds")
    print(f"Results will be saved to {save_dir}")
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    # Save final metrics
    strategy.save_metrics()
    print("\nFederated training completed!")
    print(f"Best accuracy: {strategy.best_accuracy:.2f}%")


def simulate_federated_maml(
    model: nn.Module,
    client_loaders: Dict,
    num_rounds: int = 50,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    device: str = 'cpu',
    save_dir: str = "results/federated"
) -> Dict:
    """
    Simulate federated MAML training (without network communication)
    
    Useful for testing and debugging before deploying actual federated system.
    
    Args:
        model: Base model
        client_loaders: Dictionary of (support_loader, query_loader) per client
        num_rounds: Number of rounds
        inner_lr: Inner loop learning rate
        inner_steps: Number of adaptation steps
        device: Computation device
        save_dir: Directory for saving results
    
    Returns:
        Dictionary with training history
    """
    from copy import deepcopy
    from tqdm import tqdm
    
    print(f"\nSimulating Federated MAML with {len(client_loaders)} clients")
    
    # Initialize meta-model
    meta_model = model.to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'rounds': [],
        'train_loss': [],
        'train_acc': [],
        'per_client_metrics': []
    }
    
    print("\nStarting training...")
    for round_num in tqdm(range(1, num_rounds + 1), desc="Federated Rounds", ncols=100):
        if round_num % 10 == 1 or round_num == num_rounds:
            print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        meta_model.train()
        round_loss = 0.0
        round_acc = 0.0
        client_metrics = []
        client_models = []
        
        for client_id, (support_loader, query_loader) in client_loaders.items():
            # Clone model for client adaptation
            learner = deepcopy(meta_model)
            inner_optimizer = torch.optim.SGD(learner.parameters(), lr=inner_lr)
            
            # Inner loop: Adapt to client data
            learner.train()
            for _ in range(inner_steps):
                for features, labels in support_loader:
                    features, labels = features.to(device), labels.to(device)
                    
                    inner_optimizer.zero_grad()
                    outputs = learner(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    inner_optimizer.step()
            
            # Evaluate on query set
            learner.eval()
            client_loss = 0.0
            client_correct = 0
            client_total = 0
            
            with torch.no_grad():
                for features, labels in query_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = learner(features)
                    loss = criterion(outputs, labels)
                    
                    client_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    client_total += labels.size(0)
                    client_correct += (predicted == labels).sum().item()
            
            client_loss /= len(query_loader)
            client_acc = 100 * client_correct / client_total if client_total > 0 else 0
            
            round_loss += client_loss
            round_acc += client_acc
            
            client_metrics.append({
                'client_id': client_id,
                'loss': client_loss,
                'accuracy': client_acc,
                'samples': client_total
            })
            
            client_models.append(learner)
            
            if round_num % 10 == 1 or round_num == num_rounds:
                print(f"Client {client_id}: Loss={client_loss:.4f}, Acc={client_acc:.2f}%")
        
        # Meta-update: FedAvg aggregation
        with torch.no_grad():
            for param_idx, meta_param in enumerate(meta_model.parameters()):
                client_params = [list(client_model.parameters())[param_idx].data for client_model in client_models]
                meta_param.data = torch.stack(client_params).mean(dim=0)
        
        round_loss /= len(client_loaders)
        round_acc /= len(client_loaders)
        
        # Store metrics
        history['rounds'].append(round_num)
        history['train_loss'].append(round_loss)
        history['train_acc'].append(round_acc)
        history['per_client_metrics'].append(client_metrics)
        
        if round_num % 10 == 1 or round_num == num_rounds:
            print(f"Round {round_num} Average: Loss={round_loss:.4f}, Acc={round_acc:.2f}%")
    
    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': meta_model.state_dict(),
        'history': history
    }, save_path / "simulated_federated_maml.pt")
    
    print(f"\nSimulation completed! Results saved to {save_dir}")
    
    return history
