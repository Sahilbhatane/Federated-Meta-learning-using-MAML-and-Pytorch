"""
Flower Federated Learning Server for MAML

Coordinates federated meta-learning across clients.
Based on Phase 2 insights: 4 clients with heterogeneous data distribution.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Flower imports - optional, may fail on Python 3.14
try:
    import flwr as fl
    from flwr.server.strategy import FedAvg
    from flwr.common import Parameters, Scalar
    FLOWER_AVAILABLE = True
except (ImportError, TypeError):
    FLOWER_AVAILABLE = False
    fl = None
    FedAvg = object  # Placeholder base class
    Parameters = None
    Scalar = None


# Only define Flower-dependent class if Flower is available
if FLOWER_AVAILABLE:
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
    
    def save_model(self, parameters, round_num: int, metrics: Dict):
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

else:
    # Placeholder when Flower is not available
    MAMLFederatedStrategy = None


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
    if not FLOWER_AVAILABLE:
        raise RuntimeError("Flower (flwr) is not available. Use simulate_federated_maml() instead.")
    
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


# ============================================================================
# BASELINE ALGORITHMS FOR COMPARISON (Phase 4)
# ============================================================================

def simulate_fedavg(
    model: nn.Module,
    client_loaders: Dict,
    num_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 0.01,
    device: str = 'cpu',
    save_dir: str = "results/federated"
) -> Dict:
    """
    Simulate standard FedAvg (no meta-learning).
    
    Each client trains locally for `local_epochs`, then server averages.
    """
    from copy import deepcopy
    from tqdm import tqdm
    
    print(f"\nSimulating FedAvg with {len(client_loaders)} clients")
    
    global_model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'rounds': [], 'train_loss': [], 'train_acc': [], 'per_client_metrics': []}
    
    for round_num in tqdm(range(1, num_rounds + 1), desc="FedAvg Rounds", ncols=100):
        client_models = []
        client_weights = []
        round_loss = 0.0
        round_acc = 0.0
        client_metrics = []
        
        for client_id, (train_loader, test_loader) in client_loaders.items():
            # Clone global model for local training
            local_model = deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            
            # Local training
            local_model.train()
            for _ in range(local_epochs):
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on test set
            local_model.eval()
            client_loss = 0.0
            client_correct = 0
            client_total = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = local_model(features)
                    loss = criterion(outputs, labels)
                    client_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    client_total += labels.size(0)
                    client_correct += (predicted == labels).sum().item()
            
            client_loss /= len(test_loader) if len(test_loader) > 0 else 1
            client_acc = 100 * client_correct / client_total if client_total > 0 else 0
            
            round_loss += client_loss
            round_acc += client_acc
            client_metrics.append({'client_id': client_id, 'loss': client_loss, 'accuracy': client_acc, 'samples': client_total})
            
            client_models.append(local_model)
            client_weights.append(client_total)
        
        # FedAvg aggregation (weighted by sample count)
        total_samples = sum(client_weights)
        with torch.no_grad():
            for param_idx, global_param in enumerate(global_model.parameters()):
                weighted_sum = sum(
                    (w / total_samples) * list(client_model.parameters())[param_idx].data
                    for client_model, w in zip(client_models, client_weights)
                )
                global_param.data = weighted_sum
        
        round_loss /= len(client_loaders)
        round_acc /= len(client_loaders)
        
        history['rounds'].append(round_num)
        history['train_loss'].append(round_loss)
        history['train_acc'].append(round_acc)
        history['per_client_metrics'].append(client_metrics)
        
        if round_num % 10 == 1 or round_num == num_rounds:
            print(f"\nRound {round_num}: Loss={round_loss:.4f}, Acc={round_acc:.2f}%")
    
    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': global_model.state_dict(), 'history': history}, save_path / "fedavg_baseline.pt")
    print(f"\nFedAvg completed! Results saved to {save_dir}")
    
    return history


def simulate_fedprox(
    model: nn.Module,
    client_loaders: Dict,
    num_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 0.01,
    mu: float = 0.01,  # Proximal term coefficient
    device: str = 'cpu',
    save_dir: str = "results/federated"
) -> Dict:
    """
    Simulate FedProx (handles heterogeneity with proximal term).
    
    Adds L2 regularization towards global model during local training.
    """
    from copy import deepcopy
    from tqdm import tqdm
    
    print(f"\nSimulating FedProx (mu={mu}) with {len(client_loaders)} clients")
    
    global_model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'rounds': [], 'train_loss': [], 'train_acc': [], 'per_client_metrics': []}
    
    for round_num in tqdm(range(1, num_rounds + 1), desc="FedProx Rounds", ncols=100):
        client_models = []
        client_weights = []
        round_loss = 0.0
        round_acc = 0.0
        client_metrics = []
        
        # Cache global params for proximal term
        global_params = [p.detach().clone() for p in global_model.parameters()]
        
        for client_id, (train_loader, test_loader) in client_loaders.items():
            local_model = deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            
            local_model.train()
            for _ in range(local_epochs):
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(features)
                    loss = criterion(outputs, labels)
                    
                    # Add proximal term: (mu/2) * ||w - w_global||^2
                    prox_term = 0.0
                    for local_p, global_p in zip(local_model.parameters(), global_params):
                        prox_term += ((local_p - global_p) ** 2).sum()
                    loss += (mu / 2) * prox_term
                    
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            local_model.eval()
            client_loss = 0.0
            client_correct = 0
            client_total = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = local_model(features)
                    loss = criterion(outputs, labels)
                    client_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    client_total += labels.size(0)
                    client_correct += (predicted == labels).sum().item()
            
            client_loss /= len(test_loader) if len(test_loader) > 0 else 1
            client_acc = 100 * client_correct / client_total if client_total > 0 else 0
            
            round_loss += client_loss
            round_acc += client_acc
            client_metrics.append({'client_id': client_id, 'loss': client_loss, 'accuracy': client_acc, 'samples': client_total})
            
            client_models.append(local_model)
            client_weights.append(client_total)
        
        # FedAvg-style aggregation
        total_samples = sum(client_weights)
        with torch.no_grad():
            for param_idx, global_param in enumerate(global_model.parameters()):
                weighted_sum = sum(
                    (w / total_samples) * list(client_model.parameters())[param_idx].data
                    for client_model, w in zip(client_models, client_weights)
                )
                global_param.data = weighted_sum
        
        round_loss /= len(client_loaders)
        round_acc /= len(client_loaders)
        
        history['rounds'].append(round_num)
        history['train_loss'].append(round_loss)
        history['train_acc'].append(round_acc)
        history['per_client_metrics'].append(client_metrics)
        
        if round_num % 10 == 1 or round_num == num_rounds:
            print(f"\nRound {round_num}: Loss={round_loss:.4f}, Acc={round_acc:.2f}%")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': global_model.state_dict(), 'history': history}, save_path / "fedprox_baseline.pt")
    print(f"\nFedProx completed! Results saved to {save_dir}")
    
    return history


def simulate_local_only(
    model: nn.Module,
    client_loaders: Dict,
    num_epochs: int = 50,
    lr: float = 0.01,
    device: str = 'cpu'
) -> Dict:
    """
    Train separate local models (no federation) for baseline comparison.
    
    Returns per-client final accuracy.
    """
    from copy import deepcopy
    from tqdm import tqdm
    
    print(f"\nTraining local-only models for {len(client_loaders)} clients")
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for client_id, (train_loader, test_loader) in tqdm(client_loaders.items(), desc="Local Training"):
        local_model = deepcopy(model).to(device)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        
        for _ in range(num_epochs):
            local_model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = local_model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        local_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = local_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total if total > 0 else 0
        results[client_id] = {'accuracy': acc, 'samples': total}
        print(f"Client {client_id}: {acc:.2f}%")
    
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\nLocal-only average accuracy: {avg_acc:.2f}%")
    
    return {'per_client': results, 'average_accuracy': avg_acc}


def run_baseline_comparison(
    model: nn.Module,
    client_loaders: Dict,
    num_rounds: int = 50,
    device: str = 'cpu',
    save_dir: str = "results/federated"
) -> Dict:
    """
    Run all baseline algorithms and return comparison results.
    """
    from copy import deepcopy
    
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. MAML (Federated Meta-Learning)
    print("\n[1/4] Running Federated MAML...")
    maml_history = simulate_federated_maml(
        deepcopy(model), client_loaders, num_rounds=num_rounds, device=device, save_dir=save_dir
    )
    results['MAML'] = {'accuracy': maml_history['train_acc'][-1], 'history': maml_history}
    
    # 2. FedAvg
    print("\n[2/4] Running FedAvg...")
    fedavg_history = simulate_fedavg(
        deepcopy(model), client_loaders, num_rounds=num_rounds, device=device, save_dir=save_dir
    )
    results['FedAvg'] = {'accuracy': fedavg_history['train_acc'][-1], 'history': fedavg_history}
    
    # 3. FedProx
    print("\n[3/4] Running FedProx...")
    fedprox_history = simulate_fedprox(
        deepcopy(model), client_loaders, num_rounds=num_rounds, mu=0.01, device=device, save_dir=save_dir
    )
    results['FedProx'] = {'accuracy': fedprox_history['train_acc'][-1], 'history': fedprox_history}
    
    # 4. Local-only
    print("\n[4/4] Running Local-only training...")
    local_results = simulate_local_only(
        deepcopy(model), client_loaders, num_epochs=num_rounds, device=device
    )
    results['Local'] = {'accuracy': local_results['average_accuracy'], 'per_client': local_results['per_client']}
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for alg, res in results.items():
        print(f"{alg:12s}: {res['accuracy']:.2f}%")
    
    return results
