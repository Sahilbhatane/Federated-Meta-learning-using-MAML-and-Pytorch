"""
Phase 5: Scalability Testing for Federated Meta-Learning

Simulates federated learning with varying numbers of clients (10-100+)
to test system scalability and performance under load.
"""

import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class ScalabilityResult:
    """Result from a scalability test run."""
    num_clients: int
    total_time: float
    time_per_round: float
    memory_peak_mb: float
    final_accuracy: float
    final_loss: float
    convergence_round: Optional[int]


class ScalabilityTester:
    """
    Tests federated MAML scalability with varying client counts.
    """
    
    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        save_dir: str = 'results/scalability'
    ):
        """
        Args:
            model_fn: Function that creates a fresh model instance
            save_dir: Directory to save test results
        """
        self.model_fn = model_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ScalabilityResult] = []
    
    def generate_synthetic_data(
        self,
        num_clients: int,
        samples_per_client: int = 35,
        input_dim: int = 8,
        num_classes: int = 4
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Generate synthetic client data for scalability testing.
        
        Args:
            num_clients: Number of clients to simulate
            samples_per_client: Samples per client
            input_dim: Feature dimension
            num_classes: Number of classes
        
        Returns:
            Dictionary mapping client_id to data dict
        """
        clients_data = {}
        
        for client_id in range(num_clients):
            np.random.seed(42 + client_id)
            torch.manual_seed(42 + client_id)
            
            client_class = client_id % num_classes
            X = torch.randn(samples_per_client, input_dim)
            X[:, client_class % input_dim] += 2.0
            
            y = torch.randint(0, num_classes, (samples_per_client,))
            y[:samples_per_client // 2] = client_class
            
            clients_data[client_id] = {'X': X, 'y': y}
        
        return clients_data
    
    def simulate_federated_round(
        self,
        model: nn.Module,
        clients_data: Dict[int, Dict[str, torch.Tensor]],
        inner_lr: float = 0.01,
        inner_steps: int = 3,
        outer_lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Simulate one round of federated MAML training.
        
        Args:
            model: Global model
            clients_data: Client data dictionary
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner adaptation steps
            outer_lr: Outer loop learning rate
        
        Returns:
            Dictionary with round metrics
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        client_updates = []
        
        for client_id, data in clients_data.items():
            X, y = data['X'], data['y']
            
            adapted_model = self._clone_model(model)
            adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
            
            n_support = len(X) // 2
            X_support, y_support = X[:n_support], y[:n_support]
            X_query, y_query = X[n_support:], y[n_support:]
            
            for _ in range(inner_steps):
                adapted_optimizer.zero_grad()
                out = adapted_model(X_support)
                loss = criterion(out, y_support)
                loss.backward()
                adapted_optimizer.step()
            
            with torch.no_grad():
                query_out = adapted_model(X_query)
                query_loss = criterion(query_out, y_query)
                preds = query_out.argmax(dim=1)
                correct += (preds == y_query).sum().item()
                total += len(y_query)
                total_loss += query_loss.item()
            
            client_delta = {}
            for (name, global_param), (_, local_param) in zip(
                model.named_parameters(), adapted_model.named_parameters()
            ):
                client_delta[name] = local_param.data - global_param.data
            client_updates.append(client_delta)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                avg_delta = torch.stack([u[name] for u in client_updates]).mean(dim=0)
                param.data += outer_lr * avg_delta
        
        return {
            'loss': total_loss / len(clients_data),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    def run_scalability_test(
        self,
        client_counts: List[int] = [4, 10, 25, 50, 100],
        num_rounds: int = 20,
        samples_per_client: int = 35,
        input_dim: int = 8,
        num_classes: int = 4,
        convergence_threshold: float = 0.55
    ) -> List[ScalabilityResult]:
        """
        Run scalability tests with varying client counts.
        
        Args:
            client_counts: List of client counts to test
            num_rounds: Training rounds per test
            samples_per_client: Samples per client
            input_dim: Feature dimension
            num_classes: Number of classes
            convergence_threshold: Accuracy threshold for convergence
        
        Returns:
            List of ScalabilityResult objects
        """
        print("="*60)
        print("SCALABILITY TEST")
        print("="*60)
        print(f"Client counts: {client_counts}")
        print(f"Rounds per test: {num_rounds}")
        print()
        
        results = []
        
        for num_clients in client_counts:
            print(f"\n{'='*40}")
            print(f"Testing with {num_clients} clients...")
            print(f"{'='*40}")
            
            clients_data = self.generate_synthetic_data(
                num_clients, samples_per_client, input_dim, num_classes
            )
            
            model = self.model_fn()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            convergence_round = None
            final_acc = 0.0
            final_loss = 0.0
            
            for round_idx in range(num_rounds):
                round_metrics = self.simulate_federated_round(model, clients_data)
                final_acc = round_metrics['accuracy']
                final_loss = round_metrics['loss']
                
                if convergence_round is None and final_acc >= convergence_threshold:
                    convergence_round = round_idx + 1
                
                if (round_idx + 1) % 5 == 0:
                    print(f"  Round {round_idx + 1}/{num_rounds}: "
                          f"acc={final_acc:.4f}, loss={final_loss:.4f}")
            
            total_time = time.time() - start_time
            
            if torch.cuda.is_available():
                memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                import psutil
                memory_peak = psutil.Process().memory_info().rss / (1024 ** 2)
            
            result = ScalabilityResult(
                num_clients=num_clients,
                total_time=total_time,
                time_per_round=total_time / num_rounds,
                memory_peak_mb=memory_peak,
                final_accuracy=final_acc,
                final_loss=final_loss,
                convergence_round=convergence_round
            )
            results.append(result)
            
            print(f"\n  Results for {num_clients} clients:")
            print(f"    Total time: {total_time:.2f}s")
            print(f"    Time per round: {total_time/num_rounds:.3f}s")
            print(f"    Peak memory: {memory_peak:.1f} MB")
            print(f"    Final accuracy: {final_acc:.4f}")
            print(f"    Convergence round: {convergence_round or 'N/A'}")
        
        self.results = results
        self._save_results()
        self._print_summary()
        
        return results
    
    def _save_results(self):
        """Save results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'num_clients': r.num_clients,
                    'total_time': r.total_time,
                    'time_per_round': r.time_per_round,
                    'memory_peak_mb': r.memory_peak_mb,
                    'final_accuracy': r.final_accuracy,
                    'final_loss': r.final_loss,
                    'convergence_round': r.convergence_round
                }
                for r in self.results
            ]
        }
        
        filename = self.save_dir / f"scalability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _print_summary(self):
        """Print summary table of results."""
        print("\n" + "="*70)
        print("SCALABILITY SUMMARY")
        print("="*70)
        print(f"{'Clients':>10} {'Time(s)':>10} {'Time/Rnd':>10} "
              f"{'Memory(MB)':>12} {'Accuracy':>10} {'Conv.Rnd':>10}")
        print("-"*70)
        
        for r in self.results:
            conv = str(r.convergence_round) if r.convergence_round else "N/A"
            print(f"{r.num_clients:>10} {r.total_time:>10.2f} {r.time_per_round:>10.3f} "
                  f"{r.memory_peak_mb:>12.1f} {r.final_accuracy:>10.4f} {conv:>10}")


def generate_synthetic_clients(
    num_clients: int,
    samples_per_client: int = 35,
    input_dim: int = 8,
    num_classes: int = 4
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Convenience function to generate synthetic client data.
    
    Each client has slightly different data distribution (non-IID).
    """
    tester = ScalabilityTester(lambda: None)
    return tester.generate_synthetic_data(
        num_clients, samples_per_client, input_dim, num_classes
    )
