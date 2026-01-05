"""
Phase 5: Hyperparameter Tuning for Federated Meta-Learning

Provides grid search, random search, and optional Optuna integration
for tuning MAML and federated learning hyperparameters.
"""

import itertools
import random
import json
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime

# Optional Optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class HyperparameterTuner:
    """
    Hyperparameter tuning for Federated MAML.
    
    Supports grid search, random search, and Optuna-based Bayesian optimization.
    """
    
    def __init__(
        self,
        model_fn: Callable[..., nn.Module],
        train_fn: Callable[..., Dict[str, float]],
        param_space: Dict[str, List[Any]],
        metric: str = 'accuracy',
        maximize: bool = True,
        save_dir: str = 'results/tuning'
    ):
        """
        Args:
            model_fn: Function that creates a model given hyperparameters
            train_fn: Function that trains and returns metrics (must accept **kwargs)
            param_space: Dictionary mapping param names to lists of values to try
            metric: Metric name to optimize
            maximize: If True, maximize metric; if False, minimize
            save_dir: Directory to save tuning results
        """
        self.model_fn = model_fn
        self.train_fn = train_fn
        self.param_space = param_space
        self.metric = metric
        self.maximize = maximize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf') if maximize else float('inf')
    
    def _is_better(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.maximize:
            return score > self.best_score
        return score < self.best_score
    
    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate a set of hyperparameters."""
        print(f"\nEvaluating: {params}")
        
        try:
            model = self.model_fn(**params)
            metrics = self.train_fn(model=model, **params)
            score = metrics.get(self.metric, 0.0)
            
            result = {
                'params': params,
                'metrics': metrics,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            if self._is_better(score):
                self.best_score = score
                self.best_params = params.copy()
                print(f"  New best! {self.metric}={score:.4f}")
            else:
                print(f"  {self.metric}={score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"  Error: {e}")
            return float('-inf') if self.maximize else float('inf')
    
    def grid_search(self) -> Dict[str, Any]:
        """
        Exhaustive grid search over all parameter combinations.
        
        Returns:
            Dictionary with best parameters and score
        """
        print("="*60)
        print("GRID SEARCH")
        print("="*60)
        
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        print(f"Total combinations: {total_combinations}")
        
        for i, combo in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, combo))
            print(f"\n[{i+1}/{total_combinations}]", end="")
            self._evaluate(params)
        
        self._save_results('grid_search')
        return {'best_params': self.best_params, 'best_score': self.best_score}
    
    def random_search(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Random search over parameter space.
        
        Args:
            n_trials: Number of random configurations to try
        
        Returns:
            Dictionary with best parameters and score
        """
        print("="*60)
        print(f"RANDOM SEARCH ({n_trials} trials)")
        print("="*60)
        
        for i in range(n_trials):
            params = {
                name: random.choice(values)
                for name, values in self.param_space.items()
            }
            print(f"\n[{i+1}/{n_trials}]", end="")
            self._evaluate(params)
        
        self._save_results('random_search')
        return {'best_params': self.best_params, 'best_score': self.best_score}
    
    def optuna_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Bayesian optimization using Optuna.
        
        Args:
            n_trials: Number of Optuna trials
        
        Returns:
            Dictionary with best parameters and score
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not installed. Falling back to random search.")
            return self.random_search(n_trials)
        
        print("="*60)
        print(f"OPTUNA BAYESIAN OPTIMIZATION ({n_trials} trials)")
        print("="*60)
        
        def objective(trial):
            params = {}
            for name, values in self.param_space.items():
                if all(isinstance(v, int) for v in values):
                    params[name] = trial.suggest_int(name, min(values), max(values))
                elif all(isinstance(v, float) for v in values):
                    params[name] = trial.suggest_float(name, min(values), max(values))
                else:
                    params[name] = trial.suggest_categorical(name, values)
            
            score = self._evaluate(params)
            return score if self.maximize else -score
        
        direction = 'maximize' if self.maximize else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        self._save_results('optuna_search')
        return {'best_params': self.best_params, 'best_score': self.best_score}
    
    def _save_results(self, method: str):
        """Save tuning results to JSON."""
        output = {
            'method': method,
            'metric': self.metric,
            'maximize': self.maximize,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = self.save_dir / f"tuning_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")


def grid_search(
    model_fn: Callable,
    train_fn: Callable,
    param_grid: Dict[str, List],
    metric: str = 'accuracy',
    maximize: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for grid search.
    
    Example:
        results = grid_search(
            model_fn=lambda hidden_dims, dropout: HealthMonitorNet(8, hidden_dims, 4, dropout),
            train_fn=lambda model, inner_lr, num_rounds: train_maml(model, inner_lr, num_rounds),
            param_grid={
                'hidden_dims': [[32, 16], [64, 32], [128, 64]],
                'dropout': [0.1, 0.2, 0.3],
                'inner_lr': [0.001, 0.01, 0.1],
                'num_rounds': [30, 50]
            }
        )
    """
    tuner = HyperparameterTuner(model_fn, train_fn, param_grid, metric, maximize)
    return tuner.grid_search()


def random_search(
    model_fn: Callable,
    train_fn: Callable,
    param_space: Dict[str, List],
    n_trials: int = 20,
    metric: str = 'accuracy',
    maximize: bool = True
) -> Dict[str, Any]:
    """Convenience function for random search."""
    tuner = HyperparameterTuner(model_fn, train_fn, param_space, metric, maximize)
    return tuner.random_search(n_trials)


# Default hyperparameter search space for Federated MAML
DEFAULT_PARAM_SPACE = {
    'inner_lr': [0.001, 0.005, 0.01, 0.05, 0.1],
    'outer_lr': [0.0001, 0.0005, 0.001, 0.005],
    'inner_steps': [1, 3, 5, 10],
    'num_rounds': [25, 50, 100],
    'batch_size': [4, 8, 16],
    'hidden_dims': [[16, 8], [32, 16], [64, 32]],
    'dropout': [0.1, 0.2, 0.3]
}
