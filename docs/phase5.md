# Phase 5: Optimization

## Overview
Phase 5 adds optimization utilities for production-ready federated meta-learning:
- **Hyperparameter Tuning**: Grid search, random search, and Optuna integration
- **Scalability Testing**: Simulate 10-100+ clients
- **Model Compression**: Pruning, quantization, and knowledge distillation

## Status: IN PROGRESS

## Components

### 1. Hyperparameter Tuning (`src/optimization/hyperparameter_tuning.py`)

#### HyperparameterTuner Class
```python
from src.optimization import HyperparameterTuner

tuner = HyperparameterTuner(
    model_fn=lambda hidden_dims: HealthMonitorNet(9, hidden_dims, 4),
    train_fn=lambda model, inner_lr: train_maml(model, inner_lr),
    param_space={
        'hidden_dims': [[16, 8], [32, 16], [64, 32]],
        'inner_lr': [0.001, 0.01, 0.1],
        'inner_steps': [1, 3, 5]
    },
    metric='accuracy',
    maximize=True
)

# Grid search (exhaustive)
result = tuner.grid_search()

# Random search (faster)
result = tuner.random_search(n_trials=20)

# Optuna Bayesian optimization (if installed)
result = tuner.optuna_search(n_trials=50)
```

#### Default Parameter Space
```python
from src.optimization.hyperparameter_tuning import DEFAULT_PARAM_SPACE

# Includes: inner_lr, outer_lr, inner_steps, num_rounds, batch_size, hidden_dims, dropout
```

### 2. Scalability Testing (`src/optimization/scalability.py`)

#### ScalabilityTester Class
```python
from src.optimization import ScalabilityTester
from src.models import HealthMonitorNet

tester = ScalabilityTester(
    model_fn=lambda: HealthMonitorNet(9, [32, 16], 4)
)

results = tester.run_scalability_test(
    client_counts=[4, 10, 25, 50, 100],
    num_rounds=20,
    samples_per_client=35
)

# Results include:
# - total_time, time_per_round
# - memory_peak_mb
# - final_accuracy
# - convergence_round
```

#### Generate Synthetic Clients
```python
from src.optimization import generate_synthetic_clients

clients_data = generate_synthetic_clients(
    num_clients=100,
    samples_per_client=35,
    input_dim=8,
    num_classes=4
)
```

### 3. Model Compression (`src/optimization/compression.py`)

#### ModelCompressor Class
```python
from src.optimization import ModelCompressor
from src.models import HealthMonitorNet

model = HealthMonitorNet(9, [32, 16], 4)
compressor = ModelCompressor(model)

# Pruning (reduce parameters)
pruned = compressor.prune(amount=0.3, method='l1_unstructured')

# Quantization (reduce precision)
quantized = compressor.quantize(mode='dynamic', dtype=torch.qint8)

# Benchmark
benchmark = compressor.benchmark(torch.randn(1, 9), num_runs=100)

# Save compressed model
compressor.save('compressed_model.pt')
```

#### Knowledge Distillation
```python
student = HealthMonitorNet(9, [16, 8], 4)  # Smaller model
trained_student = compressor.distill(
    student_model=student,
    train_loader=train_loader,
    temperature=4.0,
    alpha=0.7,
    epochs=10
)
```

#### Convenience Functions
```python
from src.optimization import prune_model, quantize_model

# Quick pruning
pruned = prune_model(model, amount=0.3)

# Quick quantization
quantized = quantize_model(model, mode='dynamic')
```

## Configuration

Add to `configs/config.yaml`:
```yaml
optimization:
  hyperparameter_search:
    method: 'random'  # grid, random, optuna
    n_trials: 20
    metric: 'accuracy'
  
  scalability:
    client_counts: [4, 10, 25, 50, 100]
    num_rounds: 20
    samples_per_client: 35
  
  compression:
    pruning_amount: 0.3
    quantization_mode: 'dynamic'
```

## Results

Results are saved to:
- `results/tuning/` - Hyperparameter search results (JSON)
- `results/scalability/` - Scalability test results (JSON)
- `results/compression/` - Compressed models and stats

## Testing

```bash
python test/test_phase5.py
```

## Dependencies

Core (required):
- PyTorch >= 2.0

Optional:
- `optuna` - For Bayesian hyperparameter optimization
- `psutil` - For memory profiling on CPU

## Checklist

- [x] Hyperparameter tuning (grid/random search)
- [x] Optional Optuna integration
- [x] Scalability testing (variable client counts)
- [x] Synthetic client data generation
- [x] Model pruning (L1 unstructured)
- [x] Dynamic quantization
- [x] Knowledge distillation
- [x] Inference benchmarking
- [x] Test suite
- [ ] Integration with training notebook
- [ ] Full scalability report (100+ clients)
