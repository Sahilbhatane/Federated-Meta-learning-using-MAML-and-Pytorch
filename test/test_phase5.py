"""
Phase 5: Optimization Module Tests

Tests hyperparameter tuning, scalability testing, and model compression.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from datetime import datetime

ISSUES = []


class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self, input_dim=8, hidden_dims=[32, 16], num_classes=4, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def test_hyperparameter_tuning():
    """Test hyperparameter tuning module."""
    print("\n" + "="*50)
    print("TEST: Hyperparameter Tuning Module")
    print("="*50)
    
    try:
        from src.optimization.hyperparameter_tuning import (
            HyperparameterTuner, grid_search, random_search, DEFAULT_PARAM_SPACE
        )
        print("[PASS] Import successful")
        
        def model_fn(**kwargs):
            hidden = kwargs.get('hidden_dims', [32, 16])
            return SimpleNet(hidden_dims=hidden)
        
        def train_fn(model, **kwargs):
            model.eval()
            with torch.no_grad():
                X = torch.randn(20, 8)
                out = model(X)
            return {'accuracy': 0.5 + 0.1 * torch.rand(1).item(), 'loss': 0.5}
        
        param_space = {
            'hidden_dims': [[16, 8], [32, 16]],
            'inner_lr': [0.01, 0.1]
        }
        
        tuner = HyperparameterTuner(model_fn, train_fn, param_space, metric='accuracy')
        print("[PASS] HyperparameterTuner instantiation")
        
        result = tuner.random_search(n_trials=2)
        assert 'best_params' in result
        assert 'best_score' in result
        print(f"[PASS] Random search completed: best_score={result['best_score']:.4f}")
        
        print("[PASS] DEFAULT_PARAM_SPACE available with keys:", list(DEFAULT_PARAM_SPACE.keys())[:3], "...")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        ISSUES.append({
            'title': 'Hyperparameter tuning test failure',
            'description': str(e),
            'severity': 'high'
        })
        return False


def test_scalability():
    """Test scalability testing module."""
    print("\n" + "="*50)
    print("TEST: Scalability Testing Module")
    print("="*50)
    
    try:
        from src.optimization.scalability import (
            ScalabilityTester, ScalabilityResult, generate_synthetic_clients
        )
        print("[PASS] Import successful")
        
        clients_data = generate_synthetic_clients(num_clients=5, samples_per_client=20)
        assert len(clients_data) == 5
        assert 'X' in clients_data[0] and 'y' in clients_data[0]
        print("[PASS] generate_synthetic_clients works")
        
        model_fn = lambda: SimpleNet()
        tester = ScalabilityTester(model_fn, save_dir='results/scalability')
        print("[PASS] ScalabilityTester instantiation")
        
        results = tester.run_scalability_test(
            client_counts=[4, 8],
            num_rounds=3,
            samples_per_client=20
        )
        assert len(results) == 2
        assert isinstance(results[0], ScalabilityResult)
        print(f"[PASS] Scalability test completed: {len(results)} configurations tested")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        ISSUES.append({
            'title': 'Scalability test failure',
            'description': str(e),
            'severity': 'high'
        })
        return False


def test_compression():
    """Test model compression module."""
    print("\n" + "="*50)
    print("TEST: Model Compression Module")
    print("="*50)
    
    try:
        from src.optimization.compression import (
            ModelCompressor, prune_model, quantize_model
        )
        print("[PASS] Import successful")
        
        model = SimpleNet()
        compressor = ModelCompressor(model, save_dir='results/compression')
        print("[PASS] ModelCompressor instantiation")
        
        pruned = compressor.prune(amount=0.3, method='l1_unstructured')
        assert pruned is not None
        print("[PASS] Pruning completed")
        
        model2 = SimpleNet()
        compressor2 = ModelCompressor(model2)
        quantized = compressor2.quantize(mode='dynamic')
        assert quantized is not None
        print("[PASS] Dynamic quantization completed")
        
        pruned_convenience = prune_model(SimpleNet(), amount=0.2)
        assert pruned_convenience is not None
        print("[PASS] Convenience functions work")
        
        test_input = torch.randn(1, 8)
        benchmark = compressor.benchmark(test_input, num_runs=10)
        assert 'speedup' in benchmark
        print(f"[PASS] Benchmark completed: speedup={benchmark['speedup']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        ISSUES.append({
            'title': 'Compression test failure',
            'description': str(e),
            'severity': 'high'
        })
        return False


def test_integration():
    """Test Phase 5 integration with existing modules."""
    print("\n" + "="*50)
    print("TEST: Phase 5 Integration")
    print("="*50)
    
    try:
        from src.optimization import (
            HyperparameterTuner, ScalabilityTester, ModelCompressor
        )
        print("[PASS] All optimization modules import from package")
        
        from src.models.base_model import HealthMonitorNet
        model = HealthMonitorNet(8, [32, 16], 4)
        compressor = ModelCompressor(model)
        pruned = compressor.prune(amount=0.2)
        
        pruned.eval()
        test_input = torch.randn(1, 8)
        with torch.no_grad():
            out = pruned(test_input)
        assert out.shape == (1, 4)
        print("[PASS] Compression works with HealthMonitorNet")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        ISSUES.append({
            'title': 'Phase 5 integration failure',
            'description': str(e),
            'severity': 'medium'
        })
        return False


def generate_issues_md():
    """Generate ISSUES.md with any discovered issues."""
    if not ISSUES:
        print("\n[INFO] No issues found!")
        return
    
    issues_path = Path(__file__).parent.parent / 'ISSUES.md'
    
    existing_content = ""
    if issues_path.exists():
        with open(issues_path, 'r') as f:
            existing_content = f.read()
    
    new_issues = f"\n\n## Phase 5 Issues ({datetime.now().strftime('%Y-%m-%d')})\n\n"
    for i, issue in enumerate(ISSUES, 1):
        new_issues += f"### Issue P5-{i}: {issue['title']}\n"
        new_issues += f"- **Severity**: {issue['severity']}\n"
        new_issues += f"- **Description**: {issue['description']}\n\n"
    
    with open(issues_path, 'a') as f:
        f.write(new_issues)
    
    print(f"\n[INFO] Added {len(ISSUES)} issues to ISSUES.md")


def main():
    print("="*60)
    print("PHASE 5 OPTIMIZATION TESTS")
    print("="*60)
    
    results = {
        'hyperparameter_tuning': test_hyperparameter_tuning(),
        'scalability': test_scalability(),
        'compression': test_compression(),
        'integration': test_integration()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "[PASS]" if passed_flag else "[FAIL]"
        print(f"  {status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    generate_issues_md()
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
