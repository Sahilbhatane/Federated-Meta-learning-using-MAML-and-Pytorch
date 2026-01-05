"""
Phase 5: Model Compression for Federated Meta-Learning

Provides pruning, quantization, and knowledge distillation utilities
to reduce model size for edge/wearable device deployment.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import copy


class ModelCompressor:
    """
    Comprehensive model compression toolkit.
    
    Supports:
    - Magnitude-based pruning (structured and unstructured)
    - Dynamic quantization
    - Static quantization (with calibration)
    - Knowledge distillation helpers
    """
    
    def __init__(self, model: nn.Module, save_dir: str = 'results/compression'):
        """
        Args:
            model: Model to compress
            save_dir: Directory to save compressed models
        """
        self.original_model = model
        self.compressed_model: Optional[nn.Module] = None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.compression_stats: Dict[str, Any] = {}
    
    def _count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count total and non-zero parameters."""
        total = 0
        nonzero = 0
        for param in model.parameters():
            total += param.numel()
            nonzero += (param != 0).sum().item()
        return total, nonzero
    
    def _model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 ** 2)
    
    def prune(
        self,
        amount: float = 0.3,
        method: str = 'l1_unstructured',
        layers_to_prune: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Apply pruning to the model.
        
        Args:
            amount: Fraction of connections to prune (0.0 to 1.0)
            method: Pruning method ('l1_unstructured', 'random', 'ln_structured')
            layers_to_prune: List of layer names to prune (None = all linear layers)
        
        Returns:
            Pruned model
        """
        print(f"\nApplying {method} pruning (amount={amount})...")
        
        model = copy.deepcopy(self.original_model)
        
        pruning_methods = {
            'l1_unstructured': prune.l1_unstructured,
            'random': prune.random_unstructured,
            'ln_structured': lambda m, n, a: prune.ln_structured(m, n, a, n=2, dim=0)
        }
        
        if method not in pruning_methods:
            raise ValueError(f"Unknown pruning method: {method}")
        
        prune_fn = pruning_methods[method]
        
        modules_pruned = 0
        for name, module in model.named_modules():
            if layers_to_prune is not None and name not in layers_to_prune:
                continue
                
            if isinstance(module, nn.Linear):
                prune_fn(module, 'weight', amount)
                modules_pruned += 1
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
        
        total_before, _ = self._count_parameters(self.original_model)
        total_after, nonzero_after = self._count_parameters(model)
        actual_sparsity = 1 - (nonzero_after / total_after)
        
        self.compression_stats['pruning'] = {
            'method': method,
            'target_amount': amount,
            'actual_sparsity': actual_sparsity,
            'modules_pruned': modules_pruned,
            'params_before': total_before,
            'nonzero_after': nonzero_after
        }
        
        print(f"  Modules pruned: {modules_pruned}")
        print(f"  Target sparsity: {amount:.1%}")
        print(f"  Actual sparsity: {actual_sparsity:.1%}")
        print(f"  Non-zero params: {nonzero_after:,} / {total_after:,}")
        
        self.compressed_model = model
        return model
    
    def quantize(
        self,
        mode: str = 'dynamic',
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply quantization to the model.
        
        Args:
            mode: Quantization mode ('dynamic' or 'static')
            dtype: Target dtype for quantization
        
        Returns:
            Quantized model
        """
        print(f"\nApplying {mode} quantization to {dtype}...")
        
        model = copy.deepcopy(self.original_model)
        model.eval()
        
        size_before = self._model_size_mb(model)
        
        if mode == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=dtype
            )
        elif mode == 'static':
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            with torch.no_grad():
                # Infer input dimension for calibration from the first Linear layer
                input_dim = None
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        input_dim = module.in_features
                        break
                if input_dim is None:
                    raise ValueError(
                        "Unable to infer input dimension for static quantization calibration: "
                        "no nn.Linear layer with in_features found in the model."
                    )
                dummy_input = torch.randn(1, input_dim)
                for _ in range(10):
                    model(dummy_input)
            quantized_model = torch.quantization.convert(model)
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")
        
        size_after = self._model_size_mb(quantized_model)
        compression_ratio = size_before / size_after if size_after > 0 else 1.0
        
        self.compression_stats['quantization'] = {
            'mode': mode,
            'dtype': str(dtype),
            'size_before_mb': size_before,
            'size_after_mb': size_after,
            'compression_ratio': compression_ratio
        }
        
        print(f"  Size before: {size_before:.4f} MB")
        print(f"  Size after: {size_after:.4f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        self.compressed_model = quantized_model
        return quantized_model
    
    def distill(
        self,
        student_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        temperature: float = 4.0,
        alpha: float = 0.7,
        epochs: int = 10,
        lr: float = 0.01
    ) -> nn.Module:
        """
        Knowledge distillation from teacher (original) to student model.
        
        Args:
            student_model: Smaller student model to train
            train_loader: Training data loader
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss (1-alpha for hard targets)
            epochs: Training epochs
            lr: Learning rate
        
        Returns:
            Trained student model
        """
        print(f"\nKnowledge Distillation (temp={temperature}, alpha={alpha})...")
        
        teacher = self.original_model
        teacher.eval()
        student = student_model
        student.train()
        
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        criterion_hard = nn.CrossEntropyLoss()
        criterion_soft = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(epochs):
            total_loss = 0.0
            for X, y in train_loader:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = teacher(X)
                    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
                
                student_logits = student(X)
                student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)
                
                loss_hard = criterion_hard(student_logits, y)
                loss_soft = criterion_soft(student_soft, soft_targets) * (temperature ** 2)
                
                loss = alpha * loss_soft + (1 - alpha) * loss_hard
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={total_loss:.4f}")
        
        teacher_params, _ = self._count_parameters(teacher)
        student_params, _ = self._count_parameters(student)
        
        self.compression_stats['distillation'] = {
            'temperature': temperature,
            'alpha': alpha,
            'epochs': epochs,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'param_reduction': 1 - (student_params / teacher_params)
        }
        
        print(f"  Teacher params: {teacher_params:,}")
        print(f"  Student params: {student_params:,}")
        print(f"  Parameter reduction: {1 - (student_params / teacher_params):.1%}")
        
        self.compressed_model = student
        return student
    
    def save(self, filename: str = 'compressed_model.pt'):
        """Save compressed model and stats."""
        if self.compressed_model is None:
            raise ValueError("No compressed model to save. Run compression first.")
        
        model_path = self.save_dir / filename
        torch.save(self.compressed_model.state_dict(), model_path)
        
        stats_path = self.save_dir / f"{filename.replace('.pt', '_stats.json')}"
        with open(stats_path, 'w') as f:
            json.dump(self.compression_stats, f, indent=2, default=str)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Stats saved to: {stats_path}")
    
    def benchmark(
        self,
        test_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark original vs compressed model inference time.
        
        Args:
            test_input: Test input tensor
            num_runs: Number of inference runs
        
        Returns:
            Dictionary with timing results
        """
        import time
        
        if self.compressed_model is None:
            raise ValueError("No compressed model. Run compression first.")
        
        self.original_model.eval()
        self.compressed_model.eval()
        
        for _ in range(10):
            with torch.no_grad():
                self.original_model(test_input)
                self.compressed_model(test_input)
        
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                self.original_model(test_input)
        original_time = (time.time() - start) / num_runs
        
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                self.compressed_model(test_input)
        compressed_time = (time.time() - start) / num_runs
        
        speedup = original_time / compressed_time if compressed_time > 0 else 1.0
        
        results = {
            'original_time_ms': original_time * 1000,
            'compressed_time_ms': compressed_time * 1000,
            'speedup': speedup
        }
        
        print(f"\nInference Benchmark ({num_runs} runs):")
        print(f"  Original: {original_time*1000:.3f} ms")
        print(f"  Compressed: {compressed_time*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        return results


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = 'l1_unstructured'
) -> nn.Module:
    """Convenience function for pruning."""
    compressor = ModelCompressor(model)
    return compressor.prune(amount, method)


def quantize_model(
    model: nn.Module,
    mode: str = 'dynamic',
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """Convenience function for quantization."""
    compressor = ModelCompressor(model)
    return compressor.quantize(mode, dtype)
