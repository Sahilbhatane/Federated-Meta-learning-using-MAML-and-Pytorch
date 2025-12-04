"""
Phase 4 Comprehensive Test Suite
Tests all modules, configurations, and identifies issues for GitHub.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

issues_found = []

def record_issue(title: str, description: str, severity: str = "bug"):
    """Record an issue for later GitHub export"""
    issues_found.append({
        'title': title,
        'description': description,
        'severity': severity
    })
    print(f"[ISSUE] {severity.upper()}: {title}")

def test_imports():
    """Test all module imports"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML"),
    ]
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            record_issue(
                f"Missing dependency: {name}",
                f"The package `{module}` is not installed.\n\n**Error:** {e}\n\n**Fix:** `pip install {module}`",
                "bug"
            )
    
    # Optional dependencies
    optional = [
        ("opacus", "Opacus (Differential Privacy)"),
        ("tensorboard", "TensorBoard"),
    ]
    
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name} (optional)")
        except ImportError:
            record_issue(
                f"Optional dependency not installed: {name}",
                f"The optional package `{module}` is not installed. Some features will be disabled.\n\n**Fix:** `pip install {module}`",
                "enhancement"
            )
        except Exception as e:
            record_issue(
                f"Optional dependency error: {name}",
                f"The package `{module}` failed to import (possibly Python version incompatibility).\n\n**Error:** {e}",
                "enhancement"
            )
    
    # Flower - special handling due to Python 3.14 compatibility issues
    try:
        import flwr
        print(f"  ✓ Flower (Federated Learning) (optional)")
    except (ImportError, TypeError) as e:
        record_issue(
            "Flower (flwr) compatibility issue",
            f"Flower federated learning library has compatibility issues with this Python version.\n\n**Error:** {type(e).__name__}: {e}\n\n**Workaround:** Use Python 3.11 or 3.12 for full Flower support. The simulation functions in `flower_server.py` work without actual Flower networking.",
            "enhancement"
        )
    
    # Project modules
    print("\n  Project modules:")
    project_modules = [
        ("src.models.base_model", "HealthMonitorNet"),
        ("src.data.loader", "Data loaders"),
        ("src.federated.maml_trainer", "MAML Trainer"),
        ("src.federated.flower_server", "Flower Server"),
        ("src.federated.flower_client", "Flower Client"),
        ("src.utils.metrics", "Metrics"),
        ("src.utils.visualization", "Visualization"),
    ]
    
    for module, name in project_modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except Exception as e:
            record_issue(
                f"Module import error: {name}",
                f"Failed to import `{module}`.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
                "bug"
            )

def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 2: Configuration")
    print("="*60)
    
    import yaml
    config_path = project_root / "configs" / "config.yaml"
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Config loaded from {config_path}")
        
        # Validate required keys
        required_keys = ['federated', 'model', 'training', 'meta_learning', 'privacy', 'data', 'logging']
        for key in required_keys:
            if key not in config:
                record_issue(
                    f"Missing config section: {key}",
                    f"The configuration file is missing the `{key}` section.",
                    "bug"
                )
            else:
                print(f"  ✓ Config section: {key}")
        
        # Validate model dimensions match
        if 'model' in config:
            input_dim = config['model'].get('input_dim', 8)
            num_features = len(config.get('data', {}).get('features', []))
            if num_features > 0 and input_dim != num_features:
                record_issue(
                    "Config mismatch: input_dim vs feature count",
                    f"Model `input_dim` ({input_dim}) doesn't match the number of features ({num_features}) in config.",
                    "bug"
                )
        
    except FileNotFoundError:
        record_issue(
            "Configuration file not found",
            f"Expected config at `{config_path}` but file doesn't exist.",
            "bug"
        )
    except yaml.YAMLError as e:
        record_issue(
            "Invalid YAML in configuration",
            f"Failed to parse `config.yaml`.\n\n**Error:** {e}",
            "bug"
        )

def test_data_loading():
    """Test data loading and preprocessing"""
    print("\n" + "="*60)
    print("TEST 3: Data Loading")
    print("="*60)
    
    try:
        import pandas as pd
        from datasets import load_dataset
        
        # Try Hugging Face first
        try:
            dataset = load_dataset("SahilBhatane/Federated_Meta-learning_on_wearable_devices")
            df = pd.DataFrame(dataset['train'])
            print(f"  ✓ Dataset loaded from Hugging Face: {len(df)} samples")
        except Exception as e:
            # Fallback to local parquet
            parquet_path = project_root / "data" / "train.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                print(f"  ✓ Dataset loaded from local parquet: {len(df)} samples")
            else:
                record_issue(
                    "Dataset loading failed",
                    f"Could not load dataset from Hugging Face or local parquet.\n\n**Error:** {e}",
                    "bug"
                )
                return
        
        # Check columns
        print(f"  ✓ Columns: {list(df.columns)}")
        
        # Check for target column
        target_col = "Target_Blood_Pressure"
        if target_col not in df.columns:
            record_issue(
                "Target column missing from dataset",
                f"Expected column `{target_col}` not found in dataset. Available: {list(df.columns)}",
                "bug"
            )
        
        # Check for Sensor_ID (user partitioning)
        if 'Sensor_ID' not in df.columns:
            record_issue(
                "Sensor_ID column missing",
                "The `Sensor_ID` column is needed for user-based partitioning but is not present.",
                "bug"
            )
        else:
            users = df['Sensor_ID'].unique()
            print(f"  ✓ Users found: {len(users)} ({users})")
        
    except ImportError as e:
        record_issue(
            "Dataset library not installed",
            f"Failed to import datasets library.\n\n**Fix:** `pip install datasets`",
            "bug"
        )

def test_model():
    """Test model initialization and forward pass"""
    print("\n" + "="*60)
    print("TEST 4: Model")
    print("="*60)
    
    try:
        import torch
        from src.models.base_model import HealthMonitorNet
        
        model = HealthMonitorNet(input_dim=8, hidden_dims=[32, 16], num_classes=4, dropout=0.2)
        print(f"  ✓ Model created: {model.__class__.__name__}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Parameters: {num_params}")
        
        # Test forward pass
        x = torch.randn(4, 8)
        y = model(x)
        print(f"  ✓ Forward pass: input {x.shape} -> output {y.shape}")
        
        if y.shape != (4, 4):
            record_issue(
                "Model output shape mismatch",
                f"Expected output shape (4, 4) but got {y.shape}.",
                "bug"
            )
        
    except Exception as e:
        record_issue(
            "Model initialization failed",
            f"Could not create or run HealthMonitorNet.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
            "bug"
        )

def test_maml_trainer():
    """Test MAML trainer initialization"""
    print("\n" + "="*60)
    print("TEST 5: MAML Trainer")
    print("="*60)
    
    try:
        import torch
        from src.models.base_model import HealthMonitorNet
        from src.federated.maml_trainer import MAMLTrainer
        
        model = HealthMonitorNet(input_dim=8, hidden_dims=[32, 16], num_classes=4)
        
        # Test basic initialization
        trainer = MAMLTrainer(model, inner_lr=0.01, meta_lr=0.001, inner_steps=3)
        print("  ✓ MAMLTrainer initialized")
        
        # Test with DP enabled (should warn if Opacus not installed)
        trainer_dp = MAMLTrainer(model, dp_enabled=True, dp_epsilon=1.0)
        print(f"  ✓ MAMLTrainer with DP: enabled={trainer_dp.dp_enabled}")
        
        # Test with TensorBoard
        tb_dir = project_root / "results" / "tensorboard_test"
        trainer_tb = MAMLTrainer(model, tensorboard_dir=str(tb_dir))
        if trainer_tb.tb_writer:
            print("  ✓ TensorBoard logging enabled")
            trainer_tb.close()
        else:
            record_issue(
                "TensorBoard logging not available",
                "TensorBoard writer could not be initialized. Install tensorboard for logging.",
                "enhancement"
            )
        
    except Exception as e:
        record_issue(
            "MAML Trainer initialization failed",
            f"Could not initialize MAMLTrainer.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
            "bug"
        )

def test_metrics():
    """Test metrics module"""
    print("\n" + "="*60)
    print("TEST 6: Metrics")
    print("="*60)
    
    try:
        import numpy as np
        from src.utils.metrics import (
            calculate_metrics,
            calculate_comprehensive_metrics,
            compute_confusion_matrix,
            compute_roc_curves,
            aggregate_metrics
        )
        
        # Test basic metrics
        preds = np.array([0, 1, 2, 3, 0, 1])
        targets = np.array([0, 1, 2, 2, 0, 0])
        
        basic = calculate_metrics(preds, targets)
        print(f"  ✓ Basic metrics: {basic}")
        
        comprehensive = calculate_comprehensive_metrics(preds, targets, num_classes=4)
        print(f"  ✓ Comprehensive metrics: accuracy={comprehensive['accuracy']:.2f}%")
        
        cm = compute_confusion_matrix(preds, targets, num_classes=4)
        print(f"  ✓ Confusion matrix shape: {cm.shape}")
        
        # Test aggregation
        client_metrics = [{'accuracy': 80.0}, {'accuracy': 70.0}, {'accuracy': 90.0}]
        agg = aggregate_metrics(client_metrics)
        print(f"  ✓ Aggregated: {agg}")
        
    except Exception as e:
        record_issue(
            "Metrics module error",
            f"Error in metrics functions.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
            "bug"
        )

def test_visualization():
    """Test visualization module"""
    print("\n" + "="*60)
    print("TEST 7: Visualization")
    print("="*60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        from src.utils.visualization import (
            TensorBoardLogger,
            plot_training_history,
            plot_confusion_matrix,
            plot_baseline_comparison
        )
        
        print("  ✓ Visualization functions imported")
        
        # Test TensorBoard logger
        logger = TensorBoardLogger(log_dir=str(project_root / "results" / "tb_test"))
        if logger.writer:
            logger.log_scalar("test/value", 0.5, 1)
            logger.close()
            print("  ✓ TensorBoardLogger working")
        else:
            print("  ⚠ TensorBoardLogger not available (optional)")
        
    except Exception as e:
        record_issue(
            "Visualization module error",
            f"Error in visualization module.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
            "bug"
        )

def test_baseline_algorithms():
    """Test baseline algorithm implementations"""
    print("\n" + "="*60)
    print("TEST 8: Baseline Algorithms")
    print("="*60)
    
    try:
        from src.federated.flower_server import (
            simulate_federated_maml,
            simulate_fedavg,
            simulate_fedprox,
            simulate_local_only
        )
        
        print("  ✓ All baseline algorithms imported")
        print("  ✓ simulate_federated_maml")
        print("  ✓ simulate_fedavg")
        print("  ✓ simulate_fedprox")
        print("  ✓ simulate_local_only")
        
    except Exception as e:
        record_issue(
            "Baseline algorithms import error",
            f"Could not import baseline algorithms.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
            "bug"
        )

def test_data_loaders():
    """Test data loader creation"""
    print("\n" + "="*60)
    print("TEST 9: Data Loaders")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        from datasets import load_dataset
        from src.data.loader import WearableDataset, partition_by_user, create_client_loaders
        
        # Load data
        try:
            dataset = load_dataset("SahilBhatane/Federated_Meta-learning_on_wearable_devices")
            df = pd.DataFrame(dataset['train'])
        except:
            df = pd.DataFrame({
                'Sensor_ID': ['S1']*35 + ['S2']*35 + ['S3']*35 + ['S4']*35,
                'Temperature (°C)': np.random.randn(140),
                'Systolic_BP (mmHg)': np.random.randn(140),
                'Diastolic_BP (mmHg)': np.random.randn(140),
                'Heart_Rate (bpm)': np.random.randn(140),
                'SpO2 (%)': np.random.randn(140),
                'Respiratory_Rate (breaths/min)': np.random.randn(140),
                'Pulse_Rate (bpm)': np.random.randn(140),
                'Device_Battery_Level (%)': np.random.randn(140),
                'Target_Blood_Pressure': np.random.randint(0, 4, 140)
            })
            print("  ⚠ Using synthetic data for loader test")
        
        feature_cols = [
            'Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)',
            'Heart_Rate (bpm)', 'SpO2 (%)', 'Respiratory_Rate (breaths/min)',
            'Pulse_Rate (bpm)', 'Device_Battery_Level (%)'
        ]
        label_col = 'Target_Blood_Pressure'
        
        # Check feature columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            record_issue(
                "Missing feature columns in dataset",
                f"The following feature columns are missing: {missing}",
                "bug"
            )
            return
        
        if label_col not in df.columns:
            record_issue(
                "Missing label column in dataset",
                f"Label column `{label_col}` not found.",
                "bug"
            )
            return
        
        # Create loaders
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        X = scaler.fit_transform(df[feature_cols].values)
        y = label_encoder.fit_transform(df[label_col].values)
        
        print(f"  ✓ Features shape: {X.shape}")
        print(f"  ✓ Labels shape: {y.shape}, classes: {len(label_encoder.classes_)}")
        
        # Test partition
        client_partitions = partition_by_user(df, user_col='Sensor_ID')
        print(f"  ✓ Partitioned into {len(client_partitions)} clients")
        
        for cid, indices in client_partitions.items():
            print(f"    Client {cid}: {len(indices)} samples")
        
    except Exception as e:
        record_issue(
            "Data loader error",
            f"Error creating data loaders.\n\n**Error:** {e}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```",
            "bug"
        )

def test_notebook_syntax():
    """Check notebook for common issues"""
    print("\n" + "="*60)
    print("TEST 10: Notebook Check")
    print("="*60)
    
    import json
    
    notebook_path = project_root / "notebooks" / "federated_maml_training.ipynb"
    
    if not notebook_path.exists():
        record_issue(
            "Training notebook not found",
            f"Expected notebook at `{notebook_path}` but file doesn't exist.",
            "bug"
        )
        return
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print(f"  ✓ Notebook loaded: {len(nb.get('cells', []))} cells")
        
        # Check for common issues in code cells
        code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
        
        for i, cell in enumerate(code_cells):
            source = ''.join(cell.get('source', []))
            
            # Check for hardcoded paths
            if 'C:\\' in source or '/home/' in source:
                record_issue(
                    f"Hardcoded path in notebook cell {i+1}",
                    "Notebook contains hardcoded absolute paths which may not work on other systems.",
                    "enhancement"
                )
            
            # Check for feature leakage
            if 'Target_Blood_Pressure' in source and 'feature_cols' in source:
                if 'Target_Blood_Pressure' in source.split('feature_cols')[1].split('\n')[0]:
                    record_issue(
                        "Potential feature leakage in notebook",
                        "Target column may be included in feature list.",
                        "bug"
                    )
        
        print("  ✓ Notebook syntax check passed")
        
    except json.JSONDecodeError as e:
        record_issue(
            "Invalid notebook JSON",
            f"Could not parse notebook as JSON.\n\n**Error:** {e}",
            "bug"
        )

def generate_issues_document():
    """Generate GitHub issues document"""
    print("\n" + "="*60)
    print("GENERATING ISSUES DOCUMENT")
    print("="*60)
    
    if not issues_found:
        print("  No issues found! :)")
        return
    
    output_path = project_root / "ISSUES.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# GitHub Issues for Contributors\n\n")
        f.write("This document contains issues identified during Phase 4 testing.\n")
        f.write("Contributors can pick up these issues and submit PRs.\n\n")
        f.write("---\n\n")
        
        bugs = [i for i in issues_found if i['severity'] == 'bug']
        enhancements = [i for i in issues_found if i['severity'] == 'enhancement']
        
        if bugs:
            f.write("## Bugs\n\n")
            for i, issue in enumerate(bugs, 1):
                f.write(f"### Issue {i}: {issue['title']}\n\n")
                f.write(f"**Labels:** `bug`\n\n")
                f.write(f"{issue['description']}\n\n")
                f.write("---\n\n")
        
        if enhancements:
            f.write("## Enhancements\n\n")
            for i, issue in enumerate(enhancements, 1):
                f.write(f"### Issue {len(bugs)+i}: {issue['title']}\n\n")
                f.write(f"**Labels:** `enhancement`, `good first issue`\n\n")
                f.write(f"{issue['description']}\n\n")
                f.write("---\n\n")
    
    print(f"  ✓ Issues document saved to: {output_path}")
    print(f"  Total issues: {len(issues_found)} ({len(bugs)} bugs, {len(enhancements)} enhancements)")

def main():
    print("="*60)
    print("PHASE 4 COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    test_imports()
    test_config()
    test_data_loading()
    test_model()
    test_maml_trainer()
    test_metrics()
    test_visualization()
    test_baseline_algorithms()
    test_data_loaders()
    test_notebook_syntax()
    
    generate_issues_document()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if issues_found:
        print(f"  Found {len(issues_found)} issues")
        for issue in issues_found:
            severity = "🐛" if issue['severity'] == 'bug' else "✨"
            print(f"  {severity} {issue['title']}")
    else:
        print("  All tests passed! ✓")
    
    return len([i for i in issues_found if i['severity'] == 'bug'])

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
