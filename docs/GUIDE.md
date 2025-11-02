# Federated Meta Learning - Project Guide

Last Updated: Phase 3 - ✓ In Progress (MAML Implementation Complete, Testing Pending)

---

## Project Overview

Federated meta-learning system for personalized health monitoring using wearable device data. Combines privacy-preserving federated learning with fast-adaptation meta-learning algorithms.

---

## Tech Stack

**Core Framework**
- Python 3.8+
- PyTorch 2.0+
- Flower 1.8+ (Federated Learning)
- **learn2learn** (Meta-Learning: MAML, Reptile) **← CRITICAL**

**⚠️ IMPORTANT**: This project uses `learn2learn`, NOT `higher`. See `docs/learn2learn_setup.md` for details.

**Data & ML**
- Hugging Face Datasets (cloud dataset access)
- NumPy, Pandas
- scikit-learn

**Privacy**
- Opacus (Differential Privacy)

**Visualization**
- Jupyter Notebook
- Matplotlib, Seaborn
- TensorBoard (training monitoring - Phase 4)

**Development**
- Docker (client simulation - Phase 3)

---

## Installation

**Step 1: Install PyTorch**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Step 2: Install learn2learn (CRITICAL)**
```bash
pip install learn2learn
```
If this fails, see `docs/learn2learn_setup.md` for troubleshooting.

**Step 3: Install Other Dependencies**
```bash
pip install flwr flwr-datasets
pip install opacus
pip install datasets huggingface_hub
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install jupyter notebook
pip install tensorboard
```

**Or install all at once:**
```bash
pip install -r requirements.txt
```

**Windows PyTorch DLL Error Fix**:
If you encounter DLL initialization errors on Windows:
```bash
pip uninstall torch torchvision -y
pip cache purge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Also ensure Visual C++ Redistributable is installed from: https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Dataset

**Source**: SahilBhatane/Federated_Meta-learning_on_wearable_devices (Hugging Face)

**Access**:
```python
from datasets import load_dataset
dataset = load_dataset("SahilBhatane/Federated_Meta-learning_on_wearable_devices")
```

**Characteristics**:
- Total samples: 200 (140 train, 60 test)
- Users: 4 different subjects
- Wearable metrics: Heart Rate, Blood Pressure, Temperature, SpO2, Respiratory Rate, Battery
- Target: Health Status (Healthy/Unhealthy)
- Non-IID partitioning by user ID (natural data heterogeneity)
- Perfect for federated learning with personalization

---

## Project Structure

```
Federated Meta Learning/
├── GUIDE.md                          # This file
├── Planning.md                       # Architecture & research planning
├── requirements.txt                  # Python dependencies
├── notebooks/
│   ├── phase2_data_exploration.ipynb # Data analysis & visualization
│   └── phase3_maml_training.ipynb    # Meta-learning implementation
├── src/
│   ├── data/
│   │   ├── loader.py                 # Dataset loading & partitioning
│   │   └── preprocessor.py           # Data preprocessing
│   ├── models/
│   │   ├── base_model.py             # Neural network architectures
│   │   └── maml.py                   # MAML implementation
│   ├── federated/
│   │   ├── client.py                 # Flower client implementation
│   │   └── server.py                 # Flower server & aggregation
│   └── utils/
│       ├── metrics.py                # Evaluation metrics
│       └── visualization.py          # Plotting utilities
├── configs/
│   └── config.yaml                   # Hyperparameters & settings
└── results/
    └── experiments/                  # Saved models & logs
```

---

## Running the Project

### Phase 2: Data Exploration (Completed)

```bash
cd "Federated Meta Learning"
jupyter notebook notebooks/phase2_data_exploration.ipynb
```

Run cells sequentially to:
1. Load dataset from Hugging Face
2. Explore data distributions
3. Visualize user heterogeneity
4. Test non-IID partitioning
5. Save processed metadata

**Outputs**:
- Visual analysis of data distributions
- User heterogeneity plots
- Non-IID partition validation
- Saved metadata in results/experiments/

---

## Development Phases

**Phase 1: Planning** ✓
- Architecture design
- Literature review
- Tech stack selection

**Phase 2: Dataset & Foundation** ✓
- Dataset integration from Hugging Face
- Data exploration & visualization in Jupyter notebook
- Non-IID partitioning implementation
- Data loaders and preprocessing modules
- Base model architecture (HealthMonitorNet)
- Utility functions for metrics and visualization

**Phase 3: MAML Implementation** ✓ COMPLETE
- MAML algorithm with learn2learn
- Flower client-server architecture
- Federated training simulation
- Per-client adaptation and evaluation
- Visualization of training dynamics

**Phase 4: Evaluation**
- Privacy mechanisms (Opacus)
- Baseline comparisons
- Metrics & visualization dashboard
- TensorBoard integration

**Phase 5: Optimization**
- Hyperparameter tuning
- Scalability testing
- Documentation finalization

---

## Key Commands

**Start Jupyter Notebook**:
```bash
jupyter notebook
```

**Run Federated Simulation** (Phase 3+):
```bash
python -m src.federated.server
```

**Launch TensorBoard** (Phase 4+):
```bash
tensorboard --logdir=results/tensorboard
```

---

## Phase 2 Implementation Details

**Phase 2 Modules**:
- `src/data/loader.py`: Dataset loading, client partitioning, few-shot splits ✓ Updated
- `src/data/preprocessor.py`: Data normalization and cleaning
- `src/models/base_model.py`: HealthMonitorNet (9→32→16→4 architecture) ✓ Updated
- `src/utils/metrics.py`: Evaluation metrics
- `src/utils/visualization.py`: Plotting functions

**Phase 3 Modules** ✓ NEW:
- `src/federated/maml_trainer.py`: MAML training logic with inner/outer loops
- `src/federated/flower_client.py`: Flower client for federated MAML
- `src/federated/flower_server.py`: Flower server with custom aggregation strategy
- `docs/phase2_insights.md`: Key findings from data exploration

**Configuration**:
- `configs/config.yaml`: Hyperparameters for federated training

**Notebooks**:
- `notebooks/data exploration.ipynb`: Complete Phase 2 analysis (27 cells tested)
- `notebooks/federated_maml_training.ipynb`: Phase 3 MAML training pipeline ✓ NEW

---

## Phase 3 Implementation Summary

**Completed:**
✓ Analyzed Phase 2 visualizations for design insights
✓ Updated model architecture (9 input features, small network for limited data)
✓ Implemented MAML trainer with learn2learn
✓ Created Flower federated components
✓ Built end-to-end training notebook
✓ Configured for 4 heterogeneous clients (30-42 samples each)

**Key Parameters:**
- Inner LR: 0.01 (adaptation)
- Meta LR: 0.001 (meta-update)
- Inner Steps: 3 (fast adaptation)
- Batch Size: 8 (small datasets)
- Rounds: 50 (federated)

**Next: Run Training** (Phase 3 Validation)

---

## Running Phase 3

**Jupyter Notebook** (Recommended):
```bash
cd notebooks
jupyter notebook federated_maml_training.ipynb
```

**Federated Server** (Full Deployment):
```bash
python -m src.federated.flower_server
```

---

## Notes

- All data stays local on clients (federated learning principle)
- Meta-learning enables fast personalization with <5 gradient steps
- Small dataset regime (30-42 samples) ideal for few-shot MAML
- High heterogeneity (variance 83.84) requires personalization
- Dataset: SahilBhatane/Federated_Meta-learning_on_wearable_devices

---

End of Guide
