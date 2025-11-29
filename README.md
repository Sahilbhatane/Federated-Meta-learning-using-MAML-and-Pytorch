# Federated Meta Learning for Health Monitoring

Privacy-preserving personalized health monitoring system using federated meta-learning on wearable device data.

## Overview

This project combines Federated Learning and Meta-Learning to create a system that can generalize across multiple users while adapting rapidly to individual physiological patterns, all without sharing raw health data.

## Key Features

- Federated learning architecture with privacy preservation
- Meta-learning (MAML) for fast personalization
- Non-IID data partitioning across simulated clients
- Wearable device health data analysis
- Differential privacy integration
- Interactive visualization dashboard

## Tech Stack

- PyTorch 2.0+
- Flower (Federated Learning)
- **learn2learn** (Meta-Learning) **← We use learn2learn, NOT higher**
- Opacus (Differential Privacy)
- TensorBoard (Training Visualization)
- Hugging Face Datasets

> **Note**: This project specifically uses `learn2learn` for MAML implementation. If you encounter installation issues, see `docs/learn2learn_setup.md`.

## Installation

```bash
# Quick install (all dependencies)
pip install -r requirements.txt
```

## Quick Start

1. Explore the dataset:
```bash
jupyter notebook notebooks/phase2_data_exploration.ipynb
```

2. Train federated model (Phase 3+):
```bash
python -m src.federated.server
```

3. Launch TensorBoard (Phase 4+):
```bash
tensorboard --logdir=results/tensorboard
```

## Project Structure

```
Federated Meta Learning/
├── notebooks/           # Jupyter notebooks for analysis
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Neural network architectures
│   ├── federated/      # Flower client/server
│   └── utils/          # Metrics and visualization
├── configs/            # Configuration files
└── results/            # Experiment outputs
```

## Dataset

**Source**: `SahilBhatane/Federated_Meta-learning_on_wearable_devices` (Hugging Face)

**Specifications**:
- 200 samples: 140 train, 60 test
- 4 users with heterogeneous health patterns
- Features: Heart Rate, Blood Pressure, Temperature, SpO2, Respiratory Rate, Battery
- Target: Health Status (Healthy/Unhealthy)
- Non-IID distribution by user (natural data heterogeneity)

## Documentation

See `GUIDE.md` for detailed setup and usage instructions.

See `Planning.md` for architecture and research documentation.

## Development Status

- Phase 1: Planning (done)
- Phase 2: Dataset & Foundation (done)
- Phase 3: Implementation (done) (63.26% accuracy achieved)
- Phase 4: Evaluation ← Currently working
- Phase 5: Optimization

## License

MIT License
