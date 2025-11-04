# Phase 2 Insights for Phase 3 MAML Design

## Key Findings from Data Exploration

### 1. Data Distribution Characteristics

**Non-IID Heterogeneity**
- Label distribution variance: 83.84 (high heterogeneity)
- Each user has distinct health patterns
- Strong evidence for personalization need

**User/Client Statistics**
- 4 clients (Sensor_ID: 1, 2, 3, 4)
- Partition sizes: 30-42 samples per client
- Mean: 35 samples, Std: 4.4
- Small dataset regime - ideal for meta-learning

**Label Distribution Per User** (from visualization):
- User 1: Mixed labels across 4 classes (120, 130, 140, 150)
- User 2: Dominated by higher values (140, 150)
- User 3: More balanced across classes
- User 4: Strong presence of 140, 150 labels
- Confirms need for personalized models

### 2. Feature Analysis

**Input Features** (9 total):
- Patient_ID
- Temperature (°C)
- Systolic_BP (mmHg)
- Diastolic_BP (mmHg)
- Heart_Rate (bpm)
- Device_Battery_Level (%)
- Target_Blood_Pressure
- Target_Heart_Rate
- Battery_Level (%)

**Feature Correlations**:
- Strong correlation: Systolic_BP ↔ Diastolic_BP (0.94)
- Weak correlations elsewhere
- Features are mostly independent → good for learning

**Target Variable**:
- Column: Target_Health_Status
- Classes: Healthy, Unhealthy (binary classification)
- Or: Target_Blood_Pressure (120, 130, 140, 150) - multi-class

### 3. Design Implications for MAML

**Architecture Requirements**:
1. Input size: 9 features (after preprocessing)
2. Small model needed (limited data per user)
3. Fast adaptation critical (30-42 samples per user)
4. Binary or multi-class classification head

**Meta-Learning Strategy**:
1. **Inner Loop** (personalization):
   - Use 5-10 support samples per user
   - 1-3 gradient steps for adaptation
   - Per-user fine-tuning on local data

2. **Outer Loop** (meta-training):
   - Aggregate across 4 clients
   - Learn initialization that generalizes
   - Handle high heterogeneity (variance 83.84)

**Few-Shot Learning Setup**:
- Support set: 5-10 samples (for adaptation)
- Query set: Remaining samples (for evaluation)
- K-shot: 5-way or 2-way classification
- Ideal for small per-user datasets

**Federated Considerations**:
1. Non-IID partitioning: Natural user-based split
2. Unbalanced clients: 30-42 samples (acceptable variance)
3. Privacy: Each user's data stays local
4. Communication: Share model parameters only

### 4. Recommended Hyperparameters

**Model Architecture**:
```python
input_dim = 9
hidden_dims = [32, 16]  # Small network for limited data
output_dim = 2  # Binary classification (or 4 for multi-class)
dropout = 0.2  # Regularization for small datasets
```

**MAML Parameters**:
```python
meta_lr = 0.001  # Outer loop (meta-learner)
inner_lr = 0.01  # Inner loop (adaptation)
inner_steps = 3  # Gradient steps per adaptation
first_order = False  # Use second-order MAML
```

**Training Setup**:
```python
num_clients = 4
num_rounds = 50  # Federated rounds
local_epochs = 5  # Per client
batch_size = 8  # Small due to limited data
```

**Few-Shot Configuration**:
```python
k_shot = 5  # Support samples per class
n_query = 10  # Query samples for evaluation
```

### 5. Expected Outcomes

**Success Metrics**:
1. Personalized accuracy > Global model accuracy
2. Fast adaptation: <5 gradient steps
3. Generalization: Good performance on new users
4. Privacy: No data leakage across clients

**Visualization Goals** (TensorBoard):
1. Meta-learning curves (meta-train/meta-val loss)
2. Per-client adaptation progress
3. Global vs personalized accuracy comparison
4. Convergence across federated rounds

### 6. Implementation Checklist

- [ ] Update data loader to use Sensor_ID for partitioning
- [ ] Create HealthMonitorNet with 9 input features
- [ ] Implement MAML with learn2learn library
- [ ] Setup Flower clients (4) with user data
- [ ] Configure few-shot splits (support/query)
- [ ] Add TensorBoard logging
- [ ] Test adaptation on held-out user
- [ ] Validate privacy preservation

---

**Next Steps**: Implement Phase 3 with these insights integrated into MAML architecture and federated setup.
