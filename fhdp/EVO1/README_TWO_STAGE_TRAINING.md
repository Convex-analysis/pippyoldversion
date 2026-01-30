# EVO-1 Two-Stage Training Strategy

This document explains the implementation of the two-stage training strategy for EVO-1 as described in the original paper.

## Background

The EVO-1 paper proposes a two-stage training approach to effectively integrate pretrained vision-language models with action experts:

### Stage 1: Action Expert Alignment
- **Goal**: Align randomly initialized action expert weights with multimodal embedding space
- **Method**: Freeze the entire vision-language backbone, train only action expert and integration module
- **Benefit**: Prevents noisy gradients from damaging pretrained features

### Stage 2: Full-scale Fine-Tuning  
- **Goal**: Joint refinement of all components
- **Method**: Unfreeze all components for end-to-end training
- **Benefit**: Deeper integration and better adaptation to specific tasks

## Implementation

### Configuration

Enable two-stage training in your configuration:

```python
from EVO1.utils.config import EVO1DrivingConfig

config = EVO1DrivingConfig()
config.training.use_stage_training = True
config.training.stage1_rounds = 50      # Rounds for Stage 1
config.training.stage2_rounds = 50      # Rounds for Stage 2  
config.training.stage1_lr = 1e-4       # Learning rate for Stage 1
config.training.stage2_lr = 5e-5       # Learning rate for Stage 2
config.training.aggregation_rounds = 100 # Total rounds
```

### Key Features

#### 1. Automatic Stage Management
The trainer automatically handles stage transitions:
```python
# In FederatedEVO1Trainer
def _check_stage_transition(self, round_idx: int):
    if round_idx == self.config.training.stage1_rounds:
        logging.info("Transitioning from Stage 1 to Stage 2")
        self._initialize_training_stage(round_idx)
```

#### 2. Parameter Freezing/Unfreezing
Model components are automatically frozen/unfrozen:
```python
# Stage 1: Freeze backbone, train action expert
def set_stage1_mode(self):
    for param in self.vl_embedder.parameters():
        param.requires_grad = False  # Freeze backbone
    for param in self.action_head.parameters():
        param.requires_grad = True   # Train action expert

# Stage 2: Unfreeze all
def set_stage2_mode(self):
    for param in self.parameters():
        param.requires_grad = True   # Train all
```

#### 3. Stage-Aware Aggregation
Only trainable parameters are aggregated:
```python
def aggregate_client_updates(self, client_updates):
    for param_name in client_updates[first_client]:
        if not self._is_stage1_trainable_param(param_name):
            continue  # Skip frozen parameters in Stage 1
```

## Usage Examples

### Separated Stage Training (Recommended for Limited GPU Resources)

#### Train Stage 1 Only
```bash
# Default settings (50 rounds, 4 clients, batch size 4)
python3 fhdp/EVO1/examples/stage1_training.py

# Custom settings
python3 fhdp/EVO1/examples/stage1_training.py \
  --rounds 30 \
  --clients 2 \
  --batch-size 2 \
  --gpu cuda
```

#### Train Stage 2 Only (with Stage 1 checkpoint)
```bash
# After Stage 1 completes, automatically find checkpoint
python3 fhdp/EVO1/examples/stage2_training.py --auto-resume

# Or specify checkpoint explicitly
python3 fhdp/EVO1/examples/stage2_training.py \
  --resume ./outputs/evo1_stage1/checkpoints/stage_1_model_final.pt

# Custom settings
python3 fhdp/EVO1/examples/stage2_training.py \
  --rounds 30 \
  --clients 2 \
  --batch-size 2 \
  --gpu cuda
```

#### Sequential Training (Both Stages)
```bash
# Train both stages sequentially
python3 fhdp/EVO1/examples/sequential_training.py --mode both

# Train only Stage 1
python3 fhdp/EVO1/examples/sequential_training.py --mode stage1

# Train only Stage 2 (finds Stage 1 checkpoint automatically)
python3 fhdp/EVO1/examples/sequential_training.py --mode stage2 --auto-resume
```

#### Test Trained Models
```bash
# Test Stage 1 model
python3 fhdp/EVO1/examples/test_model.py \
  --model ./outputs/evo1_stage1/checkpoints/stage_1_model_final.pt \
  --speed-test

# Test Stage 2 model
python3 fhdp/EVO1/examples/test_model.py \
  --model ./outputs/evo1_stage2/checkpoints/stage_2_model_final.pt \
  --speed-test --real-data
```

### Integrated Two-Stage Training
```bash
# Original integrated approach
python3 fhdp/EVO1/examples/evo1_stage1_jetson_training.py
```

### Custom Configuration
```python
# Custom two-stage setup
config.training.use_stage_training = True
config.training.stage1_rounds = 30
config.training.stage2_rounds = 70
config.training.stage1_lr = 2e-4   # Higher for faster alignment
config.training.stage2_lr = 1e-5   # Lower for fine-tuning
```

## Training Progress

### Stage 1 Logging
```
[STAGE1] Freezing vision-language backbone, training only action expert and integration
[STAGE1] Trainable parameters: 12,345,678/123,456,789 (10.0%)
[CLIENT_TRAINER] client_0 - Round 0 - Stage 1: Training action expert only
[AGGREGATION] Stage 1 (action expert only) - Aggregated 45 parameters
```

### Stage Transition
```
[TRANSITION] Transitioning from Stage 1 to Stage 2 at round 50
[STAGE2] Unfreezing all components for full-scale fine-tuning
[STAGE2] Trainable parameters: 123,456,789/123,456,789 (100.0%)
```

### Stage 2 Logging
```
[CLIENT_TRAINER] client_0 - Round 50 - Stage 2: Full fine-tuning
[AGGREGATION] Stage 2 (full model) - Aggregated 256 parameters
```

## Performance Benefits

### Computational Efficiency
- **Stage 1**: Only ~10% of parameters trained (faster, less memory)
- **Stage 2**: Full training on aligned model (more stable)

### Training Stability
- Prevents catastrophic forgetting of pretrained features
- Gradual alignment of action expert
- Reduced gradient noise in early training

### Expected Performance
- Better final accuracy than single-stage training
- Faster convergence in Stage 2
- More robust to hyperparameter changes

## Configuration Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_stage_training` | Enable two-stage strategy | `True` | `True` |
| `stage1_rounds` | Rounds for Stage 1 | `50` | `30-50` |
| `stage2_rounds` | Rounds for Stage 2 | `50` | `50-100` |
| `stage1_lr` | Learning rate Stage 1 | `1e-4` | `1e-4` |
| `stage2_lr` | Learning rate Stage 2 | `5e-5` | `5e-5` |

## Best Practices

### 1. Learning Rate Selection
- Stage 1: Higher LR (1e-4) for faster alignment
- Stage 2: Lower LR (5e-5) for stable fine-tuning

### 2. Round Allocation
- Start with 30-50 rounds for Stage 1
- Use remaining rounds for Stage 2
- Monitor validation loss to adjust if needed

### 3. Model Size Considerations
- Large models (>1B parameters): Benefit most from two-stage
- Small models: May not need two-stage approach

### 4. Monitoring
- Watch for stage transition logs
- Monitor trainable parameter count changes
- Compare loss curves between stages

## Troubleshooting

### Common Issues

1. **Stage Transition Not Working**
   - Ensure `use_stage_training = True`
   - Check `stage1_rounds` > 0
   - Verify proper round counting

2. **Memory Issues in Stage 2**
   - Reduce batch size for Stage 2
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Slow Convergence**
   - Increase Stage 1 rounds
   - Adjust learning rates
   - Check parameter freezing

### Debugging
```python
# Check current stage
current_stage = "Stage 1" if round_idx < config.stage1_rounds else "Stage 2"

# Check trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
```

## GPU Resource Management

### Benefits of Separated Training

#### Limited GPU Resources
- **Stage 1 Only**: Train action expert with minimal memory usage
- **Stage 2 Later**: Fine-tune when more resources available
- **Testing Intermediate**: Validate Stage 1 performance before committing to Stage 2

#### Resource Optimization
```bash
# For limited GPU memory (≤8GB)
python3 fhdp/EVO1/examples/stage1_training.py --clients 2 --batch-size 2
python3 fhdp/EVO1/examples/stage2_training.py --auto-resume --clients 2 --batch-size 2

# For medium GPU memory (16GB)  
python3 fhdp/EVO1/examples/stage1_training.py --clients 4 --batch-size 4
python3 fhdp/EVO1/examples/stage2_training.py --auto-resume --clients 4 --batch-size 6

# For high GPU memory (>24GB)
python3 fhdp/EVO1/examples/stage1_training.py --clients 8 --batch-size 8
python3 fhdp/EVO1/examples/stage2_training.py --auto-resume --clients 8 --batch-size 8
```

### Memory Usage Comparison

| Stage | Trainable Parameters | Memory Usage | Training Speed |
|--------|-------------------|---------------|----------------|
| Stage 1 | ~10% of total | Low | Fast |
| Stage 2 | 100% of total | High | Moderate |

### Testing Intermediate Models

You can test your Stage 1 model before proceeding to Stage 2:

```bash
# Test Stage 1 performance
python3 fhdp/EVO1/examples/test_model.py \
  --model ./outputs/evo1_stage1/checkpoints/stage_1_model_final.pt \
  --speed-test --real-data

# If performance is good, proceed to Stage 2
python3 fhdp/EVO1/examples/stage2_training.py --auto-resume

# If not, tune Stage 1 hyperparameters
python3 fhdp/EVO1/examples/stage1_training.py --rounds 30 --stage1-lr 2e-4
```

## References

- EVO-1 Paper: "EVO-1: Empowering Vision-Language Models for Embodied AI"
- Two-stage training strategy: Section 3.2 of the original paper
- Implementation based on federated learning adaptation