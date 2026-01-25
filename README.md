# VSA Training Optimizer

Training optimization toolkit using Vector Symbolic Architecture (VSA), ternary math, and gradient prediction for efficient large model training on consumer GPUs.

## Features

- **VSA Gradient Compression**: Hyperdimensional computing for ~90% memory savings on gradient storage
- **Ternary Optimization**: `{-1, 0, +1}` gradient accumulation with stochastic rounding for unbiased estimates
- **Gradient Prediction**: Predict gradients from history, reducing compute by ~80% with periodic corrections
- **Phase-based Training**: Cycle between `FULL → PREDICT → CORRECT` phases for accelerated training

## Installation

```bash
pip install vsa-training-optimizer
```

Or from source:

```bash
git clone https://github.com/tzervas/vsa-training-optimizer.git
cd vsa-training-optimizer
pip install -e ".[dev]"
```

## Quick Start

### Phase-based Training (Recommended)

The `PhaseTrainer` orchestrates all optimization techniques automatically:

```python
import torch
from vsa_optimizer import PhaseTrainer, PhaseConfig

# Your model and optimizer
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Configure phase training
config = PhaseConfig(
    full_steps=10,      # Steps of full gradient computation
    predict_steps=40,   # Steps using predicted gradients
    correct_every=10,   # Correction frequency
)

trainer = PhaseTrainer(model, optimizer, config)

# Training loop
def compute_loss(model, batch):
    return model(batch["x"]).sum()

for batch in dataloader:
    stats = trainer.train_step(batch, compute_loss)
    print(f"Step {stats['total_step']}: loss={stats['loss']:.4f}, speedup={stats['speedup']:.2f}x")
```

### VSA Gradient Compression

Compress gradients for efficient storage and communication:

```python
from vsa_optimizer import VSAGradientCompressor, VSAConfig

# Create compressor
param_count = sum(p.numel() for p in model.parameters())
compressor = VSAGradientCompressor(param_count, VSAConfig(compression_ratio=0.1))

# After computing gradients
gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
shapes = {name: g.shape for name, g in gradients.items()}

# Compress (90% memory reduction)
compressed, metadata = compressor.compress(gradients)

# Later: decompress
reconstructed = compressor.decompress(compressed, metadata, shapes)
```

### Ternary Gradient Accumulation

Memory-efficient gradient accumulation using ternary representation:

```python
from vsa_optimizer import TernaryOptimizer, TernaryConfig

optimizer = torch.optim.AdamW(model.parameters())
ternary_opt = TernaryOptimizer(
    model, optimizer,
    TernaryConfig(accumulation_steps=8)
)

for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()

    if ternary_opt.step():  # Returns True when actual update happens
        print("Optimizer step performed")
```

### Gradient Prediction

Predict gradients to skip expensive backward passes:

```python
from vsa_optimizer import GradientPredictor, PredictionConfig

predictor = GradientPredictor(model, PredictionConfig(
    history_size=5,
    prediction_steps=4,
))

for batch in dataloader:
    if predictor.should_compute_full():
        loss.backward()
        predictor.record_gradient()
    else:
        predicted = predictor.predict_gradient()
        predictor.apply_predicted(predicted)

    optimizer.step()
```

## How It Works

### VSA Gradient Compression

Uses hyperdimensional computing principles:
- **Random Projection**: Johnson-Lindenstrauss lemma preserves gradient direction
- **Ternary Quantization**: Further compress to `{-1, 0, +1}` with scale factors
- **Bundling**: Combine multiple gradients into single hypervector

### Phase Training Cycle

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   FULL (N steps) ──► PREDICT (M steps) ───┐    │
│       ▲                     │              │    │
│       │                     ▼              │    │
│       └────── CORRECT (periodic) ◄────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
```

- **FULL**: Standard backpropagation with full gradient computation
- **PREDICT**: Use momentum-based gradient extrapolation (no backward pass)
- **CORRECT**: Compute full gradient and apply accumulated corrections

### Memory Savings

| Technique | Memory Reduction | Compute Reduction |
|-----------|-----------------|-------------------|
| VSA Compression | ~90% gradient storage | - |
| Ternary Accumulation | ~93% during accumulation | - |
| Gradient Prediction | - | ~80% backward passes |
| Combined (Phase Training) | ~90% | ~60-80% |

## Configuration Reference

### PhaseConfig

```python
@dataclass
class PhaseConfig:
    full_steps: int = 10        # Full training steps per cycle
    predict_steps: int = 40     # Predicted training steps per cycle
    correct_every: int = 10     # Correction frequency during predict
    adaptive_phases: bool = True  # Auto-adjust based on loss
    max_grad_norm: float = 1.0  # Gradient clipping
```

### VSAConfig

```python
@dataclass
class VSAConfig:
    dimension: int = 8192       # Hypervector dimension
    compression_ratio: float = 0.1  # Target compression
    use_ternary: bool = True    # Enable ternary quantization
    seed: int = 42              # Reproducibility
```

### TernaryConfig

```python
@dataclass
class TernaryConfig:
    accumulation_steps: int = 8     # Steps before optimizer update
    use_stochastic_rounding: bool = True  # Unbiased quantization
```

### PredictionConfig

```python
@dataclass
class PredictionConfig:
    history_size: int = 5       # Past gradients to keep
    prediction_steps: int = 4   # Steps to predict before correction
    momentum: float = 0.9       # Prediction momentum
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{vsa_training_optimizer,
  author = {Zervas, Tyler},
  title = {VSA Training Optimizer: Efficient Training with Hyperdimensional Computing},
  year = {2025},
  url = {https://github.com/tzervas/vsa-training-optimizer}
}
```

## References

- Kanerva (2009): Hyperdimensional Computing
- Rahimi et al. (2016): High-Dimensional Computing as a Nanoscalable Paradigm
- Johnson-Lindenstrauss Lemma for random projections
