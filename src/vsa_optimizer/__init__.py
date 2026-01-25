"""VSA Training Optimizer - Efficient training with hyperdimensional computing.

Training optimization toolkit for efficient large model training.

Why: Enables training of massive parameter models on consumer GPUs by:
1. VSA (Vector Symbolic Architecture) gradient compression
2. Ternary math acceleration for gradient operations
3. Predictive training with correction cycles
4. Memory-efficient gradient accumulation

This module can be extracted as a standalone training optimization toolkit.

Key techniques:
- Gradient prediction: Predict multiple steps, apply corrections
- VSA compression: Hyperdimensional computing for gradient approximation
- Ternary acceleration: {-1, 0, +1} operations for speedup
- Phase cycling: Full train → compressed predict → correction → repeat

Example usage:
    >>> from vsa_optimizer import PhaseTrainer, PhaseConfig
    >>> config = PhaseConfig(full_steps=10, predict_steps=40)
    >>> trainer = PhaseTrainer(model, optimizer, config)
    >>> for batch in dataloader:
    ...     stats = trainer.train_step(batch, compute_loss_fn)
    ...     print(f"Step {stats['total_step']}: loss={stats['loss']:.4f}, speedup={stats['speedup']:.2f}x")
"""

from vsa_optimizer.gradient_predictor import (
    GradientPredictor,
    PredictionConfig,
    PredictiveTrainer,
)
from vsa_optimizer.phase_trainer import (
    PhaseConfig,
    PhaseTrainer,
    TrainingPhase,
)
from vsa_optimizer.ternary_optimizer import (
    TernaryConfig,
    TernaryGradientAccumulator,
    TernaryOptimizer,
)
from vsa_optimizer.vsa_compression import (
    VSAConfig,
    VSAGradientCompressor,
    hyperdimensional_bind,
    hyperdimensional_bundle,
    ternary_quantize,
)

__version__ = "0.1.0"

__all__ = [
    # Gradient prediction
    "GradientPredictor",
    "PredictionConfig",
    "PredictiveTrainer",
    # Ternary optimization
    "TernaryConfig",
    "TernaryGradientAccumulator",
    "TernaryOptimizer",
    # VSA compression
    "VSAConfig",
    "VSAGradientCompressor",
    "hyperdimensional_bind",
    "hyperdimensional_bundle",
    "ternary_quantize",
    # Phase-based training
    "PhaseConfig",
    "PhaseTrainer",
    "TrainingPhase",
]
