"""General utilities shared across algorithms and training loops.

Use this package for stable cross-cutting types, config dataclasses, Flax
TrainState helpers, and small update utilities. Avoid adding experiment-specific
train logic here.
"""

from il.utils.config import TrainingConfig
from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput, StepRecord

__all__ = [
    "ControllerId",
    "GateDecision",
    "GateReason",
    "PolicyOutput",
    "StepRecord",
    "TrainingConfig",
]
