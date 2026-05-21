"""Rollout and training-loop orchestration.

Loops connect envs, policies, gates, buffers, and trainable algos. They own the
data-flow decisions: what action is executed, what is stored, and when updates
run. Network definitions and low-level losses should stay in `algo`/`networks`.
"""

from il.loops.online import choose_action
from il.loops.recipe import run_train_loop

__all__ = ["choose_action", "run_train_loop"]
