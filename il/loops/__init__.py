"""Rollout and training-loop orchestration.

Loops connect envs, policies, gates, buffers, and trainable algos. Import concrete
helpers from their modules, e.g. `il.loops.rollout` or `il.loops.train_loop`. Keeping
this package init import-light avoids circular imports between train loops and
evaluation helpers.
"""

__all__: list[str] = []
