"""Trainable algorithm implementations.

This package is for objects that can update parameters from batches:
loss functions, optimizer state, target-network updates, and `agent.update`.

Do not put environment rollout, gating decisions, W&B logging, or experiment
path logic here. Those belong in `loops` or the top-level train entrypoint.

Important boundary:
- `algo` is for learning.
- `policies` is for action sampling only, especially restored experts.
"""

from il.algo.bc import BCFlowAgent, BCMLPAgent, get_bc_flow_config, get_bc_mlp_config
from il.algo.rl import ACRLPDAgent, get_rlpd_config

__all__ = [
    "ACRLPDAgent",
    "BCFlowAgent",
    "BCMLPAgent",
    "get_bc_flow_config",
    "get_bc_mlp_config",
    "get_rlpd_config",
]
