"""Behavior-cloning algorithms.

BC agents train a policy from supervised action targets. They may be used for
offline BC, DAgger relabeling, or intervention data imitation. They should not
own environment rollout logic; loops decide which target action key to train on.
"""

from il.algo.bc.flow import BCFlowAgent, get_config as get_bc_flow_config
from il.algo.bc.mlp import BCMLPAgent, get_config as get_bc_mlp_config

__all__ = [
    "BCFlowAgent",
    "BCMLPAgent",
    "get_bc_flow_config",
    "get_bc_mlp_config",
]
