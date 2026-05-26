"""Reinforcement-learning algorithms.

RL agents own critic/actor losses, optimizers, target-network updates, and
batch update APIs. Rollout code should call policies or loop helpers instead of
embedding environment interaction inside RL algorithm files.
"""

from il.algo.rl.rlpd import ACRLPDAgent, get_config as get_rlpd_config
from il.algo.rl.residual_rlpd import ResidualRLPDAgent, get_config as get_residual_rlpd_config
from il.algo.rl.residual_td3 import ResidualTD3Agent, get_config as get_residual_td3_config

__all__ = [
    "ACRLPDAgent",
    "ResidualRLPDAgent",
    "ResidualTD3Agent",
    "get_rlpd_config",
    "get_residual_rlpd_config",
    "get_residual_td3_config",
]
