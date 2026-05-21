"""Reinforcement-learning algorithms.

RL agents own critic/actor losses, optimizers, target-network updates, and
batch update APIs. Rollout code should call policies or loop helpers instead of
embedding environment interaction inside RL algorithm files.
"""

from il.algo.rl.rlpd import ACRLPDAgent, get_config as get_rlpd_config

__all__ = ["ACRLPDAgent", "get_rlpd_config"]
