from __future__ import annotations

import jax

from il.buffers.mixed import MixedReplaySampler, MixedSamplingSpec
from il.utils.config import TrainingConfig


def sample_rl_update_batch(replay_buffer, config: TrainingConfig):
    """Sample a QC-style update batch with configurable update-to-data ratio."""
    sampler = replay_buffer
    if config.sampling_fractions is not None:
        sampler = MixedReplaySampler(replay_buffer, MixedSamplingSpec(config.sampling_fractions))
    batch = sampler.sample_sequence(
        config.batch_size * config.utd_ratio,
        sequence_length=config.horizon_length,
        discount=config.discount,
    )
    return jax.tree_util.tree_map(
        lambda x: x.reshape((config.utd_ratio, config.batch_size) + x.shape[1:]),
        batch,
    )
