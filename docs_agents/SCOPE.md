# Project Scope

## Purpose

Build a clean, standalone intervention learning project derived from useful QC
components without inheriting unrelated FQL, QC-FQL, diffusion-RL, or BT
experiments.

## Reference Repositories

- Experimental QC repo: `/home/junhyeong/repos/qc`
- Clean QC reference: `/home/junhyeong/repos/qc_base`
- This project: `/home/junhyeong/repos/intervention_learning`

Treat QC repos as references only. Copy or adapt only the minimum code needed for
this project.

## In Scope For v0

- Environment: `square-mh-low_dim`
- Learner: RLPD/SAC-style online learner
- Expert: restored RLPD/SAC checkpoint policy
- Gate: random probability gate and expert-Q gap intervention gate
- Replay: online/demo/intervention buffers, executed action, learner proposal, expert proposal, controller id, gate metadata, log-probs
- Mixed sampling: configurable ratios across `online`, `intervention`, and `demo`
- N-step backup: expose `sample_sequence(batch_size, sequence_length, discount)`
- Frame stack: keep the parameter but default to `1`
- Critic ensemble size: configurable through `num_qs`, default `2`
- Update-to-data ratio: configurable through `utd_ratio`, default `1`
- TD target Q aggregation: configurable through `target_q_agg`, default `"min"`
- TD target critic count: fixed to the first 2 target critics
- Action chunking: ignored for v0; assume `horizon_length=1`

## Out Of Scope For v0

- FQL
- QC-FQL
- BT model
- OGBench/cube tasks
- Human keyboard/UI intervention
- Action chunking support
- New gate families based on uncertainty, disagreement, learned classifiers, or human input without an explicit design pass

Diffusion/flow-matching actors are in scope only as BC policy components. Do not
bring full FQL/QC-FQL or BT logic unless explicitly requested.

## Implementation Constraints

- Do not commit checkpoints, replay buffers, videos, logs, W&B files, or generated experiment artifacts.
- Gradient clipping is disabled by default. Enable it only through explicit config.
- Layer normalization is allowed inside simple MLP actor/critic networks.
- Do not reintroduce FiLM, BT, or value-auxiliary branches for v0.

## Current Scaffold

- `il/gating/`: controller gate interface, random gate, and expert-Q gap gate
- `il/policies/`: minimal policy protocol shared by learner and expert
- `il/buffers/`: replay buffers, transition schema, and episode routing
- `il/datasets/`: offline dataset loaders and dataset transforms, currently reserved
- `il/algo/rl/rlpd.py`: simple QC-base-style RLPD/SAC agent
- `il/algo/bc/flow.py`: BC-only flow-matching actor, not full FQL/QC-FQL
- `il/algo/bc/mlp.py`: deterministic MLP BC actor
- `il/networks/`: network blocks adapted from `qc_base`
- `il/distributions/`: action distributions adapted from `qc_base`
- `il/utils/`: shared dataclasses/enums, top-level training config, Flax train state utilities, UTD batch sampling
- `il/loops/`: rollout action selection, training loop, and update runner
- `il/evaluation/`: learner-only evaluation helper
- `il/logger/`: interval metric logger
- `il/train.py`: recipe-driven train/build entrypoint
