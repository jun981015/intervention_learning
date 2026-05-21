# Intervention Learning Project Brief

## Purpose

Build a clean, standalone intervention learning project derived from the useful
parts of the QC codebase without inheriting unrelated FQL/QC-FQL/diffusion/BT
experiments.

The immediate goal is an end-to-end online intervention pipeline for Robomimic
Square where both learner and expert actions are sampled at every environment
step, a gate chooses which controller acts, and replay stores all relevant
metadata.

## Document Map

- Current implementation and handoff snapshot: [STATUS_2026-05-18.md](STATUS_2026-05-18.md)
- Scope, constraints, and current scaffold: [SCOPE.md](SCOPE.md)
- Online intervention step logic and BC routing: [PIPELINE.md](PIPELINE.md)
- DAgger learner rollout and expert relabel baseline: [DAGGER_BASELINE.md](DAGGER_BASELINE.md)
- Pretrained policy artifacts and generic load examples: [PRETRAINED_POLICIES.md](PRETRAINED_POLICIES.md)
- Replay schema, n-step backup, UTD, and target Q aggregation: [REPLAY_AND_UPDATES.md](REPLAY_AND_UPDATES.md)
- Network defaults and MLP option policy: [NETWORKS.md](NETWORKS.md)
- Completed work, validation commands, and next implementation steps: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Environment setup: [INSTALL.md](INSTALL.md)

## Non-negotiable Constraints

- Treat `/home/junhyeong/repos/qc` and `/home/junhyeong/repos/qc_base` as references only.
- Do not commit checkpoints, replay buffers, videos, logs, W&B files, or experiment outputs.
- Do not merge full FQL, QC-FQL, or BT logic unless explicitly requested.
- Keep v0 simple: Robomimic Square, RLPD/SAC learner, RLPD/SAC expert, random gate, `horizon_length=1`.
- Preserve learner action, expert action, executed action, and gate metadata separately in replay.

## Current Status

Simulator-free intervention data flow is verified by `scripts/smoke_test.py`.
The current smoke covers gate/replay, intervention routing, mixed replay
sampling, RLPD update, RLPD checkpoint policy loading, BC MLP update, BC Flow
update, and BC Flow checkpoint policy loading.

The next blocker is real Robomimic online rollout support. Follow
[IMPLEMENTATION_PLAN.md#next-implementation-steps](IMPLEMENTATION_PLAN.md#next-implementation-steps)
for the next tasks.
