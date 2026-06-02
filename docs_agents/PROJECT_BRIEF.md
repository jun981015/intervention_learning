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

- Latest implementation and handoff snapshot: [STATUS_2026-05-28.md](STATUS_2026-05-28.md)
- Previous DAgger/pretrained/replay/logger snapshot: [STATUS_2026-05-21.md](STATUS_2026-05-21.md)
- Scope, constraints, and current scaffold: [SCOPE.md](SCOPE.md)
- Online intervention step logic and BC routing: [PIPELINE.md](PIPELINE.md)
- DAgger learner rollout and expert relabel baseline: [DAGGER_BASELINE.md](DAGGER_BASELINE.md)
- Pretrained policy artifacts and generic load examples: [PRETRAINED_POLICIES.md](PRETRAINED_POLICIES.md)
- Replay schema, n-step backup, UTD, and target Q aggregation: [REPLAY_AND_UPDATES.md](REPLAY_AND_UPDATES.md)
- Network defaults and MLP option policy: [NETWORKS.md](NETWORKS.md)
- Completed work, validation commands, and next implementation steps: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Active code-review follow-up: [CODE_REVIEW_2026-05-26.md](CODE_REVIEW_2026-05-26.md)
- Environment setup: [INSTALL.md](INSTALL.md)

## Collaboration Protocol

For non-trivial code changes, do not proceed silently through multiple design
steps. Propose a short plan first, implement only the agreed unit, report the
changed files and validation, then wait before continuing to the next unit.
Do not run training jobs, change config architecture, or do large refactors
without explicit user confirmation.

## Non-negotiable Constraints

- Treat `/home/junhyeong/repos/qc` and `/home/junhyeong/repos/qc_base` as references only.
- Do not commit checkpoints, replay buffers, videos, logs, W&B files, or experiment outputs.
- Do not merge full FQL, QC-FQL, or BT logic unless explicitly requested.
- Keep v0 simple: Robomimic Square, RLPD/SAC learner/expert, random gate plus `expert_q_gap` gate, `horizon_length=1` by default.
- Preserve learner action, expert action, executed action, and gate metadata separately in replay.
- Dataset adapters are explicit for offline demo/prefill canonicalization; do not implicitly copy `actions` to `expert_actions` without adapter semantics.
- Action chunk queue TODO: prefer `collections.deque` per learner/expert, allow different learner/expert horizons, canonical `full_action_chunk=(horizon, action_dim)`, and clear queues on controller switch.
- Horizon TODO: separate BC action chunk horizon from RL n-step TD horizon. Do not let one `horizon_length` implicitly control both losses once RL+BC mixed updates are added.

## Current Status

Simulator-free intervention data flow is verified by `scripts/smoke_test.py`.
The current code has a recipe-driven `python -m il.train` entrypoint, real
Robomimic Square smoke coverage for DAgger relabeling and `expert_q_gap`, and
residual RLPD/TD3 rollout/update paths. As of 2026-05-28, residual action
composition can also feed intervention gates: the gate sees the full
`clip(base + residual)` learner proposal, while replay keeps residual metadata
for both learner-selected and expert-selected steps. Simulator-free smoke still
covers gate/replay, intervention routing, mixed replay sampling, RLPD, BC MLP,
and BC Flow update paths.

The active work is no longer basic env wiring. Treat
`docs/CODE_REVIEW_2026-05-26.md` and the agent-facing
`CODE_REVIEW_2026-05-26.md` as the current code-quality backlog, and
`docs/IMPLEMENTATION_TODO.md` as the broader project backlog.
