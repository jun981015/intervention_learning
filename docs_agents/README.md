# Agent Documentation

This directory is for coding agents such as Codex and Claude.

Human-facing project notes live in `docs/` and are written in Korean. Agent-facing
instructions live here and should be written in English for clearer automated
code work.

## Files

- `INSTALL.md`: Conda environment setup, editable install, and validation commands.
- `STATUS_2026-05-18.md`: Older implementation, decisions, validation status, and remaining work.
- `STATUS_2026-05-21.md`: Current DAgger v0, pretrained loading, replay, and logger snapshot.
- `PROJECT_BRIEF.md`: Short project brief and document map.
- `SCOPE.md`: Scope, constraints, out-of-scope items, and current scaffold.
- `PIPELINE.md`: Online intervention step logic and demo/intervention buffer routing.
- `DAGGER_BASELINE.md`: DAgger baseline with learner rollout and expert relabeling.
- `PRETRAINED_POLICIES.md`: Available pretrained expert/learner artifacts and loading examples.
- `REPLAY_AND_UPDATES.md`: Replay schema, n-step backup, UTD, and target Q aggregation.
- `NETWORKS.md`: Shared MLP options and algorithm-specific network defaults.
- `JAX_FLAX_GUIDE.md`: JAX/Flax mental model, `TrainState`, and gradient-flow rules for agents.
- `LOGGING_AND_METRICS.md`: Interval logging behavior and next metric TODOs.
- `EXTENSIBILITY_REVIEW_2026-05-21.md`: Hardcoded defaults/prefixes and extensibility risks.
- `CODE_REVIEW_2026-05-26.md`: Current code-review findings, resolved/open status, and next coding order.
- `IMPLEMENTATION_PLAN.md`: Completed work, validation commands, next steps, and pending risks.

## Agent Rules

- Treat `/home/junhyeong/repos/qc` and `/home/junhyeong/repos/qc_base` as references only.
- Do not merge full QC-FQL, FQL, or BT logic into this project unless explicitly requested.
- Diffusion/flow-matching code is allowed only as a BC policy component.
- Keep the v0 pipeline simple: Robomimic Square, RLPD/SAC or BCFlow learner, RLPD/SAC checkpoint expert, random gate plus `expert_q_gap`, `horizon_length=1` by default.
- Preserve the separation between learner action, expert action, and executed action in replay.
- Do not commit generated logs, videos, replay buffers, checkpoints, W&B files, or experiment outputs.
- Keep `PROJECT_BRIEF.md` as a short navigation page. Put detailed notes in task-specific docs.

