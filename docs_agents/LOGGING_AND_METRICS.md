# Logging and Metrics Notes

This document is for coding agents continuing work on `intervention_learning`.
The human-facing version is `docs/LOGGING_AND_METRICS.md`.

## Current Behavior

`il/logger/logger.py` implements `MetricLogger`.

The logger records scalar metrics every environment step but writes only once per logging interval:

- `MetricLogger.record(metrics, step, force_flush=False)` accumulates scalar metrics.
- At `log_interval`, it flushes one row to JSONL, CSV, stdout, and optionally W&B.
- Losses, gradients, and numeric diagnostics are averaged over the interval.
- Routing event counts are summed over the interval, e.g. `routing/demo_added`.
- State-like counters keep the last value: replay sizes, episode count, step, throughput, recent env stats, and `*_total` counters.
- Each flushed row includes `train/log_interval_records`.
- Eval metrics use `log_immediate()` so they do not pollute the train accumulator.

## Important Files

- `il/logger/logger.py`
- `il/loops/train_loop.py`
- `docs/LOGGING_AND_METRICS.md`
- `docs/STATUS_2026-05-21.md`

## Next Metrics To Add

High-priority:

- `update/num_updates`
- `update/skip_count`
- `update/env_time_seconds`
- `update/sample_time_seconds`
- `update/update_time_seconds`
- `batch/source_online_fraction`
- `batch/source_demo_fraction`
- `batch/source_intervention_fraction`
- `batch/terminal_fraction`
- `batch/timeout_fraction`
- `batch/mask_mean`
- `action/learner_expert_l2_mean`
- `action/learner_expert_l2_max`
- `action/clip_fraction`
- `action/executed_variance`
- `action/learner_variance`
- `action/expert_variance`
- `action/policy_entropy_mean`
- `state/norm_mean`
- `state/running_mean_shift`
- `state/running_std_shift`
- `state/out_of_dataset_z_fraction`
- configurable CSV write frequency / append policy
- `gate/intervention_rate`
- `gate/expert_execute_rate`
- `gate/score_mean`

Medium-priority:

- `episode/success_length_mean`
- `episode/failure_length_mean`
- `episode/timeout_rate`
- `model/actor_param_norm`
- `model/grad_to_param_ratio`
- BCFlow chunk-index diagnostics.
- RLPD Q / alpha / log-prob diagnostics.

## Rules

- Do not write metrics every step to disk.
- The logging interval row is the main unit for JSONL, W&B, and stdout.
- CSV write frequency should be configurable; avoid rewriting a large CSV every flush in long runs.
- Keep stdout compact; put detailed metrics in JSONL/CSV/W&B.
- Keep metric prefixes stable: `train/`, `env/`, `routing/`, `batch/`, `action/`, `gate/`, `episode/`, `model/`, `eval/`.
