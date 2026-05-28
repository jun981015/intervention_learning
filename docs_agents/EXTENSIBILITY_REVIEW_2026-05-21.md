# Extensibility Review 2026-05-21

Base git commit: `f1730d4` (`Initialize intervention learning scaffold`)

This is a read-only review of hardcoded defaults, prefixes, and assumptions that may limit future extensibility.
No code was changed for this review.

## Summary

The repo is a usable v0 research scaffold for DAgger-style relabeling and future intervention learning.
It is not yet a general-purpose algorithm framework. Most assumptions are acceptable for the current Robomimic Square + BCFlow learner + RLPD expert vertical slice.

## High-Priority Assumptions

- `il/builders/config.py::DEFAULT_RECIPE` is strongly tied to Square, BCFlow, RLPD, and `expert_actions`.
- Public YAML configs are converted into a legacy internal recipe by `new_schema_to_legacy_recipe()`.
- `replay.sampling` currently selects one primary sampling spec, preferring `bc` over `rl`; named multi-batch sampling is not fully wired.
- Update objective inference has been removed. Update specs now define target actor, replay source, sampling knobs, and optional `target_action_key`; the agent kind owns its loss in `agent.update(batch)`.
- `update_interval`, `updates_per_step`, `save_final`, and eval video fields are now consumed at runtime. Remaining mostly declarative fields include `keep_last` and `storage.store_*`.
- Env registry currently contains only `robomimic_lowdim`.
- Robomimic dataset paths and sparse success reward are hardcoded in `il/envs/robomimic_lowdim.py`.
- Current actor builder rejects image-only observations and supports low-dimensional policies only.
- Expert actors require `pretrained_path`; scripted/human/random experts need a separate expert provider abstraction.
- Activation fields in YAML are currently dropped by the config adapter.
- Eval is now split into generic `evaluate_policy(policy, env, ...)` and a TrainContext adapter; context eval still evaluates the learner policy rather than a gated controller.
- Update scheduling now respects initial collection, `update_interval`, and `updates_per_step`.
- Buffer-underfilled update skip depends on matching a `ValueError` string.
- Final checkpoint saving respects `save_final`; replay saving still follows the train config save flag.
- Stdout summary is tied to BCFlow metric names.
- Chunk policies execute only the first primitive action from a sampled chunk.
- RLPD target critic aggregation uses the first two critics only.
- Experiment scripts are Square-specific utilities; the main path should be `python -m il.train --config ...`.

## Suggested Order To Relax Assumptions

1. Run real-env residual+intervention gate smoke after the new wiring.
2. Make stdout logging metric selection algorithm-agnostic.
3. Support true named multi-batch replay sampling for RL+BC hybrid learners.
4. Add a second env registry entry and make dataset path/reward mode configurable.
5. Implement action chunk queues when chunk semantics become central.
6. Add image-policy network builders after lowdim replay/env paths are stable.

## Current Judgment

The scaffold is ready for DAgger relabeling baseline work and for adding new algorithms incrementally.
It should not yet be treated as a fully generic framework.
