# Implementation Plan


## Current Status Update - 2026-05-28

This file preserves the early implementation plan. For current work, use:

- `docs/IMPLEMENTATION_TODO.md` for the broader backlog.
- `docs_agents/STATUS_2026-05-28.md` for the latest agent-facing handoff.
- `docs/REAL_ENV_SMOKE_TESTS.md` for real Robomimic smoke results.
- `docs/CODE_REVIEW_2026-05-26.md` and `docs_agents/CODE_REVIEW_2026-05-26.md` for the active code-review follow-up.

The original Robomimic env wiring, recipe-driven `il.train` entrypoint, DAgger
relabel real-env smoke, expert-Q gap real-env smoke, residual RLPD/TD3 paths,
gate runtime Protocol cleanup, runtime update scheduling fields, generic eval
video helper, state-only dict observation mode, and residual+gate rollout wiring
are no longer pending scaffold tasks.

## Completed Scaffold

- Project scaffold and replay schema.
- Random gate and minimal policy interface.
- N-step `ReplayBuffer.sample_sequence()` smoke coverage.
- Minimal QC-base-style RLPD/SAC agent adapter.
- BC Flow actor adapter.
- BC MLP actor adapter.
- Function/class docstrings across `il/`.
- Simulator-free intervention routing smoke coverage.
- Mixed replay sampler for online/demo/intervention ratio sampling.
- Generic `RLPDPolicy.from_checkpoint()` loader.
- Generic `BCFlowPolicy.from_checkpoint()` loader.
- Checkpoint save/load smoke coverage for RLPD and BC Flow policies.
- DAgger learner-rollout plus expert-relabel helper.
- `target_action_key` support for BC Flow.

## Current Validation Commands

```bash
conda run -n il python -m compileall -q il scripts
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n il python scripts/smoke_test.py
```

Expected smoke output:

```text
gate/replay smoke ok
intervention routing smoke ok
mixed replay sampling smoke ok
dagger baseline smoke ok
rlpd smoke ok
rlpd policy checkpoint smoke ok
bc mlp smoke ok
bc flow smoke ok
bc flow policy checkpoint smoke ok
```

## Next Implementation Steps

1. Run a real Robomimic build-only and short rollout for residual+intervention gate, starting with a random gate before `expert_q_gap`.
2. Add explicit dataset adapters for offline demo/prefill canonicalization.
3. Add replay save/load round-trip tests, including real-env generated replay files.
4. Replace duplicated residual-kind checks with a small actor/agent registry before adding another residual family.
5. Decide whether the next gate family can fit the current `ControllerGate` Protocol or needs a `GateContext`.

## Pending Risks

- Residual+intervention gate wiring has only dummy-policy smoke coverage; run real-env build-only and short rollout before long jobs.
- Replay save/load round-trip test is missing.
- Dataset semantics are still partly implicit in the Robomimic prefill loader.
- `keep_last` is intentionally not wired; `storage.store_*` remains mostly declarative rather than controlling canonical replay writes.
- General learner/expert action chunk queues are not implemented.
- `PolicyOutput.info` still carries implicit contracts for chunk and residual metadata.
