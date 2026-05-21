# Implementation Plan

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

1. Add Robomimic Square environment construction.
2. Run a 100-step online rollout smoke with fresh or restored learner, restored expert, and random gate.
3. Verify saved replay contains learner/expert/executed actions and gate metadata.
4. Reproduce no-intervention RLPD baseline.
5. Run random-intervention baseline.
6. Add BC losses from `demo_buffer` and `intervention_buffer`.
7. Then add smarter gating, human UI, and action chunking.

## Pending Risks

- Real Robomimic env rollout is not wired yet.
- Replay save/load round-trip test is missing.
- Online training loop CLI is still a placeholder.
- Demo/intervention BC losses are not yet connected to learner updates.
