# Pretrained Policy Artifacts

This document records pretrained policy artifacts currently available for
intervention-learning experiments.

## Rules

- Do not add source-repo-specific loaders to the runtime path.
- Checkpoints must match this repo's agent state-dict layout and config before use.
- Do not commit checkpoints, replay buffers, videos, logs, W&B files, or generated experiment outputs.
- Current artifacts live under `exp/pretrained/`, which is ignored by git.

## Square RLPD Expert

Artifact:

```text
exp/pretrained/rlpd_square_bc03_seed0_2m/
  params_2000000.pkl
  config.json
  metadata.json
```

Source:

```text
/home/junhyeong/repos/qc/exp/qc/square-rlpd-sparse-bc03-2m-seed0/square-mh-low_dim/sd00020260507_123338/params_2000000.pkl
```

Shape/config:

```text
obs_dim=23
action_dim=7
horizon_length=1
actor_layer_norm=False
critic_layer_norm=True
```

Load example:

```python
import json
from pathlib import Path

from il.policies import RLPDPolicy

run_dir = Path("exp/pretrained/rlpd_square_bc03_seed0_2m")
config = json.loads((run_dir / "config.json").read_text())
metadata = json.loads((run_dir / "metadata.json").read_text())

expert = RLPDPolicy.from_checkpoint(
    run_dir / "params_2000000.pkl",
    config=config,
    obs_dim=metadata["obs_dim"],
    action_dim=metadata["action_dim"],
    seed=metadata["seed"],
)
```

Legacy validation (before critic-contract fix):

```text
Restored from .../rlpd_square_bc03_seed0_2m/params_2000000.pkl
action_shape (7,)
log_prob 16.3937
```

## ToolHang Residual TD3 Expert

The current ToolHang expert artifact is the 1.5M checkpoint from the residual TD3 run with residual scale `0.2`, BC regularization `0.1`, actor LR `5e-5`, and warmup `0.1`.

Reusable artifact:

```text
exp/pretrained/residual_td3_tool_hang_ph_scale02_bc01_actorlr5e5_warmup01_seed0_1500k/
  params_1500000.pkl
  config.json
  metadata.json
```

Source run and full checkpoint set:

```text
exp/runs/intervention_learning/tool_hang_residual_online/tool_hang-ph-low_dim/
  tool_hang_residual_td3_bcflow_top200_mixed50_shiftm1_nstep5_scale02_bc01_actorlr5e5_warmup01_seed0_2m/
    params_100000.pkl
    params_200000.pkl
    ...
    params_1500000.pkl
    ...
    params_2000000.pkl
```

Required base policy:

```text
exp/pretrained/bcflow_tool_hang_ph_top200_actorln_seed0_1m/
  params_1000000.pkl
```

Shape/config:

```text
env_obs_dim=53
base_action_dim=7
residual_actor_obs_dim=60
action_dim=7
residual_scale=0.2
checkpoint_step=1500000
```

Config using this expert:

```text
config/tool_hang_residual_td3_scale02_ckpt1500k_expert_random_gate_smoke.yaml
```

Notes:

- A residual TD3 expert does not act from plain state alone. It consumes `state + base_action`, then executes `clip(base_action + residual_scale * raw_residual)`.
- Use this ToolHang residual expert with the BCFlow base checkpoint listed above.
- The reusable expert artifact has `exploration_noise=0.0` in `config.json` for deterministic expert proposals.
- This 1.5M checkpoint critic was trained with the pre-2026-05-29 contract, `Q([state, base_action], executed_action)`. Do not use it for Q-gap critic routing under the corrected code contract; retraining is required. Full-agent restore can also hit critic shape mismatch, and actor-only reuse would need a separate partial-restore path.

Legacy validation (before critic-contract fix):

```text
Restored from .../residual_td3_tool_hang_ph_scale02_bc01_actorlr5e5_warmup01_seed0_1500k/params_1500000.pkl
expert=residual_td3
gate=random
real-env ToolHang 20-step smoke passed
interventions=8/20
intervention_action_matches_expert_max_abs_err=0
```

## Flow BC Learner

Artifact:

```text
exp/pretrained/bcflow_square_actorln_seed0_1m/
  params_1000000.pkl
  config.json
  metadata.json
```

Source:

```text
/home/junhyeong/repos/qc/exp/qc/qc-grid-base-offline-actorln/square-mh-low_dim/sd00020260512_134241/params_1000000.pkl
```

Shape/config:

```text
obs_dim=23
action_dim=7
horizon_length=5
full_action_dim=35
actor_layer_norm=True
flow_steps=10
```

Load example:

```python
import json
from pathlib import Path

from il.policies import BCFlowPolicy

run_dir = Path("exp/pretrained/bcflow_square_actorln_seed0_1m")
config = json.loads((run_dir / "config.json").read_text())
metadata = json.loads((run_dir / "metadata.json").read_text())

learner = BCFlowPolicy.from_checkpoint(
    run_dir / "params_1000000.pkl",
    config=config,
    obs_dim=metadata["obs_dim"],
    action_dim=metadata["action_dim"],
    seed=metadata["seed"],
)
```

Legacy validation (before critic-contract fix):

```text
Restored from .../bcflow_square_actorln_seed0_1m/params_1000000.pkl
action_shape (7,)
log_prob nan
full_chunk_shape (5, 7)
```

`log_prob=nan` is expected for implicit flow sampling.

## Chunk Behavior

The v0 pipeline does not implement action chunk queues yet. Chunked flow policies
produce a full chunk internally, but the default `PolicyOutput.action` is the
first primitive action.

```text
full chunk: (5, 7)
executed action: (7,)
```

Future chunk queue work should use `PolicyOutput.info["full_action_chunk"]`.

## Current Combination

RLPD expert and Flow BC learner both return `PolicyOutput(action, log_prob, info)`.
The online training path uses `il.loops.rollout.choose_rollout_action()` to sample learner/expert proposals, run the gate, and select the executed action.

For direct policy checks, call each policy on the same observation:

```python
learner_output = learner.sample_action(obs, rng=learner_rng)
expert_output = expert.sample_action(obs, rng=expert_rng)
```

## Notes

- The current RLPD artifact has no actor LayerNorm.
- The GPU6 run `square-rlpd-actorln-bc03-2m-seed0` has actor LayerNorm enabled in qc. Once checkpoints are available, prepare a new ignored artifact for this repo.
- The one-off artifact preparation scripts are intentionally not checked into this repo.
