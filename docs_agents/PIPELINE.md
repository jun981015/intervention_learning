# Online Intervention Pipeline

## Core Step Logic

```python
learner_output = learner.sample_action(obs, rng=learner_rng)
expert_output = expert.sample_action(obs, rng=expert_rng)

decision = gate.decide(
    step=step,
    observation=obs,
    learner=learner_output,
    expert=expert_output,
    rng=gate_rng,
)

action = expert_output.action if decision.use_expert else learner_output.action
next_obs, reward, terminated, truncated, info = env.step(action)
```

The learner and expert must be sampled before gating so replay can store both
proposed actions at the same state.

## RLPD Expert Loading

Expert policies should be loaded through this repo's generic `RLPDPolicy`.
External pretrained weights must already match this repo's `ACRLPDAgent`
state-dict layout and config. Do not add source-specific loader code to the
runtime path.

```python
from il.policies import RLPDPolicy

expert = RLPDPolicy.from_checkpoint(
    "/path/to/params_2000000.pkl",
    config=rlpd_config,
    obs_dim=23,
    action_dim=7,
    seed=0,
)
```

`RLPDPolicy` creates `ACRLPDAgent`, restores `params_*.pkl`, and exposes
`PolicyOutput(action, log_prob, info)` for the shared rollout loop.

Flow-matching BC experts use the same source-agnostic pattern.

```python
from il.policies import BCFlowPolicy

expert = BCFlowPolicy.from_checkpoint(
    "/path/to/params_1000000.pkl",
    config=bc_flow_config,
    obs_dim=23,
    action_dim=7,
    seed=0,
)
```

`BCFlowPolicy` creates `BCFlowAgent`, restores `params_*.pkl`, and exposes the
same `PolicyOutput` interface. External weights must already match this repo's
`BCFlowAgent` state-dict layout and config.

## Buffers

`online_buffer` stores every online transition collected by the learner.

`demo_buffer` stores clean expert-label data. It can contain autonomous success
trajectories, offline expert demonstrations, or scripted expert demonstrations.
With `demo_insert_mode="replace_longest_if_better"`, a shorter successful
episode can replace the current longest demo episode.

`intervention_buffer` stores the suffix starting at the first intervention.
Failed expert intervention suffixes are optional through a config flag.

## Verified Smoke Coverage

`scripts/smoke_test.py::smoke_intervention_routing()` currently verifies the
pipeline without a simulator:

- autonomous success episodes go to `demo_buffer`
- intervention success episodes store only the suffix after the first intervention in `intervention_buffer`
- failed intervention suffixes are included or skipped by `include_failed_interventions`
- an intervention transition has `actions == expert_actions` and `actions != learner_actions`

This is data-flow validation. Actual online rollout is handled by the
recipe-driven `il/train.py` entrypoint.

## Unified Train Loop

The entrypoint is `il/train.py`. It builds components, then delegates the
env-step loop to `il/loops/recipe.py::run_train_loop()`.

Loop v0 behavior:

- Reset env, sample learner/expert proposals.
- Use `rollout.execute` set to `learner`, `expert`, or `gate` to decide the executed action.
- Store every transition in `online_buffer`.
- On episode end, route the episode into demo/intervention buffers through `route_episode_to_buffers()`.
- Run configured `updates` by sampling from the requested replay source and updating the target actor.
- Save trainable actor checkpoints at `save_interval`; save all replay buffers as `.npz` at the end.

Run:

```bash
conda run -n il python -m il.train --config config/my_run.yaml
```

Build-only check:

```bash
conda run -n il python -m il.train --config config/my_run.yaml --build-only
```

Current limitations:

- Action chunk queues are not yet modeled carefully in the v0 loop. Use primitive-action configs by default.
- Image observations can pass through env/replay, but current actor updates are lowdim-state based.

## Image Observation Status

The Robomimic env wrapper can expose lowdim, image-only, or image+state
observations.

```yaml
env:
  observation_mode: pixels_state
  render_offscreen: true
  image_size: 64
  camera_names:
    - agentview
    - sideview
    - robot0_eye_in_hand
```

Multi-camera observations are stored as dict observations:

```python
obs = {
    "state": low_dim,
    "agentview": image,
    "sideview": image,
    "robot0_eye_in_hand": image,
}
```

Verified Square camera names include `frontview`, `birdview`, `agentview`,
`sideview`, `robot0_robotview`, and `robot0_eye_in_hand`.

Important limitation: env and replay plumbing are image-ready, but current
policies/networks are still lowdim-only. Pixel encoders and feature fusion are
tracked in `NETWORKS.md` as future work.
