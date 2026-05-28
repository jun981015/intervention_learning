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

## Residual Learner With Gates

Residual action composition can now be combined with intervention gates. When
`training.action_composition: residual` is enabled, the learner proposal passed
to the gate is the full executable action, not the raw residual:

```text
base_action = base_policy(obs)
raw_residual = residual_learner(concat(state, base_action))
learner_output.action = clip(base_action + residual_scale * raw_residual)
```

If `intervention.enabled: true`, config conversion sets `rollout.execute =
"gate"`, but `il.train.build_context()` still builds `actors.base` because
residual composition needs it. The gate then chooses between the full learner
action above and the expert action.

Replay metadata for residual composition is stored even when the gate chooses
the expert:

```text
base_actions      = base policy action at current obs
next_base_actions = base policy action at next obs
residual_actions  = executed_action - base_actions
```

For expert-selected steps, `residual_actions` is therefore the residual needed
to reproduce the expert's executed action from the frozen base policy.

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


## Expert-Q Gap Gate

`expert_q_gap` is an intervention trigger, not a policy-query selector. It
compares expert and learner proposals at the same state with the expert's
action-value function:

```text
q_gap = Q_expert(s, a_expert) - Q_expert(s, a_learner)
signal = q_gap > threshold
```

If the signal is true, intervention starts with probability
`intervention_prob`. Once started, the expert controls for
`intervention_horizon` consecutive environment steps. There is no `p_off` in
this v0 design.

This gate must not depend on `kind == "rlpd"`. Keep the boundary explicit.

Gate responsibilities:

- Compare learner and expert action proposals at the same state.
- Pass the `q_agg` string to the expert Q API.
- Never inspect expert internals such as `critic`, `q`, or `qf` module names.

Expert agent/adapter responsibilities:

- Expose `evaluate_q(observations, actions, q_agg=...)` or `q_values(observations, actions, q_agg=...)`.
- Own multi-Q head shapes and `min|mean|max` aggregation semantics.
- Optionally expose `evaluate_q_heads(observations, actions)` for diagnostics.

SAC/RLPD and TD3-BC-style experts can support this by matching the explicit API in their agent/adapter. PPO value-only
experts cannot compute the gap unless they add an action-value head or adapter.

Example:

```yaml
intervention:
  enabled: true
  expert_query: always
  gate:
    kind: expert_q_gap
    threshold: 0.5
    intervention_prob: 0.9
    intervention_horizon: 10
    q_agg: min
```

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
env-step loop to `il/loops/train_loop.py::run_train_loop()`.

Loop v0 behavior:

- Reset env, sample learner/expert proposals.
- Use `rollout.execute` set to `learner`, `expert`, or `gate` to decide the executed action.
- If `action_composition == "residual"`, build the learner proposal as `clip(base + residual)` before routing or gating.
- Store every transition in `online_buffer`.
- On episode end, route the episode into demo/intervention buffers through `route_episode_to_buffers()`.
- Run configured `updates` after `initial_collect` is complete, respecting `update_interval` and `updates_per_step`.
- Save trainable actor checkpoints at `save_interval`; save the final checkpoint only when `save_final` is true; save replay buffers as `.npz` at the end when enabled.

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

The Robomimic env wrapper can expose lowdim, state-only dict, image-only, or
image+state observations.

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

`observation_mode: state` returns `{"state": low_dim}` without pixels. This is
useful for exercising dict-observation plumbing before policy image encoders are
implemented.

Important limitation: env and replay plumbing are image-ready, but current
policies/networks are still lowdim-only. Pixel encoders and feature fusion are
tracked in `NETWORKS.md` as future work.
