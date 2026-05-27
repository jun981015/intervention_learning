# Replay And Update Design

## Replay Transition Schema

Required fields:

- `observations`
- `actions`
- `learner_actions`
- `expert_actions`
- `rewards`
- `terminals`
- `masks`
- `next_observations`
- `controller_ids`
- `episode_ids`
- `episode_steps`
- `gating_reasons`
- `gating_scores`
- `learner_action_log_probs`
- `expert_action_log_probs`
- `interventions`
- `base_actions`
- `residual_actions`
- `next_base_actions`

`actions` is always the action actually executed in the environment.

For residual policies, `actions` is the combined executed action
`clip(base_actions + residual_actions)`. `base_actions` and
`next_base_actions` must be finite for residual RL updates. Non-residual
transitions may keep these residual metadata fields as NaN placeholders.

`episode_ids` identifies the episode containing the transition, and
`episode_steps` is the 0-based step index within that episode. Image replay uses
these fields to reconstruct next image observations from the following frame.

`controller_ids`:

- `0`: learner
- `1`: expert

`gating_reasons`:

- `0`: none
- `1`: random gate

## Terminals And Masks

`terminals` marks any episode boundary, including timeout/truncation.

`masks` marks true termination for bootstrap control. Timeouts/truncations should
not automatically zero the bootstrap mask.

## Image Replay Storage

Do not store image observations twice under both `observations` and
`next_observations`. `ReplayBuffer` splits image leaves into a separate
`image_observations[camera_name]` store and keeps only the current frame.

Storage example:

```python
buffer.data["observations"]["state"]         # low-dimensional state
buffer.image_data["agentview"]               # current image frame
buffer.image_data["sideview"]
```

Sampling preserves the public batch API:

```python
batch["observations"]["state"]
batch["observations"]["agentview"]           # image at transition i
batch["next_observations"]["state"]
batch["next_observations"]["agentview"]      # image from transition i + 1
batch["image_next_valid"]                    # 1 if i+1 is the next step in the same episode
```

The next image frame is pulled from the next transition only when `episode_ids`
match and `episode_steps` is exactly `current + 1`. If the next transition is
not available or crosses an episode boundary, the image is zero-filled and
`image_next_valid=0`.

## N-step Backup

The buffer should provide QC-style n-step sampling:

```python
batch = replay_buffer.sample_sequence(batch_size, sequence_length=n, discount=gamma)
```

The returned `batch["rewards"][:, -1]` should be the discounted n-step return
from the sampled start state, and `batch["next_observations"][:, -1]` should be
the bootstrap observation after the final sampled transition.

Expected critic target:

```text
target = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1}
       + gamma^n * mask_chain * Q_target(s_{t+n}, a_{t+n})
```

Start with `sequence_length=1` for v0, but keep the buffer API general.

## Boundary Policy

For v0, match `qc_base`: if an episode boundary appears in the middle of an
n-step window, do not use that sample for the critic update. The sampler may
still return the sample, but `valid[:, -1]` must be `0`, so the loss is masked
out.

Future ablation:

```text
boundary_mode = "drop"
  QC-base behavior. Mask out samples with a boundary inside the n-step window.

boundary_mode = "truncate"
  Build a shorter n-step target at the boundary. Bootstrap at timeout/truncation
  boundaries; do not bootstrap at true termination boundaries.
```

## Update-to-data Ratio

Expose `utd_ratio` as a training config parameter. To match the QC update path,
sample `batch_size * utd_ratio` transitions/sequences and reshape each batch
array to `(utd_ratio, batch_size, ...)` before calling `agent.batch_update()`.

Default values:

- `num_qs = 2`
- `utd_ratio = 1`

## Mixed Buffer Sampling

Physical replay buffers are split by role:

- `online`: all transitions collected by learner online rollout
- `intervention`: first-intervention suffixes from intervention episodes
- `demo`: clean expert-label data such as autonomous success trajectories, offline expert demos, or scripted demos

`MixedReplaySampler` samples from named buffers according to a ratio spec.

```python
buffers = ReplayBufferCollection(
    online=online_buffer,
    intervention=intervention_buffer,
    demo=demo_buffer,
)

sampler = MixedReplaySampler(
    buffers,
    MixedSamplingSpec({"online": 0.5, "intervention": 0.25, "demo": 0.25}),
)

batch = sampler.sample_sequence(batch_size=256, sequence_length=1, discount=0.99)
```

`TrainingConfig.sampling_fractions` wires the same mechanism into
`sample_rl_update_batch()`.

```python
config = TrainingConfig(
    batch_size=256,
    utd_ratio=1,
    horizon_length=1,
    discount=0.99,
    sampling_fractions={"online": 0.5, "intervention": 0.25, "demo": 0.25},
)
batch = sample_rl_update_batch(buffers, config)
```

Fractions do not need to sum to 1. Counts are normalized and rounded so the
total count matches the requested batch size.

## Initial Replay Prefill

`online`, `intervention`, and `demo` buffers can optionally be prefilled at
startup from an existing replay dataset. The currently supported format is a
schema-compatible `.npz` saved by `ReplayBuffer.save_npz()`.

Example:

```yaml
replay:
  online_size: 500000
  intervention_size: 500000
  demo_size: 500000
  prefill:
    demo:
      path: /path/to/demo_replay.npz
      format: npz
      max_transitions: 100000  # optional
      cache_base_actions: true  # optional, residual RL only; requires actors.base
```

`prefill` can target any physical buffer: `online`, `intervention`, or `demo`.
The buffers remain physically separate; prefill only initializes the target
buffer's contents, `size`, and `pointer`.
For simple cases, a direct path string such as `demo: /path/to/demo_replay.npz`
is also accepted.

Image replay is restored from the same `.npz` format. Keys such as
`image_observations/<camera>` are restored into `ReplayBuffer.image_data[camera]`,
and sampling reconstructs next images from the `i+1` frame.


For residual RL, `cache_base_actions: true` runs the frozen `actors.base` policy
over the prefilled observations and fills `base_actions`, `next_base_actions`,
and diagnostic `residual_actions`. This can be expensive on large datasets and
must stay opt-in.

High-priority TODO: add explicit dataset adapters / canonicalization. Offline
demo sources do not all give `actions` the same meaning. The loader should not
silently copy `actions` into `expert_actions` unless an adapter such as
`demo_actions_are_expert` states that semantic explicitly. Schema-compatible
saved replay should use a separate `replay_npz`-style adapter that preserves the
saved fields as-is.

## Demo Episode Insert Mode

The demo buffer is not assumed to be a perfect expert dataset. If online
training produces a shorter successful trajectory, it can replace the current
longest success trajectory in the demo pool.

Supported modes:

- `none`: do not insert online success episodes into the demo buffer. Use this
  when the demo buffer should stay fixed to prefilled/offline demos.
- `append`: append the success episode transition-by-transition.
- `replace_longest_if_better`: regardless of free capacity, replace the current
  longest stored episode when the new success episode is shorter. If there is
  no comparable stored episode yet, append.

Example:

```python
route_episode_to_buffers(
    episode,
    demo_buffer=demo_buffer,
    intervention_buffer=intervention_buffer,
    include_failed_interventions=False,
    demo_insert_mode="replace_longest_if_better",
)
```

This mode relies on valid `episode_ids` and `episode_steps`. Avoid collisions
between prefilled demo episode ids and online rollout episode ids.

Implementation detail: `ReplayBuffer` lazily builds an episode index when
`replace_longest_if_better` is used. The cache stores `episode_id -> length/indices`
plus a longest-first worst-order list. It does not rescan the full buffer on
every replacement decision; it rebuilds the index after replay mutation when
needed. The current worst metric is episode `length`.

## TD Target Q Aggregation

Expose `target_q_agg` for the target critic ensemble used in TD backups.
Regardless of `num_qs`, TD target computation uses only the first two target
critics: `q_tar1` and `q_tar2`.

Supported values:

- `"mean"`: use the mean of `q_tar1` and `q_tar2`
- `"min"`: use `min(q_tar1, q_tar2)`, default

Keep the actor objective aggregation as mean in v0 unless explicitly requested.
`num_qs` remains configurable, but it must be at least `2`.

## Frame Stack

Keep a `frame_stack` parameter in the buffer/env pipeline. The default is `1`,
meaning no history stacking. Support for `frame_stack > 1` can be added later.
