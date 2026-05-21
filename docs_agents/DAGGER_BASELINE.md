# DAgger Baseline

DAgger is intentionally separate from the intervention-learning pipeline in this repo.
The expert does not execute environment actions. The learner acts, and the expert only labels
the learner-visited state.

## Difference From Intervention Learning

Intervention learning:

- Sample both learner and expert actions.
- A gate chooses whether learner or expert controls the environment.
- If the expert is selected, the environment advances with the expert action.
- Intervention suffixes are routed into `intervention_buffer`.

DAgger:

- Sample both learner and expert actions.
- Always execute the learner action in the environment.
- Store the expert action only as a supervised relabel target.
- Store every visited state in `online_buffer`, regardless of success or failure.
- Train the learner with BC against `expert_actions`.

## Replay Semantics

DAgger uses the canonical replay schema.

- `actions`: executed learner action.
- `learner_actions`: learner proposal.
- `expert_actions`: expert relabel target.
- `controller_ids`: always `LEARNER`.
- `interventions`: always 0.

This preserves both the behavior action and the expert label for later analysis.

`DAggerConfig.store_expert_action=True` is the default. If it is `False`, rollout does not call
the expert and stores a NaN placeholder in `expert_actions`. This keeps a clean switch for a future
update-time relabeling ablation.

## Implementation

- `il.loops.dagger.choose_dagger_action()`: returns the learner action for execution and the expert action as a label.
- `il.buffers.dagger.add_dagger_episode_to_online_buffer()`: stores a full episode in the online buffer.
- `BCMLPAgent`: set `target_action_key="expert_actions"` for DAgger BC.
- `BCFlowAgent`: now has a `target_action_key` option. Default is `"actions"` for backward compatibility; use `"expert_actions"` for DAgger.

## Step Example

```python
from il.loops.dagger import choose_dagger_action

action, learner_output, expert_output, decision = choose_dagger_action(
    step=step,
    observation=obs,
    learner=learner,
    expert=expert,
    learner_rng=learner_rng,
    expert_rng=expert_rng,
    config=dagger_config,
)

next_obs, reward, terminated, truncated, info = env.step(action)
```

Build a `StepRecord` and call `step_record_to_transition()` to use the standard replay schema.

## BC Update

MLP BC:

```python
config = get_bc_mlp_config()
config.target_action_key = "expert_actions"
agent = BCMLPAgent.create(seed, ex_observations, ex_actions, config)
batch = online_buffer.sample(config.batch_size)
agent, info = agent.update(batch)
```

Flow BC:

```python
config = get_bc_flow_config()
config.action_chunking = False
config.horizon_length = 1
config.target_action_key = "expert_actions"
agent = BCFlowAgent.create(seed, ex_observations, ex_actions, config)
batch = online_buffer.sample(config.batch_size)
agent, info = agent.update(batch)
```

v0 keeps `horizon_length=1`; action chunk queues are out of scope for now.

## Smoke Coverage

`scripts/smoke_test.py` checks:

- DAgger executes the learner action.
- Expert actions are stored as `expert_actions`.
- `store_expert_action=False` leaves a NaN expert-action placeholder.
- `controller_ids=LEARNER` and `interventions=0`.
- BC MLP can update from the online buffer.
- BC Flow can use `target_action_key="expert_actions"`.

Validation command:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n il python scripts/smoke_test.py
```
