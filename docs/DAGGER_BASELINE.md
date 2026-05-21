# DAgger Baseline

DAgger는 이 repo의 intervention learning pipeline과 의도적으로 분리한다. 핵심 차이는 expert가
실제 env action을 실행하지 않는다는 점이다.

## Intervention Learning과 차이

Intervention learning:

- learner와 expert action을 둘 다 뽑는다.
- gate가 실제 실행 controller를 learner/expert 중에서 고른다.
- expert가 선택된 step에서는 env가 expert action으로 진행된다.
- intervention suffix를 별도 `intervention_buffer`로 routing한다.

DAgger:

- learner와 expert action을 둘 다 뽑는다.
- 실제 env에는 항상 learner action을 넣는다.
- expert action은 같은 state에 대한 label로만 저장한다.
- 성공/실패/intervention 여부와 무관하게 visited state를 `online_buffer`에 저장한다.
- learner update는 BC objective로 `expert_actions`를 맞춘다.

## Replay 의미

DAgger transition도 canonical replay schema를 그대로 쓴다.

- `actions`: env에 실제 실행된 learner action
- `learner_actions`: learner proposal
- `expert_actions`: expert relabel target
- `controller_ids`: 항상 `LEARNER`
- `interventions`: 항상 0

이렇게 저장하면 나중에 같은 데이터로 “learner가 실제로 한 행동”과 “expert가 달았을 label”을
분리해서 분석할 수 있다.

`DAggerConfig.store_expert_action=True`가 기본값이다. `False`로 두면 rollout 때 expert를 호출하지 않고
`expert_actions`에는 NaN placeholder를 넣는다. 이 옵션은 나중에 update-time relabeling ablation을
할 때 쓰기 위한 것이다.

## 구현 위치

- `il.loops.dagger.choose_dagger_action()`: learner action을 실행 action으로 반환하고 expert action을 label로 같이 반환한다.
- `il.buffers.dagger.add_dagger_episode_to_online_buffer()`: episode 전체를 `online_buffer`에 저장한다.
- `BCMLPAgent`: `target_action_key="expert_actions"`로 DAgger BC 학습이 가능하다.
- `BCFlowAgent`: `target_action_key` 옵션을 추가했다. 기본값은 기존과 같은 `"actions"`이고, DAgger에서는 `"expert_actions"`로 바꾼다.

## 기본 Step

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

이후 `StepRecord`와 `step_record_to_transition()`을 쓰면 기존 replay schema로 변환된다.

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

v0에서는 action chunk queue를 아직 구현하지 않았으므로 DAgger도 `horizon_length=1`을 기본으로 둔다.

## 현재 Smoke

`scripts/smoke_test.py`에서 다음을 확인한다.

- DAgger 실행 action이 learner action인지
- expert action이 `expert_actions` label로 저장되는지
- `store_expert_action=False`에서 expert action이 NaN placeholder로 남는지
- `controller_ids=LEARNER`, `interventions=0`인지
- online buffer에서 sample한 batch로 BC MLP update가 가능한지
- BC Flow가 `target_action_key="expert_actions"` batch를 받을 수 있는지

검증 명령:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n il python scripts/smoke_test.py
```
