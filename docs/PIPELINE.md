# Online Intervention Pipeline

## 초기 파이프라인

1. env reset 후 observation을 받는다.
2. learner policy에서 `learner_action`을 샘플링한다.
3. expert policy에서 `expert_action`을 샘플링한다.
4. gate가 learner/expert 중 실행할 controller를 결정한다.
5. 선택된 action을 env에 넣는다.
6. 실제 실행 action, learner proposal, expert proposal, gate metadata를 모두 replay에 저장한다.
7. episode 종료 시 demo/intervention buffer로 routing한다.
8. learner는 online/demo/intervention buffer를 실험 조건에 맞게 섞어 update한다.

## Step Logic

```python
learner_output = learner.sample_action(obs, rng=learner_rng)
expert_output = expert.sample_action(obs, rng=expert_rng)

decision = gate.decide(
    step=step,
    observation=obs,
    learner=learner_output,
    expert=expert_output,
    rng=gate_rng,
    context=gate_context,  # optional: only diagnostic gates use this
)

action = expert_output.action if decision.use_expert else learner_output.action
next_obs, reward, terminated, truncated, info = env.step(action)
```

learner와 expert action을 gate 전에 둘 다 뽑는 것이 중요하다. 그래야 같은 state에서
learner proposal, expert proposal, 실제 실행 action을 모두 저장할 수 있다.

`GateContext`는 uncertainty처럼 policy를 다시 샘플링해야 하는 diagnostic gate용 optional context다. 기존
random gate와 expert-Q gap gate는 이미 샘플된 proposal만으로 결정할 수 있지만, `action_uncertainty`는
같은 observation에서 source policy를 여러 번 다시 샘플링해야 한다. 이 재샘플링은 env step, replay write,
network update를 하지 않는다.


## Expert-Q Gap Gate

`expert_q_gap`은 expert가 action을 실행할지 정하는 intervention trigger다. policy selector나 expert query 여부 자체가 아니다.

```text
q_gap = Q_expert(s, a_expert) - Q_expert(s, a_learner)
signal = q_gap > threshold
```

signal이 켜지면 `intervention_prob` 확률로 expert intervention을 시작하고, 시작 후에는
`intervention_horizon` step 동안 expert가 연속 제어한다. 현재 설계에서는 `p_off`를 두지 않는다.

이 gate는 RLPD 전용으로 구현하지 않는다. gate의 책임과 expert agent/adapter의 책임을 분리한다.

Gate 책임:

- learner/expert action proposal을 같은 state에서 비교한다.
- `q_agg` 문자열을 expert Q API에 넘긴다.
- expert 내부의 `critic`, `q`, `qf` 같은 module 이름을 직접 탐색하지 않는다.

Expert agent/adapter 책임:

- `evaluate_q(observations, actions, q_agg=...)` 또는 `q_values(observations, actions, q_agg=...)`를 제공한다.
- multi-Q head shape과 `min|mean|max` aggregation은 agent/adapter 내부에서 처리한다.
- 필요하면 raw head 확인용으로 `evaluate_q_heads(observations, actions)`를 추가 제공한다.

SAC/RLPD, TD3-BC처럼 action-value critic이 있는 expert는 agent/adapter에서 이 API를 맞추면 같은 방식으로 쓸 수 있다. PPO처럼 V-only critic만
있는 expert는 이 gate를 바로 쓸 수 없고 action-value head 또는 adapter가 필요하다.

## Action-Uncertainty Gate

`action_uncertainty`는 같은 observation에서 하나의 policy source를 여러 번 샘플링하고 action variance가
threshold를 넘으면 expert intervention을 시작한다.

```text
samples = [pi_source(s; rng_i) for i in 1..num_samples]
score = sqrt(mean(var(samples, axis=sample)))
signal = score > threshold
```

현재 구현은 `estimator: sample_variance`, `score: rms_std`만 지원한다. `source`는 `learner`,
`expert`, `base` 중 하나다. Diffusion/flow BC, SAC/RLPD stochastic actor처럼 같은 state에서 다른
sample을 낼 수 있는 policy에 바로 쓸 수 있다. TD3처럼 deterministic actor는 exploration noise를 넣지
않으면 score가 거의 0이 된다.

SAC analytic std, policy entropy, BC ensemble variance는 아직 구현하지 않았다. 이들은 같은
`action_uncertainty` gate family의 estimator backend로 추가한다.

## RLPD Expert Weight 로딩

expert policy는 이 repo의 `ACRLPDAgent` checkpoint layout과 config에 맞춰 로드한다.
외부에서 가져온 weight는 미리 같은 state-dict layout으로 맞춰둔 뒤, runtime 코드에서는
source-specific loader를 두지 않는다.

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

`RLPDPolicy`는 특정 원본 repo를 알지 않는다. `ACRLPDAgent.create()`로 agent를 만든 뒤
`restore_agent_with_file()`로 checkpoint를 복원하고, `PolicyOutput(action, log_prob, info)`
인터페이스만 제공한다.

diffusion/flow BC expert도 같은 방식이다.

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

`BCFlowPolicy`도 원본 repo를 알지 않는다. checkpoint는 이 repo의 `BCFlowAgent`
state-dict layout과 config에 맞아야 한다.

## Buffer 역할

`online_buffer`

- 모든 online transition을 저장한다.
- `actions`는 실제 env에 실행된 action이다.
- learner가 실제로 방문한 state/action distribution이다.

`demo_buffer`

- intervention 없이 autonomous하게 성공한 episode를 저장할 수 있다.
- offline expert demo dataset 또는 scripted/expert dataset을 넣을 수도 있다.
- BC/DAgger류 학습에서 clean expert label source로 쓴다.
- `demo_insert_mode="replace_longest_if_better"`를 쓰면, 더 짧은 성공 episode가 나왔을 때 기존 demo
  pool의 가장 긴 episode를 밀어낼 수 있다.

`intervention_buffer`

- intervention이 발생한 episode의 first intervention 이후 suffix를 저장한다.
- 실패 직전/실패 상태에서 expert correction을 imitation하는 용도다.
- expert가 실패한 suffix를 포함할지는 flag로 제어한다.

## 현재 Smoke Test로 확인한 것

[scripts/smoke_test.py](../scripts/smoke_test.py)의 `smoke_intervention_routing()`에서 simulator 없이
다음 흐름을 확인한다.

- autonomous success episode는 `demo_buffer`에 전체 trajectory가 들어간다.
- intervention success episode는 first intervention 이후 suffix만 `intervention_buffer`에 들어간다.
- failed intervention suffix는 `include_failed_interventions=False`이면 버리고, `True`이면 넣는다.
- intervention transition에서 `actions == expert_actions`이고 `actions != learner_actions`임을 확인한다.

이 smoke test는 데이터 흐름 검증이다. 실제 online rollout은 `il/train.py`가 recipe 기반으로
수행한다.



Action chunk TODO: 현재 train rollout은 primitive action 기준이다. chunk policy는 나중에 `collections.deque`로
learner/expert queue를 따로 두고, queue가 비었을 때만 policy를 query하는 방식으로 정리한다. learner와 expert의
horizon은 서로 다를 수 있어야 한다. policy adapter는 `full_action_chunk`를 항상 `(horizon, action_dim)`으로 제공하고,
controller switch 시 stale chunk를 막기 위해 양쪽 queue를 clear하는 방향을 우선 검토한다.

## Unified Train Loop

실행 진입점은 `il/train.py`다. 이 파일은 build만 담당하고, 실제 env-step loop는
`il/loops/train_loop.py`의 `run_train_loop()`가 담당한다.

loop v0 동작:

- env reset 후 learner/expert proposal을 샘플링한다.
- `rollout.execute`가 `learner`, `expert`, `gate` 중 어떤 action을 실행할지 결정한다.
- 모든 transition은 항상 `online_buffer`에 저장한다.
- episode 종료 시 `route_episode_to_buffers()`로 demo/intervention routing을 수행한다.
- `updates` recipe에 따라 replay source를 sample하고 target actor를 update한다.
- `save_interval`마다 trainable actor checkpoint를 저장하고, 종료 시 세 replay buffer를 `.npz`로 저장한다.

기본 실행:

```bash
conda run -n il python -m il.train --config config/my_run.yaml
```

build만 확인:

```bash
conda run -n il python -m il.train --config config/my_run.yaml --build-only
```

현재 제한:

- action chunk queue는 아직 v0 loop에 정교하게 반영하지 않았다. 기본 설정은 primitive action 기준이다.
- image observation은 env/replay에는 들어가지만, current actor update는 lowdim state actor 기준이다.

## Image Observation 상태

Robomimic env wrapper는 `observation_mode`로 lowdim, image, image+state를 선택할 수 있다.

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

multi-camera observation은 dict로 저장된다.

```python
obs = {
    "state": low_dim,
    "agentview": image,
    "sideview": image,
    "robot0_eye_in_hand": image,
}
```

현재 확인된 Square camera 이름은 `frontview`, `birdview`, `agentview`, `sideview`,
`robot0_robotview`, `robot0_eye_in_hand`이다.

중요한 제한: env와 replay buffer는 image obs를 받을 수 있지만, policy/network 학습은 아직
lowdim-only다. image encoder와 feature fusion은 [NETWORKS.md](NETWORKS.md)의 TODO로 남긴다.
