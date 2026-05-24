# Residual Policy / ResFiT 분석

작성일: 2026-05-24

이 문서는 `/home/junhyeong/repos/residual-offpolicy-rl`의 ResFiT 구현을 보고, 이 레포에 어떤 방식으로 붙일 수 있는지 정리한 기록이다. 결론부터 말하면 ResFiT 코드를 그대로 가져오는 방식은 피하고, 핵심 구조만 JAX/Flax 기반으로 다시 구현하는 것이 맞다.

## Source Repo

- repo: `/home/junhyeong/repos/residual-offpolicy-rl`
- paper/code name: ResFiT, Residual Off-Policy RL for Finetuning Behavior Cloning Policies
- 주요 파일:
- `/home/junhyeong/repos/residual-offpolicy-rl/resfit/rl_finetuning/wrappers/residual_env_wrapper.py`
- `/home/junhyeong/repos/residual-offpolicy-rl/resfit/rl_finetuning/off_policy/rl/q_agent.py`
- `/home/junhyeong/repos/residual-offpolicy-rl/resfit/rl_finetuning/off_policy/rl/actor.py`
- `/home/junhyeong/repos/residual-offpolicy-rl/resfit/rl_finetuning/scripts/train_residual_td3.py`
- `/home/junhyeong/repos/residual-offpolicy-rl/resfit/rl_finetuning/config/residual_td3.py`

주의: ResFiT repo는 PyTorch + LeRobot + TorchRL + Hydra + vectorized DexMG/Robosuite 중심이다. 현재 이 레포는 JAX/Flax + numpy replay + YAML builder + Robomimic low-dim 중심이므로 직접 import/vendor 방식은 비용이 크다.

## 핵심 아이디어

일반 RL은 policy가 바로 env action을 낸다.

```text
actor(s) -> a
Q(s, a)
```

Residual policy는 frozen base policy 위에서 correction만 학습한다.

```text
base_policy(s) -> a_base
residual_actor(s, a_base) -> delta
a_exec = clip(a_base + delta)
Q(s, a_exec)
```

여기서 중요한 점은 critic이 residual action `delta`가 아니라 실제 env에 실행된 `a_exec`를 학습한다는 것이다. actor만 residual `delta`를 출력한다.

## Gradient Semantics

`a_base`는 Q backprop에서 detach/stop-gradient하는 것이 기본이다.

```text
a_base = stop_gradient(base_policy(s))
delta = residual_actor(s, a_base)
a_exec = clip(a_base + delta)
loss_actor = -Q(s, a_exec)
```

이렇게 해야 gradient가 residual actor로만 흐른다. base policy까지 Q gradient로 업데이트하면 "BC policy 위에 residual correction을 학습한다"는 실험 의미가 깨진다.

나중에 base policy까지 joint finetune하는 실험은 가능하지만, 그것은 ResFiT 기본 구조와 다른 ablation으로 봐야 한다.

## 일반 RL과 다른 구현 포인트

### 1. Base action이 actor input에 들어간다

Residual actor는 `s`만 보지 않고 `s, a_base`를 같이 본다. ResFiT 구현은 observation dict에 `observation.base_action`을 추가한다.

우리 레포의 low-dim v1에서는 가장 단순하게 `actor_observation = concat(state, base_action)`으로 시작할 수 있다. 다만 critic까지 같은 observation shape을 쓰게 할지, actor/critic observation을 분리할지는 결정이 필요하다.

### 2. Replay의 `actions`는 combined action이어야 한다

Replay에서 `actions`는 env에 실제 실행한 action이어야 한다.

```text
actions = a_exec = clip(a_base + delta)
```

추가로 분석과 residual loss를 위해 아래 key를 명시적으로 저장하는 것이 좋다.

```text
base_actions
residual_actions
next_base_actions
```

`next_base_actions`는 target actor로 next residual을 만들 때 필요하다.

### 3. Actor는 residual만 출력한다

Critic loss:

```text
target_action = clip(next_base_action + residual_target_actor(next_obs, next_base_action))
target_q = r + gamma^n * mask * Q_target(next_obs, target_action)
critic_loss = (Q(obs, actions) - target_q)^2
```

Actor loss:

```text
delta = residual_actor(obs, base_action)
a_for_q = clip(base_action + delta)
actor_loss = -mean(Q(obs, a_for_q))
```

### 4. Residual scale과 final init이 중요하다

ResFiT은 residual actor output scale을 작게 둔다. 예시는 `action_scale=0.1` 또는 `0.2`이고, actor final layer init도 `0.0`에 가깝게 둔다.

의도는 학습 초기에 `delta ~= 0`이 되게 해서 실행 policy가 거의 base policy로 시작하도록 만드는 것이다. 이 장치가 없으면 초반 random residual이 base policy의 행동을 망가뜨릴 수 있다.

### 5. Warmup도 base + noise가 자연스럽다

일반 off-policy RL처럼 완전 random action으로 online buffer를 채우는 것보다, residual setting에서는 아래가 더 자연스럽다.

```text
a_exec = clip(a_base + noise)
```

ResFiT에도 `use_base_policy_for_warmup`이 있고, 기본 방향은 base 주변에서 local exploration을 하는 것이다.

### 6. Offline/demo buffer 생성 방식이 다르다

Demo transition에서 `a_demo`가 있을 때, base policy를 같은 state에 돌려 `a_base`를 구해야 residual dataset이 된다.

```text
state -> base_policy -> a_base
demo action -> a_demo
residual target intuition -> a_demo - a_base
```

다만 critic은 여전히 `actions = a_demo`를 학습하면 된다. Residual actor를 BC로 pretrain하려면 `a_demo - a_base`가 target이 된다.

### 7. Action normalization/scaling

ResFiT은 dataset action min/max 기반 `ActionScaler`로 action을 `[-1, 1]`로 정규화하고, residual도 normalized action space에서 더한다.

Robomimic low-dim action은 대체로 `[-1, 1]`이므로 v1은 단순 clip으로 시작해도 된다. 하지만 task별 action range가 다르거나 gripper scale이 이상하면 action scaler가 필요하다.

### 8. Stabilization 장치

ResFiT에는 residual 구조 외에도 안정화 장치가 있다.

- critic-only warmup
- n-step backup
- multi-Q / REDQ style ensemble
- target action noise
- prioritized replay
- actor LR를 작게 설정
- residual action L2 penalty

이 중 residual 구현 v1에서 우선 넣을 만한 것은 critic warmup, residual scale, small final init, base+noise warmup이다. PER와 large Q ensemble은 후순위로 둔다.

## PER에 대한 판단

PER 자체는 구현 가능하지만 residual의 필수 요소는 아니다.

필요 요소:

```text
p_i = priority_i ** alpha
P(i) = p_i / sum_j p_j
w_i = (N * P(i)) ** (-beta)
priority_i = abs(td_error_i) + eps
```

우리 코드에 붙이려면 다음이 필요하다.

- `ReplayBuffer.sample_sequence()`에서 priority 기반 index sampling 지원
- batch에 `indices`, `importance_weights` 포함
- `rlpd.critic_loss()`가 per-sample TD error와 weighted loss를 반환
- update 후 `ReplayBuffer.update_priorities(indices, td_errors)` 호출
- mixed buffer와 aux BC batch에서는 PER 적용 범위를 명확히 분리

v1 우선순위는 낮다. 먼저 residual action path와 actor/critic loss가 맞는지 검증한 뒤 PER를 넣는 게 맞다.

## 우리 레포에 붙이는 권장 설계

v1 목표는 low-dim Robomimic/ToolHang 기준 residual RLPD를 먼저 붙이는 것이다.

최소 구현:

```text
base_policy = frozen BCFlow or RLPD policy
residual_policy = trainable residual RLPD actor
a_base = stop_gradient(base_policy(obs))
delta = residual_policy(obs, a_base)
a_exec = clip(a_base + delta)
env.step(a_exec)
```

Replay transition:

```text
actions = a_exec
base_actions = a_base
residual_actions = delta
next_base_actions = base_policy(next_obs)
```

Config 방향:

```yaml
actors:
  base:
    kind: bc_flow
    trainable: false
    pretrained_path: ...
  learner:
    kind: residual_rlpd
    trainable: true
    residual:
      scale: 0.1
      base_action_key: base_actions
      action_l2: 0.0

training:
  action_composition: residual
```

이름은 확정 전이지만, `learner.kind: residual_rlpd` 또는 `training.action_composition: residual` 중 하나로 명확히 표현해야 한다.

## Intervention Learning과의 관계

Residual policy는 intervention과 다르다.

- Intervention/gating: learner와 expert 중 누가 env를 제어할지 고른다.
- Residual policy: base policy action에 residual correction을 더해 하나의 action을 만든다.

즉 아래 두 구조는 별개다.

```text
gate: choose learner or expert
residual: execute base + learner_residual
```

나중에는 gate와 residual을 결합할 수 있다.

예시:

```text
normal step: execute base + residual
gate on: execute expert action
```

하지만 v1에서는 residual 단독 path부터 검증하는 것이 안전하다.

## 구현 로드맵

### Phase 1. 문서와 schema 결정

- residual policy config 이름 결정
- replay key 결정: `base_actions`, `residual_actions`, `next_base_actions`
- actor/critic observation shape 결정

### Phase 2. Rollout path

- base policy를 query한다.
- residual learner를 query한다.
- `a_exec = clip(a_base + delta)`를 env에 실행한다.
- replay에 base/residual/combined action을 저장한다.

### Phase 3. Residual RLPD loss

- critic은 `batch["actions"]`를 combined action으로 본다.
- target action은 `next_base_actions + target_residual_actor(...)`.
- actor loss는 `Q(obs, base_actions + residual_actor(...))`로 계산한다.
- `base_actions`는 stop-gradient 처리한다.

### Phase 4. Smoke test

- base policy: ToolHang top200 BCFlow 1M 또는 Square BCFlow pretrained
- learner: residual RLPD random init
- 100-step real env smoke
- 확인할 로그:
- `residual/action_l1`
- `residual/action_l2`
- `action/base_norm`
- `action/executed_norm`
- `critic/loss`
- `actor/loss`
- env success/reward

### Phase 5. 안정화 옵션

- critic warmup
- base + noise warmup
- residual final layer small init
- residual action L2 penalty
- PER는 그 다음 단계

## 남은 결정 사항

- actor observation에 `base_action`을 concat할지, observation dict key로 둘지.
- critic도 base action을 observation으로 볼지, 순수 `Q(s, a_exec)`만 볼지.
- base policy가 action chunk를 낼 때 queue를 어떻게 관리할지.
- stochastic base policy를 deterministic eval action으로 고정할지.
- offline demo에서 base action을 미리 cache할지, training time에 계산할지.
- action scaler를 v1부터 넣을지, Robomimic action clip으로 시작할지.
