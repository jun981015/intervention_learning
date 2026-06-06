# Config Schema Decisions

이 문서는 `intervention_learning`의 train loop를 config 중심으로 구성하기 위해 현재 합의한 schema 결정을 기록한다. 목적은 DAgger 같은 IL method부터 RLPD 같은 RL method, 나중의 RL+BC hybrid까지 같은 train entrypoint에서 다루되, 지금 시점에서 과도한 objective scheduler를 만들지 않는 것이다.

## 핵심 원칙

- `actors.learner.kind`가 algorithm class를 결정한다.
- algorithm class가 자기 `update()` rule을 책임진다.
- top-level `objectives` 필드는 두지 않는다.
- loss 종류를 config로 중복 지정하지 않는다.
- config는 data source와 sampling 방식, rollout/intervention 방식, logging/checkpointing 운영 방식을 정한다.
- hybrid method가 필요하면 새 `kind`를 추가하거나, 필요한 경우 named sampling batch를 넘긴다.

## Top-level Fields

### `experiment`

실험 식별자와 재현성 메타정보를 둔다.

```yaml
experiment:
  name: square_dagger_top50_500k
  seed: 0
  output_dir: exp/runs
  tags: [square, dagger, bcflow]
```

역할:

- run 이름
- seed
- output root
- tag
- config snapshot 기준

### `env`

환경 생성 설정을 둔다.

```yaml
env:
  kind: robomimic
  name: square-mh-low_dim
  observation_mode: lowdim
  max_episode_steps: 400
  render_offscreen: false
  reward_scale: 1.0
  reward_shift: 0.0
```

역할:

- env wrapper 선택
- task name
- observation mode
- horizon
- render/camera 관련 설정
- sparse task reward의 affine transform 설정

Reward transform:

- Robomimic 기본 task reward는 성공 여부 기반 `task_reward = float(success)`로 둔다.
- env가 반환하는 reward는 `reward = task_reward * reward_scale + reward_shift`다.
- 기본값 `reward_scale=1.0`, `reward_shift=0.0`은 기존 binary reward와 동일하다.
- 예를 들어 모든 step에 `-1`, 성공 step에 `0`을 주고 싶으면 `reward_scale=1.0`, `reward_shift=-1.0`을 쓴다.
- `info["task_reward"]`에는 transform 전 sparse task reward를 남긴다.

현재 image 관련 결정:

- env wrapper와 replay는 image input을 받을 준비를 한다.
- `observation_mode: lowdim`은 기존 호환용 array state observation이다.
- `observation_mode: state`는 `{"state": ...}` dict observation으로, image key를 나중에 추가해도 buffer/policy access pattern을 덜 바꾸기 위한 state-only mode다.
- `observation_mode: pixels_state`는 같은 dict 구조에 camera image leaf를 추가한다.
- policy/network가 image를 어떻게 쓸지는 아직 결정하지 않았다.
- 당장 train은 low-dim state policy 기준으로 진행한다.
- image policy 학습은 별도 설계 후 붙인다.

관련 문서: [NETWORKS.md#image-observation-todo](NETWORKS.md#image-observation-todo), [PIPELINE.md#image-observation-상태](PIPELINE.md#image-observation-상태)

### `actors`

learner/expert policy를 어떻게 만들지 정한다. `kind`는 registry key이며, policy/learning algorithm class를 선택한다.

```yaml
actors:
  learner:
    kind: bc_flow
    trainable: true
    pretrained_path: exp/pretrained/bcflow_square_top50_actorln_seed0_500k
    network:
      actor_hidden_dims: [512, 512, 512, 512]
      actor_activation: mish
      actor_layer_norm: true
      horizon_length: 5
      action_chunking: true
      flow_steps: 10
    optimization:
      lr: 3e-4
      batch_size: 256
      grad_clip_norm: 10.0
      weight_decay: 0.0
    update:
      target_action_key: expert_actions

  expert:
    kind: rlpd
    trainable: false
    pretrained_path: exp/pretrained/rlpd_square_bc03_seed0_2m
    network:
      actor_hidden_dims: [256, 256, 256]
      actor_activation: mish
      critic_hidden_dims: [256, 256, 256]
      critic_activation: mish
      actor_layer_norm: true
      critic_layer_norm: true
      num_qs: 2
```

결정:

- `kind`를 사용한다. `algorithm`, `type`, `class_path`보다 실험 config용 enum에 가깝고, env/gate/actor에 일관되게 쓸 수 있다.
- network 구조는 top-level이 아니라 각 actor 밑에 둔다.
- network 구조는 algorithm에 depend하므로 `actors.<role>.network`가 자연스럽다.
- activation은 network field로 둔다.
- RLPD처럼 actor/critic이 나뉘면 `actor_activation`, `critic_activation`처럼 명시한다.
- BCFlow처럼 actor만 있으면 `actor_activation` 또는 builder fallback으로 `activation`을 허용할 수 있다.
- BC 계열도 `update.train_critic: true`를 주면 보조 critic을 학습할 수 있다. 이 critic은 진단, value 시각화, gate용 Q API를 위한 것이며 actor/policy loss에는 쓰지 않는다.
- `update`에는 algorithm-specific update 옵션만 둔다. 예: BC 계열의 `target_action_key`, `train_critic`, `critic_loss_coef`.

### `training`

전체 env loop와 update timing을 정한다. 어떤 loss를 쓰는지는 learner kind가 결정한다.

```yaml
training:
  total_steps: 300000
  start_training: 1000
  initial_collect:
    unit: steps  # steps | episodes
    count: 1000
  update_interval: 1
  updates_per_step: 1
  action_mode: first_action
```

역할:

- 전체 env step 수
- update 시작 전 online replay를 얼마나 채울지
- update 주기
- env step당 gradient update 수
- action chunk 실행 방식

`initial_collect`는 online interaction으로 replay를 먼저 채우는 단계다. 이 조건을 만족하기 전에는 env step과 buffer 저장만 하고 gradient update는 하지 않는다.

- `unit: steps`이면 env step 수 기준으로 채운다.
- `unit: episodes`이면 episode 종료 개수 기준으로 채운다.
- `initial_collect`가 없으면 기존 config 호환을 위해 `start_training`을 step 단위 collect count로 해석한다.
- 둘 다 없으면 기본값은 `DEFAULT_RECIPE.train.start_training = 1000` step이다.
- `start_training`은 legacy alias로 남겨두되, 새 config에서는 `initial_collect`를 명시하는 쪽을 우선한다.

현재 action mode 결정:

- v0는 `first_action` 기준으로 시작한다.
- chunk policy가 action chunk를 출력해도 매 step 새 chunk를 뽑고 첫 action만 실행하는 receding-horizon 방식이다.
- `chunk_queue`는 나중에 추가할 수 있다.
- chunk queue를 도입하면 learner/expert 각각 독립 queue가 필요하고, gate 전환 시 queue 의미를 명확히 해야 한다.

### `replay`

buffer 크기, prefill, sampling recipe를 정한다.

```yaml
replay:
  buffers:
    online_size: 500000
    demo_size: 500000
    intervention_size: 500000
  sampling:
    bc:
      source:
        online: 1.0
      sequence_length: 5
      batch_size: 256
      boundary_mode: drop
```

결정:

- `replay.prefill.<buffer>`는 파일 `format`과 semantic `adapter`를 분리한다. `adapter`는 생략 가능하며, `npz`는 `replay_npz`, Robomimic demo format은 `demo_actions_are_expert`로 기본 추론한다.
- `adapter`가 dataset semantic을 canonical replay schema로 변환한다. loader는 파일 구조만 읽고 `actions -> expert_actions` 같은 semantic copy를 하지 않는다.
- top-level `objectives` 대신 `replay.sampling`에서 batch를 이름 붙여 뽑을 수 있게 열어둔다.
- pure BC/DAgger는 `sampling.bc`만 있으면 된다.
- pure RL은 `sampling.rl`만 있으면 된다.
- RL+BC hybrid는 `sampling.rl`, `sampling.bc`를 둘 다 제공하고, learner kind가 필요한 batch를 사용한다.

예시 RL+BC:

```yaml
replay:
  sampling:
    rl:
      source:
        online: 1.0
      sequence_length: 1
      batch_size: 256
      boundary_mode: drop
    bc:
      source:
        demo: 0.5
        intervention: 0.5
      sequence_length: 5
      batch_size: 256
      boundary_mode: drop
```

train loop는 named batches를 이렇게 넘길 수 있다.

```python
batch = {
    "rl": rl_batch,
    "bc": bc_batch,
}
learner.update(batch)
```

algorithm class는 필요한 key만 사용한다.

```python
BCFlowAgent.update(batch)  # uses batch["bc"] or batch directly
RLPDAgent.update(batch)    # uses batch["rl"] or batch directly
RLPDBC.update(batch)       # uses both batch["rl"] and batch["bc"]
```

이 방식은 top-level objective scheduler 없이도 future hybrid method를 열어둔다.

### `intervention`

expert가 언제 개입하는지와 expert query 정책을 정한다. gate는 policy selector가 아니라 expert intervention trigger다.

```yaml
intervention:
  enabled: false
  expert_query: always
  gate:
    kind: always_off
```

개념:

```text
g_t = gate(s_t, history, learner_info, maybe env_info)

g_t = 0 -> learner 계속 실행
g_t = 1 -> expert intervention

pi(a | s) = (1 - g_t) pi_learner(a | s) + g_t pi_expert(a | s)
```

중요한 해석:

- gate는 “어떤 policy를 query할지”가 아니다.
- gate는 “expert가 개입해야 하는 시점인지”를 판단하는 함수다.
- learner가 expert trajectory/manifold에서 너무 벗어났거나, unsafe/low-value/high-uncertainty이면 expert가 개입한다.

`expert_query`는 gate와 분리한다.

```text
always
  DAgger label 저장용. expert가 실행하지 않아도 매 step expert action을 저장한다.

on_intervention
  gate가 켜질 때만 expert를 호출한다.

never
  expert를 호출하지 않는다.
```

DAgger v0:

```yaml
intervention:
  enabled: false
  expert_query: always
  gate:
    kind: always_off
```

진짜 intervention method 예시:

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

`expert_q_gap` 의미:

```text
q_gap = Q_expert(s, a_expert) - Q_expert(s, a_learner)
signal = q_gap > threshold
if signal: expert intervention starts with probability intervention_prob
```

- `intervention_prob`은 signal이 켜졌을 때만 적용한다. 별도 `p_off`는 두지 않는다.
- `intervention_horizon`은 intervention이 시작된 뒤 expert가 연속으로 제어하는 env step 수다.
- `q_agg`는 ensemble Q head를 scalar로 줄이는 방식이며 `min`, `mean`, `max`를 지원한다.
- gate는 expert 내부 network/module 이름을 모른다. `critic`, `q`, `qf` 같은 이름을 직접 탐색하지 않는다.
- gate는 `q_agg` 문자열을 expert API에 넘길 뿐, Q head shape이나 aggregation 구현을 가정하지 않는다.
- expert agent 또는 adapter가 `evaluate_q(obs, action, q_agg=...)` 또는 `q_values(obs, action, q_agg=...)`를 제공해야 한다.
- multi-Q 알고리즘은 필요하면 `evaluate_q_heads(obs, action)`를 추가로 제공하고, `evaluate_q(..., q_agg=...)`에서 scalar Q를 반환한다.
- 이 gate는 RLPD 전용이 아니다. SAC, TD3-BC 등도 adapter가 위 API를 맞추면 같은 방식으로 쓸 수 있다.
- PPO처럼 value-only critic만 있는 expert는 `Q(s, a_expert) - Q(s, a_learner)`를 계산할 수 없으므로 action-value head 또는 별도 adapter가 필요하다.

action uncertainty gate 예시:

```yaml
intervention:
  enabled: true
  expert_query: always
  gate:
    kind: action_uncertainty
    source: learner
    estimator: sample_variance
    num_samples: 8
    score: rms_std
    threshold: 0.15
    intervention_prob: 1.0
    intervention_horizon: 10
```

`action_uncertainty` 의미:

```text
samples = [pi_source(s; rng_i) for i in 1..num_samples]
score = sqrt(mean(var(samples, axis=sample)))
signal = score > threshold
if signal: expert intervention starts with probability intervention_prob
```

- 현재 구현된 estimator는 `sample_variance` 하나다.
- `source`는 `learner`, `expert`, `base`를 지원한다. `base`는 residual rollout에서 base policy uncertainty를 따로 볼 때 쓴다.
- `score`는 현재 `rms_std`만 지원한다. action dimension별 variance 평균의 square root를 gate score로 기록한다.
- 이 score는 executed action space 기준 variance다. Diffusion/flow BC처럼 analytic std가 없는 policy에도 적용할 수 있고, SAC/RLPD처럼 stochastic actor가 있는 policy에도 적용할 수 있다.
- SAC actor의 analytic `log_std` / entropy, BC ensemble disagreement 같은 backend는 아직 구현하지 않았다. 같은 gate family의 estimator로 추가하는 것이 다음 확장 방향이다.
- gate가 policy를 여러 번 다시 샘플링해야 하므로 rollout은 `GateContext`를 통해 policy diagnostic sampler를 넘긴다. 이 diagnostic sampling은 env를 step하거나 replay를 쓰지 않는다.

### `storage`

rollout에서 나온 정보를 replay에 무엇까지 저장할지 정한다.

```yaml
storage:
  store_executed_action: true
  store_learner_action: true
  store_expert_action: true
  store_log_prob: true
  store_gate_score: true
```

의미:

```text
actions          = 실제 env에 실행된 action
learner_actions  = learner proposal
expert_actions   = expert proposal/label, 없으면 NaN placeholder 가능
controller_ids   = 실제 실행 controller
interventions    = expert intervention 여부
gating_scores    = gate score, deviation, uncertainty 등
```

DAgger에서는 일반적으로:

```text
actions == learner_actions
expert_actions = BC target label
interventions = 0
```

Intervention에서는:

```text
interventions = 1인 step에서 actions == expert_actions
interventions = 0인 step에서 actions == learner_actions
```

### `evaluation`

학습 중 평가 설정을 둔다.

```yaml
evaluation:
  interval: 50000
  episodes: 20
  seed: 1000
  render_video: false
  video_episodes: 5
```

역할:

- eval 주기
- eval episode 수
- eval seed
- video 저장 여부
- `render_video: true`와 `video_episodes > 0`이면 evaluator가 metric episode 뒤에 별도 video episode를 굴리고 `env.render()` frame을 mp4로 저장한다. 이 video episode는 eval 통계에 포함하지 않는다.
- video 저장을 요청했는데 env가 `render()`를 제공하지 않으면 runtime error를 낸다. Robomimic eval env는 video 요청 시 offscreen render가 켜진 eval env로 만든다.
- success/return/length metric 생성

### `logging`

metric 기록 방식을 둔다.

```yaml
logging:
  stdout_interval: 1000
  csv: true
  jsonl: true
  wandb:
    enabled: true
    project: intervention_learning
    group: square_dagger
```

결정:

- stdout, CSV, JSONL, WandB를 같은 metric dict에서 동시에 기록한다.
- CSV는 논문 그래프용이므로 metric key를 안정적으로 유지한다.
- JSONL은 full metric/debug record용으로 둔다.
- WandB는 실시간 모니터링용이다.

### `checkpointing`

weight와 replay 저장 정책을 둔다.

```yaml
checkpointing:
  interval: 100000
  save_final: true
  save_replay: true
  keep_last: 3
```

역할:

- learner checkpoint 저장 주기
- final checkpoint 저장 여부
- replay buffer 저장 여부
- 오래된 checkpoint 보존 정책

## DAgger v0 Target Config

Concrete YAML draft: [`config/dagger.yaml`](../config/dagger.yaml).

당장 구현할 baseline은 아래 의미를 갖는다.

```text
learner = BCFlow top50 500k
expert = RLPD bc0.3
training.action_mode = first_action
intervention.enabled = false
intervention.expert_query = always
replay.sampling.bc.source = online
actors.learner.update.target_action_key = expert_actions
logging = stdout + csv + jsonl + wandb
checkpoint = 100k interval
```

예시:

```yaml
experiment:
  name: square_dagger_bcflow_top50_500k
  seed: 0
  output_dir: exp/runs

env:
  kind: robomimic
  name: square-mh-low_dim
  observation_mode: lowdim
  max_episode_steps: 400

actors:
  learner:
    kind: bc_flow
    trainable: true
    pretrained_path: exp/pretrained/bcflow_square_top50_actorln_seed0_500k
    network:
      actor_hidden_dims: [512, 512, 512, 512]
      actor_activation: mish
      actor_layer_norm: true
      horizon_length: 5
      action_chunking: true
      flow_steps: 10
    optimization:
      lr: 3e-4
      batch_size: 256
      grad_clip_norm: 10.0
    update:
      target_action_key: expert_actions

  expert:
    kind: rlpd
    trainable: false
    pretrained_path: exp/pretrained/rlpd_square_bc03_seed0_2m

training:
  total_steps: 300000
  start_training: 1000
  update_interval: 1
  updates_per_step: 1
  action_mode: first_action

replay:
  buffers:
    online_size: 500000
    demo_size: 500000
    intervention_size: 500000
  sampling:
    bc:
      source:
        online: 1.0
      sequence_length: 5
      batch_size: 256
      boundary_mode: drop

intervention:
  enabled: false
  expert_query: always
  gate:
    kind: always_off

storage:
  store_executed_action: true
  store_learner_action: true
  store_expert_action: true
  store_log_prob: true
  store_gate_score: true

evaluation:
  interval: 50000
  episodes: 20
  seed: 1000
  render_video: false

logging:
  stdout_interval: 1000
  csv: true
  jsonl: true
  wandb:
    enabled: true
    project: intervention_learning
    group: square_dagger

checkpointing:
  interval: 100000
  save_final: true
  save_replay: true
  keep_last: 3
```

## What To Finish Next

1. Config loader/schema validation을 만든다.
2. 위 DAgger v0 config를 실제 YAML 파일로 만든다.
3. builder가 `actors.learner.kind`, `actors.expert.kind`로 policy를 만든다.
4. train loop가 `intervention.expert_query=always`를 보고 learner rollout 중 expert label을 저장한다.
5. replay sampler가 `replay.sampling.bc`를 읽어 named batch를 만든다.
6. learner `kind=bc_flow`가 `batch["bc"]` 또는 single batch를 받아 `target_action_key=expert_actions`로 update한다.
7. stdout/CSV/JSONL/WandB logger를 같은 metric dict 기반으로 붙인다.
8. checkpoint/replay 저장 경로를 `experiment.output_dir` 아래로 정리한다.
9. 100-step smoke를 먼저 돌려 replay 저장 key와 update loss를 검증한다.
10. smoke 통과 후 300k DAgger train을 실행한다.

## Deferred Decisions

- image policy 학습 방식.
- smart intervention gate.
- human UI.
- `chunk_queue` action execution.
- `boundary_mode=truncate`.
- top-level `objectives` scheduler.
- RL+BC hybrid class 이름과 내부 update 순서.
