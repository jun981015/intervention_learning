# Implementation TODO Bookmark

이 문서는 TODO 자체를 길게 설명하는 곳이 아니라, 남은 작업을 한눈에 보고 세부 설계 문서로 바로
이동하기 위한 책갈피다. 실제 설계 근거와 상세 설명은 각 링크 문서를 기준으로 본다.

## 우선순위 높은 작업


### 현재 상태 스냅샷

2026-05-21 기준으로 DAgger v0, pretrained loading, replay/buffer, env/image, logger 상태를 별도 문서에 정리했다.

- current status: [STATUS_2026-05-21.md](STATUS_2026-05-21.md)
- logging and metrics: [LOGGING_AND_METRICS.md](LOGGING_AND_METRICS.md)
- residual policy integration: [RESIDUAL_POLICY_ANALYSIS.md](RESIDUAL_POLICY_ANALYSIS.md)

### Config schema 결정 사항

현재 config schema, DAgger v0 표현 방식, `actors.learner.kind`가 update rule을 결정한다는 원칙,
`replay.sampling` named batch 구조, intervention gate의 의미는 별도 결정 문서에 정리한다.

- config schema decisions: [CONFIG_SCHEMA_DECISIONS.md](CONFIG_SCHEMA_DECISIONS.md)

### Residual policy / ResFiT integration

ResFiT의 핵심은 frozen BC/diffusion base policy 위에 residual actor를 올리는 것이다. 일반 RL과 달리
actor는 residual `delta`만 출력하고, critic은 실제 실행 action `a_exec = clip(a_base + delta)`를
학습한다. 세부 분석과 구현 로드맵은 [RESIDUAL_POLICY_ANALYSIS.md](RESIDUAL_POLICY_ANALYSIS.md)를 본다.

구현 완료:

- `a_base`는 Q backprop에서 stop-gradient 처리한다.
- replay schema에 `base_actions`, `residual_actions`, `next_base_actions`를 추가한다.
- rollout에 base policy query와 residual composition path를 추가한다.
- `residual_rlpd` 또는 동등한 config 표현을 정하고 actor/critic loss를 구현한다.
- actor `horizon_length`와 replay/update `sequence_length`를 분리한다.
- optional `cache_base_actions`로 prefill dataset의 residual metadata를 미리 채운다.
- residual critic-only warmup update를 지원한다.
- residual rollout에서 base+noise warmup을 지원한다.
- residual actor output head를 작게 초기화하는 `actor_final_fc_init_scale`을 지원한다.
- `residual_td3`를 추가했다. deterministic residual actor, target actor, target policy smoothing, delayed actor update, UTD=4 config path를 지원한다.

남은 우선순위:

- 실제 large Robomimic/ToolHang demo prefill에서 `cache_base_actions` runtime과 메모리 비용을 측정한다.
- residual BC regularization / pretraining은 code-level smoke를 통과했다. 다음은 실제 실험 config에서 성능과 runtime을 검증한다.
- learner/expert 일반 action chunk queue와 vector env별 queue를 설계한다.
- ActionScaler는 Robomimic `[-1, 1]` action clip 밖의 task가 필요해질 때 추가한다.
- PER와 large Q ensemble은 residual 구현 검증 후 안정화 옵션으로 미룬다.


### 1. 현재 문서와 코드 상태 동기화

현재 `IMPLEMENTATION_PLAN`과 `STATUS`에는 이미 일부 구현된 항목도 TODO로 남아 있다. 새 작업을
시작하기 전에 문서가 현재 코드 상태를 정확히 반영하도록 정리한다.

- 전체 구현 순서: [IMPLEMENTATION_PLAN.md#다음-구현-순서](IMPLEMENTATION_PLAN.md#다음-구현-순서)
- 문서상 아직 안 된 것: [IMPLEMENTATION_PLAN.md#아직-안-된-것](IMPLEMENTATION_PLAN.md#아직-안-된-것)
- 2026-05-18 기준 남은 작업: [STATUS_2026-05-18.md#아직-남은-작업](STATUS_2026-05-18.md#아직-남은-작업)
- 작업 진행 방식과 기준 원칙: [WORKFLOW.md#기준-원칙](WORKFLOW.md#기준-원칙)

### Expert-Q gap gate smoke

`expert_q_gap` gate는 실제 pretrained expert checkpoint를 붙인 Robomimic 100-step rollout smoke를 통과했다. 결과와 재실행 명령은 [REAL_ENV_SMOKE_TESTS.md#expert-q-gap-gate-real-env-smoke](REAL_ENV_SMOKE_TESTS.md#expert-q-gap-gate-real-env-smoke)를 본다.

- gate config와 의미: [CONFIG_SCHEMA_DECISIONS.md#intervention](CONFIG_SCHEMA_DECISIONS.md#intervention)
- rollout에서 gate가 호출되는 위치: [PIPELINE.md#unified-train-loop](PIPELINE.md#unified-train-loop)
- 예시 config: [../config/expert_q_gap_square.yaml](../config/expert_q_gap_square.yaml)

검증 포인트:

- RLPD expert는 `evaluate_q(obs, action, q_agg=...)` 경로로 Q gap을 계산한다.
- future SAC/TD3-BC expert는 `evaluate_q(..., q_agg=...)` 또는 `q_values(..., q_agg=...)`만 제공하면 같은 gate를 쓸 수 있어야 한다.
- PPO처럼 V-only expert는 action-value head가 없으면 이 gate를 쓰지 못한다는 에러가 나와야 한다.
- `gate/q_expert`, `gate/q_learner`, `gate/q_gap`, `gate/signal`, `gate/intervention_started`가 로그에 찍히는지 확인한다.


### 2. Dataset adapter / canonicalization interface 추가

offline demo dataset은 source마다 `actions`의 의미가 다르다. `ReplayBuffer` loader가 암묵적으로
`actions -> expert_actions`를 복사하지 말고, dataset 종류별 adapter가 명시적으로 우리 canonical replay
schema로 변환해야 한다. 이 작업은 offline demo prefill과 DAgger/BC target 안정성에 직접 연결되므로
우선순위를 높게 둔다.

- initial replay prefill: [REPLAY_AND_UPDATES.md#initial-replay-prefill](REPLAY_AND_UPDATES.md#initial-replay-prefill)
- transition schema: [REPLAY_AND_UPDATES.md#transition-schema](REPLAY_AND_UPDATES.md#transition-schema)
- buffer 역할: [PIPELINE.md#buffer-역할](PIPELINE.md#buffer-역할)

결정해야 할 것:

- `adapter: replay_npz`는 schema-compatible saved replay만 그대로 로드한다.
- `adapter: demo_actions_are_expert` 같은 명시적 adapter에서만 `expert_actions = actions`를 채운다.
- raw robomimic hdf5, saved replay npz, intervention replay npz를 각각 어떤 adapter로 둘지 정한다.
- missing `learner_actions`, log-prob, controller/gate metadata를 어떤 default로 채울지 정한다.
- adapter 적용 위치는 `build_buffers()` prefill 경로가 자연스럽다.

주의:

- dataset semantics 없이 loader 내부에서 `actions -> expert_actions`를 자동 복사하지 않는다.
- `target_action_key="expert_actions"` update를 쓰는 prefill dataset은 adapter 단계에서 finite expert labels를 보장해야 한다.

### 3. Online rollout smoke를 실제 Robomimic env에서 검증

restored learner/expert와 gate를 실제 Square env에 붙인 100-step smoke는 DAgger relabel과 expert-Q gap 둘 다 통과했다. 결과와 재실행 명령은 [REAL_ENV_SMOKE_TESTS.md](REAL_ENV_SMOKE_TESTS.md)를 본다.

남은 것은 real-env 산출물 replay save/load round-trip, DAgger update-on smoke, video/action sync 검증이다.

- 초기 intervention pipeline: [PIPELINE.md#초기-파이프라인](PIPELINE.md#초기-파이프라인)
- step 단위 로직: [PIPELINE.md#step-logic](PIPELINE.md#step-logic)
- unified train loop 상태: [PIPELINE.md#unified-train-loop](PIPELINE.md#unified-train-loop)
- transition schema: [REPLAY_AND_UPDATES.md#transition-schema](REPLAY_AND_UPDATES.md#transition-schema)
- terminal/mask 처리: [REPLAY_AND_UPDATES.md#terminal과-mask](REPLAY_AND_UPDATES.md#terminal과-mask)

검증 포인트:

- `actions`는 실제 env에 실행된 action이어야 한다.
- `learner_actions`, `expert_actions`는 같은 state에서 나온 proposal이어야 한다.
- `controller_ids`, `gating_reasons`, `gating_scores`, `interventions`가 episode 흐름과 맞아야 한다.
- timeout/truncation에서 `masks`가 bootstrap 가능한 형태로 유지되는지 확인한다.

### 4. Replay save/load round-trip test 추가

온라인 rollout으로 저장한 replay buffer를 다시 로드했을 때 schema, shape, episode metadata,
image metadata가 보존되는지 테스트한다.

- replay schema: [REPLAY_AND_UPDATES.md#transition-schema](REPLAY_AND_UPDATES.md#transition-schema)
- image replay 저장 방식: [REPLAY_AND_UPDATES.md#image-replay-storage](REPLAY_AND_UPDATES.md#image-replay-storage)
- initial replay prefill: [REPLAY_AND_UPDATES.md#initial-replay-prefill](REPLAY_AND_UPDATES.md#initial-replay-prefill)
- demo insert mode와 episode index: [REPLAY_AND_UPDATES.md#demo-episode-insert-mode](REPLAY_AND_UPDATES.md#demo-episode-insert-mode)

검증 포인트:

- `ReplayBuffer.save_npz()` 후 load했을 때 transition count가 유지되는지 확인한다.
- `episode_ids`, `episode_steps`가 유지되는지 확인한다.
- image observation은 current frame만 저장하고, sample 시 `i+1` frame으로 `next_observations`가 복원되는지 확인한다.
- prefill된 demo와 online rollout의 episode id 충돌을 피하는지 확인한다.

### 5. DAgger / BC update 경로 검증

rollout 중 저장된 `expert_actions`를 이용해서 BCFlow learner update가 실제로 도는지 확인한다.

- DAgger baseline 흐름: [DAGGER_BASELINE.md](DAGGER_BASELINE.md)
- BC Flow target action key 결정: [STATUS_2026-05-18.md#dagger-baseline](STATUS_2026-05-18.md#dagger-baseline)
- unified train loop update 위치: [PIPELINE.md#unified-train-loop](PIPELINE.md#unified-train-loop)
- mixed buffer sampling: [REPLAY_AND_UPDATES.md#mixed-buffer-sampling](REPLAY_AND_UPDATES.md#mixed-buffer-sampling)

검증 포인트:

- `target_action_key="expert_actions"`일 때 loss가 NaN 없이 감소하는지 확인한다.
- 완료: `expert_actions`에 NaN/Inf가 있으면 BC update 직전에 `ValueError`로 중단한다.
- online/demo/intervention buffer 중 어떤 source에서 BC batch를 뽑는지 config로 명확히 한다.

### 6. Demo / intervention buffer에서 BC loss를 learner update에 섞기

intervention learning의 핵심 경로다. RL update와 별개로 demo/intervention에서 BC loss를 뽑아
learner policy에 추가하는 recipe를 명확히 구현한다.

- buffer 역할: [PIPELINE.md#buffer-역할](PIPELINE.md#buffer-역할)
- mixed buffer sampling: [REPLAY_AND_UPDATES.md#mixed-buffer-sampling](REPLAY_AND_UPDATES.md#mixed-buffer-sampling)
- demo episode insert mode: [REPLAY_AND_UPDATES.md#demo-episode-insert-mode](REPLAY_AND_UPDATES.md#demo-episode-insert-mode)
- update-to-data ratio: [REPLAY_AND_UPDATES.md#update-to-data-ratio](REPLAY_AND_UPDATES.md#update-to-data-ratio)

결정해야 할 것:

- BC source를 `demo`, `intervention`, `demo+intervention`, `online+intervention` 중 config로 고르는 방식.
- RL update와 BC update를 같은 step에서 둘 다 할지, schedule을 따로 둘지.
- intervention suffix에서 실패 expert correction을 포함할지 여부.

## 다음 단계 TODO

### 7. Action chunk queue 정교화

현재 v0 train loop는 primitive action 기준이다. BCFlow/RLPD policy가 action chunk를 출력하는 경우,
매 step 새 chunk를 뽑고 첫 action만 쓰면 chunk policy의 temporal semantics가 깨진다. Python
`collections.deque` 기반으로 learner/expert action queue를 따로 관리하는 방향으로 둔다.

- pipeline 제한 사항: [PIPELINE.md#unified-train-loop](PIPELINE.md#unified-train-loop)
- action chunking을 미룬 이유: [STATUS_2026-05-18.md#아직-남은-작업](STATUS_2026-05-18.md#아직-남은-작업)
- n-step과 action chunk 관계: [REPLAY_AND_UPDATES.md#n-step-backup](REPLAY_AND_UPDATES.md#n-step-backup)

현재 상태:

- replay/update path는 `sequence_length`를 TD backup 길이로 사용한다.
- actor `horizon_length`는 action chunk output 길이로 남아 있다.
- RLPD critic과 BC auxiliary critic은 실제 sampled `rewards.shape[-1]`를 TD exponent로 사용한다.
- 아직 남은 문제는 learner/expert rollout queue다. residual frozen base policy queue는 구현됐지만, 일반 learner/expert policy가 서로 다른 horizon을 가질 때의 queue는 아직 별도 설계가 필요하다.

구현 방향:

- policy adapter는 raw actor output을 canonical `full_action_chunk` shape `(horizon, action_dim)`으로 제공한다.
- horizon=1 policy도 `full_action_chunk`를 `(1, action_dim)`으로 제공하면 queue 로직이 단순해진다.
- learner와 expert의 horizon이 서로 달라도 같은 queue logic으로 처리해야 한다. 예: learner horizon=5, expert horizon=1 또는 그 반대.
- rollout state는 `learner_queue: deque`, `expert_queue: deque`, `last_controller`를 가진다.
- queue가 비었을 때만 해당 policy를 query하고, chunk를 queue에 `extend`한다.
- gate 판단은 각 queue의 현재 candidate `queue[0]`를 사용한다.
- 실제 env action은 선택된 controller queue에서만 `popleft()`한다.
- controller가 learner에서 expert로, 또는 expert에서 learner로 바뀌면 stale chunk 방지를 위해 양쪽 queue를 clear한다.
- `return_full_chunk=True`를 env step action으로 직접 쓰는 경로는 train rollout에서는 피한다.

검증 포인트:

- horizon=3이면 같은 controller가 유지되는 동안 policy query 1번으로 env step 3번을 진행한다.
- learner horizon과 expert horizon이 서로 달라도 각자 queue refill/pop 주기가 독립적으로 동작한다.
- gate가 expert로 바뀔 때 learner/expert queue가 clear되고 새 state 기준 chunk를 다시 샘플한다.
- expert와 learner 모두 proposal은 저장하되, 실제 execute action은 gate decision과 일치해야 한다.
- chunk index와 chunk length를 replay에 저장할지 결정한다.

### 7. N-step episode boundary mode 확장

현재 기본은 qc_base처럼 중간 boundary sample을 drop한다. timeout/truncation에서는 짧은 n-step target으로
bootstrap하는 `truncate` mode를 추가할 수 있다.

- n-step backup: [REPLAY_AND_UPDATES.md#n-step-backup](REPLAY_AND_UPDATES.md#n-step-backup)
- episode boundary 처리: [REPLAY_AND_UPDATES.md#episode-boundary-처리](REPLAY_AND_UPDATES.md#episode-boundary-처리)
- terminal/mask 설계: [REPLAY_AND_UPDATES.md#terminal과-mask](REPLAY_AND_UPDATES.md#terminal과-mask)

나중에 추가할 option:

```text
boundary_mode = "drop"
boundary_mode = "truncate"
```

`truncate` mode에서는 timeout이면 boundary `next_observation`에서 bootstrap하고, true termination이면
bootstrap하지 않는다.

### 8. Image observation policy 학습 지원

env와 replay는 image observation 저장을 지원하지만, actor/critic network는 아직 lowdim state만
지원한다. image encoder와 feature fusion은 별도 작업으로 남긴다.

현재 정리:

- env wrapper는 `lowdim`, `pixels`, `pixels_state` observation mode와 multi-camera render를 지원한다.
- replay buffer는 image leaf를 별도 저장하고, sample 시 `i+1` frame으로 `next_observations` image를 복원한다.
- 여기까지는 "image input을 받을 수 있는 env/replay 인프라" 단계다.
- policy/network가 image를 어떻게 쓸지는 아직 정하지 않았다.
- 따라서 당장 train은 low-dim state policy 기준으로 진행하고, image policy 학습은 별도 설계 후 붙인다.

- image observation 상태: [PIPELINE.md#image-observation-상태](PIPELINE.md#image-observation-상태)
- image observation TODO: [NETWORKS.md#image-observation-todo](NETWORKS.md#image-observation-todo)
- image replay storage: [REPLAY_AND_UPDATES.md#image-replay-storage](REPLAY_AND_UPDATES.md#image-replay-storage)

필요 작업:

- `EnvSpec.pixel_keys` 기반 CNN encoder builder 추가.
- shared camera encoder와 separate camera encoder 옵션 추가.
- image feature와 state feature concat 후 actor/critic head에 전달.
- nested observation dict를 JAX PyTree로 update에 넘기는 경로 검증.

### 9. Smarter gating / human UI

현재는 random gate와 learner-only 실행이 기본이다. 실제 intervention 연구로 가려면 gate function을
교체하기 쉬운 구조를 유지하면서 uncertainty, value, discriminator, human input 등을 붙인다.

- 초기 pipeline에서 gate 위치: [PIPELINE.md#step-logic](PIPELINE.md#step-logic)
- buffer routing smoke: [PIPELINE.md#현재-smoke-test로-확인한-것](PIPELINE.md#현재-smoke-test로-확인한-것)
- project scope: [PROJECT_SCOPE.md](PROJECT_SCOPE.md)

후보:

- random probability gate.
- value/Q uncertainty gate.
- discriminator/novelty gate.
- human keyboard/UI gate.


### 10. Logger metric 확장

현재 logger는 매 step metric을 accumulate하고 `log_interval`마다 interval row를 기록한다. 다음 단계는 DAgger/intervention/RL 분석에 필요한 metric을 추가하는 것이다. CSV write frequency, action entropy/variance, state distribution health metric은 우선순위 높게 본다.

- logger 현재 구현과 metric TODO: [LOGGING_AND_METRICS.md](LOGGING_AND_METRICS.md)

우선순위 높은 작업:

- env step time, replay sample time, update time을 분리해서 기록한다.
- update skip count와 실제 gradient update count를 기록한다.
- batch source fraction, terminal/timeout/mask 통계를 기록한다.
- learner/expert action L2, action clipping fraction, gripper stat을 기록한다.
- action entropy와 action variance를 기록해서 policy collapse 또는 과도한 randomization을 본다.
- state distribution health를 기록한다. 우선 online running stat과 demo/offline baseline stat의 mean/std drift, out-of-dataset z-score fraction을 본다.
- state normalization은 후속 구현으로 둔다. 기본 방향은 offline/demo dataset stat을 baseline으로 만들고, online rollout 중 running stat을 따로 추적해서 normalize 적용 여부와 distribution drift logging을 분리하는 것이다.
- CSV write frequency와 append/rotate 정책을 config로 분리한다. 긴 run에서 flush마다 전체 CSV rewrite를 피한다.
- gate intervention rate와 expert execute rate를 기록한다.

## 최근 실험 후속

### Top-K BCFlow pretraining 결과

최근 Square MH shortest-demo subset으로 BCFlow policy를 학습했다.

```text
500k: top10 0.13, top30 0.37, top50 0.43
1M:   top10 0.14, top30 0.40, top50 0.41
```

후속 작업:

- top50 500k를 DAgger learner 초기값 후보로 쓸지 검토한다.
- top50 500k와 top50 1M failure video를 비교해서 overfit 또는 mode collapse 징후를 본다.
- top-K shortest-demo selection이 너무 짧은 trajectory bias를 만드는지 확인한다.
- top50 metadata는 `exp/pretrained/bcflow_square_top50_actorln_seed0_500k/metadata.json`에 있다.

관련 산출물:

- `logs/topk_500k_eval/*_100ep.json`
- `logs/topk_1m_eval/*_100ep.json`
- `videos/failure_rollouts/bcflow_square_top*_actorln_seed0_500k`
- `videos/failure_rollouts/bcflow_square_top*_actorln_seed0_1m`

## Git / 산출물 관리

작업을 더 진행하기 전에 git 상태를 정리한다.

- git 운영 원칙: [IMPLEMENTATION_PLAN.md#git-운영](IMPLEMENTATION_PLAN.md#git-운영)
- workflow의 git 운영: [WORKFLOW.md#git-운영](WORKFLOW.md#git-운영)

주의:

- `exp/`, `logs/`, `videos/`, `wandb/`, replay buffer, checkpoint는 커밋하지 않는다.
- 현재 구조 변경이 많으므로 기능 단위로 작은 커밋을 나누는 것이 좋다.
- 문서 TODO와 실제 코드 상태가 어긋난 부분은 먼저 정리한다.
