# Implementation TODO Bookmark

이 문서는 TODO 자체를 길게 설명하는 곳이 아니라, 남은 작업을 한눈에 보고 세부 설계 문서로 바로
이동하기 위한 책갈피다. 실제 설계 근거와 상세 설명은 각 링크 문서를 기준으로 본다.

## 우선순위 높은 작업


### 현재 상태 스냅샷

2026-05-21 기준으로 DAgger v0, pretrained loading, replay/buffer, env/image, logger 상태를 별도 문서에 정리했다.

- current status: [STATUS_2026-05-21.md](STATUS_2026-05-21.md)
- logging and metrics: [LOGGING_AND_METRICS.md](LOGGING_AND_METRICS.md)

### Config schema 결정 사항

현재 config schema, DAgger v0 표현 방식, `actors.learner.kind`가 update rule을 결정한다는 원칙,
`replay.sampling` named batch 구조, intervention gate의 의미는 별도 결정 문서에 정리한다.

- config schema decisions: [CONFIG_SCHEMA_DECISIONS.md](CONFIG_SCHEMA_DECISIONS.md)


### 1. 현재 문서와 코드 상태 동기화

현재 `IMPLEMENTATION_PLAN`과 `STATUS`에는 이미 일부 구현된 항목도 TODO로 남아 있다. 새 작업을
시작하기 전에 문서가 현재 코드 상태를 정확히 반영하도록 정리한다.

- 전체 구현 순서: [IMPLEMENTATION_PLAN.md#다음-구현-순서](IMPLEMENTATION_PLAN.md#다음-구현-순서)
- 문서상 아직 안 된 것: [IMPLEMENTATION_PLAN.md#아직-안-된-것](IMPLEMENTATION_PLAN.md#아직-안-된-것)
- 2026-05-18 기준 남은 작업: [STATUS_2026-05-18.md#아직-남은-작업](STATUS_2026-05-18.md#아직-남은-작업)
- 작업 진행 방식과 기준 원칙: [WORKFLOW.md#기준-원칙](WORKFLOW.md#기준-원칙)

### 2. Online rollout smoke를 실제 Robomimic env에서 검증

restored learner/expert와 gate를 실제 Square env에 붙여 100 step 정도를 돌리고, replay에 의도한
metadata가 모두 저장되는지 확인한다.

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

### 3. Replay save/load round-trip test 추가

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

### 4. DAgger / BC update 경로 검증

rollout 중 저장된 `expert_actions`를 이용해서 BCFlow learner update가 실제로 도는지 확인한다.

- DAgger baseline 흐름: [DAGGER_BASELINE.md](DAGGER_BASELINE.md)
- BC Flow target action key 결정: [STATUS_2026-05-18.md#dagger-baseline](STATUS_2026-05-18.md#dagger-baseline)
- unified train loop update 위치: [PIPELINE.md#unified-train-loop](PIPELINE.md#unified-train-loop)
- mixed buffer sampling: [REPLAY_AND_UPDATES.md#mixed-buffer-sampling](REPLAY_AND_UPDATES.md#mixed-buffer-sampling)

검증 포인트:

- `target_action_key="expert_actions"`일 때 loss가 NaN 없이 감소하는지 확인한다.
- 완료: `expert_actions`에 NaN/Inf가 있으면 BC update 직전에 `ValueError`로 중단한다.
- online/demo/intervention buffer 중 어떤 source에서 BC batch를 뽑는지 config로 명확히 한다.

### 5. Demo / intervention buffer에서 BC loss를 learner update에 섞기

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

### 6. Action chunk queue 정교화

현재 v0 train loop는 primitive action 기준이다. BCFlow/RLPD policy가 action chunk를 출력하는 경우,
learner와 expert가 각각 독립적인 chunk queue를 가져야 한다.

- pipeline 제한 사항: [PIPELINE.md#unified-train-loop](PIPELINE.md#unified-train-loop)
- action chunking을 미룬 이유: [STATUS_2026-05-18.md#아직-남은-작업](STATUS_2026-05-18.md#아직-남은-작업)
- n-step과 action chunk 관계: [REPLAY_AND_UPDATES.md#n-step-backup](REPLAY_AND_UPDATES.md#n-step-backup)

검증 포인트:

- gate가 expert로 바뀌어도 learner chunk queue의 의미가 깨지지 않아야 한다.
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

현재 logger는 매 step metric을 accumulate하고 `log_interval`마다 평균 row를 기록한다. 다음 단계는 DAgger/intervention/RL 분석에 필요한 metric을 추가하는 것이다.

- logger 현재 구현과 metric TODO: [LOGGING_AND_METRICS.md](LOGGING_AND_METRICS.md)

우선순위 높은 작업:

- env step time, replay sample time, update time을 분리해서 기록한다.
- update skip count와 실제 gradient update count를 기록한다.
- batch source fraction, terminal/timeout/mask 통계를 기록한다.
- learner/expert action L2, action clipping fraction, gripper stat을 기록한다.
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
