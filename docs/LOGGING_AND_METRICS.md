# Logging and Metrics

이 문서는 train loop에서 어떤 metric을 어떻게 기록할지 정리한다. 현재 목표는 매 step 파일을 쓰지 않고,
`logging.stdout_interval` 또는 내부 legacy `train.log_interval`마다 interval 평균을 저장하는 것이다.

## 현재 구현 상태

현재 logger 구현 파일은 `il/logging.py`다.

현재 동작:

- train loop는 매 env step마다 scalar metric payload를 `MetricLogger.record()`로 넘긴다.
- logger는 내부 accumulator에 metric을 모은다.
- `log_interval`마다 JSONL, CSV, W&B, stdout에 한 번만 기록한다.
- loss, grad norm, reward-like scalar는 interval 평균으로 기록한다.
- replay size, episode count, step, throughput, recent env stat, routing stat은 마지막 값을 기록한다.
- `train/log_interval_records`를 같이 저장해서 한 row가 몇 step 평균인지 확인할 수 있다.
- eval metric은 train accumulator와 섞지 않기 위해 `log_immediate()`로 즉시 기록한다.

주의:

- `env/recent_*` 값은 이미 최근 episode window의 rolling stat이므로 interval 평균이 아니라 마지막 값을 남긴다.
- `routing/*` 값도 episode routing 결과의 마지막 상태값으로 본다.
- CSV header는 나중에 새 metric key가 생겨도 다시 써서 누락되지 않게 한다.

## 현재 확인한 smoke

짧은 logger unit smoke:

```text
loss=1,2,3,4,5 with interval=5 -> logged loss=3.0
train/log_interval_records=5
```

짧은 DAgger smoke:

```text
steps=40
start_training=20
log_interval=10
```

확인된 출력 형태:

```text
[metrics] step=10 train/log_interval_records=10 train/online_size=10 ...
[metrics] step=20 train/log_interval_records=10 ... learner_bc/actor/bc_flow_loss=0.7054 learner_bc/actor/grad/norm=12.11
[metrics] step=30 train/log_interval_records=10 ... learner_bc/actor/bc_flow_loss=0.2215 learner_bc/actor/grad/norm=3.021
[metrics] step=40 train/log_interval_records=10 ... learner_bc/actor/bc_flow_loss=0.1557 learner_bc/actor/grad/norm=1.219
```

## 지금 기록되는 핵심 metric

Train / replay:

- `train/step`
- `train/log_interval_records`
- `train/online_size`
- `train/demo_size`
- `train/intervention_size`
- `train/episodes`
- `train/interval_sps`
- `train/total_sps`

Environment:

- `env/recent_return`
- `env/recent_length`
- `env/recent_success_rate`

Routing:

- `routing/demo_added`
- `routing/demo_removed`
- `routing/demo_skipped`
- `routing/intervention_added`

BCFlow learner update:

- `learner_bc/actor/bc_flow_loss`
- `learner_bc/actor/grad/norm`
- `learner_bc/actor/grad/max`
- `learner_bc/actor/grad/min`
- `learner_bc/actor/flow_pred_mean`
- `learner_bc/actor/flow_vel_mean`

Eval:

- `eval/return`
- `eval/length`
- `eval/success_rate`

## 추가하면 좋은 metric

### Update / optimizer

우선순위 높음:

- `update/num_updates`: 실제 gradient update 횟수.
- `update/utd_ratio`: 현재 step에서 적용한 update-to-data ratio.
- `update/skip_count`: buffer가 부족해서 update를 skip한 횟수.
- `update/time_seconds`: update에 걸린 시간.
- `update/env_time_seconds`: env step에 걸린 시간.
- `update/sample_time_seconds`: replay sample에 걸린 시간.

이유:

- 지금 느린 부분이 env인지, replay sampling인지, JAX update인지 분리해서 봐야 한다.
- DAgger에서는 초반 buffer 부족으로 update skip이 생길 수 있으므로 명시적으로 기록하는 게 좋다.

### Replay / sampling

우선순위 높음:

- `batch/source_online_fraction`
- `batch/source_demo_fraction`
- `batch/source_intervention_fraction`
- `batch/valid_sequence_fraction`
- `batch/terminal_fraction`
- `batch/timeout_fraction`
- `batch/reward_mean`
- `batch/reward_max`
- `batch/mask_mean`

이유:

- online/demo/intervention 조합이 실험의 핵심이므로 실제 batch 구성비를 기록해야 한다.
- n-step boundary drop 때문에 effective batch가 줄어드는지 확인해야 한다.
- timeout bootstrap 처리를 검증하려면 terminal/timeout/mask 비율이 필요하다.

### Action / policy behavior

우선순위 높음:

- `action/executed_norm_mean`
- `action/learner_norm_mean`
- `action/expert_norm_mean`
- `action/learner_expert_l2_mean`
- `action/learner_expert_l2_max`
- `action/clip_fraction`
- `action/gripper_mean`
- `action/gripper_sign_flip_fraction`

이유:

- DAgger에서는 learner가 expert action에 가까워지는지가 핵심이다.
- action clipping이 자주 발생하면 policy output scale이나 action distribution 문제가 있다.
- robomimic gripper는 action 해석이 중요하므로 따로 봐야 한다.

### Gate / intervention

우선순위 높음:

- `gate/intervention_rate`
- `gate/expert_execute_rate`
- `gate/learner_execute_rate`
- `gate/score_mean`
- `gate/score_max`
- `gate/sticky_steps_mean`
- `episode/has_intervention_rate`
- `episode/intervention_success_rate`

이유:

- intervention learning에서는 gate가 언제 expert를 부르는지가 방법론의 핵심이다.
- 단순 success rate만 보면 expert intervention이 얼마나 들어갔는지 모른다.

### Episode outcome

우선순위 중간:

- `episode/return_mean`
- `episode/length_mean`
- `episode/success_rate`
- `episode/failure_rate`
- `episode/success_length_mean`
- `episode/failure_length_mean`
- `episode/timeout_rate`

이유:

- 현재 `env/recent_*`가 있지만 success/failure별 length를 분리하면 failure mode가 보인다.

### Model / parameter diagnostics

우선순위 중간:

- `model/actor_param_norm`
- `model/critic_param_norm`
- `model/value_param_norm`
- `model/actor_update_norm`
- `model/critic_update_norm`
- `model/grad_to_param_ratio`

이유:

- 이전 QC 실험에서 gradient 폭주와 plasticity 문제를 많이 봤기 때문에, 새 repo에서도 기본 진단을 남기는 게 좋다.

### BCFlow-specific

우선순위 중간:

- `bc_flow/time_mean`
- `bc_flow/noise_norm_mean`
- `bc_flow/pred_velocity_norm_mean`
- `bc_flow/target_velocity_norm_mean`
- `bc_flow/action_chunk_l2_by_index/*`

이유:

- action chunk policy에서는 첫 action만 좋아지고 뒤 action이 망가지는지 볼 필요가 있다.
- flow matching time이나 velocity scale이 이상하면 loss만으로는 알기 어렵다.

### RLPD / SAC-specific

우선순위 중간:

- `rl/critic_loss`
- `rl/actor_loss`
- `rl/alpha_loss`
- `rl/alpha`
- `rl/q_mean`
- `rl/q_min`
- `rl/q_max`
- `rl/target_q_mean`
- `rl/bc_loss`
- `rl/action_log_prob_mean`
- `rl/action_entropy_mean`

이유:

- expert나 learner가 RLPD일 때 SAC entropy term, Q scale, BC regularization이 성능에 직접 영향을 준다.

## 구현 TODO

1. `MetricLogger`에 metric category별 aggregation policy를 config로 받을 수 있게 한다.
2. train loop에서 env step time, sample time, update time을 분리해서 기록한다.
3. replay sampler가 batch source id와 terminal/timeout/mask 통계를 반환하게 한다.
4. policy sampling path에서 learner/expert/action clipping 통계를 계산한다.
5. gate decision을 interval accumulator에 넣어 intervention rate를 기록한다.
6. episode 종료 시 success/failure/timeout별 episode stat을 따로 업데이트한다.
7. BCFlow update info에 chunk index별 loss 또는 action l2를 추가한다.
8. RLPD update info에 Q/alpha/log_prob 관련 scalar를 표준 이름으로 맞춘다.
9. `docs_agents/LOGGING_AND_METRICS.md`와 동기화한다.

## 원칙

- 매 step 파일 write는 하지 않는다.
- interval 평균 row 하나가 WandB/CSV/JSONL의 기본 단위다.
- loss와 grad는 평균, replay size와 step counter는 마지막 값이다.
- 사람이 보는 stdout은 짧게 유지하고, 자세한 값은 JSONL/CSV/W&B에서 본다.
- 새 metric은 가능한 한 prefix를 고정한다: `train/`, `env/`, `routing/`, `batch/`, `action/`, `gate/`, `episode/`, `model/`, `eval/`.
