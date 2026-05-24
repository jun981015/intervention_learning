# Real Env Smoke Tests

이 문서는 simulator 없는 unit/smoke test가 아니라 실제 Robomimic env를 띄워서 확인한 짧은 검증 기록이다.
목적은 성능 평가가 아니라 env 생성, pretrained restore, rollout, gate, replay 저장, update 경로가 같이 깨지지 않는지 보는 것이다.

## 공통 실행 환경

- conda env: `il`
- env: `square-mh-low_dim`
- 실제 robosuite env name: `NutAssemblySquare`
- observation/action: lowdim, `obs_dim=23`, `action_dim=7`
- wandb: disabled
- JAX: CPU 강제 사용

```bash
env JAX_PLATFORM_NAME=cpu WANDB_MODE=disabled conda run -n il python -m il.train --config <config>
```

## BCFlow + Auxiliary Critic Real-env Smoke

Config: [../config/smoke_bc_critic_square.yaml](../config/smoke_bc_critic_square.yaml)

목적:

- 실제 Robomimic Square env에서 random-init BCFlow learner가 rollout 가능한지 확인한다.
- online replay에서 sample한 batch로 BC actor update와 optional auxiliary critic update가 같이 도는지 확인한다.
- critic은 actor objective에 쓰지 않고, value/Q diagnostics 용도로만 학습된다.

실행:

```bash
env JAX_PLATFORM_NAME=cpu WANDB_MODE=disabled conda run -n il python -m il.train --config config/smoke_bc_critic_square.yaml
```

결과:

- run dir: `exp/smoke/intervention_learning/smoke/square-mh-low_dim/square_bc_critic_real_env_smoke_seed0`
- steps: 20
- env build: 성공
- actor update: 성공, `learner_bc/actor/bc_flow_loss` 기록됨
- critic update: 성공, `learner_bc/critic/critic_loss`, `q_mean`, `target_q_mean`, `td_error_abs_mean` 기록됨
- saved files: `params_20.pkl`, `online_replay_buffer.npz`, `demo_replay_buffer.npz`, `intervention_replay_buffer.npz`, `metrics.jsonl`, `metrics.csv`

주의:

- 이 smoke는 pretrained expert 없이 `target_action_key=actions`로 self-label BC update를 돌린다.
- 따라서 DAgger 성능이나 expert label 품질 검증이 아니다.

## DAgger Relabel Real-env Smoke

Config: [../config/smoke_dagger_square.yaml](../config/smoke_dagger_square.yaml)

목적:

- pretrained BCFlow learner와 pretrained RLPD expert를 실제 env에 같이 붙인다.
- env에는 learner action만 실행한다.
- 같은 state에서 expert action도 query해서 replay의 `expert_actions` label로 저장되는지 확인한다.

실행:

```bash
env JAX_PLATFORM_NAME=cpu WANDB_MODE=disabled conda run -n il python -m il.train --config config/smoke_dagger_square.yaml
```

결과:

- run dir: `exp/smoke/intervention_learning/smoke/square-mh-low_dim/square_dagger_real_env_smoke_seed0`
- steps: 100
- pretrained learner restore: `exp/pretrained/bcflow_square_top50_actorln_seed0_500k/params_500000.pkl`
- pretrained expert restore: `exp/pretrained/rlpd_square_bc03_seed0_2m/params_2000000.pkl`
- online replay size: 100
- `controller_ids`: learner 100 / expert 0
- `interventions`: 0 for all 100 steps
- finite checks: `actions`, `learner_actions`, `expert_actions` all finite
- executed action check: `max_abs(actions - learner_actions) = 0.0`
- learner/expert action difference: mean L2 approximately `0.369`

주의:

- 이 smoke는 relabel storage 검증용이라 gradient update는 일부러 꺼뒀다.
- DAgger update까지 검증하려면 같은 config에서 `start_training`을 낮추고 `trainable=true`로 별도 smoke를 돌린다.

## Expert-Q Gap Gate Real-env Smoke

Config: [../config/smoke_expert_q_gap_square.yaml](../config/smoke_expert_q_gap_square.yaml)

목적:

- pretrained BCFlow learner와 pretrained RLPD expert를 실제 env에 같이 붙인다.
- expert RLPD의 `evaluate_q(obs, action, q_agg=...)`로 `Q(s, a_expert) - Q(s, a_learner)`를 계산한다.
- q-gap signal, probabilistic intervention, sticky intervention horizon이 실제 rollout에서 동작하는지 확인한다.

실행:

```bash
env JAX_PLATFORM_NAME=cpu WANDB_MODE=disabled conda run -n il python -m il.train --config config/smoke_expert_q_gap_square.yaml
```

결과:

- run dir: `exp/smoke/intervention_learning/smoke/square-mh-low_dim/square_expert_q_gap_real_env_smoke_seed0`
- steps: 100
- pretrained learner restore: `exp/pretrained/bcflow_square_top50_actorln_seed0_500k/params_500000.pkl`
- pretrained expert restore: `exp/pretrained/rlpd_square_bc03_seed0_2m/params_2000000.pkl`
- online replay size: 100
- `controller_ids`: learner 50 / expert 50
- `interventions`: 50 false / 50 true
- `gating_reasons`: `EXPERT_Q_GAP` for all 100 steps
- `gating_scores`: mean approximately `0.348`, min approximately `-0.447`, max approximately `0.653`
- interval log totals: expert executed 50 steps, intervention started 5 times
- finite checks: `actions`, `learner_actions`, `expert_actions` all finite

중요 해석:

- metric logger는 20-step interval 평균/합으로 기록한다. 예를 들어 `gate/signal=0.5`는 해당 interval 20 step 중 10 step에서 signal이 켜졌다는 뜻이다.
- `gate/expert_execute_steps_total=50`, `gate/intervention_started_total=5`가 최종 row에 기록됐다.
- threshold `0.5`, intervention probability `0.9`, horizon `10`은 검증된 실험 default가 아니라 smoke용 시작값이다.

## 아직 남은 실제 env 검증

- DAgger에서 실제 BC update까지 켠 100-step smoke.
- Expert-Q gap에서 threshold/probability/horizon 값을 바꿨을 때 intervention rate sanity check.
- replay save/load round-trip을 real-env 산출물로 검증.
- video render와 replay action/state sync 검증.
