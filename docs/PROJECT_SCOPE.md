# 프로젝트 범위

## 프로젝트 목적

`qc` 실험 코드와 분리된 독립 intervention learning 프로젝트를 만든다. `qc`와
`qc_base`에서 필요한 구성은 가져오되, FQL/QC-FQL/BT 등 현재 목표와 무관한 실험 코드는
섞지 않는다.

## 기준 원칙

- `qc`의 FQL, QC-FQL, BT model은 가져오지 않는다.
- conda 환경은 새 PC에서 `qc` 환경 없이 설치 가능한 standalone requirements를 기준으로 한다.
- diffusion/flow-matching actor는 BC policy 후보로만 가져온다.
- v0는 Robomimic `square-mh-low_dim`만 대상으로 한다.
- v0는 `horizon_length=1`로 고정하고 action chunking은 나중에 다룬다.
- n-step backup은 학습 핵심 기능으로 유지한다.
- frame stack은 구조상 인자를 두되 기본값 `1`로 둔다.
- expert와 learner는 같은 policy/algorithm interface를 공유한다.
- 매 step learner action과 expert action을 둘 다 뽑아서 저장한다.
- gating function은 어떤 controller를 실행할지 결정한다.
- 초기 gating은 random probability gate로만 테스트한다.
- 초기 expert baseline은 RLPD checkpoint를 load한 policy로 둔다.
- diffusion/flow-matching weight는 나중에 BC actor 또는 expert 후보로 쓸 수 있게 둔다.
- grad clipping은 기본적으로 끈다. 필요한 실험에서만 `grad_clip_norm`을 명시한다.
- layer norm은 단순 MLP 네트워크 안에서 사용한다. 복잡한 FiLM/BT/value-aux 구조는 쓰지 않는다.

## v0 포함 범위

- Environment: `square-mh-low_dim`
- Learner: RLPD/SAC-style online learner
- Expert: restored RLPD/SAC checkpoint policy
- Gate: random probability gate
- Replay: online/demo/intervention buffers, executed action, learner proposal, expert proposal, controller id, gate metadata, log-probs
- Mixed sampling: `online`, `intervention`, `demo` buffer를 config 비율대로 섞어 batch 구성
- N-step backup: `sample_sequence(batch_size, sequence_length, discount)` API 유지
- Frame stack: 기본값 `1`
- Critic ensemble size: `num_qs`로 조정, 기본값 `2`
- Update-to-data ratio: `utd_ratio`로 조정, 기본값 `1`
- TD target Q aggregation: `target_q_agg`, 기본값 `"min"`

## v0 제외 범위

- FQL
- QC-FQL
- BT model
- OGBench/cube tasks
- Human keyboard/UI intervention
- Action chunking support
- Q/uncertainty/disagreement 기반 smart gate

## 현재 Scaffold

- `il/gating/`: controller gate interface와 random gate
- `il/policies/`: learner/expert가 공유하는 minimal policy protocol
- `il/buffers/`: replay buffer, transition schema, episode routing
- `il/datasets/`: offline dataset loader와 transform용 예약 폴더
- `il/algo/rl/rlpd.py`: qc_base 스타일 RLPD/SAC agent
- `il/algo/bc/flow.py`: BC-only flow-matching actor
- `il/algo/bc/mlp.py`: deterministic MLP BC actor
- `il/networks/`: qc_base에서 가져온 network block
- `il/distributions/`: action distribution head
- `il/utils/`: shared dataclass, config, Flax train state, UTD sampling
- `il/loops/`: single-step action selection helper
- `il/train.py`: 아직 연결 전인 placeholder entrypoint
