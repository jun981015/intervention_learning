# Code Review — intervention_learning

> Reviewed: 2026-05-22  
> Status: pre-first-run (end-to-end 학습 아직 미실행)  
> Reviewer: Claude Opus 4.6

---

## 1. 프로젝트 개요

Robomimic Square 환경에서 config-driven intervention learning 파이프라인.
알고리즘별 클래스 파일을 만드는 대신, learner/expert/gate/replay/update를 config로 조합하여 DAgger, HG-DAgger, expert-Q gap intervention 등을 표현한다.

핵심 데이터 흐름:
```
sample proposals → gate decide → env step → schema transition → buffer routing → update spec
```

---

## 2. 런타임 버그 (P0 — 즉시 수정)

### 2.1 smoke_test에서 존재하지 않는 모듈 import

**파일**: `scripts/smoke_test.py:27-28`

```python
from il.policies.bc_flow import BCFlowPolicy
from il.policies.rlpd import RLPDPolicy
```

`il/policies/` 디렉토리에는 `base.py`, `agent_view.py`, `__init__.py`만 있다.
`BCFlowPolicy`, `RLPDPolicy` 모듈이 존재하지 않아서 smoke test 실행 시 `ImportError`.

**조치**: 해당 policy adapter 파일을 생성하거나, `AgentPolicyView`를 통해 우회하도록 smoke test를 수정.

### 2.2 `np.random.seed()` global state 오염

**파일**: `il/envs/robomimic_lowdim.py:188-192`

```python
def seed(self, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
```

`np.random.seed`는 global state이므로, evaluation에서 `eval_env.reset(options={"seed": ...})`을 호출할 때마다 train replay sampling(`np.random.randint`)의 재현성이 깨진다.

**조치**: `np.random.default_rng(seed)`로 env-local RNG를 사용. replay buffer의 `sample()`/`sample_sequence()`도 동일하게 local RNG로 전환 권장.

---

## 3. 논리적 불안정성 (P1 — 첫 실행 전 수정 권장)

### 3.1 Eval RNG와 Train RNG 미분리

**파일**: `il/evaluation/evaluator.py:27, 46`

```python
rng = context.rng          # train RNG를 가져감
# ... eval 루프 ...
context.rng = rng          # 소모된 RNG를 다시 씀
```

Eval 에피소드 수에 따라 train loop의 JAX RNG 시퀀스가 달라진다. `eval_interval`을 바꾸면 동일 seed에서도 train 궤적이 달라짐.

**조치**: `context.rng`에서 eval 전용 sub-key를 split하고, train RNG는 건드리지 않는다.

```python
eval_rng = jax.random.fold_in(context.rng, step)
# eval 루프에서는 eval_rng만 사용
```

### 3.2 BC loss에서 episode boundary를 넘는 target action

**파일**: `il/buffers/replay_buffer.py:560`, `il/algo/bc/flow.py:55-57`

`sample_sequence`는 연속 인덱스를 뽑으며 episode 경계를 넘는 시퀀스를 허용한다.
RL critic loss는 `valid` mask로 걸러지지만, BC flow loss의 `action_chunking=False` path는:

```python
flow_loss = jnp.mean((pred - vel) ** 2)  # episode 경계 무시
```

다른 episode의 action을 target으로 학습하여 BC loss가 오염된다.

**조치 옵션**:
- `sample_sequence`에 `episode_ids` 기반 episode-boundary check 추가
- 또는 BC loss에서 `valid` mask를 항상 적용

### 3.3 `BCFlowAgent.bc_flow_loss`에서 `valid` 키 의존

**파일**: `il/algo/bc/flow.py:55`

`action_chunking=True`일 때 `batch["valid"]`를 참조하는데, `ReplayBuffer.sample()` (non-sequence)로 배치를 뽑으면 `valid` 키가 없어서 `KeyError`.

현재 config에서는 `sequence_length=5`를 쓰므로 당장 안 터지지만, 다른 update source 연결 시 즉시 발생.

**조치**: `valid`가 없으면 `jnp.ones`로 대체하는 fallback 추가.

---

## 4. 비효율 / 성능 문제 (P2)

### 4.1 `replace_episode`의 O(N) 전체 복사

**파일**: `il/buffers/replay_buffer.py:411-416`

```python
kept = [self._transition_at(index) for index in range(self.size) if index not in remove_indices]
self._clear()
for transition in kept:
    self.add_transition(transition)
```

Demo buffer에서 episode replacement 시 전체 버퍼를 Python-level로 복사 → 클리어 → 재삽입한다.
500k 크기 buffer에서 매 성공 에피소드마다 호출 시 극심한 bottleneck.

현재 `demo_insert_mode: "none"`으로 안 타지만, `replace_longest_if_better`를 쓰면 발생.

**조치**: numpy-level in-place compaction 또는 lazy tombstone 방식으로 교체.

### 4.2 Circular buffer overwrite 후 episode index 불일치

**파일**: `il/buffers/replay_buffer.py:310-311`

버퍼가 꽉 찬 후 circular overwrite 시, `episode_records`에 저장된 indices가 이미 다른 episode 데이터로 덮어씌워졌을 수 있다. `_rebuild_episode_index`가 `episode_ids[:self.size]`만 보지만, overwrite 후에는 이전 episode의 일부 transition만 남아 있거나 indices가 엉킨 episode가 생길 수 있다.

**조치**: Demo buffer에서 episode-level 연산을 쓸 거면 circular overwrite를 비활성화하거나, episode index rebuild 시 연속성 검증 추가.

### 4.3 CSV 전체 rewrite

**파일**: `il/logger/logger.py:161-170`

`self.csv_rows`에 모든 행을 누적하고 매 flush마다 전체 파일을 rewrite한다.
필드가 동적으로 늘어나서 rewrite가 필요한 설계이지만, 장기 실험에서 메모리 누적 + I/O 비용.

**조치**: 필드 목록이 안정화되면 append-only로 전환하거나, 새 필드 추가 시에만 rewrite.

### 4.4 매 step gate metric 연산

**파일**: `il/loops/train_loop.py:194`

`_gate_metric_payload(decision)`을 매 env step마다 호출하여 10+ key를 accumulator에 넣는다.
`expert_q_gap`의 경우 `decision.info`에 12개 키가 들어간다. log interval이 아닌 step에서는 불필요.

**조치**: `force_log` step에서만 gate metric을 구성하거나, accumulator에 raw decision만 저장.

---

## 5. 설계 / 아키텍처 의견

### 5.1 Config 스키마 이중성 — 가장 큰 구조 문제

새 스키마(`dagger.yaml`, `expert_q_gap_square.yaml`)와 구 스키마(`DEFAULT_RECIPE`)가 공존한다.
`new_schema_to_legacy_recipe` (약 110줄)가 변환을 담당하는데:

- 변환 과정에서 정보 손실: `training.update_interval`, `training.updates_per_step`, `checkpointing.keep_last`, `storage.*` 등 새 스키마 키들이 변환 후 무시됨
- 디버깅 시 "사용자가 쓴 config" vs "실제 적용 config" 양쪽을 봐야 함
- config-driven 설계의 핵심 장점("config를 보면 알고리즘이 뭔지 안다")이 약화됨

**권장**: 구 스키마를 제거하고 builder들이 새 스키마를 직접 읽도록 마이그레이션.

### 5.2 Update spec의 표현력 한계

현재 update spec은 "누구를(target) 어떤 데이터로(source) 학습시킬지"만 표현한다.
**어떤 loss function을 쓸지**는 agent 내부의 `total_loss`에 하드코딩되어 있다.

```python
# BCFlowAgent → 항상 flow loss
# ACRLPDAgent → 항상 critic + actor + alpha loss
```

향후 mixed objective (예: BC + RL을 동시에, 다른 source에서)를 구현하려면 agent 클래스를 새로 만들거나 내부에 분기를 추가해야 한다.

**인지 사항**: 당장은 문제 아님. 연구가 mixed loss 방향으로 가면 loss function도 config에서 선택할 수 있는 구조가 필요.

### 5.3 ActorBundle ↔ AgentPolicyView mutable 동기화

**파일**: `il/loops/updates.py:67-71`

```python
def _set_bundle_agent(bundle: ActorBundle, agent) -> None:
    bundle.agent = agent
    if bundle.policy is not None and hasattr(bundle.policy, "agent"):
        bundle.policy.agent = agent
```

JAX의 functional update 패턴과 Python mutable reference가 충돌하는 지점.
`_set_bundle_agent` 외의 경로에서 `bundle.agent`를 직접 교체하면 policy view가 stale agent를 참조한다.

**권장**: `AgentPolicyView`가 agent를 직접 보유하지 않고, bundle에서 매번 참조하는 구조로 변경.

### 5.4 Gate에 expert_agent 전체 전달

`ExpertQGapGate.decide`가 `expert_agent`를 받아서 직접 `evaluate_q`를 호출한다.
Gate가 특정 agent 구현(action-value Q function 보유)에 결합되어 있다.

**권장**: Q evaluation을 callable로 주입.
```python
# 예시
gate.decide(..., q_fn=lambda obs, act: expert.evaluate_q(obs, act))
```

### 5.5 Replay buffer 고정 schema

모든 buffer(online, demo, intervention)가 동일한 16-key schema를 사용한다.
Demo buffer에 `gating_scores`, `interventions` 등 불필요한 필드가 할당된다.

Low-dim에서는 observation이 작아서 무시 가능하지만, pixel observation 확장 시 메모리 낭비가 커진다.

---

## 6. 코드 스타일 / 소소한 사항

### 6.1 `_update`의 `@staticmethod` + `self` naming

**파일**: `il/algo/rl/rlpd.py:164-165`

```python
@staticmethod
def _update(self, batch):
```

`jax.lax.scan` 호환을 위한 패턴이지만, `self`라는 인자명이 Python convention과 충돌.
`BCFlowAgent`에서는 이미 `agent`로 쓰고 있으므로 통일 권장.

### 6.2 `il/utils/updates.py` vs `il/loops/updates.py` 중복

`sample_rl_update_batch`가 `il/utils/updates.py`에 있고, 실질적으로 같은 역할의 로직이 `il/loops/updates.py`에도 있다. smoke test는 전자를, train loop은 후자를 사용. 한쪽만 수정하면 불일치 발생.

**조치**: 하나로 통합.

### 6.3 Evaluation에서 action chunking 미처리

**파일**: `il/evaluation/evaluator.py:37`

`AgentPolicyView._format_action`이 chunk의 첫 action만 반환하므로 evaluation에서 action chunking의 multi-step execution 이점이 사라진다. `dagger.yaml`에 `action_mode: first_action`이 있지만 eval 코드에서는 이 설정을 읽지 않는다.

**조치**: 의도적이라면 주석으로 명시. 아니라면 eval에서도 action chunk execution 지원.

---

## 7. 종합 평가

### 잘 된 부분

- **핵심 추상화** (ControllerGate protocol, PolicyOutput, GateDecision, StepRecord): intervention 연구의 확장점이 명확하고, 새 gate 함수를 추가할 때 인터페이스만 맞추면 됨
- **데이터 흐름**: `choose_rollout_action → step_record_to_transition → route_episode_to_buffers → run_update_spec` 체인이 단순하고 읽기 좋음
- **Replay transition schema**: learner/expert 양쪽 action과 gate metadata를 모두 저장하여 사후 분석에 유리
- **Smoke test 커버리지**: gate, routing, mixed sampling, replay prefill, agent update, checkpoint restore 등 핵심 경로를 커버

### 수정 우선순위

| 우선도 | 항목 | 카테고리 |
|--------|------|----------|
| **P0** | smoke_test 없는 모듈 import | 런타임 버그 |
| **P0** | `np.random.seed` global state | 재현성 파괴 |
| **P1** | eval/train RNG 분리 | 재현성 |
| **P1** | BC loss episode boundary | 학습 오염 |
| **P1** | config 스키마 통일 | 유지보수성 |
| **P2** | `replace_episode` O(N) | 성능 |
| **P2** | CSV rewrite / gate metric 매 step | 성능 |
| **P2** | `utils/updates.py` 중복 제거 | 정리 |

### 한 줄 요약

파이프라인의 뼈대는 잘 잡혀 있고, gate 중심의 intervention 실험을 빠르게 돌리기에 적합한 구조다.
Config 스키마 통일과 RNG/episode-boundary 문제를 정리하면, 나머지는 연구 진행하면서 자연스럽게 확장할 수 있는 상태.
