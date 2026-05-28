# Code Review — 2026-05-26

> Reviewer: Claude Opus 4.7 (`/code-review high`)
> Scope: `il/` 전체 패키지 (PR diff 없음 — main 브랜치 working tree 기준)
> Method: 3 angle (line-by-line / abstraction / cross-file) finder agent 병렬 → 1-vote verify → recall-biased
> Raw 후보 18개 → REFUTED 6개 → 최종 finding 9개

이전 리뷰 `code_review.md` (2026-05-22, Opus 4.6) 이후의 변경분 — residual policy path, residual TD3 추가, intervention pipeline checkpoint — 포함.

---


## 현재 해결 상태 — 2026-05-28

이 문서는 리뷰 당시 발견 사항을 보존한다. 아래 표는 현재 코드 기준 진행 상태다.
최신 전체 스냅샷은 [STATUS_2026-05-28.md](STATUS_2026-05-28.md)를 본다.

| id | 상태 | 코드 기준 확인 |
| --- | --- | --- |
| P0-1 ExpertQGapGate episode reset | 해결 | `ControllerGate`가 `@runtime_checkable` Protocol이고, `TrainContext.gate` / `build_gate()` 타입이 `ControllerGate | None`이다. build-time Protocol validation이 있고, `ExpertQGapGate`와 `RandomGate` 모두 `reset_episode()`를 구현한다. |
| P0-2 residual_scale train/eval 불일치 | 해결 | `il/loops/rollout.py::resolve_residual_scale()`가 train rollout과 context eval에서 공유된다. |
| P0-3 buffer 부족 예외 string match | 의도적으로 유지 | 사용자가 `BufferTooSmall` custom exception 제거를 요청했다. replay가 아직 `sequence_length`보다 작을 때는 기존 `"smaller than sequence_length"` `ValueError` 문자열 기반 skip을 유지한다. |
| P1-1 residual rollout hardcoding | 부분 해결 | residual learner proposal 생성은 residual-only와 residual+gate path가 같은 helper를 쓴다. 다만 `rollout.execute == "residual"` 분기와 `PolicyOutput.info` key contract는 아직 남아 있다. |
| P1-2 gate Protocol이 expert_agent 중심 | 부분 해결 | Protocol runtime contract는 정리했다. 다만 `decide(..., expert_agent=..., action_dim=...)` 시그니처는 아직 expert-agent 중심이다. `GateContext`는 learner/base/history가 필요한 새 gate family가 들어올 때 도입한다. |
| P1-3 hasattr dispatch | 미해결 | critic-only update, Q API, policy sampling 쪽에 `hasattr` dispatch가 남아 있다. |
| P1-4 residual kind set 중복 | 미해결 | `{"residual_rlpd", "residual_td3"}` literal set이 actor builder 여러 위치에 남아 있다. 새 residual family 추가 전 registry/spec 정리가 필요하다. |
| P1-5 `PolicyOutput.info` implicit schema | 미해결 | `full_action_chunk`, `base_action`, `residual_action`, `raw_residual_action` 등 info key contract가 typed field/Protocol로 승격되지 않았다. |
| P2-1 critic loss normalization | 해결 | RLPD, residual RLPD, residual TD3, BC critic loss가 valid sample 수 기준 normalizer를 사용한다. |

현재 다음 작은 작업 후보는 아래 셋이다.

```text
1. residual+intervention gate real-env build-only / short rollout smoke를 먼저 돌린다.
2. 새 residual family 전에 il/builders/actors.py의 residual kind literal set을 registry/spec로 정리한다.
3. PolicyOutput.info의 residual/chunk metadata contract를 typed helper나 작은 dataclass로 정리한다.
```

---
## TL;DR

**구조**: RL+intervention 코드베이스로서 평균 이상. 모듈 경계, YAML recipe, replay buffer routing, NaN 모니터링이 단단함.

**가장 큰 약점**: `"residual"`이라는 한 가지 알고리즘 변형이 너무 깊이 박혀 있음. `rollout.execute == "residual"` string과 `{"residual_rlpd", "residual_td3"}` set이 4–5개 파일에 산재. 5–6개 algorithm × 3–4개 gate variant ablation 단계로 가려면 한 번 refactor 필요.

**리뷰 당시 즉시 고칠 버그 2개**는 현재 해결됐다.
1. Gate 상태 episode reset은 `ControllerGate.reset_episode()` runtime contract로 정리했다.
2. Eval/train의 `residual_scale` fallback은 `resolve_residual_scale()` 공유 helper로 맞췄다.

---

## 1. P0 — 즉시 수정 (logic bug)

### 1.1 ExpertQGapGate 상태가 episode 경계를 넘어 누수

**파일**: `il/gating/expert_q_gap.py:32-33`
**관련**: `il/loops/rollout.py:61-63` (`reset_rollout_state`), `il/loops/train_loop.py:240`

**증상**: `ExpertQGapGate`는 `_remaining_steps`, `_last_info`를 dataclass 필드로 들고 있는 stateful 객체. `reset_rollout_state(context)`는 `context.rollout_state` dict만 비울 뿐, `context.gate`는 건드리지 않음. Gate에는 `reset()` 메서드 자체가 없음.

**시나리오**:
- Episode 1, step 50에서 `intervention_horizon=4`로 intervention 시작 → `_remaining_steps=3`
- Episode 1이 step 51에서 terminated/truncated → `_remaining_steps=3` 그대로 남음
- Episode 2 첫 `decide()` 호출 → `_remaining_steps > 0` True → `_horizon_decision()`이 EXPERT 강제 반환
- Episode 2의 step 0~2가 Q-gap 신호 없이 expert 통제됨
- `gate/intervention_started_count`는 증가하지 않으므로 silent

**수정 방향**:
- `ControllerGate` Protocol에 `reset_episode()` (no-op 기본) 추가
- `reset_rollout_state(context)`에서 `if context.gate is not None and hasattr(context.gate, 'reset_episode'): context.gate.reset_episode()` 호출
- 또는 train_loop.py의 episode-end 분기에서 직접 reset

---

### 1.2 Eval과 train의 `residual_scale` fallback 경로 다름

**파일**: `il/evaluation/evaluator.py:47`
**비교 대상**: `il/loops/rollout.py:285`

**증상**:
```python
# rollout.py:285 (train)
residual_scale = float(
    context.learner.config.get("residual_scale",
        context.config["rollout"].get("residual_scale", 1.0))
)

# evaluator.py:47 (eval)
residual_scale = float(context.learner.config.get("residual_scale", 1.0))
```

YAML에서 `rollout.residual_scale=0.1`만 설정하고 `learner.config`에 없으면, train은 0.1로 실행되지만 eval은 1.0으로 실행됨 → 10배 큰 perturbation.

**수정 방향**:
- 헬퍼 `_resolve_residual_scale(context)`를 `loops/rollout.py`에 추가하고 양쪽에서 호출
- 또는 `build_context`에서 한 번 resolve해서 `context.residual_scale`에 저장

---

### 1.3 Exception을 string match로 swallow

**파일**: `il/loops/train_loop.py:253-255`
**관련**: `il/buffers/replay_buffer.py:562`

**증상**:
```python
except ValueError as exc:
    if "smaller than sequence_length" not in str(exc):
        raise
```

`replay_buffer.py:562`의 에러 메시지가 reword 되면 silent breakage. 반대로, 무관한 ValueError가 우연히 이 문자열을 포함하면 silent suppression.

**수정 방향(리뷰 당시 제안)**: `class BufferTooSmall(ValueError)` 같은 전용 예외를 정의하고 isinstance 체크.

**현재 결정**: 사용자가 `BufferTooSmall`을 제거하라고 했으므로 전용 예외는 도입하지 않는다. 기존 문자열 기반 skip은 남겨둔다.

---

## 2. P1 — 추상화 / 확장성 (새 알고리즘 추가 시 비용 큼)

### 2.1 `"residual"`이 1-class citizen으로 박혀 있음 (가장 큰 추상화 문제)

**박힌 위치**:
- `il/train.py:52` — `if rollout_execute == "residual"`
- `il/loops/train_loop.py:187` — `if context.config["rollout"].get("execute") == "residual"` + `learner_output.info`에서 `base_action`, `residual_action` 직접 추출
- `il/loops/rollout.py:350` — `if execute == "residual": return _choose_residual_action(...)`
- `il/evaluation/evaluator.py:40` — eval 경로에서도 동일 분기
- `il/builders/components.py` — buffer prefill 시 residual base action 캐싱

**박힌 set**:
- `il/builders/actors.py:108, 128, 169, 204` — `{"residual_rlpd", "residual_td3"}` 4번 등장
  - 108: `residual_policy=True`, `base_obs_dim` 세팅
  - 128: `obs_dim += action_dim`
  - 169: metadata 검증의 expected_obs_dim 계산
  - 204: `AgentPolicyView.obs_dim` 계산

**새 residual variant (예: ResidualSAC, ResidualIQL) 추가 시 비용**:
1. `il/algo/rl/residual_sac.py` 작성
2. `il/builders/actors.py` 5군데 수정 (`default_agent_config`, `create_agent`, 그리고 위 4개 set 모두)
3. set을 빠뜨리면 obs_dim mismatch가 JAX 깊은 곳에서 터짐

**수정 방향**:
- `default_agent_config` 등록을 `AGENT_REGISTRY: dict[str, AgentSpec]`로 변환. `AgentSpec`이 `is_residual`, `make_default_config`, `create_agent` 등을 들고 있음
- 또는 config 레벨에 `residual_policy: bool` 플래그를 두고 set 검사 대신 이 플래그를 봄

---

### 2.2 `rollout.execute` string dispatch → `RolloutStrategy` 추상이 없음

**현재**: `il/loops/rollout.py:346-373`의 `choose_rollout_action`이 `execute` 문자열로 분기. `residual` 경로는 별도 함수 `_choose_residual_action`. 둘이 반환 형식만 같지 내부 로직은 완전히 다름.

**새 rollout topology 추가 시 비용**: 예) hierarchical residual, mixture-of-policies, learnable gate가 controller 결정 자체에 영향 → train_loop, rollout, evaluator 모두 새 분기 추가 필요.

**수정 방향**:
```python
class RolloutStrategy(Protocol):
    def step(self, context, observation, *, step: int) -> RolloutStep: ...

@dataclass
class RolloutStep:
    action: np.ndarray
    learner_output: PolicyOutput
    expert_output: PolicyOutput
    decision: GateDecision
    transition_metadata: dict  # base_action, residual_action, next_base_action, ...
```

train_loop는 strategy 객체만 호출. 새 알고리즘은 새 strategy 추가만 하면 됨.

---

### 2.3 `ControllerGate.decide`에 `expert_agent`가 박혀 있음

**파일**: `il/gating/base.py:13-25`, `il/gating/expert_q_gap.py:73-84`

**문제**: Gate Protocol이 `expert_agent` 파라미터를 받고, ExpertQGapGate가 `hasattr(expert_agent, "evaluate_q")` 또는 `"q_values"`로 sniff. "expert critic으로 Q-gap을 본다"는 한 가지 패턴이 Protocol 레벨에 새겨짐.

**막혀 있는 확장**:
- Learner ensemble disagreement gate (학습 중 critic 분산 클 때 개입)
- Learnable gate (gate 자체를 신경망으로 학습)
- State-only gate (obs만 보고 결정)

**수정 방향**:
```python
@dataclass
class GateContext:
    step: int
    observation: np.ndarray
    learner: PolicyOutput
    expert: PolicyOutput
    rng: np.random.Generator
    expert_agent: Any | None = None
    learner_agent: Any | None = None     # 새 필드 추가 가능
    history: GateHistory | None = None   # 누적 통계
    action_dim: int | None = None

class ControllerGate(Protocol):
    def decide(self, ctx: GateContext) -> GateDecision: ...
    def reset_episode(self) -> None: ...  # 1.1 버그도 같이 해결
```

---

### 2.4 `hasattr` sniffing dispatch가 산재

**위치**:
- `il/loops/updates.py:179-182` — `hasattr(agent, "batch_update_critic_only")` / `"update_critic_only"`
- `il/gating/expert_q_gap.py:75-78` — `hasattr(expert_agent, "evaluate_q")` / `"q_values"`
- `il/policies/agent_view.py` — agent attribute sniffing

**문제**: Protocol에 없으니 IDE/타입체커가 못 잡음. 새 algo 작성자가 어떤 메서드를 publish해야 하는지 알 방법이 없음.

**수정 방향**:
```python
class CriticOnlyUpdatable(Protocol):
    def update_critic_only(self, batch): ...
    def batch_update_critic_only(self, batch): ...

class QEvaluable(Protocol):
    def evaluate_q(self, obs, action, *, q_agg: str): ...
```

---

### 2.5 `PolicyOutput.info`가 implicit type system

**파일**: `il/utils/types.py:26-31` (`info: dict[str, Any]`)

**암묵 contract 키**:
- `base_action`, `residual_action`, `raw_residual_action` — residual rollout
- `full_action_chunk` — chunked base policy (`il/loops/rollout.py:79`, `il/gating/expert_q_gap.py:62`에서 모두 기대)
- `base_chunk_index`, `base_chunk_size`, `base_queue_refill`, `base_queue_remaining_after_pop`
- `residual_warmup`, `residual_warmup_steps`, `warmup_noise_scale`
- `missing` — `_missing_policy_output`에서 사용

**문제**: 새 chunked base policy 작성자가 `full_action_chunk` 키 이름·shape를 어디서 알 수 있나? rollout.py를 역공학해야 함.

**수정 방향**: 자주 쓰는 메타데이터는 `PolicyOutput`의 typed field로 승격하거나, `ResidualPolicyOutput(PolicyOutput)` 같은 subclass 분리.

---

## 3. P2 — 일관성 / 작은 이슈

### 3.1 Critic loss normalization 불일치

**현재 상태**: 해결됨. RLPD, residual RLPD, residual TD3, BC critic loss가 valid sample 수 기준 normalizer를 사용한다.

**위치(리뷰 당시)**: `il/algo/rl/residual_rlpd.py:91` vs `il/algo/bc/critic.py:113-115`

```python
# residual_rlpd.py:91
critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()
# → num_qs * batch_size로 나눔. valid_fraction 낮으면 loss magnitude 같이 줄어듦.

# bc/critic.py:113-115
squared_error = jnp.square(td_error) * valid
normalizer = jnp.maximum(jnp.sum(valid) * q.shape[0], 1.0)
critic_loss = jnp.sum(squared_error) / normalizer
# → (num_qs * sum(valid))로 나눔. valid_fraction에 무관.
```

**영향**: 짧은 에피소드 + 큰 n-step → valid_fraction 떨어지면 residual_rlpd의 effective critic LR이 비례해서 줄어듦. Hyperparameter tuning 결과가 episode 길이에 따라 silent하게 바뀜.

**수정 방향**: 둘을 `sum(squared * valid) / (sum(valid) * num_qs)`로 통일.

---

### 3.2 `docs/` vs `docs_agents/` 이중화

**위치**: 리포 루트
- `docs/` (15+ 파일)
- `docs_agents/` (13개 파일, `docs/`와 상당 부분 겹침)

의도된 분리인지, 옛 파일을 미정리한 것인지 확인 필요. Agent용 문서는 `docs_agents/`로 통일하거나, 한 쪽을 정리해야 함.

---

## 4. REFUTED (finder가 올렸지만 검증 결과 false positive)

검토 과정에서 false positive로 판명된 후보들. 비슷한 의심이 나중에 다시 올라올 수 있으니 기록:

### 4.1 ❌ Replay buffer wrap state에서 `next_idxs < self.size` 잘못됨
**위치**: `il/buffers/replay_buffer.py:540`
**reasoning**: 표준 append/wrap 버퍼에서 `size < max_size`이면 `pointer == size`이고 populated indices는 `0..size-1`. `size == max_size` (wrap 완료)이면 모든 슬롯 populated, `next_idxs < max_size` 항상 True. `replace_episode`는 `_clear()` 후 재삽입이라 비연속 wrap 상태가 발생하지 않음. 추가로 `episode_ids`/`episode_steps` 검증이 같이 걸려 있어 false positive 한 번 더 막힘.

### 4.2 ❌ BC critic loss normalizer가 `* batch_size`로 너무 큼
**위치**: `il/algo/bc/critic.py:114`
**reasoning**: `q.shape[0]`은 ensemble의 `num_qs` (보통 2~10). `batch_size` 아님. `squared_error`는 `[num_qs, batch_size]` 전체 합이므로 `sum(valid) * num_qs`로 나누는 게 올바른 (head, valid_sample) 평균.

### 4.3 ❌ HG-DAgger horizon off-by-one
**위치**: `il/gating/expert_q_gap.py:106, 153`
**reasoning**: Trigger step은 `_horizon_decision()` 거치지 않고 main `decide()` 경로에서 직접 EXPERT 반환. `_remaining_steps = horizon - 1` 세팅 후 다음 호출부터 decrement. horizon=3이면 trigger(1) + horizon_decision×2 = 총 3 expert step. 수학적으로 맞음.

### 4.4 ❌ Residual TD3 vs RLPD BC target scaling 모순
**위치**: `residual_td3.py:149-152` vs `residual_rlpd.py:144-147`
**reasoning**: 수학적으로 동등. 두 식 모두 actor의 raw output을 `delta/scale`로 학습시킴. TD3는 `||scale*raw - delta||²`, RLPD는 `-log_prob(delta/scale)` — 둘 다 optimum이 `raw = delta/scale`.

### 4.5 ❌ `residual_action` fallback이 scaled vs raw 혼동
**위치**: `il/loops/train_loop.py:189`
**reasoning**: `info["residual_action"]`도 scaled, `learner_output.action`도 scaled (rollout.py:314). 둘이 같은 값. fallback 무해.

### 4.6 ❌ `include_failed_interventions=False`일 때 expert segment drop
**위치**: `il/buffers/routing.py:106-111`
**reasoning**: 플래그 이름이 명시적으로 opt-in. 의도된 동작이고, 별도 ablation knob으로 제공됨.

---

## 5. 강점 (유지할 것)

코드 변경 시 다음 패턴은 지키는 게 좋음:

1. **`train.py` orchestration / `train_loop.py` env-step / `algo/` gradient / `buffers/` storage 책임 분리**
2. **YAML recipe driven** — 새 실험을 코드 수정 없이 돌릴 수 있음
3. **`StepRecord → step_record_to_transition → buffer`로 schema 일원화** (`il/buffers/schema.py`)
4. **`_rollout_health_metrics`, `_array_health`로 매 스텝 NaN/Inf 모니터링** — RL에서 매우 중요
5. **`_assert_finite_target_actions`, `_assert_residual_metadata`** — JAX 진입 전 fast-fail
6. **`gating_reasons`, `gating_scores`, `interventions` 필드를 transition에 저장** — 사후 분석 가능
7. **`update_specs`가 list라 여러 update (critic warmup, BC aux 등) 조합 가능**

---

## 6. 에이전트가 봐야 할 경로 / 명령 모음

### 6.1 우선순위별 파일

**P0 (지금 고치기)**:
- `il/gating/expert_q_gap.py` (32, 73-176)
- `il/loops/rollout.py` (61-63 `reset_rollout_state`)
- `il/loops/train_loop.py` (239-244 episode 종료 분기, 253-255 exception swallow)
- `il/evaluation/evaluator.py` (47)

**P1 (리팩토링)**:
- `il/builders/actors.py` (108, 128, 169, 204 — residual set 4곳)
- `il/loops/rollout.py` (346-373 — execute string dispatch)
- `il/loops/train_loop.py` (185-192 — residual info-key 추출)
- `il/gating/base.py` (ControllerGate Protocol 시그니처)
- `il/loops/updates.py` (172-183 — hasattr dispatch)
- `il/utils/types.py` (`PolicyOutput.info` schema화)

**P2 (consistency)**:
- `il/algo/rl/residual_rlpd.py:91` vs `il/algo/bc/critic.py:113-115`

### 6.2 빠른 grep 명령

```bash
# residual이 박혀 있는 모든 위치
grep -rn '"residual"' il/ --include='*.py'
grep -rn 'residual_rlpd\|residual_td3' il/ --include='*.py'

# hasattr 기반 dispatch
grep -rn 'hasattr(' il/ --include='*.py'

# info dict 키 contract
grep -rn 'info\.get\|info\[' il/ --include='*.py'

# Gate 관련
grep -rn 'gate\|Gate' il/loops/ il/gating/ --include='*.py'

# residual_scale 사용처
grep -rn 'residual_scale' il/ --include='*.py'
```

### 6.3 검증 절차 (수정 후)

```bash
# 1. residual 관련 변경: smoke 테스트
python scripts/train_dagger_square.py --config config/smoke_residual_square.yaml --build-only
python -m il.train --config config/smoke_residual_square.yaml --build-only

# 2. Gate reset 버그 수정 후: 짧은 episode + 큰 horizon으로 검증
#    config/smoke_expert_q_gap_square.yaml에서 intervention_horizon=8, 짧은 max_steps로 설정
#    gate/intervention_started_count와 gate/expert_execute_rate 비교

# 3. eval residual_scale 수정 후: rollout.residual_scale != 1.0인 config로
#    train + eval 한 사이클 돌려서 eval/return이 train의 recent_return 근처에 있는지 확인

# 4. 일반 회귀: 기존 smoke 전부
ls config/smoke_*.yaml | xargs -I {} python -m il.train --config {} --build-only
```

### 6.4 관련 문서

- `docs/PIPELINE.md` — 전체 흐름
- `docs/CONFIG_SCHEMA_DECISIONS.md` — YAML schema 결정 사항
- `docs/EXTENSIBILITY_REVIEW_2026-05-21.md` — 이전 확장성 리뷰
- `code_review.md` (리포 루트) — 2026-05-22 이전 리뷰 (Opus 4.6)

---

## 7. 머신 판독용 finding JSON

자동화/스크립트가 파싱하기 좋은 형식. 본 리뷰의 9개 finding 전체.

```json
[
  {
    "id": "P0-1",
    "file": "il/gating/expert_q_gap.py",
    "line": 32,
    "severity": "high",
    "category": "logic_bug",
    "summary": "ExpertQGapGate is stateful (_remaining_steps, _last_info) but the gate is never reset on episode boundaries — reset_rollout_state() at il/loops/train_loop.py:240 clears only the rollout dict, and the gate has no reset() method.",
    "failure_scenario": "Episode 1 triggers intervention at step 50 with horizon=4; episode terminates at step 51 with _remaining_steps=3 buffered. Episode 2's first decide() sees _remaining_steps>0, forces EXPERT on episode 2 step 0 without any Q-gap signal. gate/intervention_started_count is not bumped so the corruption is silent.",
    "related": ["il/loops/rollout.py:61", "il/loops/train_loop.py:240"]
  },
  {
    "id": "P0-2",
    "file": "il/evaluation/evaluator.py",
    "line": 47,
    "severity": "high",
    "category": "logic_bug",
    "summary": "Eval reads residual_scale ONLY from learner.config (default 1.0); train rollout falls back to context.config['rollout']['residual_scale']. If residual_scale is set only under rollout: in YAML, train and eval execute different policies.",
    "failure_scenario": "YAML sets rollout.residual_scale=0.1, learner.config has no residual_scale. Training executes base + 0.1*residual. Eval executes base + 1.0*residual — 10x larger perturbation. eval/success_rate looks broken even when training is healthy.",
    "related": ["il/loops/rollout.py:285"]
  },
  {
    "id": "P0-3",
    "file": "il/loops/train_loop.py",
    "line": 253,
    "severity": "medium",
    "category": "logic_bug",
    "summary": "train_loop swallows update errors by substring-matching the exception message ('smaller than sequence_length'). Rewording the source error string at il/buffers/replay_buffer.py:562 silently breaks warmup behavior.",
    "failure_scenario": "Someone tweaks the error to 'smaller than the requested sequence length'. The except clause no longer matches; train loop dies at step 1. Or a future unrelated ValueError containing the magic substring gets silently suppressed.",
    "related": ["il/buffers/replay_buffer.py:562"]
  },
  {
    "id": "P1-1",
    "file": "il/loops/train_loop.py",
    "line": 187,
    "severity": "high",
    "category": "abstraction",
    "summary": "Train loop hard-codes `rollout.execute == 'residual'` and reaches into learner_output.info for keys ('base_action', 'residual_action') that form an undocumented contract. The loop knows both the algo and the info-dict shape.",
    "failure_scenario": "Adding hierarchical_residual, mixture-of-policies, learnable-gate execution modes requires editing train_loop.py to add new branches AND knowing which info keys to extract. A new residual-like algo that emits 'delta_action' instead of 'residual_action' silently writes nans to the buffer via the get() default.",
    "related": ["il/train.py:52", "il/loops/rollout.py:346", "il/evaluation/evaluator.py:40"]
  },
  {
    "id": "P1-2",
    "file": "il/gating/expert_q_gap.py",
    "line": 75,
    "severity": "medium",
    "category": "abstraction",
    "summary": "ControllerGate.decide() takes `expert_agent` parameter and ExpertQGapGate sniffs hasattr for 'evaluate_q' / 'q_values'. The gate Protocol bakes in 'the only critic worth querying lives on the expert', blocking learner-side disagreement gates, ensemble-variance gates, and learnable gates.",
    "failure_scenario": "Implementing a 'learner ensemble disagreement' gate (natural HG-DAgger extension) requires either threading a `learner_agent` parameter into the Protocol (breaks every existing gate) or smuggling it through expert_agent. A learnable gate that needs its own optimizer state has nowhere to put it.",
    "related": ["il/gating/base.py:13"]
  },
  {
    "id": "P1-3",
    "file": "il/loops/updates.py",
    "line": 179,
    "severity": "medium",
    "category": "abstraction",
    "summary": "Critic-only warmup dispatch uses hasattr(agent, 'batch_update_critic_only') / 'update_critic_only'. New RL algos without these method names raise 'does not support critic-only warmup updates' even if critic_warmup_steps was never requested.",
    "failure_scenario": "Adding IQL or CQL (often have no separable critic-only pretrain) requires implementing dummy update_critic_only methods just to satisfy hasattr, or wrapping every call site. Dispatch should route by whether the spec actually asks for actor=False.",
    "related": []
  },
  {
    "id": "P1-4",
    "file": "il/builders/actors.py",
    "line": 108,
    "severity": "medium",
    "category": "abstraction",
    "summary": "Literal set {'residual_rlpd', 'residual_td3'} appears at 4 sites in actors.py (lines 108, 128, 169, 204) to gate residual-specific behavior. Adding a new residual variant requires touching all four.",
    "failure_scenario": "Adding ResidualSAC: developer adds 'residual_sac' to default_agent_config and create_agent but forgets the set at 169 or 204. create_agent builds obs_dim correctly, but metadata validation and AgentPolicyView.obs_dim drop the action-dim concat. policy_view at inference receives a tensor of wrong size and errors deep inside JAX.",
    "related": ["il/builders/actors.py:128", "il/builders/actors.py:169", "il/builders/actors.py:204"]
  },
  {
    "id": "P1-5",
    "file": "il/loops/rollout.py",
    "line": 79,
    "severity": "medium",
    "category": "abstraction",
    "summary": "_enqueue_base_policy_output requires base policies emit info['full_action_chunk'] of shape [horizon, action_dim], but no Protocol declares this contract. ExpertQGapGate at il/gating/expert_q_gap.py:62 also relies on the same key.",
    "failure_scenario": "Adding a base policy whose chunk format is [horizon, action_dim_chunked] flattened, the queue treats the flat action as primitive: line 82 reshapes by action_dim, masking the bug only if action.size happens to be a multiple of action_dim.",
    "related": ["il/gating/expert_q_gap.py:62", "il/utils/types.py:26"]
  },
  {
    "id": "P2-1",
    "file": "il/algo/rl/residual_rlpd.py",
    "line": 91,
    "severity": "low",
    "category": "consistency",
    "summary": "Residual RLPD critic loss is (jnp.square(q - target_q) * valid[..., -1]).mean(), averaging over num_qs * batch_size regardless of valid_fraction. il/algo/bc/critic.py:114 instead divides by sum(valid) * num_qs.",
    "failure_scenario": "With n-step targets that frequently cross episode boundaries (large nstep, short episodes), valid[..., -1].mean() can drop to 0.5 or lower. residual_rlpd critic_loss magnitude halves, effectively cutting critic LR. Tuning that worked with long episodes silently degrades.",
    "related": ["il/algo/bc/critic.py:113"]
  }
]
```

---

## 8. Refactor 우선순위 제안

2026-05-28 현재 기준 작업 순서:

1. **Residual+gate real-env smoke** — 더미 smoke는 통과했으므로 실제 Robomimic config에서 build-only와 짧은 rollout을 확인한다.
2. **P1-4 agent registry** — `AGENT_REGISTRY` 또는 `AgentSpec` 패턴을 도입해 residual kind set 중복을 제거한다.
3. **P1-5 PolicyOutput metadata contract** — `base_action`, `residual_action`, `raw_residual_action`, `full_action_chunk` key contract를 typed helper나 작은 dataclass로 정리한다.
4. **P1-3 Protocol 정리** — critic-only update, Q evaluation, policy sampling의 `hasattr` dispatch를 필요한 Protocol로 좁힌다.
5. **Dataset adapter / canonicalization** — offline demo/prefill 의미를 adapter에서 명시한다.
6. **Replay save/load round-trip** — 실제 env 산출 replay까지 schema와 episode/image metadata round-trip을 검증한다.
7. **Action chunk queue** — learner/expert 일반 action chunk queue는 residual base queue와 별도 설계로 처리한다.
8. **Image policy** — env/replay plumbing은 있으나 policy image encoder가 없으므로 별도 설계 후 진행한다.
