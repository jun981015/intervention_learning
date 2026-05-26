# Extensibility Review 2026-05-21

기준 git commit: `f1730d4` (`Initialize intervention learning scaffold`)

이 문서는 현재 코드에서 prefix/default/hardcoded assumption 때문에 나중에 확장성이 떨어질 수 있는 지점을 정리한다.
코드는 수정하지 않고, 다음 확장 작업 전에 확인할 체크리스트로 남긴다.

## 결론

현재 repo는 DAgger relabeling baseline과 intervention-learning scaffold로는 충분히 틀이 잡혀 있다.
다만 범용 프레임워크라기보다는 Robomimic Square, BCFlow learner, RLPD expert, lowdim observation 중심의 v0 scaffold다.

당장 blocker는 아니다. 다른 알고리즘을 붙이면서 아래 항목 중 필요한 것부터 풀면 된다.

## 우선순위 높은 hardcoded 지점

### 1. Default recipe가 Square DAgger에 강하게 묶여 있음

파일: `il/builders/config.py`

관련 위치:

- `DEFAULT_RECIPE.run.group = "square_recipe_v1"`
- `DEFAULT_RECIPE.env.name = "square-mh-low_dim"`
- `DEFAULT_RECIPE.learner.kind = "bc_flow"`
- `DEFAULT_RECIPE.expert.kind = "rlpd"`
- `DEFAULT_RECIPE.learner.config.target_action_key = "expert_actions"`
- pretrained path가 Square weight 경로로 박혀 있음

영향:

- config 없이 실행하면 사실상 Square DAgger run이 된다.
- 다른 env/algorithm으로 가려면 YAML을 항상 명시해야 한다.

판단:

- v0에서는 괜찮다.
- 나중에 CLI default는 minimal/no-op config로 바꾸거나, default config를 실험별 YAML로만 관리하는 편이 낫다.

### 2. Public schema를 legacy recipe로 변환하는 adapter가 아직 중간 구조임

파일: `il/builders/config.py`

관련 위치:

- `load_recipe()`가 `experiment:`가 있으면 `new_schema_to_legacy_recipe()`로 변환한다.
- `config/dagger.yaml`은 public schema인데 runtime은 legacy shape를 쓴다.

영향:

- config 문서와 runtime 내부 config가 다르다.
- 새 field를 추가하면 adapter에도 따로 매핑해야 한다.

판단:

- 지금은 빠르게 scaffold를 돌리기 위한 현실적인 선택이다.
- 나중에는 builders가 public schema를 직접 읽도록 옮기는 것이 맞다.

### 3. Replay sampling은 named batch 여러 개를 실제로 다 쓰지 않음

파일: `il/builders/config.py`

관련 위치:

- `_first_sampling_spec()`는 `replay.sampling.bc`를 우선 선택하고, 없으면 `rl` 하나만 선택한다.
- `new_schema_to_legacy_recipe()`는 결국 단일 `updates` spec 하나를 만든다.

영향:

- `replay.sampling.bc`와 `replay.sampling.rl`을 동시에 넘기는 hybrid update가 아직 안 된다.
- RL batch와 BC batch를 따로 샘플해서 learner에 넘기는 구조가 열려 있지만 runtime은 v0 단일 batch다.

판단:

- DAgger relabeling에는 충분하다.
- RL+BC, residual policy, diffusion+RL로 가려면 먼저 풀어야 한다.

### 4. Update objective inference 제거됨

파일: `il/builders/config.py`, `il/loops/updates.py`

현재 상태:

- runtime update path는 더 이상 `objective=bc|rl`을 보지 않는다.
- `learner_kind.startswith("bc_")`로 objective를 추론하지 않는다.
- update spec은 target actor, replay source, sampling knob, optional `target_action_key`만 정한다.
- 실제 loss는 `agent.update(batch)`를 구현한 agent kind가 결정한다.
- DAgger/BC relabeling은 `target_action_key="expert_actions"`로 표현한다.

남은 영향:

- named multi-batch RL+BC hybrid는 아직 별도 작업이다.
- `target_action_key`는 supervised-action label이 필요한 agent에만 의미가 있다.

### 5. Config에 있지만 runtime에서 아직 안 쓰는 field가 있음

파일: `config/dagger.yaml`, `il/loops/train_loop.py`

현재 미반영 또는 약하게 반영된 예:

- `training.update_interval`
- `training.updates_per_step`
- `training.reset_on_done`
- `evaluation.action_mode`
- `evaluation.render_video`
- `evaluation.video_episodes`
- `checkpointing.save_final`
- `checkpointing.save_replay`
- `checkpointing.keep_last`
- `storage.store_*`

영향:

- YAML만 보고 기대한 동작과 runtime 동작이 다를 수 있다.

판단:

- 실험 시작 전에는 특히 update/checkpoint/eval 관련 field를 맞춰두는 게 좋다.

### 6. Env registry가 Robomimic lowdim 하나뿐임

파일: `il/envs/__init__.py`, `il/builders/config.py`

관련 위치:

- `ENV_BUILDERS = {"robomimic_lowdim": ...}`
- public `env.kind: robomimic`은 내부에서 `robomimic_lowdim`으로 매핑된다.

영향:

- OGBench, custom gym env, IsaacLab env를 붙이려면 registry를 추가해야 한다.

판단:

- Square v0에는 충분하다.
- 다음 env 확장 시 가장 먼저 손볼 부분이다.

### 7. Robomimic dataset path와 reward가 고정적임

파일: `il/envs/robomimic_lowdim.py`

관련 위치:

- dataset path: `~/.robomimic/{task}/{dataset_type}/low_dim_v15.hdf5`
- reward base: `float(task_success)`
- reward transform: `reward = task_reward * reward_scale + reward_shift`
- success 시 terminate
- horizon은 task name prefix로 결정

영향:

- 다른 dataset 위치, image hdf5, custom reward, dense reward, shaped reward에는 바로 안 맞는다.

판단:

- 현재 reward hacking 우려 때문에 sparse binary reward를 의도적으로 둔 것으로 이해하면 된다.
- `reward_scale` / `reward_shift` affine transform은 config화했다. transform 전 값은 `info["task_reward"]`에 남긴다.
- dense reward나 task-specific shaped reward mode, dataset path config화는 아직 별도 작업이다.

### 8. Actor builder는 lowdim-only를 강하게 요구함

파일: `il/builders/actors.py`

관련 위치:

- `resolve_agent_config()`에서 `env_spec.obs_dim is None`이면 `NotImplementedError`.
- `create_agent()`도 image observations를 막는다.

영향:

- env/replay는 image 준비가 되어 있어도 policy/network가 image를 받을 수 없다.

판단:

- image input은 천천히 해도 되므로 현재 blocker는 아니다.
- image policy를 시작할 때는 여기와 network builder가 첫 수정 지점이다.

### 9. Expert는 무조건 pretrained path가 있어야 함

파일: `il/builders/actors.py`

관련 위치:

```text
if name == "expert" and not spec.get("pretrained_path"):
    raise ValueError(...)
```

영향:

- scripted expert, random expert, human expert, external service expert를 actor builder로 표현하기 어렵다.

판단:

- 현재 RLPD expert baseline에는 맞다.
- intervention learning으로 갈 때 expert 종류가 늘어나면 expert builder를 policy/provider registry로 분리하는 게 좋다.

### 10. Activation config가 문서/YAML에는 있지만 runtime에서 drop됨

파일: `il/builders/config.py`, `config/dagger.yaml`

관련 위치:

- `actor_activation: mish`가 YAML에 있음.
- `_actor_config_from_new_schema()`에서 `activation`, `actor_activation`, `critic_activation`을 제거한다.

영향:

- config를 바꿔도 activation은 바뀌지 않는다.
- 문서/YAML과 실제 동작이 다르다.

판단:

- activation ablation을 할 계획이면 빨리 정리해야 한다.
- 당장 고정 MLP면 큰 문제는 아니다.

## Train loop 관련 지점

### 11. Eval은 learner-only

파일: `il/loops/train_loop.py`

영향:

- expert eval, gate/intervention eval, video eval, action mode eval은 현재 없다.

판단:

- DAgger learner 성능 확인에는 충분하다.
- intervention method 비교에는 부족하다.

### 12. Update schedule이 단순함

파일: `il/loops/train_loop.py`

현재 동작:

- `step >= start_training`이면 매 step 모든 `update_specs`를 한 번씩 돈다.
- `update_interval`, `updates_per_step`는 현재 반영되지 않는다.

영향:

- UTD, sparse update, BC/RL alternating schedule을 config로 제대로 표현하지 못한다.

판단:

- DAgger v0에는 충분하다.
- RL+BC 실험 전에 풀어야 한다.

### 13. Buffer 부족 skip이 exception string에 의존함

파일: `il/loops/train_loop.py`

관련 동작:

- `ValueError` 메시지에 `"smaller than sequence_length"`가 있으면 update skip으로 처리한다.

영향:

- 에러 메시지가 바뀌면 skip logic이 깨진다.
- 다른 `ValueError`와 구분이 약하다.

판단:

- 작은 guardrail로는 동작한다.
- 나중에 `InsufficientReplayError` 같은 custom exception이 낫다.

### 14. Save behavior가 config flag를 무시함

파일: `il/loops/train_loop.py`

현재 동작:

- 마지막 checkpoint는 항상 저장.
- replay buffer는 `checkpointing.save_replay`가 true일 때만 저장한다.
- `save_final`, `keep_last`는 아직 반영되지 않는다.

영향:

- `save_replay: false`로 긴 실험의 replay 저장 용량을 줄일 수 있다.

판단:

- `save_replay`는 반영됐다.
- `save_final`, `keep_last`는 실제 long run 전 추가 검토가 필요하다.

## Logging 관련 지점

### 15. stdout summary가 BCFlow metric 이름에 묶여 있음

파일: `il/logger/logger.py`

관련 위치:

- stdout summary key에 `learner_bc/actor/bc_flow_loss`, `learner_bc/actor/grad/norm`이 직접 들어가 있다.

영향:

- 다른 알고리즘 metric은 JSONL/CSV/W&B에는 남아도 stdout에는 안 보일 수 있다.

판단:

- logger 자체는 interval 평균으로 잘 잡혀 있다.
- stdout key는 config로 받거나 prefix scan으로 바꾸면 더 좋다.

## Action chunk 관련 지점

### 16. Policy view는 chunk의 첫 action만 실행함

파일: `il/policies/agent_view.py`, `il/policies/bc_flow.py`, `il/policies/rlpd.py`

현재 동작:

- action chunk가 나오면 `chunk[0]`만 action으로 반환한다.
- full chunk는 info에 저장된다.

영향:

- true chunk execution, chunk queue, expert/learner 별 독립 queue는 아직 없다.

판단:

- 사용자가 말한 것처럼 구현 디테일이다.
- 현재 relabeling pipeline이 먼저라면 큰 blocker는 아니다.

## Algorithm 관련 지점

### 17. RLPD target critic은 앞 2개만 씀

파일: `il/algo/rl/rlpd.py`

관련 위치:

- `TARGET_NUM_QS = 2`
- `aggregate_target_qs()`는 첫 2개 critic만 사용한다.

영향:

- `num_qs`를 늘려도 target backup은 첫 2개 기준이다.

판단:

- 의도한 결정이면 괜찮다.
- ensemble ablation을 하려면 옵션화가 필요하다.

### 18. RLPD actor update는 critic ensemble 평균을 씀

파일: `il/algo/rl/rlpd.py`

현재 동작:

- actor loss에서 `q = jnp.mean(qs, axis=0)`.

영향:

- actor objective의 Q aggregation이 고정이다.

판단:

- 현재 우리가 논의했던 방향과 맞다.
- min/mean 비교를 하려면 config화하면 된다.

## Legacy/experiment scripts 관련 지점

### 19. Scripts는 Square hardcode가 많음

파일:

- `scripts/train_dagger_square.py`
- `scripts/train_bcflow_topk_robomimic.py`
- `scripts/eval_bcflow_policy.py`

예:

- default env가 `square-mh-low_dim`
- default dataset이 `~/.robomimic/square/mh/low_dim_v15.hdf5`
- run name에 `square`, `topk`, `bcflow`가 들어감

영향:

- scripts는 일반 framework entrypoint라기보다 실험 helper다.

판단:

- 메인 경로는 `python -m il.train --config ...`로 보는 게 맞다.
- scripts는 reference/utility로 남기면 된다.

## 다음에 풀면 좋은 순서

1. `training.update_interval`, `training.updates_per_step`, `checkpointing.save_*`를 runtime에 반영한다.
2. stdout logger key를 algorithm-agnostic하게 만든다.
3. replay sampling을 named multi-batch로 넘길 수 있게 한다.
4. env registry에 두 번째 env를 추가할 때 dataset path/reward mode를 config화한다.
5. action chunk queue는 별도 작업으로 구현한다.
6. image policy는 env/replay가 안정된 뒤 network builder부터 추가한다.

## 현재 판단

현재 상태는 "다른 알고리즘을 꽂을 수 있는 연구 scaffold"로는 충분하다.
다만 "알고리즘만 추가하면 모든 실험이 바로 되는 범용 프레임워크"는 아니다.
지금 단계에서 강한 default는 속도를 위해 괜찮고, 실제 확장 시 위 항목을 하나씩 풀면 된다.
