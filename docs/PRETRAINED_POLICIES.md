# Pretrained Policy Artifacts

이 문서는 외부 실험에서 가져온 pretrained policy weight를 이 repo에서 어떻게 쓰는지 기록한다.

## 원칙

- `intervention_learning` runtime 코드에는 source repo 전용 loader를 넣지 않는다.
- weight 파일은 이 repo의 agent state-dict layout과 config에 맞아 있어야 한다.
- git에는 checkpoint, replay, video, log, wandb 산출물을 넣지 않는다.
- artifact는 현재 `exp/pretrained/` 아래에 두며 `.gitignore` 대상이다.

## 현재 Artifact

### Square RLPD Expert

경로:

```text
exp/pretrained/rlpd_square_bc03_seed0_2m/
  params_100000.pkl
  params_200000.pkl
  ...
  params_2000000.pkl
  config.json
  metadata.json
```

용도:

- expert policy 후보
- square `bc_alpha=0.3` RLPD checkpoint sweep
- `params_100000.pkl`부터 `params_2000000.pkl`까지 100k 단위 checkpoint가 있다.
- `metadata.checkpoint_step`은 기본 restore용으로 `2000000`을 가리킨다. 다른 step을 쓰려면 actor config에서 `checkpoint_step`을 명시한다.

source:

```text
/home/junhyeong/repos/qc/exp/qc/square-rlpd-sparse-bc03-2m-seed0/square-mh-low_dim/sd00020260507_123338/params_2000000.pkl
```

확인된 shape:

```text
obs_dim=23
action_dim=7
horizon_length=1
actor_layer_norm=False
critic_layer_norm=True
```

로드 예시:

```python
import json
from pathlib import Path

from il.policies import RLPDPolicy

run_dir = Path("exp/pretrained/rlpd_square_bc03_seed0_2m")
config = json.loads((run_dir / "config.json").read_text())
metadata = json.loads((run_dir / "metadata.json").read_text())

expert = RLPDPolicy.from_checkpoint(
    run_dir / "params_2000000.pkl",
    config=config,
    obs_dim=metadata["obs_dim"],
    action_dim=metadata["action_dim"],
    seed=metadata["seed"],
)
```

과거 검증 결과(critic contract 수정 전):

```text
Restored from .../rlpd_square_bc03_seed0_2m/params_2000000.pkl
action_shape (7,)
log_prob 16.3937
```

### ToolHang Residual TD3 Expert

현재 ToolHang expert로 쓰는 residual TD3 weight는 scale `0.2`, BC regularization `0.1`, actor LR `5e-5`, warmup `0.1` run의 1.5M checkpoint다.

재사용 artifact 경로:

```text
exp/pretrained/residual_td3_tool_hang_ph_scale02_bc01_actorlr5e5_warmup01_seed0_1500k/
  params_1500000.pkl
  config.json
  metadata.json
```

원본 run 경로와 전체 checkpoint:

```text
exp/runs/intervention_learning/tool_hang_residual_online/tool_hang-ph-low_dim/
  tool_hang_residual_td3_bcflow_top200_mixed50_shiftm1_nstep5_scale02_bc01_actorlr5e5_warmup01_seed0_2m/
    params_100000.pkl
    params_200000.pkl
    ...
    params_1500000.pkl
    ...
    params_2000000.pkl
```

같이 쓰는 base policy:

```text
exp/pretrained/bcflow_tool_hang_ph_top200_actorln_seed0_1m/
  params_1000000.pkl
```

확인된 shape:

```text
env_obs_dim=53
base_action_dim=7
residual_actor_obs_dim=60
action_dim=7
residual_scale=0.2
checkpoint_step=1500000
```

사용 config:

```text
config/tool_hang_residual_td3_scale02_ckpt1500k_expert_random_gate_smoke.yaml
```

주의:

- residual TD3 expert는 plain state만으로 action을 내지 않는다. `state + base_action`을 입력으로 받고, 실행 action은 `clip(base_action + residual_scale * raw_residual)`이다.
- 따라서 ToolHang residual expert는 위 base BCFlow checkpoint와 같이 써야 한다.
- expert role로 쓸 때 artifact `config.json`은 deterministic proposal을 위해 `exploration_noise=0.0`으로 둔다.
- 이 1.5M checkpoint의 critic은 2026-05-29 이전 contract인 `Q([state, base_action], executed_action)`로 학습되었다. 현재 수정된 code contract에서는 Q-gap critic 용도로 쓰지 말고 재학습해야 한다. full-agent restore도 critic shape mismatch가 날 수 있으며, actor만 재사용하려면 별도 partial restore가 필요하다.

과거 검증 결과(critic contract 수정 전):

```text
Restored from .../residual_td3_tool_hang_ph_scale02_bc01_actorlr5e5_warmup01_seed0_1500k/params_1500000.pkl
expert=residual_td3
gate=random
real-env ToolHang 20-step smoke passed
interventions=8/20
intervention_action_matches_expert_max_abs_err=0
```

### Diffusion / Flow BC Learner

경로:

```text
exp/pretrained/bcflow_square_actorln_seed0_1m/
  params_1000000.pkl
  config.json
  metadata.json
```

용도:

- learner 초기 policy 후보
- square offline diffusion/flow BC actor-LayerNorm 1M checkpoint

source:

```text
/home/junhyeong/repos/qc/exp/qc/qc-grid-base-offline-actorln/square-mh-low_dim/sd00020260512_134241/params_1000000.pkl
```

확인된 shape:

```text
obs_dim=23
action_dim=7
horizon_length=5
full_action_dim=35
actor_layer_norm=True
flow_steps=10
```

로드 예시:

```python
import json
from pathlib import Path

from il.policies import BCFlowPolicy

run_dir = Path("exp/pretrained/bcflow_square_actorln_seed0_1m")
config = json.loads((run_dir / "config.json").read_text())
metadata = json.loads((run_dir / "metadata.json").read_text())

learner = BCFlowPolicy.from_checkpoint(
    run_dir / "params_1000000.pkl",
    config=config,
    obs_dim=metadata["obs_dim"],
    action_dim=metadata["action_dim"],
    seed=metadata["seed"],
)
```

과거 검증 결과(critic contract 수정 전):

```text
Restored from .../bcflow_square_actorln_seed0_1m/params_1000000.pkl
action_shape (7,)
log_prob nan
full_chunk_shape (5, 7)
```

`log_prob=nan`은 flow policy가 명시적 likelihood를 제공하지 않기 때문에 정상이다.

## Chunk 동작

현재 v0에서는 chunk queue를 아직 구현하지 않았다. 따라서 chunked flow policy는 내부적으로
`(horizon_length, action_dim)` chunk를 만들지만, `PolicyOutput.action`으로는 첫 primitive action만
반환한다.

```text
full chunk: (5, 7)
executed action: (7,)
```

나중에 action chunk queue를 구현하면 `PolicyOutput.info["full_action_chunk"]`를 이용해 같은 chunk를
여러 step에 걸쳐 실행하도록 바꿀 수 있다.

## 현재 가능한 조합

RLPD expert + Flow BC learner는 둘 다 `PolicyOutput(action, log_prob, info)`를 반환한다.
실제 online train path에서는 `il.loops.rollout.choose_rollout_action()`이 learner/expert proposal을 샘플링하고 gate decision을 거쳐 executed action을 고른다.

직접 policy를 확인할 때는 각 policy를 같은 observation에서 호출하면 된다.

```python
learner_output = learner.sample_action(obs, rng=learner_rng)
expert_output = expert.sample_action(obs, rng=expert_rng)
```

## 주의점

- RLPD artifact는 actor LayerNorm 없는 기존 weight다.
- GPU6에서 새로 돌리는 `square-rlpd-actorln-bc03-2m-seed0`는 actor LayerNorm이 실제로 들어간 run이다. checkpoint가 나오면 같은 layout 방식으로 새 artifact를 만들어야 한다.
- artifact 생성에 쓴 일회성 스크립트는 repo에 남기지 않았다.
- `exp/pretrained/`는 git ignore 대상이므로 다른 머신에서는 별도 복사나 artifact sync가 필요하다.
