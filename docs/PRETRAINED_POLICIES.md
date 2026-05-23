# Pretrained Policy Artifacts

이 문서는 외부 실험에서 가져온 pretrained policy weight를 이 repo에서 어떻게 쓰는지 기록한다.

## 원칙

- `intervention_learning` runtime 코드에는 source repo 전용 loader를 넣지 않는다.
- weight 파일은 이 repo의 agent state-dict layout과 config에 맞아 있어야 한다.
- git에는 checkpoint, replay, video, log, wandb 산출물을 넣지 않는다.
- artifact는 현재 `exp/pretrained/` 아래에 두며 `.gitignore` 대상이다.

## 현재 Artifact

### RLPD Expert

경로:

```text
exp/pretrained/rlpd_square_bc03_seed0_2m/
  params_2000000.pkl
  config.json
  metadata.json
```

용도:

- expert policy 후보
- square `bc_alpha=0.3` RLPD 2M checkpoint

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

검증 결과:

```text
Restored from .../rlpd_square_bc03_seed0_2m/params_2000000.pkl
action_shape (7,)
log_prob 16.3937
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

검증 결과:

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
