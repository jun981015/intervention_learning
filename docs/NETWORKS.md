# Network 기본값과 옵션

## 공용 MLP

공용 MLP는 [il/networks/mlp.py](../il/networks/mlp.py)에 있다. RLPD, BC MLP, BC Flow가 모두
이 클래스를 사용한다.

기본값:

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `activations` | `nn.relu` | hidden layer activation |
| `activate_final` | `False` | 마지막 layer에도 activation을 적용할지 여부 |
| `kernel_init` | `None` | 모든 Dense layer에 직접 넣을 initializer |
| `use_layer_norm` | `False` | LayerNorm 사용 여부 |
| `layer_norm_after_activation` | `False` | LayerNorm을 activation 뒤에 둘지 여부 |
| `scale_final` | `None` | 마지막 layer kernel initializer scale override |
| `dropout_rate` | `None` | hidden layer dropout |
| `use_pnorm` | `False` | 최종 feature를 L2 norm으로 나눠 unit-norm에 가깝게 만드는 옵션 |
| `sow_intermediate_feature` | `False` | 마지막 출력 직전 hidden feature를 `intermediates/feature`에 저장 |

`scale_final`은 출력값을 직접 clipping하는 옵션이 아니다. 마지막 Dense layer의 weight
초기화 scale을 조절하고, 그 결과 초기 output scale이 작아지는 경향을 만든다.

`use_pnorm`은 loss penalty가 아니다. forward pass에서 feature 자체를 `x / ||x||_2`로
바꾸는 normalization/projection에 가깝다.

`sow_intermediate_feature`는 forward 값에는 영향을 주지 않는다. Flax `mutable=["intermediates"]`
로 호출할 때 representation 분석용 feature를 꺼내기 위한 hook이다.

## 알고리즘별 MLP 사용 방식

| 사용처 | 파일 | 주요 옵션 |
| --- | --- | --- |
| RLPD critic | `il/algo/rl/rlpd.py` | `activations=relu`, `activate_final=True`, `use_layer_norm=config["layer_norm"]` |
| RLPD actor | `il/algo/rl/rlpd.py` | `activations=relu`, `activate_final=True`, `use_layer_norm=config["actor_layer_norm"]` |
| BC MLP actor | `il/algo/bc/mlp.py` | `activations=relu`, `activate_final=True`, `use_layer_norm=config["actor_layer_norm"]` |
| BC Flow vector field | `il/networks/flow.py` | `activations=gelu`, `kernel_init=variance_scaling`, `use_layer_norm=config["actor_layer_norm"]`, `layer_norm_after_activation=True`, `sow_intermediate_feature=True` |

## LayerNorm 위치

공용 기본값은 `LayerNorm -> activation` 순서다.

BC Flow는 qc_base flow network 동작을 보존하기 위해 `activation -> LayerNorm` 순서를 쓴다.
그래서 `layer_norm_after_activation=True`를 켠다.

## 왜 Flow 전용 MLP를 없앴는가

qc_base에는 flow network 파일 내부에 별도 MLP가 있었다. 기능적으로는 공용 MLP와 크게
다르지 않았고, 차이는 옵션으로 표현 가능했다. 그래서 이 repo에서는 공용 MLP 하나로 통합했다.

보존한 flow 쪽 차이:

- `gelu` activation
- `variance_scaling(scale, fan_avg, uniform)` initializer
- activation 뒤 LayerNorm
- `intermediates/feature` 저장 hook

## MLPResNet 상태

`MLPResNetV2`는 [il/networks/mlp_resnet.py](../il/networks/mlp_resnet.py)에 남아 있지만 현재
agent에서는 사용하지 않는다. qc/qc_base에도 실험용으로 구현만 남아있는 형태에 가깝다.
나중에 residual MLP ablation을 할 때만 연결한다.

## Image Observation TODO

현재 policy/network 구현은 low-dimensional state 입력만 학습 대상으로 지원한다. Env와 replay
buffer는 `pixels_state`처럼 image + state dict observation을 받을 수 있게 되어 있지만,
`BCFlowAgent`, `BCMLPAgent`, `ACRLPDAgent`는 아직 image encoder를 붙이지 않았다.

따라서 image observation recipe로 현재 low-dim agent를 만들면 builder에서 명시적으로
`NotImplementedError`를 낸다. 다음 단계에서 필요한 작업은 다음과 같다.

- `EnvSpec.pixel_keys`를 읽는 CNN encoder builder 추가
- camera별 shared encoder 또는 separate encoder 선택 옵션 추가
- image feature와 state feature를 concat한 뒤 actor/critic head에 전달
- replay batch의 nested observation dict를 JAX PyTree로 agent update에 넘기는 경로 검증

지금 image render/multi-camera 지원은 데이터 수집과 buffer 구조 준비 단계이며, policy 학습
지원은 의도적으로 TODO로 남겨둔다.
