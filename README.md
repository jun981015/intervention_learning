# Intervention Learning

Robomimic Square에서 human/expert intervention learning을 실험하기 위한 독립 프로젝트다.

이 repo는 `qc`와 섞지 않는다. `qc`, `qc_base`는 참조용이고, 여기에는 intervention
learning에 필요한 코드만 가져오거나 새로 정리한다.

## 문서 규칙

- `docs/`: 사람이 읽는 문서. 한국어로 작성한다.
- `docs_agents/`: Codex, Claude 같은 coding agent가 참조하는 문서. 영어로 작성한다.

## 초기 목표

- 환경: `square-mh-low_dim`
- learner: RLPD/SAC 계열 online learner
- expert baseline: 학습된 RLPD checkpoint를 불러온 policy
- BC actor 후보: diffusion/flow-matching policy
- gating baseline: random probability gate
- action chunking: v0에서는 제외하고 `horizon_length=1`로 둔다.
- 매 step learner action과 expert action을 둘 다 샘플링해서 저장한다.
- gate가 실제 실행할 controller를 learner/expert 중에서 고른다.

## 핵심 루프

```python
learner_action = learner.sample_action(obs)
expert_action = expert.sample_action(obs)

decision = gate.decide(
    step=step,
    observation=obs,
    learner=learner_action,
    expert=expert_action,
)

action = expert_action if decision.use_expert else learner_action
next_obs, reward, terminated, truncated, info = env.step(action)
```

## 참조 코드

- 현재 실험 코드: `/home/junhyeong/repos/qc`
- 원본 QC clean reference: `/home/junhyeong/repos/qc_base`

## 설치

`qc` conda 환경이 전혀 없는 새 PC를 기준으로 설치한다. 필요한 것은 conda와 git이다.
GPU 학습/eval에서는 PC에 깔린 CUDA toolkit보다 NVIDIA driver와 JAX wheel 조합이 더
중요하다. 기본 설치는 로컬 CUDA toolkit을 쓰지 않고 pip의 CUDA runtime wheel을 사용한다.

기본 설치:

```bash
git clone <YOUR_REPO_URL> intervention_learning
cd intervention_learning

conda create -n il python=3.10 pip -y
conda activate il
python -m pip install --upgrade pip setuptools wheel
```

GPU 머신에서는:

```bash
python -m pip install -r requirements-cuda12.txt
```

CPU-only 머신에서는:

```bash
python -m pip install -r requirements-cpu.txt
```

그 다음 local package를 설치한다.

```bash
python -m pip install -e . --no-build-isolation
```

CUDA 12 JAX wheel은 NVIDIA driver `>=525`가 필요하다. driver가 낮으면 driver를 업데이트하거나
CPU 모드로 설치한다.
로컬 CUDA toolkit을 직접 쓰는 설치는 기본 지원하지 않는다. PC마다 `/usr/local/cuda`가
다르게 잡혀 있으면 재현성이 떨어지기 때문이다.
`requirements-cuda12.txt`와 `requirements-cpu.txt`는 constraints를 포함한다. Flax/Orbax 같은
transitive dependency가 JAX를 더 높은 버전으로 올리지 못하게 하기 위해서다.

한 번에 설치하는 helper도 있다. README의 명령어를 그대로 자동화한 convenience script다.

```bash
JAX_BACKEND=auto bash scripts/install.sh il
```

설치 검증:

```bash
conda run -n il python --version
conda run -n il python -c "import jax; print(jax.__version__); print(jax.devices())"
conda run -n il python -c "import flax, optax, distrax, robomimic, robosuite, mujoco; print('ok')"
conda run -n il python -m py_compile il/algo/rl/rlpd.py il/algo/bc/flow.py il/algo/bc/mlp.py
```

Robomimic/Robosuite 렌더링 또는 online eval을 돌릴 때는 보통 아래 환경변수를 둔다.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

JAX가 잘못된 CUDA library를 잡으면 `LD_LIBRARY_PATH`가 원인일 수 있다. 로컬 CUDA를
일부러 쓰는 상황이 아니면 `/usr/local/cuda/lib64` 같은 경로를 `LD_LIBRARY_PATH`에 넣지 않는다.

## 현재 상태

아직 학습 entrypoint는 연결하지 않았다. 먼저 replay schema, gating interface,
learner/expert policy interface를 고정하고, 그 다음 RLPD/SAC와 flow-matching BC
adapter를 붙인다.

기존 학습 weight는 외부 checkpoint 경로로 참조한다. checkpoint, replay, video, log,
wandb 산출물은 git에 넣지 않는다.
