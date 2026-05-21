# 설치 가이드

이 프로젝트는 당장 Robomimic Square + RLPD/SAC online intervention pipeline을
목표로 한다. 설치 문서는 `qc` conda 환경이 전혀 없는 새 PC를 기준으로 작성한다.

## 전제 조건

- Conda 또는 Mambaforge/Miniforge
- Git
- GPU 학습/eval을 할 경우 NVIDIA driver

Python package 설치는 repo의 requirements 파일만 사용한다. 기본 방침은 로컬 CUDA toolkit
(`/usr/local/cuda`, `nvcc`)에 의존하지 않는 것이다. JAX의 pip CUDA wheel이 CUDA/cuDNN runtime을
가져오고, PC에는 그 wheel을 실행할 수 있는 NVIDIA driver만 있으면 된다.

## 기본 설치

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
CPU mode로 설치한다.

`requirements-cuda12.txt`와 `requirements-cpu.txt`는 내부에서 constraints 파일을 사용한다.
이 constraints는 Flax가 가져오는 최신 `orbax-checkpoint` 같은 transitive dependency가 JAX를
더 높은 버전으로 올리지 못하게 막는다. 이 고정이 없으면 driver 550 환경에서도 JAX/CUDA 13
조합이 설치될 수 있다.

로컬 CUDA toolkit을 직접 사용하는 `jax[cuda12-local]`류 설치는 기본 지원하지 않는다. PC마다
toolkit/cuDNN/NCCL 조합이 달라서 재현성이 떨어지고, `LD_LIBRARY_PATH` 충돌이 잦기 때문이다.
정말 필요한 경우에는 해당 PC에서 별도 install override로 처리한다.

## Optional helper script

아래 스크립트는 위 설치 절차를 자동화한 convenience helper다. 일반적인 repo 설치 흐름은
위 수동 명령어를 기준으로 한다.

```bash
JAX_BACKEND=auto bash scripts/install.sh il
JAX_BACKEND=cuda12 bash scripts/install.sh il
JAX_BACKEND=cpu bash scripts/install.sh il
```

## 검증

```bash
conda run -n il python --version
conda run -n il python -c "import jax; print(jax.__version__); print(jax.devices())"
conda run -n il python -c "import flax, optax, distrax, robomimic, robosuite, mujoco; print('ok')"
conda run -n il python -m py_compile il/algo/rl/rlpd.py il/algo/bc/flow.py il/algo/bc/mlp.py
```

GPU가 보여야 하는 검증:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n il python -c "import jax; print(jax.devices())"
```

GPU 설치인데 여기서 CPU만 보이면 JAX CUDA wheel 또는 NVIDIA driver 조합이 맞지 않는 것이다.
먼저 `nvidia-smi`의 driver version을 확인한다.

## Robomimic/Robosuite 실행 환경변수

headless GPU rendering이나 online eval을 돌릴 때는 보통 아래 환경변수를 둔다.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

JAX가 잘못된 CUDA library를 잡으면 `LD_LIBRARY_PATH`가 원인일 수 있다. 로컬 CUDA를 일부러
쓰는 상황이 아니면 `/usr/local/cuda/lib64` 같은 경로를 `LD_LIBRARY_PATH`에 넣지 않는다.

`robosuite` macro 경고가 나오면 fatal error가 아닌 경우가 많다. 다만 렌더링이나 camera
설정에서 문제가 생기면 해당 conda 환경 안에서 robosuite macro setup을 다시 확인한다.

Ubuntu에서 OpenGL 관련 import/render 에러가 나면 system package가 부족할 수 있다.
관리자 권한이 있으면 아래를 먼저 확인한다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglfw3 libglew2.2 libosmesa6 patchelf
```
