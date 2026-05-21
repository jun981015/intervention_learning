# Environment Setup

This setup assumes a fresh machine with no existing `qc` conda environment.
Installation must be reproducible from this repository alone.

## Prerequisites

- Conda, Mambaforge, or Miniforge
- Git
- For GPU training/evaluation: an NVIDIA driver

The default policy is to avoid depending on a locally installed CUDA toolkit
(`/usr/local/cuda`, `nvcc`). JAX's pip CUDA wheel should provide the CUDA/cuDNN
runtime. The machine only needs a compatible NVIDIA driver.

## Default Install

```bash
git clone <YOUR_REPO_URL> intervention_learning
cd intervention_learning

conda create -n il python=3.10 pip -y
conda activate il
python -m pip install --upgrade pip setuptools wheel
```

On GPU machines:

```bash
python -m pip install -r requirements-cuda12.txt
```

On CPU-only machines:

```bash
python -m pip install -r requirements-cpu.txt
```

Then install the local package:

```bash
python -m pip install -e . --no-build-isolation
```

The CUDA 12 JAX wheel requires NVIDIA driver `>=525`. If the driver is older,
update the driver or install in CPU mode.

`requirements-cuda12.txt` and `requirements-cpu.txt` include constraints files.
These constraints prevent transitive dependencies such as `orbax-checkpoint`
from upgrading JAX to a newer CUDA stack.

Do not use `jax[cuda12-local]` by default. Local CUDA/cuDNN/NCCL installations
vary across machines and are easy to break with `LD_LIBRARY_PATH`.

## Optional Helper Script

The helper script is only a convenience wrapper around the manual commands above.
Prefer documenting and debugging the explicit install commands first.

```bash
JAX_BACKEND=auto bash scripts/install.sh il
JAX_BACKEND=cuda12 bash scripts/install.sh il
JAX_BACKEND=cpu bash scripts/install.sh il
```

## Validation

```bash
conda run -n il python --version
conda run -n il python -c "import jax; print(jax.__version__); print(jax.devices())"
conda run -n il python -c "import flax, optax, distrax, robomimic, robosuite, mujoco; print('ok')"
conda run -n il python -m py_compile il/algo/rl/rlpd.py il/algo/bc/flow.py il/algo/bc/mlp.py
```

For GPU visibility:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n il python -c "import jax; print(jax.devices())"
```

Common runtime environment variables:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

If JAX picks the wrong CUDA libraries, check `LD_LIBRARY_PATH`. Unless local CUDA
is explicitly intended, do not include paths such as `/usr/local/cuda/lib64`.

If Robomimic/Robosuite rendering fails on Ubuntu, system OpenGL packages may be
missing:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglfw3 libglew2.2 libosmesa6 patchelf
```
