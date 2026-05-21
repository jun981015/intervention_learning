#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-il}"
BACKEND="${JAX_BACKEND:-auto}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

if [[ "${BACKEND}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="cuda12"
  else
    BACKEND="cpu"
  fi
fi

case "${BACKEND}" in
  cuda12)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "JAX_BACKEND=cuda12 requested, but nvidia-smi was not found." >&2
      exit 1
    fi
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
    if [[ "${DRIVER_MAJOR}" -lt 525 ]]; then
      echo "CUDA 12 JAX wheels need a CUDA 12 compatible NVIDIA driver." >&2
      echo "Detected driver: ${DRIVER_VERSION}. Update the driver or use JAX_BACKEND=cpu." >&2
      exit 1
    fi
    if [[ "${LD_LIBRARY_PATH:-}" == *cuda* ]]; then
      echo "Warning: LD_LIBRARY_PATH contains CUDA paths." >&2
      echo "JAX pip CUDA wheels are intended to use their bundled CUDA libraries." >&2
      echo "If import fails, unset LD_LIBRARY_PATH and retry." >&2
    fi
    REQUIREMENTS="requirements-cuda12.txt"
    ;;
  cpu)
    REQUIREMENTS="requirements-cpu.txt"
    ;;
  *)
    echo "Unknown JAX_BACKEND=${BACKEND}. Use auto, cuda12, or cpu." >&2
    exit 1
    ;;
esac

echo "Installing intervention_learning into conda env '${ENV_NAME}' with JAX_BACKEND=${BACKEND}"

conda create -n "${ENV_NAME}" python=3.10 pip -y
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS}"
conda run -n "${ENV_NAME}" python -m pip install -e . --no-build-isolation

conda run -n "${ENV_NAME}" python --version
conda run -n "${ENV_NAME}" python -c "import jax; print(jax.__version__); print(jax.devices())"
conda run -n "${ENV_NAME}" python -c "import flax, optax, distrax, robomimic, robosuite, mujoco; print('ok')"
conda run -n "${ENV_NAME}" python -m py_compile il/algo/rl/rlpd.py il/algo/bc/flow.py il/algo/bc/mlp.py
