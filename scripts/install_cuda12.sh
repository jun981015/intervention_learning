#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-il}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JAX_BACKEND=cuda12 bash "${ROOT_DIR}/scripts/install.sh" "${ENV_NAME}"
