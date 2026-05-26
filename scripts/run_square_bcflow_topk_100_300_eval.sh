#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-6}"
PYTHON_BIN="${PYTHON_BIN:-/home/junhyeong/miniconda3/envs/il/bin/python}"
STEPS="${STEPS:-1000000}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
TOP_KS="${TOP_KS:-100 150 200 250 300}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1

mkdir -p logs logs/topk_1m_eval exp/pretrained

for K in ${TOP_KS}; do
  RUN_NAME="bcflow_square_top${K}_actorln_seed${SEED}_1m"
  RUN_DIR="exp/pretrained/${RUN_NAME}"
  TRAIN_LOG="logs/${RUN_NAME}_gpu${GPU_ID}.log"
  EVAL_JSON="logs/topk_1m_eval/${RUN_NAME}_${EPISODES}ep.json"
  EVAL_LOG="logs/topk_1m_eval/${RUN_NAME}_${EPISODES}ep_gpu${GPU_ID}.log"

  echo "[pipeline] top_k=${K} run=${RUN_NAME} gpu=${GPU_ID}"

  TRAIN_ARGS=(
    scripts/train_bcflow_topk_robomimic.py
    --top-k "${K}"
    --steps "${STEPS}"
    --seed "${SEED}"
    --run-name "${RUN_NAME}"
    --wandb-name "${RUN_NAME}"
    --run-group "square-bcflow-topk-pretrain"
    --wandb-tags "square,bcflow,pretrain,topk,top${K},1m"
    --wandb
  )

  if [[ -f "${RUN_DIR}/params_${STEPS}.pkl" ]]; then
    echo "[pipeline] train skip existing ${RUN_DIR}/params_${STEPS}.pkl"
  else
    if compgen -G "${RUN_DIR}/params_*.pkl" > /dev/null; then
      LATEST_STEP="$(find "${RUN_DIR}" -maxdepth 1 -name 'params_*.pkl' -printf '%f\n' | sed -E 's/params_([0-9]+)\.pkl/\1/' | sort -n | tail -n 1)"
      echo "[pipeline] resume ${RUN_NAME} from ${LATEST_STEP}"
      TRAIN_ARGS+=(--resume-dir "${RUN_DIR}" --resume-step "${LATEST_STEP}")
    fi
    "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"
  fi

  echo "[pipeline] eval ${RUN_NAME}"
  "${PYTHON_BIN}" scripts/eval_bcflow_policy.py \
    --env-name square-mh-low_dim \
    --policy-dir "${RUN_DIR}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --output-json "${EVAL_JSON}" 2>&1 | tee "${EVAL_LOG}"
done

echo "[pipeline] done"
