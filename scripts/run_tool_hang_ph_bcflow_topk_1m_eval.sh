#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/junhyeong/miniconda3/envs/il/bin/python}"
DATASET="${DATASET:-/home/junhyeong/.robomimic/tool_hang/ph/low_dim_v141.hdf5}"
STEPS="${STEPS:-1000000}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
TOP_KS="${TOP_KS:-10 15 20 30 50 100 150 200}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1

mkdir -p logs logs/tool_hang_ph_topk_1m_eval exp/pretrained

for K in ${TOP_KS}; do
  RUN_NAME="bcflow_tool_hang_ph_top${K}_actorln_seed${SEED}_1m"
  RUN_DIR="exp/pretrained/${RUN_NAME}"
  TRAIN_LOG="logs/${RUN_NAME}_gpu${GPU_ID}.log"
  EVAL_JSON="logs/tool_hang_ph_topk_1m_eval/${RUN_NAME}_${EPISODES}ep.json"
  LEGACY_EVAL_JSON="logs/eval_${RUN_NAME}_${EPISODES}ep_seed${SEED}.json"
  EVAL_LOG="logs/tool_hang_ph_topk_1m_eval/${RUN_NAME}_${EPISODES}ep_gpu${GPU_ID}.log"

  echo "[pipeline] dataset=${DATASET} top_k=${K} run=${RUN_NAME} gpu=${GPU_ID}"

  TRAIN_ARGS=(
    scripts/train_bcflow_topk_robomimic.py
    --dataset "${DATASET}"
    --top-k "${K}"
    --steps "${STEPS}"
    --seed "${SEED}"
    --run-name "${RUN_NAME}"
    --wandb-name "${RUN_NAME}"
    --run-group "tool-hang-ph-bcflow-topk-pretrain"
    --wandb-tags "tool_hang,ph,bcflow,pretrain,topk,top${K},1m"
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

  if [[ -f "${EVAL_JSON}" ]]; then
    echo "[pipeline] eval skip existing ${EVAL_JSON}"
    continue
  fi
  if [[ -f "${LEGACY_EVAL_JSON}" ]]; then
    echo "[pipeline] eval skip existing legacy ${LEGACY_EVAL_JSON}"
    continue
  fi

  echo "[pipeline] eval ${RUN_NAME}"
  "${PYTHON_BIN}" scripts/eval_bcflow_policy.py \
    --env-name tool_hang-ph-low_dim \
    --policy-dir "${RUN_DIR}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --output-json "${EVAL_JSON}" 2>&1 | tee "${EVAL_LOG}"
done

echo "[pipeline] done"
