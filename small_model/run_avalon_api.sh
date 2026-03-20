#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 只使用物理显卡 5（对 Python 来说是 cuda:0）
export CUDA_VISIBLE_DEVICES=2

# 如有虚拟环境，在此处激活，例如：
# source /path/to/venv/bin/activate

# 配置 Qwen Embedding 模型路径（按需修改为你的实际路径）
QWEN_EMB_PATH="/data2/AVALON/model/Qwen/Qwen3-Embedding-0___6B"
export QWEN_EMB_PATH

# 推理模型权重路径（可选覆盖）
# 默认与 `avalon_infer_api.py` 内一致：/data1/shy/SoDe_RL/output/avalon_soft_policy.pt
# 如需覆盖可直接修改此处，或在外部环境中 export AVALON_POLICY_WEIGHTS=...
AVALON_POLICY_WEIGHTS="${AVALON_POLICY_WEIGHTS:-/data2/AVALON/SoDe_Avalon_3/small_model/avalon_soft_policy_epoch380.pt}"
export AVALON_POLICY_WEIGHTS

exec uvicorn avalon_infer_api:app --host 0.0.0.0 --port 8001

