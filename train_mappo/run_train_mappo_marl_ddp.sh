#!/bin/bash
set -euo pipefail

# ================= 配置区域 (Configuration) =================
WORK_DIR="/data2/AVALON/SoDe_Avalon_3"
CONDA_ENV="sc2_place"

TRAIN_PY="${WORK_DIR}/train_mappo/train_entry.py"

# ---- 多卡训练 (torchrun) ----
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- LLM API (discussion phase 需要) ----
# 推荐使用环境变量传入：LLM_MODEL_NAME / LLM_API_KEY / LLM_BASE_URL
: "${LLM_MODEL_NAME:?Need env LLM_MODEL_NAME}"
: "${LLM_API_KEY:?Need env LLM_API_KEY}"
: "${LLM_BASE_URL:?Need env LLM_BASE_URL}"

LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_TOP_P="${LLM_TOP_P:-1.0}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"
LLM_REASONING="${LLM_REASONING:-false}"  # true/false

# ---- Qwen embedding ----
# QWEN_EMB_PATH: embedding 模型路径（可选）
QWEN_EMB_PATH="${QWEN_EMB_PATH:-}"
USE_TEXT_CACHE="${USE_TEXT_CACHE:-true}" # true 表示跳过 speech 文本编码（默认行为）

# ---- 初始策略/价值网络 checkpoint (可选) ----
INIT_ACTOR_GOOD_PATH="${INIT_ACTOR_GOOD_PATH:-}"
INIT_CRITIC_GOOD_PATH="${INIT_CRITIC_GOOD_PATH:-}"
INIT_ACTOR_EVIL_PATH="${INIT_ACTOR_EVIL_PATH:-}"
INIT_CRITIC_EVIL_PATH="${INIT_CRITIC_EVIL_PATH:-}"

# ---- 两阶段训练参数 ----
STAGE1_ITERS="${STAGE1_ITERS:-200}"
STAGE2_ITERS="${STAGE2_ITERS:-200}"
EPISODES_PER_ITER="${EPISODES_PER_ITER:-2}"

# ---- PPO/MAPPO 核心超参 ----
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.99}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
VALUE_COEF="${VALUE_COEF:-0.5}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ---- Advantage / GAE warmup（稀疏奖励场景）----
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
GAE_WARMUP_ITERS="${GAE_WARMUP_ITERS:-0}"
GAE_WARMUP_LAMBDA="${GAE_WARMUP_LAMBDA:-1.0}"
CRITIC_ONLY_WARMUP_ITERS="${CRITIC_ONLY_WARMUP_ITERS:-0}"

# ---- Elo Pool 采样（stage1 & stage2 对手）----
POOL_MAX_SIZE="${POOL_MAX_SIZE:-5}"
POOL_TOP_K="${POOL_TOP_K:-3}"
INITIAL_ELO="${INITIAL_ELO:-1000}"
ELO_K_FACTOR="${ELO_K_FACTOR:-32}"

# ---- stage2 good/evil 样本权重 ----
STAGE2_GOOD_WEIGHT="${STAGE2_GOOD_WEIGHT:-1.0}"
STAGE2_EVIL_WEIGHT="${STAGE2_EVIL_WEIGHT:-1.0}"

# ================= 构造命令参数 =================
ARGS=(
  --ddp
  --device "cuda"
  --seed "$SEED"
  --iters "$STAGE1_ITERS"
  --episodes_per_iter "$EPISODES_PER_ITER"
  --stage1_iters "$STAGE1_ITERS"
  --stage2_iters "$STAGE2_ITERS"
  --gamma "$GAMMA"
  --clip_ratio "$CLIP_RATIO"
  --value_coef "$VALUE_COEF"
  --entropy_coef "$ENTROPY_COEF"
  --ppo_epochs "$PPO_EPOCHS"
  --minibatch_size "$MINIBATCH_SIZE"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --gae_lambda "$GAE_LAMBDA"
  --gae_warmup_iters "$GAE_WARMUP_ITERS"
  --gae_warmup_lambda "$GAE_WARMUP_LAMBDA"
  --critic_only_warmup_iters "$CRITIC_ONLY_WARMUP_ITERS"
  --stage2_good_weight "$STAGE2_GOOD_WEIGHT"
  --stage2_evil_weight "$STAGE2_EVIL_WEIGHT"
  --pool_max_size "$POOL_MAX_SIZE"
  --pool_top_k "$POOL_TOP_K"
  --initial_elo "$INITIAL_ELO"
  --elo_k_factor "$ELO_K_FACTOR"
  --llm_model_name "$LLM_MODEL_NAME"
  --llm_api_key "$LLM_API_KEY"
  --llm_base_url "$LLM_BASE_URL"
  --llm_temperature "$LLM_TEMPERATURE"
  --llm_top_p "$LLM_TOP_P"
  --llm_max_tokens "$LLM_MAX_TOKENS"
)

if [[ "$LLM_REASONING" == "true" ]]; then
  ARGS+=(--llm_reasoning)
fi

if [[ -n "$QWEN_EMB_PATH" ]]; then
  ARGS+=(--qwen_emb_path "$QWEN_EMB_PATH")
fi

if [[ "$USE_TEXT_CACHE" != "true" ]]; then
  # use_text_cache=True 时会跳过 speech 的文本编码；若你想启用 qwen emb，请改成 false
  ARGS+=(--no_use_text_cache)
fi

if [[ -n "$INIT_ACTOR_GOOD_PATH" ]]; then
  ARGS+=(--init_actor_good_path "$INIT_ACTOR_GOOD_PATH")
fi
if [[ -n "$INIT_CRITIC_GOOD_PATH" ]]; then
  ARGS+=(--init_critic_good_path "$INIT_CRITIC_GOOD_PATH")
fi
if [[ -n "$INIT_ACTOR_EVIL_PATH" ]]; then
  ARGS+=(--init_actor_evil_path "$INIT_ACTOR_EVIL_PATH")
fi
if [[ -n "$INIT_CRITIC_EVIL_PATH" ]]; then
  ARGS+=(--init_critic_evil_path "$INIT_CRITIC_EVIL_PATH")
fi

# ================= 启动 =================
if ! command -v tmux &> /dev/null; then
  echo "Warning: tmux not found; running directly."
fi

if command -v conda &> /dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "$CONDA_ENV" || true
fi

echo "[DDP] NPROC_PER_NODE=$NPROC_PER_NODE MASTER_PORT=$MASTER_PORT"
echo "[DDP] stage1=$STAGE1_ITERS stage2=$STAGE2_ITERS"
echo "[DDP] qwen_emb_path=$QWEN_EMB_PATH use_text_cache=$USE_TEXT_CACHE"

torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  "$TRAIN_PY" \
  "${ARGS[@]}"

