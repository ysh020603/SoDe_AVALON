#!/bin/bash

# ================= 配置区域 (Configuration) =================

WORK_DIR="/data2/AVALON/SoDe_Avalon_3"
CONDA_ENV="sc2_place"
PYTHON_SCRIPT="run_simulation_avalon.py"

# --- 实验通用参数 ---
TOTAL_ROUNDS=200    # 总运行轮数
PARALLEL_WORKERS=2   # 并行窗口数量
ROUNDS_PER_WORKER=$((TOTAL_ROUNDS / PARALLEL_WORKERS))

# --- 游戏参数 ---
PLAYER_NUM=7       # 游戏人数 (5-10)

# --- 模型 A 配置 (Model A) ---
# MODEL_A_NAME="Qwen3-32B"
# MODEL_A_API_KEY="EMPTY"
# MODEL_A_BASE_URL="http://172.18.30.165:8815/v1"

MODEL_A_NAME="Qwen3-235B-A22B"
MODEL_A_API_KEY="EMPTY"
MODEL_A_BASE_URL="http://172.18.30.177:2325/v1"

# 模型 A 推理参数 (填 "None" 代表使用默认值)
MODEL_A_TEMP="0.7"
MODEL_A_TOP_P="0.8"
MODEL_A_MAX_TOKENS="8192"
MODEL_A_USE_REASONING="False" # True: 使用 Reasoning; False: 不使用
 

# MODEL_B_NAME="Qwen3-235B-A22B"
# MODEL_B_API_KEY="EMPTY"
# MODEL_B_BASE_URL="http://172.18.30.177:2325/v1"

MODEL_B_NAME="glm-4.7"
MODEL_B_API_KEY="df2af4f04d184c8cae1e0c70bbed26e0.5yVd73IGL743VEsF"
MODEL_B_BASE_URL="https://open.bigmodel.cn/api/paas/v4"


# 模型 B 推理参数 (填 "None" 代表使用默认值)
MODEL_B_TEMP="0.7"
MODEL_B_TOP_P="0.8"
MODEL_B_MAX_TOKENS="8192"
MODEL_B_USE_REASONING="False" # True: 使用 Reasoning; False: 不使用

# 定义日志后缀（这里仅将模型名中的 . 替换为 _，只影响日志与 session 名字）
LOG_TAG="${MODEL_A_NAME//./_}_VS_${MODEL_B_NAME//./_}_${PLAYER_NUM}Players"

SESSION_NAME="Avalon_Exp_Parallel_${LOG_TAG}"

# ==========================================================

# 检查 tmux
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed."
    exit 1
fi

# 重建 Session
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION_NAME already exists. Killing it..."
    tmux kill-session -t $SESSION_NAME
fi

echo "Creating session $SESSION_NAME..."
tmux new-session -d -s $SESSION_NAME -n "Worker_0"

# 定义初始化函数
init_and_run() {
    local target=$1
    local worker_id=$2
    
    # 1. 进入目录
    tmux send-keys -t $target "cd $WORK_DIR" C-m
    
    # 2. 激活环境
    tmux send-keys -t $target "source activate $CONDA_ENV || conda activate $CONDA_ENV" C-m
    
    # 3. 设置 Proxy (如果需要)
    # tmux send-keys -t $target "export NO_PROXY=localhost,127.0.0.1,10.119.141.215" C-m
    
    # 4. 构建 Python 命令
    cmd="python $PYTHON_SCRIPT \
        --rounds $ROUNDS_PER_WORKER \
        --worker_id $worker_id \
        --log_tag \"$LOG_TAG\" \
        --player_num $PLAYER_NUM \
        --model_a_name '$MODEL_A_NAME' \
        --model_a_key '$MODEL_A_API_KEY' \
        --model_a_url '$MODEL_A_BASE_URL' \
        --model_a_temp '$MODEL_A_TEMP' \
        --model_a_top_p '$MODEL_A_TOP_P' \
        --model_a_max_tokens '$MODEL_A_MAX_TOKENS' \
        --model_a_reasoning '$MODEL_A_USE_REASONING' \
        --model_b_name '$MODEL_B_NAME' \
        --model_b_key '$MODEL_B_API_KEY' \
        --model_b_url '$MODEL_B_BASE_URL' \
        --model_b_temp '$MODEL_B_TEMP' \
        --model_b_top_p '$MODEL_B_TOP_P' \
        --model_b_max_tokens '$MODEL_B_MAX_TOKENS' \
        --model_b_reasoning '$MODEL_B_USE_REASONING'"

    echo "Starting Worker $worker_id in $target (Rounds: $ROUNDS_PER_WORKER)..."
    tmux send-keys -t $target "$cmd" C-m
}

# 启动 Worker 0 (Session 创建时自带的窗口)
init_and_run "$SESSION_NAME:0" "0"

# 启动其余 Worker (创建新窗口)
for ((i=1; i<PARALLEL_WORKERS; i++)); do
    tmux new-window -t $SESSION_NAME -n "Worker_$i"
    init_and_run "$SESSION_NAME:$i" "$i"
done

echo "=================================================="
echo "Experiment started: Avalon ($TOTAL_ROUNDS rounds total)"
echo "Players: $PLAYER_NUM"
echo "Split into $PARALLEL_WORKERS windows ($ROUNDS_PER_WORKER rounds each)."
echo "Model A: $MODEL_A_NAME (Reasoning: $MODEL_A_USE_REASONING)"
echo "Model B: $MODEL_B_NAME (Reasoning: $MODEL_B_USE_REASONING)"
echo "Log Tag: $LOG_TAG"
echo "Check output: tmux attach -t $SESSION_NAME"
echo "=================================================="