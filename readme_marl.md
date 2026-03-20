# SoDe_Avalon_3 MAPPO（两阶段多智能体自博弈）训练框架说明

## 1. 训练目标概述
Avalon 在好人（Good）与坏人（Evil）阵营上存在：
1) 完全不同的胜利条件  
2) 完全不同的信息不对称  

因此训练代码使用两阶段策略：
- **阶段 1（专职对抗 + 池构建）**：使用 `actor_good/critic_good` 与 `actor_evil/critic_evil` 各自建模，并通过 `SelfPlayPool` 分别维护 `good_pool/evil_pool`。
- **阶段 2（共享模型同时学习）**：创建一个共享 `shared_actor/shared_critic`，并将阶段 1 的池用于对手采样。阶段 2 中将两次 rollout 的数据合并为**同一个大 batch**，仅做**一次 PPO update**，并通过 `sample_weight` 控制好/坏对损失的贡献比例。

## 2. 稳定性关键实现细节
### 2.1 Advantage Normalization（非常重要）
在 PPO 的每个 mini-batch 内对 advantage 做标准化：
`adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)`

作用：避免某个 batch 的 advantage 数值过大导致策略更新过剧烈，破坏 PPO 稳定性。

### 2.2 GAE（稀疏终局奖励）
Avalon 终局奖励非常稀疏，因此阶段 2 使用 `GAE(lambda)` 来更平滑地分配信用（通过 critic 的价值预测做 bootstrap）。

为了缓解冷启动（critic 初期为噪声导致 advantage 方差很大），提供 warmup：
- `--gae_warmup_iters`：前 N 个 stage2 iter 使用 `--gae_warmup_lambda`（默认 1.0，退化为更接近 Monte-Carlo 的行为）
- `--critic_only_warmup_iters`：前 N 个 stage2 iter 只更新 Critic（降低 actor 冷启动风险）

## 3. Elo 对手池（good/evil pool）
`train_mappo/self_play.py` 中的 `SelfPlayPool` 维护每条策略的：
- `state_dict`
- `rating`（Elo）
- `name`

采样策略：
- `strategy=elo_topk`：按 Elo 从高到低取 top-k，然后在 top-k 内随机抽样。

对局后更新：
- 以“受控侧（controlled side）是否获胜”为 score 更新 controlled 与 opponent 两边的 Elo。

## 4. 多卡训练（torchrun + DDP）
使用方式：
- 启动脚本基于 `torchrun --standalone`
- 代码参数使用 `--ddp` 开关
- 每个 rank 会独立维护自己的 self-play 池（不做跨 rank 同步，降低复杂度）
- DDP 只负责同步反向传播的梯度，用于让共享模型收敛到一致的参数更新方向

## 5. 文件结构与关键入口
- 训练入口：`train_mappo/train_entry.py`
- PPO/MAPPO 更新与稳定性实现：`train_mappo/mappo/algo_adapter.py`
- Elo 池：`train_mappo/self_play.py`
- 启动脚本：`train_mappo/run_train_mappo_marl_ddp.sh`

## 6. 启动示例
### 6.1 环境变量示例
```bash
export LLM_MODEL_NAME="qwen/your-model"
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="http://your-llm-server/v1"

export QWEN_EMB_PATH="/path/to/qwen-emb"

# 可选：初始化 checkpoint（如果不提供将从随机初始化开始）
export INIT_ACTOR_GOOD_PATH="/path/actor_good.pt"
export INIT_CRITIC_GOOD_PATH="/path/critic_good.pt"

export NPROC_PER_NODE=4
export STAGE1_ITERS=200
export STAGE2_ITERS=200
```

### 6.2 执行
```bash
bash train_mappo/run_train_mappo_marl_ddp.sh
```

## 7. 参数说明（train_entry.py）
### 7.1 阶段控制
- `--stage1_iters`：阶段 1 迭代次数（负责构建 good/evil pool）
- `--stage2_iters`：阶段 2 迭代次数（合并 batch + 单次 update 的共享模型训练）
- `--episodes_per_iter`：每个 iter 内 rollout 的对局数量

### 7.2 PPO/优化
- `--gamma`：折扣因子
- `--clip_ratio`：PPO clip 比例
- `--value_coef`：value loss 系数
- `--entropy_coef`：熵正则系数
- `--ppo_epochs`：同一 batch 上重复更新次数
- `--minibatch_size` / `--batch_size`
- `--lr` / `--weight_decay` / `--grad_clip`

### 7.3 GAE 与 warmup
- `--gae_lambda`：阶段 2 的 GAE(lambda)
- `--gae_warmup_iters`：warmup 迭代数
- `--gae_warmup_lambda`：warmup 使用的 lambda（默认 1.0）
- `--critic_only_warmup_iters`：warmup 阶段只更新 critic

### 7.4 stage2 好/坏样本权重
- `--stage2_good_weight`
- `--stage2_evil_weight`

### 7.5 Elo 池与采样
- `--pool_max_size`：池最大容量
- `--pool_top_k`：Elo top-k 采样
- `--initial_elo`：初始 Elo
- `--elo_k_factor`：Elo 更新 K 系数

### 7.6 Qwen embedding 与文本编码
- `--qwen_emb_path`：Qwen embedding 模型路径（可选）
- `--use_text_cache`：默认 True（跳过 speech 文本编码）
- `--no_use_text_cache`：禁用 text cache（更可能真正用到 qwen emb）

### 7.7 LLM API（discussion 阶段需要调用）
- `--llm_model_name` / `--llm_api_key` / `--llm_base_url`
- `--llm_temperature` / `--llm_top_p` / `--llm_max_tokens`
- `--llm_reasoning`：允许 reasoning（若模型支持）

### 7.8 初始 checkpoint（可选）
- `--init_actor_good_path` / `--init_critic_good_path`
- `--init_actor_evil_path` / `--init_critic_evil_path`

### 7.9 DDP
- `--ddp`：torchrun 多卡开关

## 8. 训练流程摘要（建议阅读）
```mermaid
flowchart LR
  A[Stage1: good/evil 专职网络] --> B[维护 good_pool/evil_pool]
  B --> C[Stage2: shared网络]
  C --> D[rollout_good + rollout_evil]
  D --> E[合并 transitions + sample_weight]
  E --> F[AdvNorm + GAE(λ) + PPO 单次 update]
```

