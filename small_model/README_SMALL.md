# SoDe_Avalon_3 Small Model（Avalon Soft Policy）部署与调用

本目录提供“小模型（软策略）推理服务”的**服务端**与**客户端示例**，用于给 `SoDe_Avalon_3` 的 Agent 决策提供一个额外的 `suggestion`（先验动作/偏好）。

## 1. 目录内容

- `avalon_infer_api.py`：FastAPI 推理服务（`POST /predict_from_log`）
- `run_avalon_api.sh`：启动脚本（设置 GPU / 权重 / embedding）
- `test_api.py`：客户端脚本（读取 JSON 文件 → 请求 API → 解码 logits → 输出 action）

## 2. 服务端部署（启动推理 API）

### 2.1 依赖（典型）

服务端主要依赖：
- `torch`
- `fastapi`
- `uvicorn`
- `pydantic`
- （可选）`transformers`：当启用 SPEECH 文本编码时会用到

### 2.2 关键环境变量

- **`AVALON_POLICY_WEIGHTS`**：模型权重路径  
  默认：`/data1/shy/SoDe_RL/output/avalon_soft_policy.pt`
- **`QWEN_EMB_PATH`**：embedding 模型路径（如果启用 transformers 文本编码）
- **`CUDA_VISIBLE_DEVICES`**：选择 GPU

### 2.3 一键启动

在本目录下执行：

```bash
bash run_avalon_api.sh
```

默认监听：`0.0.0.0:8000`  
对外接口：`http://127.0.0.1:8000/predict_from_log`

## 3. API 接口说明

### 3.1 Endpoint

- **`POST /predict_from_log`**

### 3.2 Request JSON

```json
{
  "log": { "...整份对局日志dict..." },
  "phase": "discussion",
  "player": 5,
  "round_number": 4,
  "attempt_index": 1
}
```

字段含义：
- **`log`**：完整对局日志 dict（顶层必须是 object/dict），通常包含 `players_config`、`game_timeline` 等
- **`phase`**：决策阶段（推荐字符串）：
  - `"discussion"`：发言（对应大模型侧 `speech`）
  - `"proposal"`：组队
  - `"voting"`：投票
  - `"mission"`：任务出牌（对应大模型侧 `execution`）
  - `"assassination"`：刺杀
- **`player`**：actor id（1..7）
- **`round_number`**：轮次（通常 1..5）
- **`attempt_index`**：该轮内尝试（通常 1..5）

### 3.3 Response JSON（核心字段）

服务端返回包含 logits 的结果（用于 debug 的 token 细节也会返回），核心字段：
- `phase_id` / `phase_name`
- `round_number` / `attempt_index`
- `actor_id` / `role` / `model_name`
- `logits`

客户端（`test_api.py` 或 `SoDe_Avalon_3/Tool/call_small_model.py`）会把 `logits` 解码为 **action**：
- discussion：`identity_label_text / intent_semantic / attitude_semantic`
- proposal：`propose_targets`（长度 7 的 0/1 mask）
- voting：`vote_label_text`（Approve/Reject）
- mission：`mission_label_text`（Success/Fail）
- assassination：`kill_target`

## 4. 客户端调用（脚本验证）

在服务端已启动后，执行：

```bash
python test_api.py \
  --api_url http://127.0.0.1:8000/predict_from_log \
  --json_path /path/to/*_game_condition.json \
  --phase discussion \
  --player 5 \
  --round_number 4 \
  --attempt_index 1
```

## 5. 在 SoDe_Avalon_3 中的集成方式

### 5.1 Tool 封装

`SoDe_Avalon_3/Tool/call_small_model.py` 会把 `test_api.py` 的客户端逻辑封装为**直接接收 dict** 的调用（不再读 JSON 文件）：
- 输入：`game_condition: Dict[str, Any]` + selector（phase/player/round/attempt）+ `api_url`
- 输出：结构化结果（包含 `output` 即 action）

### 5.2 Agent 注入 suggestion

`SoDe_Avalon_3/Agents` 下新增的 Agent 会在每次决策时：
- 根据当前 `phase` 映射到小模型 phase（`speech->discussion`, `execution->mission`）
- 发送 `game_condition` + selector 请求
- 将返回的 action 注入到 prompt，例如：

```text
suggestion:
{...小模型返回的 action...}
```

