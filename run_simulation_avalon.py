import os
import sys
import random
import datetime
import json
import traceback
import argparse
import time

# ================= 参数解析 (Argument Parsing) =================

parser = argparse.ArgumentParser(description="Run Avalon Simulation (Parallel Worker)")

# 基础运行参数
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds for this worker to run")
parser.add_argument("--worker_id", type=str, default="0", help="ID of this tmux worker (for logging)")
parser.add_argument("--log_tag", type=str, default="", help="Tag for log folder suffix")

# 游戏参数
parser.add_argument("--player_num", type=int, default=5, choices=[5, 6, 7, 8, 9, 10], help="Number of players")

# 模型 A 参数
parser.add_argument("--model_a_name", type=str, required=True)
parser.add_argument("--model_a_key", type=str, required=True)
parser.add_argument("--model_a_url", type=str, required=True)
parser.add_argument("--model_a_temp", type=str, default="None")
parser.add_argument("--model_a_top_p", type=str, default="None")
parser.add_argument("--model_a_max_tokens", type=str, default="None")
parser.add_argument("--model_a_reasoning", type=str, default="False", help="True/False string")

# 模型 B 参数
parser.add_argument("--model_b_name", type=str, required=True)
parser.add_argument("--model_b_key", type=str, required=True)
parser.add_argument("--model_b_url", type=str, required=True)
parser.add_argument("--model_b_temp", type=str, default="None")
parser.add_argument("--model_b_top_p", type=str, default="None")
parser.add_argument("--model_b_max_tokens", type=str, default="None")
parser.add_argument("--model_b_reasoning", type=str, default="False", help="True/False string")

args = parser.parse_args()

# ================= 全局配置 (Global Config) =================

GAME_TYPE = "Avalon"
TOTAL_ROUNDS = args.rounds
WORKER_ID = args.worker_id
PLAYER_NUM = args.player_num

# --- 辅助函数：构建推理配置 ---
def build_inference_config(model_name, temp, top_p, max_tokens, use_reasoning_str):
    config = {
        "model": model_name
    }
    
    # 处理数值参数 (支持 "None")
    if temp.lower() != "none":
        config["temperature"] = float(temp)
    if top_p.lower() != "none":
        config["top_p"] = float(top_p)
    if max_tokens.lower() != "none":
        config["max_tokens"] = int(max_tokens)
        
    # 处理 Reasoning (思考) 模板
    # 如果 use_reasoning 为 True，代表使用思考，不需要添加 extra_body
    # 如果 use_reasoning 为 False，代表不使用思考，需要添加 extra_body 并设 enable_thinking 为 False
    is_reasoning = use_reasoning_str.lower() == "true"
    
    if not is_reasoning:
        if "glm" in model_name:
            config["extra_body"] ={
            "thinking": {
                "type": "disabled",
            },}
        else:
            config["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        
    return config

# 构造模型配置字典
CONFIG_MODEL_A = {
    "name": args.model_a_name,
    "api_url_config": {
        "api_key": args.model_a_key,
        "base_url": args.model_a_url
    },
    "inference_config": build_inference_config(
        args.model_a_name, 
        args.model_a_temp, 
        args.model_a_top_p, 
        args.model_a_max_tokens, 
        args.model_a_reasoning
    )
}

CONFIG_MODEL_B = {
    "name": args.model_b_name,
    "api_url_config": {
        "api_key": args.model_b_key,
        "base_url": args.model_b_url
    },
    "inference_config": build_inference_config(
        args.model_b_name, 
        args.model_b_temp, 
        args.model_b_top_p, 
        args.model_b_max_tokens, 
        args.model_b_reasoning
    )
}

# 构造日志目录名称
LOG_ROOT = "ex_log"
if args.log_tag:
    LOG_DIR_NAME = args.log_tag
else:
    LOG_DIR_NAME = f"{args.model_a_name}_VS_{args.model_b_name}"

FULL_LOG_PATH = os.path.join(LOG_ROOT, LOG_DIR_NAME)

print(f"[{datetime.datetime.now()}] Worker {WORKER_ID} initialized.")
print(f"Game: Avalon ({PLAYER_NUM} players) | Rounds: {TOTAL_ROUNDS}")
print(f"Matchup: {CONFIG_MODEL_A['name']} vs {CONFIG_MODEL_B['name']}")
print(f"Log Directory: {FULL_LOG_PATH}")

# ================= 导入模块 (Imports) =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from Tool.callopenai import api_call_format
    from Agents.Agent import Agent
    from Game.Avalon_multiturn import Game_Avalon_Multiturn
except ImportError as e:
    print("Error importing game modules. Please run this script from the root of the project structure.")
    print(f"Details: {e}")
    sys.exit(1)

# ================= 游戏配置 (Game Definitions) =================
# 根据人数生成角色配置
CONFIG_MAP = {
    5: {"Merlin": 1, "Percival": 1, "Loyal Servant": 1, "Morgana": 1, "Assassin": 1},
    6: {"Merlin": 1, "Percival": 1, "Loyal Servant": 2, "Morgana": 1, "Assassin": 1},
    7: {"Merlin": 1, "Percival": 1, "Loyal Servant": 2, "Morgana": 1, "Assassin": 1, "Oberon": 1},
    8: {"Merlin": 1, "Percival": 1, "Loyal Servant": 3, "Morgana": 1, "Assassin": 1, "Minion": 1},
    9: {"Merlin": 1, "Percival": 1, "Loyal Servant": 4, "Morgana": 1, "Assassin": 1, "Mordred": 1},
    10: {"Merlin": 1, "Percival": 1, "Loyal Servant": 4, "Morgana": 1, "Assassin": 1, "Mordred": 1, "Oberon": 1}
}

def get_role_list(num_players):
    if num_players not in CONFIG_MAP:
        raise ValueError(f"Unsupported player number: {num_players}. Must be 5-10.")
    
    role_counts = CONFIG_MAP[num_players]
    roles = []
    for role, count in role_counts.items():
        roles.extend([role] * count)
    return roles

ROLE_DEFINITIONS = {
    "Avalon": {
        "positive": ["Merlin", "Percival", "Loyal Servant"],
        "negative": ["Morgana", "Assassin", "Minion", "Oberon", "Mordred"],
        # 这里不再存储固定的 roles_config，而是动态生成
        "requires_words": False
    }
}

# ================= 辅助函数 (Helper Functions) =================

def ensure_log_dir():
    os.makedirs(FULL_LOG_PATH, exist_ok=True)

def log_experiment(info_dict):
    ensure_log_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_suffix = random.randint(1000, 9999)
    filename = f"log_Avalon_{timestamp}_w{WORKER_ID}_{rnd_suffix}.json"
    log_file = os.path.join(FULL_LOG_PATH, filename)
    
    summary_file = os.path.join(FULL_LOG_PATH, "experiment_summary.jsonl")
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=4, ensure_ascii=False)
        
    try:
        content = json.dumps(info_dict, ensure_ascii=False)
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(content + "\n")
    except Exception:
        pass
    
    print(f"[Worker {WORKER_ID}] Log saved: {filename}")

def setup_agents(roles_list, pos_config, neg_config):
    agents = []
    ids = [i for i in range(1, len(roles_list)+1)]
    
    def_info = ROLE_DEFINITIONS["Avalon"]
    pos_roles = def_info["positive"]
    neg_roles = def_info["negative"]
    
    assigned_configs = {} 
    
    for i, role in enumerate(roles_list):
        pid = ids[i]
        if role in pos_roles:
            config = pos_config
            side = "Positive"
        elif role in neg_roles:
            config = neg_config
            side = "Negative"
        else:
            config = pos_config 
            side = "Neutral"
            
        agent = Agent(pid, role, config.copy(), api_call_format)
        agents.append(agent)
        assigned_configs[f"P{pid}_{role}"] = f"{config.get('name', 'Unknown')} ({side})"
        
    return agents, assigned_configs

# ================= 主循环 (Main Loop) =================

def run_simulation():
    ensure_log_dir()
    
    swap_point = TOTAL_ROUNDS // 2
    
    for i in range(TOTAL_ROUNDS):
        print(f"\n>>> [Worker {WORKER_ID}] Round {i+1}/{TOTAL_ROUNDS} <<<")
        
        # 1. 确定模型阵营
        if i < swap_point:
            model_pos = CONFIG_MODEL_A
            model_neg = CONFIG_MODEL_B
            setup_desc = f"Run 1-{swap_point}: {model_pos['name']} (Pos) vs {model_neg['name']} (Neg)"
        else:
            model_pos = CONFIG_MODEL_B
            model_neg = CONFIG_MODEL_A
            setup_desc = f"Run {swap_point+1}-{TOTAL_ROUNDS}: {model_pos['name']} (Pos) vs {model_neg['name']} (Neg)"
            
        print(f"[Setup] {setup_desc}")

        # 2. 动态生成角色并随机打乱 (根据 PLAYER_NUM)
        try:
            roles_list = get_role_list(PLAYER_NUM)
            random.shuffle(roles_list)
        except ValueError as e:
            print(f"[Error] {e}")
            break
        
        # 3. 初始化 Agents
        agents, agent_log_info = setup_agents(roles_list, model_pos, model_neg)
        
        # 4. 初始化游戏
        try:
            game_instance = Game_Avalon_Multiturn(agents, log_tag=args.log_tag)
            
            start_time = datetime.datetime.now()
            
            # 5. 运行游戏
            if hasattr(game_instance, 'run_game'):
                game_instance.run_game()
            elif hasattr(game_instance, 'run'):
                game_instance.run()
            else:
                print("Error: No run method found for Game instance.")

            # 6. 记录日志
            log_data = {
                "worker_id": WORKER_ID,
                "iteration_in_worker": i + 1,
                "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "game_type": GAME_TYPE,
                "player_count": PLAYER_NUM,
                "model_setup_desc": setup_desc,
                "model_positive": model_pos['name'],
                "model_negative": model_neg['name'],
                "model_a_config": CONFIG_MODEL_A['inference_config'],
                "model_b_config": CONFIG_MODEL_B['inference_config'],
                "agent_assignments": agent_log_info,
            }
            log_experiment(log_data)
            
        except Exception as e:
            print(f"[Error] Worker {WORKER_ID} failed at round {i+1}: {e}")
            traceback.print_exc()
            err_data = {
                "worker_id": WORKER_ID,
                "iteration": i + 1,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            log_experiment(err_data)

        time.sleep(1)

if __name__ == "__main__":
    run_simulation()