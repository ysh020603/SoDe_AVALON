import os
import sys
import random
import datetime
import json
import traceback
import argparse
import time
from typing import Any, Dict, List, Tuple


parser = argparse.ArgumentParser(description="Run Avalon Simulation (SmallModel compare)")

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

# small model compare 参数
parser.add_argument(
    "--smallmodel_for",
    type=str,
    default="A",
    choices=["A", "B", "none", "both"],
    help="Which model(s) use SmallModelAgent: A|B|none|both",
)
parser.add_argument(
    "--smallmodel_api_url",
    type=str,
    default="http://127.0.0.1:8000/predict_from_log",
    help="Small model API url",
)
parser.add_argument(
    "--smallmodel_timeout",
    type=int,
    default=600,
    help="Small model request timeout seconds",
)

args = parser.parse_args()


GAME_TYPE = "Avalon"
TOTAL_ROUNDS = args.rounds
WORKER_ID = args.worker_id
PLAYER_NUM = args.player_num


def build_inference_config(model_name: str, temp: str, top_p: str, max_tokens: str, use_reasoning_str: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {"model": model_name}

    if temp.lower() != "none":
        config["temperature"] = float(temp)
    if top_p.lower() != "none":
        config["top_p"] = float(top_p)
    if max_tokens.lower() != "none":
        config["max_tokens"] = int(max_tokens)

    is_reasoning = use_reasoning_str.lower() == "true"
    if not is_reasoning:
        if "glm" in model_name:
            config["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            config["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return config


CONFIG_MODEL_A: Dict[str, Any] = {
    "name": args.model_a_name,
    "api_url_config": {"api_key": args.model_a_key, "base_url": args.model_a_url},
    "inference_config": build_inference_config(
        args.model_a_name,
        args.model_a_temp,
        args.model_a_top_p,
        args.model_a_max_tokens,
        args.model_a_reasoning,
    ),
}

CONFIG_MODEL_B: Dict[str, Any] = {
    "name": args.model_b_name,
    "api_url_config": {"api_key": args.model_b_key, "base_url": args.model_b_url},
    "inference_config": build_inference_config(
        args.model_b_name,
        args.model_b_temp,
        args.model_b_top_p,
        args.model_b_max_tokens,
        args.model_b_reasoning,
    ),
}


LOG_ROOT = "ex_log"
if args.log_tag:
    LOG_DIR_NAME = args.log_tag
else:
    LOG_DIR_NAME = f"{args.model_a_name}_VS_{args.model_b_name}"
FULL_LOG_PATH = os.path.join(LOG_ROOT, LOG_DIR_NAME)

print(f"[{datetime.datetime.now()}] Worker {WORKER_ID} initialized.")
print(f"Game: Avalon ({PLAYER_NUM} players) | Rounds: {TOTAL_ROUNDS}")
print(f"Matchup: {CONFIG_MODEL_A['name']} vs {CONFIG_MODEL_B['name']}")
print(f"SmallModelAgent for: {args.smallmodel_for} | api={args.smallmodel_api_url}")
print(f"Log Directory: {FULL_LOG_PATH}")


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from Tool.callopenai import api_call_format
    from Agents.Agent import Agent
    from Agents.SmallModelAgent import SmallModelAgent
    from Game.Avalon_multiturn import Game_Avalon_Multiturn
except ImportError as e:
    print("Error importing game modules. Please run this script from the root of the project structure.")
    print(f"Details: {e}")
    sys.exit(1)


CONFIG_MAP = {
    5: {"Merlin": 1, "Percival": 1, "Loyal Servant": 1, "Morgana": 1, "Assassin": 1},
    6: {"Merlin": 1, "Percival": 1, "Loyal Servant": 2, "Morgana": 1, "Assassin": 1},
    7: {"Merlin": 1, "Percival": 1, "Loyal Servant": 2, "Morgana": 1, "Assassin": 1, "Oberon": 1},
    8: {"Merlin": 1, "Percival": 1, "Loyal Servant": 3, "Morgana": 1, "Assassin": 1, "Minion": 1},
    9: {"Merlin": 1, "Percival": 1, "Loyal Servant": 4, "Morgana": 1, "Assassin": 1, "Mordred": 1},
    10: {"Merlin": 1, "Percival": 1, "Loyal Servant": 4, "Morgana": 1, "Assassin": 1, "Mordred": 1, "Oberon": 1},
}


def get_role_list(num_players: int) -> List[str]:
    if num_players not in CONFIG_MAP:
        raise ValueError(f"Unsupported player number: {num_players}. Must be 5-10.")
    role_counts = CONFIG_MAP[num_players]
    roles: List[str] = []
    for role, count in role_counts.items():
        roles.extend([role] * count)
    return roles


ROLE_DEFINITIONS = {
    "Avalon": {
        "positive": ["Merlin", "Percival", "Loyal Servant"],
        "negative": ["Morgana", "Assassin", "Minion", "Oberon", "Mordred"],
        "requires_words": False,
    }
}


def ensure_log_dir() -> None:
    os.makedirs(FULL_LOG_PATH, exist_ok=True)


def log_experiment(info_dict: Dict[str, Any]) -> None:
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


def setup_agents_compare(
    roles_list: List[str],
    pos_config: Dict[str, Any],
    neg_config: Dict[str, Any],
    *,
    use_small_for_pos: bool,
    use_small_for_neg: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    """按阵营区分：只有选定的模型（A 或 B）使用 SmallModelAgent，另一阵营用 Agent。"""
    agents: List[Any] = []
    ids = [i for i in range(1, len(roles_list) + 1)]

    def_info = ROLE_DEFINITIONS["Avalon"]
    pos_roles = def_info["positive"]
    neg_roles = def_info["negative"]

    assigned_configs: Dict[str, Any] = {}

    for i, role in enumerate(roles_list):
        pid = ids[i]
        if role in pos_roles:
            config = pos_config
            side = "Positive"
            use_small = use_small_for_pos
        elif role in neg_roles:
            config = neg_config
            side = "Negative"
            use_small = use_small_for_neg
        else:
            config = pos_config
            side = "Neutral"
            use_small = use_small_for_pos

        model_name = config.get("name", "Unknown")

        if use_small:
            agent = SmallModelAgent(
                pid,
                role,
                config.copy(),
                api_call_format,
                small_model_api_url=args.smallmodel_api_url,
                small_model_timeout=args.smallmodel_timeout,
                enable_small_model=True,
            )
            agent_type = "SmallModelAgent"
        else:
            agent = Agent(pid, role, config.copy(), api_call_format)
            agent_type = "Agent"

        agents.append(agent)
        assigned_configs[f"P{pid}_{role}"] = {
            "model": model_name,
            "side": side,
            "agent_type": agent_type,
            "use_smallmodel": bool(use_small),
        }

    return agents, assigned_configs


def run_simulation() -> None:
    ensure_log_dir()

    swap_point = TOTAL_ROUNDS // 2
    # 只有选定的模型用 SmallModelAgent：按“当前正/反阵营是 A 还是 B”决定，不按名字（避免 A/B 同名时全用 Small）
    use_small_a = args.smallmodel_for in ("A", "both")
    use_small_b = args.smallmodel_for in ("B", "both")

    for i in range(TOTAL_ROUNDS):
        print(f"\n>>> [Worker {WORKER_ID}] Round {i+1}/{TOTAL_ROUNDS} <<<")

        # 1. 确定模型阵营（与原脚本一致：半程交换 A/B 的正反）
        if i < swap_point:
            model_pos = CONFIG_MODEL_A
            model_neg = CONFIG_MODEL_B
            use_small_for_pos = use_small_a
            use_small_for_neg = use_small_b
            setup_desc = f"Run 1-{swap_point}: {model_pos['name']} (Pos) vs {model_neg['name']} (Neg)"
        else:
            model_pos = CONFIG_MODEL_B
            model_neg = CONFIG_MODEL_A
            use_small_for_pos = use_small_b
            use_small_for_neg = use_small_a
            setup_desc = f"Run {swap_point+1}-{TOTAL_ROUNDS}: {model_pos['name']} (Pos) vs {model_neg['name']} (Neg)"

        print(f"[Setup] {setup_desc} (SmallModel: Pos={use_small_for_pos}, Neg={use_small_for_neg})")

        # 2. 动态生成角色并随机打乱
        try:
            roles_list = get_role_list(PLAYER_NUM)
            random.shuffle(roles_list)
        except ValueError as e:
            print(f"[Error] {e}")
            break

        # 3. 初始化 Agents：只有选定阵营用 SmallModelAgent，另一阵营用 Agent
        agents, agent_log_info = setup_agents_compare(
            roles_list,
            model_pos,
            model_neg,
            use_small_for_pos=use_small_for_pos,
            use_small_for_neg=use_small_for_neg,
        )

        # 4. 初始化游戏并运行
        try:
            game_instance = Game_Avalon_Multiturn(agents, log_tag=args.log_tag)
            start_time = datetime.datetime.now()

            if hasattr(game_instance, "run_game"):
                game_instance.run_game()
            elif hasattr(game_instance, "run"):
                game_instance.run()
            else:
                print("Error: No run method found for Game instance.")

            log_data: Dict[str, Any] = {
                "worker_id": WORKER_ID,
                "iteration_in_worker": i + 1,
                "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "game_type": GAME_TYPE,
                "player_count": PLAYER_NUM,
                "model_setup_desc": setup_desc,
                "model_positive": model_pos["name"],
                "model_negative": model_neg["name"],
                "model_a_config": CONFIG_MODEL_A["inference_config"],
                "model_b_config": CONFIG_MODEL_B["inference_config"],
                "smallmodel_for": args.smallmodel_for,
                "smallmodel_api_url": args.smallmodel_api_url,
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
                "traceback": traceback.format_exc(),
            }
            log_experiment(err_data)

        time.sleep(1)


if __name__ == "__main__":
    run_simulation()

