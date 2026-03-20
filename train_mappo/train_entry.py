import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "SoDe_Avalon_3"))


def _build_roles_7() -> List[str]:
    # Avalon 7-player 固定角色数量（对应 Game_Avalon_Multiturn 校验）
    return [
        "Merlin",
        "Percival",
        "Loyal Servant",
        "Loyal Servant",
        "Morgana",
        "Assassin",
        "Oberon",
    ]


def _choose_train_side(iter_idx: int) -> str:
    # 简单 alternation：偶数轮训练 good，奇数轮训练 evil
    return "good" if iter_idx % 2 == 0 else "evil"


def _terminal_reward(train_side: str, final_result: str) -> float:
    # Game_Avalon_Multiturn 的 game_condition["meta"]["final_result"]：good_win / evil_win_missions / evil_win_assassination
    if train_side == "good":
        return 1.0 if final_result == "good_win" else -1.0
    return 1.0 if final_result != "good_win" else -1.0


@dataclass
class TrainConfig:
    device: torch.device
    seed: int
    player_num: int = 7

    # rollout
    iters: int = 1
    episodes_per_iter: int = 2
    max_game_step_failsafe: int = 5000

    # PPO/MAPPO
    gamma: float = 0.99
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 2
    minibatch_size: int = 16
    batch_size: int = 32

    # model
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 4
    max_seq_len: int = 512
    max_text_len: int = 256
    use_causal_mask: bool = True
    use_text_cache: bool = True

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # swanlab
    use_swanlab: bool = False
    swanlab_project: str = "avalon_mappo"
    swanlab_run_name: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SoDe_Avalon_3 MAPPO training (new code only)")

    # LLM config (discussion phase 需要调用)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--llm_api_key", type=str, required=True)
    parser.add_argument("--llm_base_url", type=str, required=True)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_top_p", type=float, default=1.0)
    parser.add_argument("--llm_max_tokens", type=int, default=1024)
    parser.add_argument("--llm_reasoning", action="store_true", help="Allow model reasoning if supported.")

    # training config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--episodes_per_iter", type=int, default=2)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--minibatch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--use_causal_mask", action="store_true")
    parser.add_argument("--no_use_causal_mask", dest="use_causal_mask", action="store_false")
    parser.set_defaults(use_causal_mask=True)
    parser.add_argument("--use_text_cache", action="store_true", help="Skip speech text encoding (fast).")

    # swanlab
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="avalon_mappo")
    parser.add_argument("--swanlab_run_name", type=str, default="")

    return parser.parse_args()


def build_llm_config(args: argparse.Namespace) -> Dict[str, Any]:
    inference_config: Dict[str, Any] = {
        "temperature": float(args.llm_temperature),
        "top_p": float(args.llm_top_p),
        "max_tokens": int(args.llm_max_tokens),
    }
    # 兼容已有逻辑：部分模型使用 extra_body/chat_template_kwargs
    if not args.llm_reasoning:
        if "glm" in args.llm_model_name.lower():
            inference_config["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            inference_config["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    return {
        "name": args.llm_model_name,
        "api_url_config": {"api_key": args.llm_api_key, "base_url": args.llm_base_url},
        "inference_config": inference_config,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_cfg = TrainConfig(
        device=device,
        seed=args.seed,
        iters=args.iters,
        episodes_per_iter=args.episodes_per_iter,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        max_text_len=args.max_text_len,
        use_causal_mask=args.use_causal_mask,
        # 训练/rollout 默认跳过 speech 文本编码，避免依赖 transformers embedding 权重
        use_text_cache=True,
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        swanlab_run_name=args.swanlab_run_name,
    )

    llm_config = build_llm_config(args)

    # Lazy imports：避免其它新模块尚未就绪时 import 阶段失败
    from SoDe_Avalon_3.Tool.callopenai import api_call_format
    from SoDe_Avalon_3.Game.Avalon_multiturn import Game_Avalon_Multiturn

    from SoDe_Avalon_3.train_mappo.agents.rl_mappo_agent import (
        RLSmallModelMAPPOAgent,
    )
    from SoDe_Avalon_3.train_mappo.models.actor_avalon import AvalonActor
    from SoDe_Avalon_3.train_mappo.models.critic_avalon import AvalonCritic
    from SoDe_Avalon_3.train_mappo.mappo.algo_adapter import MAPPOCTDEAdapter
    from SoDe_Avalon_3.train_mappo.self_play import SelfPlayPool
    from SoDe_Avalon_3.train_mappo.swanlab_logging.swanlab_logger import SwanLabLogger

    sw_logger = SwanLabLogger(train_cfg) if train_cfg.use_swanlab else None
    if sw_logger is not None:
        sw_logger.init()

    # Actor/Critic 初始化：good/evil 两套（训练时只更新其中一套）
    actor_good = AvalonActor(
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers,
        max_round_attempt_id=64,
        max_seq_len=train_cfg.max_seq_len,
        max_text_len=train_cfg.max_text_len,
        use_causal_mask=train_cfg.use_causal_mask,
        use_text_cache=train_cfg.use_text_cache,
        device=train_cfg.device,
    )
    critic_good = AvalonCritic(
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers,
        max_round_attempt_id=64,
        max_seq_len=train_cfg.max_seq_len,
        max_text_len=train_cfg.max_text_len,
        use_causal_mask=train_cfg.use_causal_mask,
        use_text_cache=train_cfg.use_text_cache,
        device=train_cfg.device,
    )
    actor_evil = copy.deepcopy(actor_good)
    critic_evil = copy.deepcopy(critic_good)

    opt_good = torch.optim.AdamW(actor_good.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    opt_critic_good = torch.optim.AdamW(critic_good.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    opt_evil = torch.optim.AdamW(actor_evil.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    opt_critic_evil = torch.optim.AdamW(critic_evil.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # 对手池：初始把当前策略都塞进去，避免空池
    pool_good = SelfPlayPool(max_size=5)
    pool_evil = SelfPlayPool(max_size=5)
    pool_good.push("init", actor_good.state_dict())
    pool_evil.push("init", actor_evil.state_dict())

    mappo_adapter_good = MAPPOCTDEAdapter(
        actor=actor_good,
        critic=critic_good,
        actor_optimizer=opt_good,
        critic_optimizer=opt_critic_good,
        cfg=train_cfg,
        side="good",
        device=train_cfg.device,
    )
    mappo_adapter_evil = MAPPOCTDEAdapter(
        actor=actor_evil,
        critic=critic_evil,
        actor_optimizer=opt_evil,
        critic_optimizer=opt_critic_evil,
        cfg=train_cfg,
        side="evil",
        device=train_cfg.device,
    )

    for iter_idx in range(train_cfg.iters):
        train_side = _choose_train_side(iter_idx)

        # 训练侧选用对应 adapter
        adapter = mappo_adapter_good if train_side == "good" else mappo_adapter_evil

        if sw_logger is not None:
            sw_logger.log_iter_start(iter_idx=iter_idx, train_side=train_side)

        all_transitions: List[Any] = []
        episode_stats: List[Dict[str, Any]] = []

        for ep_idx in range(train_cfg.episodes_per_iter):
            roles = _build_roles_7()
            random.shuffle(roles)

            # 对手 actor：从池中采样（不更新）
            if train_side == "good":
                opponent_actor_sd = pool_evil.sample_state_dict()
                controlled_actor = actor_good
            else:
                opponent_actor_sd = pool_good.sample_state_dict()
                controlled_actor = actor_evil

            # 构造对手 actor 副本（保证采样一致性，不把梯度图挂到训练 actor）
            opponent_actor = copy.deepcopy(controlled_actor)  # 结构一致
            opponent_actor.load_state_dict(opponent_actor_sd, strict=False)

            # 对手 critic 不需要；但 RL Agent 构造时要求 critic 对象，我们传 None 并禁用 record
            # (agent 内不会用 critic 来算动作 log_prob，只在 record mode 采样时才需要)
            agents: List[Any] = []
            ids = [i for i in range(1, len(roles) + 1)]
            for pid, role in zip(ids, roles):
                is_good = role in {"Merlin", "Percival", "Loyal Servant"}
                player_side = "good" if is_good else "evil"

                record = player_side == train_side

                agent = RLSmallModelMAPPOAgent(
                    player_id=pid,
                    role=role,
                    llm_config=llm_config,
                    llm_func=api_call_format,
                    actor=controlled_actor if record else opponent_actor,
                    critic=critic_good if train_side == "good" and record else (critic_evil if train_side == "evil" and record else None),
                    record=record,
                    # 只记录受训侧；对手不记录
                    use_llm_in_discussion=True,
                )
                agents.append(agent)

            # 跑一局
            game_instance = Game_Avalon_Multiturn(agents, log_tag="mappo_train")
            final_result = game_instance.run_game()

            # 收集受训侧 transitions
            episode_transitions: List[Any] = []
            for ag in agents:
                if hasattr(ag, "collected_transitions"):
                    episode_transitions.extend(ag.collected_transitions)
            for ag in agents:
                # 清理给下一局
                if hasattr(ag, "clear_episode"):
                    ag.clear_episode()

            reward = _terminal_reward(train_side=train_side, final_result=final_result)
            adapter.compute_and_set_returns(episode_transitions, terminal_reward=reward, gamma=train_cfg.gamma)
            all_transitions.extend(episode_transitions)

            episode_stats.append(
                {
                    "episode_index": ep_idx,
                    "final_result": final_result,
                    "reward_for_train_side": reward,
                    "transitions": len(episode_transitions),
                }
            )

            if sw_logger is not None:
                sw_logger.log_episode(iter_idx=iter_idx, ep_idx=ep_idx, train_side=train_side, stats=episode_stats[-1])

        # PPO/MAPPO 更新
        update_metrics = adapter.update(
            transitions=all_transitions,
            ppo_epochs=train_cfg.ppo_epochs,
            minibatch_size=train_cfg.minibatch_size,
            batch_size=train_cfg.batch_size,
        )

        if sw_logger is not None:
            sw_logger.log_iter_end(iter_idx=iter_idx, train_side=train_side, update_metrics=update_metrics)

        # 简化入池：每次 rollout 后，把训练侧当前 actor 入自己的池（top-k 策略可在后续增强）
        if train_side == "good":
            pool_good.push(f"iter{iter_idx}", actor_good.state_dict())
        else:
            pool_evil.push(f"iter{iter_idx}", actor_evil.state_dict())

    if sw_logger is not None:
        sw_logger.finish()


if __name__ == "__main__":
    main()

