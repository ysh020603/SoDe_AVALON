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

    # GAE / warmup
    gae_lambda: float = 0.95
    gae_warmup_iters: int = 0
    gae_warmup_lambda: float = 1.0  # 早期退化为 MC（缓解稀疏奖励下 advantage 方差）
    critic_only_warmup_iters: int = 0  # 早期可仅更新 Critic（降低 Actor 冷启动风险）

    # Stage2 合并 batch 的好/坏贡献权重（sample_weight）
    stage2_good_weight: float = 1.0
    stage2_evil_weight: float = 1.0

    # Self-play pool (Elo / top-k)
    pool_max_size: int = 5
    pool_top_k: int = 3
    initial_elo: float = 1000.0
    elo_k_factor: float = 32.0


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

    # init checkpoints (optional)
    parser.add_argument("--init_actor_good_path", type=str, default=None)
    parser.add_argument("--init_critic_good_path", type=str, default=None)
    parser.add_argument("--init_actor_evil_path", type=str, default=None)
    parser.add_argument("--init_critic_evil_path", type=str, default=None)

    # 两阶段训练
    parser.add_argument("--stage1_iters", type=int, default=-1, help="阶段1 iters；不填则使用 --iters")
    parser.add_argument("--stage2_iters", type=int, default=0, help="阶段2 iters（合并 batch + 单次更新）；0 表示禁用阶段2")

    # DDP
    parser.add_argument("--ddp", action="store_true", help="Enable torchrun/DDP training.")

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

    # GAE / warmup（仅作用阶段2；阶段1 可直接用 lambda=1 等价 return-to-go）
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--gae_warmup_iters", type=int, default=0)
    parser.add_argument("--gae_warmup_lambda", type=float, default=1.0)
    parser.add_argument("--critic_only_warmup_iters", type=int, default=0)

    # Stage2 合并 batch 的 good/evil 样本权重
    parser.add_argument("--stage2_good_weight", type=float, default=1.0)
    parser.add_argument("--stage2_evil_weight", type=float, default=1.0)

    # Self-play pool Elo / top-k
    parser.add_argument("--pool_max_size", type=int, default=5)
    parser.add_argument("--pool_top_k", type=int, default=3)
    parser.add_argument("--initial_elo", type=float, default=1000.0)
    parser.add_argument("--elo_k_factor", type=float, default=32.0)

    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--use_causal_mask", action="store_true")
    parser.add_argument("--no_use_causal_mask", dest="use_causal_mask", action="store_false")
    parser.set_defaults(use_causal_mask=True)
    parser.add_argument("--use_text_cache", action="store_true", help="Skip speech text encoding (fast).")
    parser.add_argument("--no_use_text_cache", dest="use_text_cache", action="store_false", help="Do NOT skip speech text encoding.")
    parser.set_defaults(use_text_cache=True)

    # qwen embedding model path (用于 speech text encoder，取决于 use_text_cache 的值)
    parser.add_argument("--qwen_emb_path", type=str, default=None, help="Qwen embedding model path.")

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

    ddp_enabled = bool(args.ddp)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if ddp_enabled:
        # torchrun will set LOCAL_RANK/RANK/WORLD_SIZE; use it to pin device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    else:
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
        use_text_cache=args.use_text_cache,
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        swanlab_run_name=args.swanlab_run_name,
        gae_lambda=args.gae_lambda,
        gae_warmup_iters=args.gae_warmup_iters,
        gae_warmup_lambda=args.gae_warmup_lambda,
        critic_only_warmup_iters=args.critic_only_warmup_iters,
        stage2_good_weight=args.stage2_good_weight,
        stage2_evil_weight=args.stage2_evil_weight,
        pool_max_size=args.pool_max_size,
        pool_top_k=args.pool_top_k,
        initial_elo=args.initial_elo,
        elo_k_factor=args.elo_k_factor,
    )

    stage1_iters = int(args.stage1_iters) if int(args.stage1_iters) >= 0 else int(train_cfg.iters)
    stage2_iters = int(args.stage2_iters)
    train_cfg.iters = stage1_iters  # 方便复用已有变量名

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
    if sw_logger is not None and ddp_enabled and rank != 0:
        # DDP 下避免多 rank 同时写日志
        sw_logger = None
    if sw_logger is not None:
        sw_logger.init()

    # Actor/Critic 初始化：good/evil 两套（训练时只更新其中一套）
    qwen_emb_path = args.qwen_emb_path
    actor_good = AvalonActor(
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers,
        max_round_attempt_id=64,
        max_seq_len=train_cfg.max_seq_len,
        max_text_len=train_cfg.max_text_len,
        use_causal_mask=train_cfg.use_causal_mask,
        use_text_cache=train_cfg.use_text_cache,
        qwen_model_path=qwen_emb_path,
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
        qwen_model_path=qwen_emb_path,
        device=train_cfg.device,
    )

    def _load_weights_into(model: torch.nn.Module, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            elif "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
                sd = sd["model_state_dict"]
            elif "module" in sd and isinstance(sd["module"], dict):
                sd = sd["module"]

        if not isinstance(sd, dict):
            raise ValueError(f"Unsupported checkpoint format: {path}")

        # strip "module." prefix if checkpoint was saved from DDP
        if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        model.load_state_dict(sd, strict=False)

    # load good init (optional)
    if args.init_actor_good_path:
        _load_weights_into(actor_good, args.init_actor_good_path)
    if args.init_critic_good_path:
        _load_weights_into(critic_good, args.init_critic_good_path)

    actor_evil = copy.deepcopy(actor_good)
    critic_evil = copy.deepcopy(critic_good)

    # load evil init (optional; if not provided, keep copied good weights)
    if args.init_actor_evil_path:
        _load_weights_into(actor_evil, args.init_actor_evil_path)
    if args.init_critic_evil_path:
        _load_weights_into(critic_evil, args.init_critic_evil_path)

    # DDP wrap：用于梯度同步（rollout 仍使用未包装的 actor/critic 实例）
    ddp_actor_good = None
    ddp_actor_evil = None
    ddp_critic_good = None
    ddp_critic_evil = None
    if ddp_enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp_actor_good = DDP(
            actor_good,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        ddp_critic_good = DDP(
            critic_good,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        ddp_actor_evil = DDP(
            actor_evil,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        ddp_critic_evil = DDP(
            critic_evil,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    opt_good = torch.optim.AdamW(actor_good.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    opt_critic_good = torch.optim.AdamW(critic_good.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    opt_evil = torch.optim.AdamW(actor_evil.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    opt_critic_evil = torch.optim.AdamW(critic_evil.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # 对手池：初始把当前策略都塞进去，避免空池
    pool_good = SelfPlayPool(max_size=train_cfg.pool_max_size)
    pool_evil = SelfPlayPool(max_size=train_cfg.pool_max_size)
    pool_good.push("init_good", actor_good.state_dict(), rating=train_cfg.initial_elo)
    pool_evil.push("init_evil", actor_evil.state_dict(), rating=train_cfg.initial_elo)

    # 维护当前“受控侧”策略的 Elo（用于给后续 push 的 rating 提供一个随训练变化的估计）
    good_controller_elo = float(train_cfg.initial_elo)
    evil_controller_elo = float(train_cfg.initial_elo)

    def _elo_expected(player_elo: float, opponent_elo: float) -> float:
        return 1.0 / (1.0 + (10.0 ** ((opponent_elo - player_elo) / 400.0)))

    def _elo_update(player_elo: float, opponent_elo: float, score: float) -> Tuple[float, float]:
        """
        score: 1 表示 player 胜；0 表示 player 负
        返回：new_player_elo, new_opponent_elo
        """
        k = float(train_cfg.elo_k_factor)
        expected_player = _elo_expected(player_elo, opponent_elo)
        expected_opponent = 1.0 - expected_player
        score_opponent = 1.0 - float(score)
        new_player = player_elo + k * (float(score) - expected_player)
        new_opponent = opponent_elo + k * (score_opponent - expected_opponent)
        return float(new_player), float(new_opponent)

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
                opponent_name, opponent_actor_sd, opponent_elo = pool_evil.sample_entry(
                    strategy="elo_topk",
                    top_k=train_cfg.pool_top_k,
                )
                controlled_actor = actor_good
                controlled_elo = good_controller_elo
            else:
                opponent_name, opponent_actor_sd, opponent_elo = pool_good.sample_entry(
                    strategy="elo_topk",
                    top_k=train_cfg.pool_top_k,
                )
                controlled_actor = actor_evil
                controlled_elo = evil_controller_elo

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

            # 更新 Elo：受控侧 vs 采样到的对手池条目
            score_controlled = 1.0 if reward > 0 else 0.0
            new_controlled_elo, new_opponent_elo = _elo_update(
                controlled_elo,
                opponent_elo,
                score_controlled,
            )
            if train_side == "good":
                good_controller_elo = new_controlled_elo
                pool_evil.update_rating(opponent_name, new_opponent_elo)
            else:
                evil_controller_elo = new_controlled_elo
                pool_good.update_rating(opponent_name, new_opponent_elo)

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
            pool_good.push(f"iter{iter_idx}", actor_good.state_dict(), rating=good_controller_elo)
        else:
            pool_evil.push(f"iter{iter_idx}", actor_evil.state_dict(), rating=evil_controller_elo)

    # =========================
    # Stage2：共享模型 + 合并 batch 单次更新
    # =========================
    if stage2_iters > 0:
        # 从阶段1的 good 模型初始化共享模型（保证 good/evil 网络架构与参数维度一致）
        shared_actor = copy.deepcopy(actor_good)
        shared_critic = copy.deepcopy(critic_good)

        ddp_shared_actor = None
        ddp_shared_critic = None
        if ddp_enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_shared_actor = DDP(
                shared_actor,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )
            ddp_shared_critic = DDP(
                shared_critic,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )

        opt_shared_actor = torch.optim.AdamW(shared_actor.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        opt_shared_critic = torch.optim.AdamW(shared_critic.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

        shared_adapter = MAPPOCTDEAdapter(
            actor=shared_actor,
            critic=shared_critic,
            actor_optimizer=opt_shared_actor,
            critic_optimizer=opt_shared_critic,
            cfg=train_cfg,
            side="shared",
            device=train_cfg.device,
        )

        shared_good_elo = float(good_controller_elo)
        shared_evil_elo = float(evil_controller_elo)

        def _run_rollout(
            controlled_side: str,
            opponent_pool: SelfPlayPool,
            controller_elo: float,
        ) -> Tuple[List[Any], float, str, float]:
            roles = _build_roles_7()
            random.shuffle(roles)

            # 对手 actor：从池中采样（不更新）
            opponent_name, opponent_actor_sd, opponent_elo = opponent_pool.sample_entry(
                strategy="elo_topk",
                top_k=train_cfg.pool_top_k,
            )
            opponent_actor = copy.deepcopy(shared_actor)  # 保证采样与 record actor 权重一致
            opponent_actor.load_state_dict(opponent_actor_sd, strict=False)

            agents: List[Any] = []
            ids = [i for i in range(1, len(roles) + 1)]
            for pid, role in zip(ids, roles):
                is_good = role in {"Merlin", "Percival", "Loyal Servant"}
                player_side = "good" if is_good else "evil"
                record = player_side == controlled_side

                agent = RLSmallModelMAPPOAgent(
                    player_id=pid,
                    role=role,
                    llm_config=llm_config,
                    llm_func=api_call_format,
                    actor=shared_actor if record else opponent_actor,
                    critic=shared_critic if record else None,
                    record=record,
                    use_llm_in_discussion=True,
                )
                agents.append(agent)

            game_instance = Game_Avalon_Multiturn(agents, log_tag="mappo_train_stage2")
            final_result = game_instance.run_game()

            # 收集受控侧 transitions
            episode_transitions: List[Any] = []
            for ag in agents:
                if hasattr(ag, "collected_transitions"):
                    episode_transitions.extend(ag.collected_transitions)
            for ag in agents:
                if hasattr(ag, "clear_episode"):
                    ag.clear_episode()

            reward = _terminal_reward(train_side=controlled_side, final_result=final_result)
            score_controlled = 1.0 if reward > 0 else 0.0
            new_controller_elo, new_opponent_elo = _elo_update(controller_elo, opponent_elo, score_controlled)
            opponent_pool.update_rating(opponent_name, new_opponent_elo)

            return episode_transitions, reward, final_result, new_controller_elo

        for stage2_idx in range(stage2_iters):
            gae_lambda_use = train_cfg.gae_warmup_lambda if stage2_idx < train_cfg.gae_warmup_iters else train_cfg.gae_lambda
            critic_only = stage2_idx < train_cfg.critic_only_warmup_iters

            all_transitions: List[Any] = []
            transition_weights: List[float] = []

            # 每个外层 iter：分别 rollout good-controlled 与 evil-controlled，然后合并 transitions
            for ep_idx in range(train_cfg.episodes_per_iter):
                transitions_good, r_good, _, shared_good_elo = _run_rollout("good", pool_evil, shared_good_elo)
                if transitions_good:
                    all_transitions.extend(transitions_good)
                    transition_weights.extend([train_cfg.stage2_good_weight] * len(transitions_good))
                    shared_adapter.compute_and_set_returns(
                        transitions_good,
                        terminal_reward=r_good,
                        gamma=train_cfg.gamma,
                        use_gae=True,
                        gae_lambda=gae_lambda_use,
                    )

                transitions_evil, r_evil, _, shared_evil_elo = _run_rollout("evil", pool_good, shared_evil_elo)
                if transitions_evil:
                    all_transitions.extend(transitions_evil)
                    transition_weights.extend([train_cfg.stage2_evil_weight] * len(transitions_evil))
                    shared_adapter.compute_and_set_returns(
                        transitions_evil,
                        terminal_reward=r_evil,
                        gamma=train_cfg.gamma,
                        use_gae=True,
                        gae_lambda=gae_lambda_use,
                    )

            update_metrics = shared_adapter.update(
                transitions=all_transitions,
                transition_weights=transition_weights,
                ppo_epochs=train_cfg.ppo_epochs,
                minibatch_size=train_cfg.minibatch_size,
                batch_size=train_cfg.batch_size,
                critic_only=critic_only,
            )

            if sw_logger is not None:
                # stage2 的 train_side 语义不完全对应 SwanLabLogger，这里仅做统一记录
                sw_logger.log_iter_end(
                    iter_idx=train_cfg.iters + stage2_idx,
                    train_side="good",
                    update_metrics=update_metrics,
                )

            # 简化入池：阶段2 后把共享 actor 的最新版本同时塞入 good/evil 池
            pool_good.push(f"stage2_iter{stage2_idx}", shared_actor.state_dict(), rating=shared_good_elo)
            pool_evil.push(f"stage2_iter{stage2_idx}", shared_actor.state_dict(), rating=shared_evil_elo)

    if sw_logger is not None:
        sw_logger.finish()


if __name__ == "__main__":
    main()

