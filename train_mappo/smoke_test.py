import argparse
import json
import random
import os
import sys
from typing import Any, Dict, List

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "SoDe_Avalon_3"))

from SoDe_Avalon_3.Game.Avalon_multiturn import Game_Avalon_Multiturn
from SoDe_Avalon_3.train_mappo.agents.rl_mappo_agent import RLSmallModelMAPPOAgent
from SoDe_Avalon_3.train_mappo.swanlab_logging.swanlab_logger import SwanLabLogger
from SoDe_Avalon_3.train_mappo.mappo.algo_adapter import MAPPOCTDEAdapter
from SoDe_Avalon_3.train_mappo.models.actor_avalon import AvalonActor
from SoDe_Avalon_3.train_mappo.models.critic_avalon import AvalonCritic
from SoDe_Avalon_3.train_mappo.self_play import SelfPlayPool


_GOOD_ROLES = {"Merlin", "Percival", "Loyal Servant"}


def dummy_llm_func(messages, api_url_config: dict, inference_config: dict, system_prompt: str = None) -> str:
    # discussion 阶段环境只要求可解析 {"statement": "..."}
    return json.dumps({"statement": "..."}, ensure_ascii=False)


def build_llm_config_dummy() -> Dict[str, Any]:
    return {
        "name": "dummy",
        "api_url_config": {"api_key": "none", "base_url": "http://localhost"},
        "inference_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16},
    }


def _terminal_reward(train_side: str, final_result: str) -> float:
    # 与 train_entry 保持一致
    if train_side == "good":
        return 1.0 if final_result == "good_win" else -1.0
    return 1.0 if final_result != "good_win" else -1.0


def _build_roles_7() -> List[str]:
    return ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin", "Oberon"]


def main() -> None:
    parser = argparse.ArgumentParser("Avalon MAPPO smoke test (new code only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_swanlab", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm_config = build_llm_config_dummy()

    # 小模型，便于 smoke test 快速跑通
    class Cfg:
        gamma = 0.99
        clip_ratio = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        grad_clip = 1.0
        swanlab_project = "avalon_mappo_smoke"
        swanlab_run_name = "smoke"

    cfg = Cfg()
    # NOTE: class 体不会捕获外层函数局部变量，所以 device 需要在 class 定义之后再挂载
    cfg.device = device

    # SoDe_RL.AvalonPolicyModel 的 token encoder 输出固定为 1024 维，
    # 因此 d_model 也必须保持 1024 以避免 token_emb 与 pos_emb 维度不匹配。
    actor_good = AvalonActor(d_model=1024, n_heads=4, n_layers=1, max_seq_len=256, use_text_cache=True, device=device)
    critic_good = AvalonCritic(d_model=1024, n_heads=4, n_layers=1, max_seq_len=256, use_text_cache=True, device=device)
    actor_evil = AvalonActor(d_model=1024, n_heads=4, n_layers=1, max_seq_len=256, use_text_cache=True, device=device)
    critic_evil = AvalonCritic(d_model=1024, n_heads=4, n_layers=1, max_seq_len=256, use_text_cache=True, device=device)

    opt_ag = torch.optim.AdamW(actor_good.parameters(), lr=1e-4)
    opt_cg = torch.optim.AdamW(critic_good.parameters(), lr=1e-4)
    opt_ae = torch.optim.AdamW(actor_evil.parameters(), lr=1e-4)
    opt_ce = torch.optim.AdamW(critic_evil.parameters(), lr=1e-4)

    adapter_good = MAPPOCTDEAdapter(actor=actor_good, critic=critic_good, actor_optimizer=opt_ag, critic_optimizer=opt_cg, cfg=cfg, side="good", device=device)
    adapter_evil = MAPPOCTDEAdapter(actor=actor_evil, critic=critic_evil, actor_optimizer=opt_ae, critic_optimizer=opt_ce, cfg=cfg, side="evil", device=device)

    for ep in range(args.episodes):
        roles = _build_roles_7()
        random.shuffle(roles)
        ids = [i for i in range(1, len(roles) + 1)]

        agents: List[Any] = []
        for pid, role in zip(ids, roles):
            is_good = role in _GOOD_ROLES
            agent = RLSmallModelMAPPOAgent(
                player_id=pid,
                role=role,
                llm_config=llm_config,
                llm_func=dummy_llm_func,
                actor=actor_good if is_good else actor_evil,
                critic=critic_good if is_good else critic_evil,
                record=True,
                use_llm_in_discussion=True,
            )
            agents.append(agent)

        game = Game_Avalon_Multiturn(agents, log_tag="mappo_smoke")
        final_result = game.run_game()

        transitions_good: List[Dict[str, Any]] = []
        transitions_evil: List[Dict[str, Any]] = []
        for ag in agents:
            for tr in getattr(ag, "collected_transitions", []):
                role = tr.get("meta", {}).get("role", "")
                if role in _GOOD_ROLES:
                    transitions_good.append(tr)
                else:
                    transitions_evil.append(tr)

        r_good = _terminal_reward("good", final_result)
        r_evil = _terminal_reward("evil", final_result)
        adapter_good.compute_and_set_returns(transitions_good, terminal_reward=r_good, gamma=cfg.gamma)
        adapter_evil.compute_and_set_returns(transitions_evil, terminal_reward=r_evil, gamma=cfg.gamma)

        print(f"[Smoke] ep={ep} final_result={final_result} transitions_good={len(transitions_good)} transitions_evil={len(transitions_evil)}")

        m_good = adapter_good.update(
            transitions=transitions_good,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            batch_size=args.batch_size,
        )
        m_evil = adapter_evil.update(
            transitions=transitions_evil,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            batch_size=args.batch_size,
        )

        # 检查 phase 覆盖：由于上面 record=True 覆盖双方，应该能覆盖 5 phase
        phase_set = {tr["phase_id"] for tr in transitions_good + transitions_evil}
        print(f"[Smoke] phases_covered={sorted(list(phase_set))} metrics_good={m_good} metrics_evil={m_evil}")

    print("[Smoke] finished.")


if __name__ == "__main__":
    main()

