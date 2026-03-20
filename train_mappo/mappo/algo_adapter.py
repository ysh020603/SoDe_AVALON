import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from SoDe_Avalon_3.train_mappo.state.action_space import (
    compute_assassination_log_prob_entropy,
    compute_discussion_log_prob_entropy,
    compute_mission_log_prob_entropy,
    compute_proposal_log_prob_entropy,
    compute_voting_log_prob_entropy,
)
from SoDe_Avalon_3.train_mappo.state.token_builder import (
    DecisionLabels,
    PHASE_ASSASSINATION,
    PHASE_DISCUSSION,
    PHASE_MISSION,
    PHASE_PROPOSING,
    PHASE_VOTING,
)


class MAPPOCTDEAdapter:
    """
    以“共享 Actor + 集中 Critic”方式实现 MAPPO/CTDE 的核心 PPO 更新。

    由于当前环境是“轮到哪个 agent 决策才推进”，我们将一次 rollout 中所有受控侧的决策点视为同一轨迹时间序列，
    在 CTDE 下用集中 critic 的 V(s) 做 advantage baseline。
    """

    def __init__(
        self,
        *,
        actor,
        critic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        cfg: Any,
        side: str,
        device: torch.device,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.cfg = cfg
        self.side = side
        self.device = device

        self.gamma = float(cfg.gamma)
        self.clip_ratio = float(cfg.clip_ratio)
        self.value_coef = float(cfg.value_coef)
        self.entropy_coef = float(cfg.entropy_coef)
        self.grad_clip = float(cfg.grad_clip)

    def compute_and_set_returns(
        self,
        transitions: List[Dict[str, Any]],
        *,
        terminal_reward: float,
        gamma: float,
    ) -> None:
        """
        稀疏终局 reward：returns = gamma^(T-1-t) * R_terminal
        advantages = returns - old_value
        """
        if not transitions:
            return
        T = len(transitions)
        for t in range(T):
            returns = (gamma ** (T - 1 - t)) * float(terminal_reward)
            old_value = float(transitions[t]["old_value"])
            transitions[t]["returns"] = float(returns)
            transitions[t]["advantages"] = float(returns - old_value)

    def _stack_obs_state(self, transitions: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
        # 由于每个决策点 token 序列长度可能不同，我们在 stack 前做 padding。
        obs_keys = set()
        state_keys = set()
        for idx in indices:
            obs_keys |= set(transitions[idx]["obs_tensors"].keys())
            state_keys |= set(transitions[idx]["state_tensors"].keys())
        obs_keys = sorted(obs_keys)
        state_keys = sorted(state_keys)

        def _pad_and_stack_list(ts: List[torch.Tensor], *, pad_value: float, pad_mask_value: Optional[bool] = None) -> torch.Tensor:
            # 每个 tensor 形状通常以 batch 维开头：(1, L, ...) 或 (1, ...)
            # 返回：(B, Lmax, ...)
            squeezed = [t.squeeze(0) for t in ts]  # remove leading batch dim
            # 计算 Lmax（对 dim>=1 且存在可变序列长度的张量）
            max_len = 0
            for x in squeezed:
                if x.dim() >= 1:
                    max_len = max(max_len, int(x.shape[0]))
            padded_items: List[torch.Tensor] = []
            for x in squeezed:
                cur_len = int(x.shape[0]) if x.dim() >= 1 else 0
                if x.dim() == 0:
                    padded_items.append(x.unsqueeze(0))
                    continue
                if cur_len == max_len:
                    padded_items.append(x)
                    continue
                pad_width = [max_len - cur_len]
                # torch.nn.functional.pad expects ... for last dims; easier: create new tensor
                new_shape = (max_len,) + tuple(x.shape[1:])
                new_tensor = torch.empty(new_shape, dtype=x.dtype)
                # padding mask key 使用 True，其他使用 pad_value
                if pad_mask_value is not None:
                    fill_val = bool(pad_mask_value)
                else:
                    fill_val = pad_value
                new_tensor.fill_(fill_val)
                new_tensor[:cur_len, ...] = x
                padded_items.append(new_tensor)
            return torch.stack(padded_items, dim=0)

        batch_obs: Dict[str, Any] = {"labels_list": []}
        batch_state: Dict[str, Any] = {}

        for k in obs_keys:
            ts = [transitions[idx]["obs_tensors"][k] for idx in indices]
            # padding_mask 的 padding 值是 True，否则 padding 的位置会被当成有效 token
            if k == "padding_mask":
                batch_obs[k] = _pad_and_stack_list(ts, pad_value=0.0, pad_mask_value=True).to(self.device)
            else:
                # 大多数离散/float 张量 padding 用 0
                batch_obs[k] = _pad_and_stack_list(ts, pad_value=0.0).to(self.device)

        for k in state_keys:
            ts = [transitions[idx]["state_tensors"][k] for idx in indices]
            if k == "padding_mask":
                batch_state[k] = _pad_and_stack_list(ts, pad_value=0.0, pad_mask_value=True).to(self.device)
            else:
                batch_state[k] = _pad_and_stack_list(ts, pad_value=0.0).to(self.device)

        for idx in indices:
            batch_obs["labels_list"].append(DecisionLabels(phase_id=int(transitions[idx]["phase_id"])))

        return [batch_obs, batch_state]

    def _compute_log_probs_for_minibatch(
        self,
        *,
        outputs_list: List[Dict[str, torch.Tensor]],
        transitions: List[Dict[str, Any]],
        indices: List[int],
    ) -> torch.Tensor:
        # new_log_prob: (B,)
        bsz = len(indices)
        new_logps = torch.zeros((bsz,), device=self.device, dtype=torch.float32)
        entropies = torch.zeros((bsz,), device=self.device, dtype=torch.float32)

        for j, idx in enumerate(indices):
            phase_id = int(transitions[idx]["phase_id"])
            action = transitions[idx]["action"]
            out = outputs_list[j]

            if phase_id == PHASE_DISCUSSION:
                # 注意：out head logits shape 通常为 (1, C)
                logp, ent = compute_discussion_log_prob_entropy(
                    identity_logits=out["identity_logits"].squeeze(0),
                    intent_logits=out["intent_logits"].squeeze(0),
                    attitude_logits=out["attitude_logits"].squeeze(0),
                    action=action,
                )
            elif phase_id == PHASE_PROPOSING:
                logp, ent = compute_proposal_log_prob_entropy(
                    propose_logits=out["propose_logits"].squeeze(0),
                    action=action,
                )
            elif phase_id == PHASE_VOTING:
                logp, ent = compute_voting_log_prob_entropy(
                    vote_logits=out["vote_logits"].squeeze(0),
                    action=action,
                )
            elif phase_id == PHASE_MISSION:
                logp, ent = compute_mission_log_prob_entropy(
                    mission_logits=out["mission_logits"].squeeze(0),
                    action=action,
                )
            elif phase_id == PHASE_ASSASSINATION:
                logp, ent = compute_assassination_log_prob_entropy(
                    kill_logits=out["kill_logits"].squeeze(0),
                    action=action,
                )
            else:
                raise ValueError(f"Unknown phase_id={phase_id}")

            new_logps[j] = logp
            entropies[j] = ent

        # 为方便上层使用，把 entropy 存起来
        self._last_entropy = entropies
        return new_logps

    def update(
        self,
        *,
        transitions: List[Dict[str, Any]],
        ppo_epochs: int,
        minibatch_size: int,
        batch_size: int,
    ) -> Dict[str, float]:
        if not transitions:
            return {}

        # 只用 precomputed returns/advantages
        N = len(transitions)
        old_logps = torch.tensor([t["old_log_prob"] for t in transitions], device=self.device, dtype=torch.float32)
        returns = torch.tensor([t["returns"] for t in transitions], device=self.device, dtype=torch.float32)
        advantages = torch.tensor([t["advantages"] for t in transitions], device=self.device, dtype=torch.float32)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        total_loss_acc = 0.0
        clip_frac_acc = 0.0
        approx_kl_acc = 0.0
        steps = 0

        indices_all = list(range(N))

        # 训练模式
        self.actor.train()
        self.critic.train()

        for epoch in range(int(ppo_epochs)):
            random.shuffle(indices_all)
            # mini-batch
            for start in range(0, N, int(minibatch_size)):
                idxs = indices_all[start : start + int(minibatch_size)]
                bsz = len(idxs)
                if bsz == 0:
                    continue

                batch_obs, batch_state = self._stack_obs_state(transitions, idxs)

                # forward：actor/critic
                outputs_list = self.actor.model(batch_obs)
                values = self.critic(batch_state)  # (B,)

                # new log_prob
                new_logps = self._compute_log_probs_for_minibatch(
                    outputs_list=outputs_list,
                    transitions=transitions,
                    indices=idxs,
                )
                entropies = getattr(self, "_last_entropy")

                old_logp_b = old_logps[idxs]
                adv_b = advantages[idxs]
                ret_b = returns[idxs]

                ratios = torch.exp(new_logps - old_logp_b)
                clipped = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -torch.min(ratios * adv_b, clipped * adv_b).mean()

                value_loss = torch.mean((values - ret_b) ** 2)
                entropy_mean = entropies.mean()

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                # backward
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # metrics
                with torch.no_grad():
                    clip_frac = float(((torch.abs(ratios - 1.0) > self.clip_ratio)).float().mean().item())
                    approx_kl = float((old_logp_b - new_logps).mean().item())

                    policy_loss_acc += float(policy_loss.item())
                    value_loss_acc += float(value_loss.item())
                    entropy_acc += float(entropy_mean.item())
                    total_loss_acc += float(total_loss.item())
                    clip_frac_acc += clip_frac
                    approx_kl_acc += approx_kl
                    steps += 1

        if steps == 0:
            return {}
        return {
            "policy_loss": policy_loss_acc / steps,
            "value_loss": value_loss_acc / steps,
            "entropy": entropy_acc / steps,
            "total_loss": total_loss_acc / steps,
            "clip_fraction": clip_frac_acc / steps,
            "approx_kl": approx_kl_acc / steps,
            "n_transitions": float(N),
        }

