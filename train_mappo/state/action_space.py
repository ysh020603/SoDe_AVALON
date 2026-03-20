import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _clamp_probs(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(p, eps, 1.0 - eps)


def _categorical_log_prob_from_logits(logits: torch.Tensor, action_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      - log_prob: scalar tensor
      - entropy: scalar tensor
    """
    # logits: (C,)
    dist = torch.distributions.Categorical(logits=logits)
    log_prob = dist.log_prob(torch.tensor(action_index, device=logits.device))
    entropy = dist.entropy()
    return log_prob, entropy


def _bernoulli_log_prob_and_entropy(logits: torch.Tensor, action01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (N,) ; action01: (N,) 取 0/1
    """
    p = torch.sigmoid(logits)
    p = _clamp_probs(p)
    # log Bernoulli
    log_prob = torch.where(action01 > 0.5, torch.log(p), torch.log(1.0 - p)).sum()
    entropy = (-p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)).sum()
    return log_prob, entropy


def sample_discussion_action(
    *,
    identity_logits: torch.Tensor,
    intent_logits: torch.Tensor,
    attitude_logits: torch.Tensor,
    rng: Optional[random.Random] = None,
    num_players: int = 7,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """
    actor heads:
      - identity_logits: (10,)
      - intent_logits: (5,)
      - attitude_logits: (MAX_PLAYERS, 3) or (10,3)
    动作：
      - identity_label: int
      - intent_multi_hot: List[int] (5,)
      - attitude_labels: List[int] (num_players,) values in {0,1,2}
    """
    device = identity_logits.device
    # identity
    dist_id = torch.distributions.Categorical(logits=identity_logits)
    identity_label = int(dist_id.sample().item())
    logp_id = dist_id.log_prob(torch.tensor(identity_label, device=device))
    ent_id = dist_id.entropy()

    # intent (multi-binary Bernoulli)
    dist_int = torch.distributions.Bernoulli(probs=torch.sigmoid(intent_logits))
    intent_sample = dist_int.sample()
    # Bernoulli.sample() 返回 float tensor，.tolist() 后元素是 Python float
    intent_multi_hot = [int(v) for v in intent_sample.detach().cpu().tolist()]
    logp_int, ent_int = _bernoulli_log_prob_and_entropy(intent_logits, intent_sample)

    # attitude: categorical per-player (none/pos/neg)
    # attitude_logits: (MAX_PLAYERS, 3) -> take first num_players
    att_logits = attitude_logits[:num_players, :]
    # sample each row independently
    logps: List[torch.Tensor] = []
    ents: List[torch.Tensor] = []
    attitude_labels: List[int] = []
    for j in range(num_players):
        dist_j = torch.distributions.Categorical(logits=att_logits[j])
        a = int(dist_j.sample().item())
        attitude_labels.append(a)
        logps.append(dist_j.log_prob(torch.tensor(a, device=device)))
        ents.append(dist_j.entropy())
    logp_att = torch.stack(logps).sum()
    ent_att = torch.stack(ents).sum()

    action = {
        "identity_label": identity_label,
        "intent_multi_hot": intent_multi_hot,
        "attitude_labels": attitude_labels,
    }
    logp = logp_id + logp_int + logp_att
    entropy = ent_id + ent_int + ent_att
    return action, logp, entropy


def compute_discussion_log_prob_entropy(
    *,
    identity_logits: torch.Tensor,
    intent_logits: torch.Tensor,
    attitude_logits: torch.Tensor,
    action: Dict[str, Any],
    num_players: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = identity_logits.device
    identity_label = int(action["identity_label"])
    logp_id, ent_id = _categorical_log_prob_from_logits(identity_logits, identity_label)

    intent_multi_hot = action["intent_multi_hot"]
    intent_action = torch.tensor(intent_multi_hot, dtype=torch.float32, device=device)
    logp_int, ent_int = _bernoulli_log_prob_and_entropy(intent_logits, intent_action)

    att_logits = attitude_logits[:num_players, :]
    attitude_labels = action["attitude_labels"]
    logps: List[torch.Tensor] = []
    ents: List[torch.Tensor] = []
    for j in range(num_players):
        a = int(attitude_labels[j])
        logp_j, ent_j = _categorical_log_prob_from_logits(att_logits[j], a)
        logps.append(logp_j)
        ents.append(ent_j)
    logp_att = torch.stack(logps).sum()
    ent_att = torch.stack(ents).sum()

    return logp_id + logp_int + logp_att, ent_id + ent_int + ent_att


def sample_proposal_action(
    *,
    propose_logits: torch.Tensor,
    team_size: int,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """
    propose_logits: (7,) 对应玩家 id 1..7
    动作：
      - team_order: list[int] 玩家 id，长度=team_size（按无放回采样顺序）
    """
    device = propose_logits.device
    logits = propose_logits
    remaining: List[int] = list(range(7))
    chosen_order: List[int] = []

    logp_total = torch.zeros((), device=device, dtype=logits.dtype)
    ent_total = torch.zeros((), device=device, dtype=logits.dtype)

    for _ in range(int(team_size)):
        # softmax over remaining
        rem_logits = logits[remaining]
        dist = torch.distributions.Categorical(logits=rem_logits)
        idx_in_rem = int(dist.sample().item())
        chosen_idx = remaining[idx_in_rem]
        chosen_order.append(chosen_idx + 1)  # player id is idx+1

        logp_total = logp_total + dist.log_prob(torch.tensor(idx_in_rem, device=device))
        ent_total = ent_total + dist.entropy()

        remaining.pop(idx_in_rem)

    action = {"team_order": chosen_order, "team_size": int(team_size)}
    return action, logp_total, ent_total


def compute_proposal_log_prob_entropy(
    *,
    propose_logits: torch.Tensor,
    action: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = propose_logits.device
    logits = propose_logits
    team_order: List[int] = list(action["team_order"])

    # 顺序严格对应 rollout 时的无放回采样顺序
    remaining: List[int] = list(range(7))
    logp_total = torch.zeros((), device=device, dtype=logits.dtype)
    ent_total = torch.zeros((), device=device, dtype=logits.dtype)

    for pid in team_order:
        chosen_idx = int(pid) - 1
        # compute distribution on remaining
        rem_logits = logits[remaining]
        dist = torch.distributions.Categorical(logits=rem_logits)
        idx_in_rem = remaining.index(chosen_idx)
        logp_total = logp_total + dist.log_prob(torch.tensor(idx_in_rem, device=device))
        ent_total = ent_total + dist.entropy()
        remaining.pop(idx_in_rem)

    return logp_total, ent_total


def sample_voting_action(
    *,
    vote_logits: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """
    vote_logits: (2,) -> category 0/1 ; 环境里 {vote:true} 要求 Approve
    我们约定：category==1 对应 vote=True（Approve）
    """
    device = vote_logits.device
    dist = torch.distributions.Categorical(logits=vote_logits)
    cat = int(dist.sample().item())
    vote_bool = bool(cat == 1)
    logp = dist.log_prob(torch.tensor(cat, device=device))
    ent = dist.entropy()
    return {"vote": vote_bool, "vote_cat": cat}, logp, ent


def compute_voting_log_prob_entropy(
    *,
    vote_logits: torch.Tensor,
    action: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = vote_logits.device
    cat = int(action.get("vote_cat", 1 if action["vote"] else 0))
    dist = torch.distributions.Categorical(logits=vote_logits)
    return dist.log_prob(torch.tensor(cat, device=device)), dist.entropy()


def sample_mission_action(
    *,
    mission_logits: torch.Tensor,
    force_success: bool,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """
    mission_logits: (2,) -> category 0 表示 Success(True), category 1 表示 Fail(False)
    force_success=true 时环境会强制 Good success=True，我们也对齐动作=Success。
    """
    device = mission_logits.device
    dist = torch.distributions.Categorical(logits=mission_logits)
    if force_success:
        cat = 0
    else:
        cat = int(dist.sample().item())
    success = bool(cat == 0)
    logp = dist.log_prob(torch.tensor(cat, device=device))
    ent = dist.entropy()
    return {"success": success, "mission_cat": cat}, logp, ent


def compute_mission_log_prob_entropy(
    *,
    mission_logits: torch.Tensor,
    action: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = mission_logits.device
    cat = int(action.get("mission_cat", 0 if action["success"] else 1))
    dist = torch.distributions.Categorical(logits=mission_logits)
    return dist.log_prob(torch.tensor(cat, device=device)), dist.entropy()


def _mask_logits_for_candidates(kill_logits: torch.Tensor, valid_target_ids: Sequence[int]) -> torch.Tensor:
    """
    kill_logits: (MAX_PLAYERS,) 对应 index i -> player_id = i+1
    valid_target_ids: e.g. [1,3,4]
    """
    logits = kill_logits.clone()
    valid_set = {int(x) for x in valid_target_ids}
    for i in range(logits.shape[0]):
        pid = i + 1
        if pid not in valid_set:
            logits[i] = -1e9
    return logits


def sample_assassination_action(
    *,
    kill_logits: torch.Tensor,
    valid_target_ids: Sequence[int],
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    device = kill_logits.device
    masked = _mask_logits_for_candidates(kill_logits, valid_target_ids)
    dist = torch.distributions.Categorical(logits=masked)
    cat = int(dist.sample().item())
    target_player_id = cat + 1
    logp = dist.log_prob(torch.tensor(cat, device=device))
    ent = dist.entropy()
    return {"target_player_id": int(target_player_id), "kill_cat": cat, "valid_target_ids": list(valid_target_ids)}, logp, ent


def compute_assassination_log_prob_entropy(
    *,
    kill_logits: torch.Tensor,
    action: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = kill_logits.device
    valid_target_ids: List[int] = list(action["valid_target_ids"])
    target_player_id = int(action["target_player_id"])
    cat = target_player_id - 1
    masked = _mask_logits_for_candidates(kill_logits, valid_target_ids)
    dist = torch.distributions.Categorical(logits=masked)
    return dist.log_prob(torch.tensor(cat, device=device)), dist.entropy()

