import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# =========================
# 常量：与现有训练/推理代码保持一致
# =========================
PHASE_DISCUSSION = 0
PHASE_PROPOSING = 1
PHASE_VOTING = 2
PHASE_MISSION = 3
PHASE_ASSASSINATION = 4

TOKEN_TYPE_CTX = 0
TOKEN_TYPE_SPEECH = 1
TOKEN_TYPE_PROPOSE = 2
TOKEN_TYPE_VOTE = 3
TOKEN_TYPE_MISSION = 4
TOKEN_TYPE_ASSASSIN = 5  # 当前训练构造里未直接用到
TOKEN_TYPE_TASK = 6

MAX_PLAYERS = 10

PHASE_STR_TO_ID: Dict[str, int] = {
    "discussion": PHASE_DISCUSSION,
    "proposal": PHASE_PROPOSING,
    "voting": PHASE_VOTING,
    "mission": PHASE_MISSION,
    "assassination": PHASE_ASSASSINATION,
}


def get_self_identity_id(role: str) -> int:
    mapping = {
        "Merlin": 0,
        "Percival": 1,
        "Loyal Servant": 2,
        "Morgana": 3,
        "Assassin": 4,
        "Oberon": 5,
    }
    return int(mapping.get(role, 0))


def make_meta_tuple(success_count: int, fail_count: int, attempt_index: int) -> Tuple[int, int, int, int, int]:
    # mission size 规则与其它代码一致（5 轮）
    mission_size_rules = [2, 3, 3, -4, 4]
    round_idx = min(success_count + fail_count, len(mission_size_rules) - 1)
    rule = mission_size_rules[round_idx]
    mission_required_size = abs(rule)
    bad_vote_tolerance = 1 if rule < 0 else 0
    return (
        int(success_count),
        int(fail_count),
        int(attempt_index),
        int(mission_required_size),
        int(bad_vote_tolerance),
    )


@dataclass
class DecisionLabels:
    phase_id: int
    identity_label: int = -1
    intent_multi_hot: Optional[torch.Tensor] = None
    attitude_labels: Optional[torch.Tensor] = None
    propose_targets: Optional[torch.Tensor] = None
    vote_label: int = -1
    mission_label: int = -1
    kill_target: int = -1


@dataclass
class TokenStruct:
    token_type: int
    actor_id: int
    round_index: int
    attempt_index: int
    meta_tuple: Tuple[int, int, int, int, int]
    body: Dict[str, Any]


def _normalize_players_info(players_config: Any) -> Dict[int, Dict[str, Any]]:
    players_info: Dict[int, Dict[str, Any]] = {}
    if not isinstance(players_config, dict):
        return players_info
    for pid_k, info in players_config.items():
        try:
            pid_int = int(pid_k)
        except Exception:
            continue
        if isinstance(info, dict):
            players_info[pid_int] = {
                "role": info.get("role", "unknown"),
                "model": info.get("model", "unknown"),
            }
    return players_info


def _parse_phase(phase: Union[str, int]) -> int:
    if isinstance(phase, int):
        if 0 <= phase <= 4:
            return phase
        raise ValueError(f"phase 整数必须在 0..4 之间，当前为 {phase}")
    if isinstance(phase, str):
        s = phase.strip().lower()
        if s in PHASE_STR_TO_ID:
            return PHASE_STR_TO_ID[s]
        raise ValueError(f"phase 字符串必须为 {list(PHASE_STR_TO_ID.keys())} 之一，当前为 '{phase}'")
    raise ValueError(f"phase 必须为 str 或 int，当前类型为 {type(phase)}")


def _build_kg_states_local(players_info: Dict[int, Dict[str, Any]], my_id: int) -> List[int]:
    """Actor local obs：战争迷雾/可见性遮罩版（与现有 SmallModel 实现一致）。"""

    def is_bad(role: str) -> bool:
        return role in {"Assassin", "Morgana", "Mordred", "Oberon", "Minion"}

    def is_good(role: str) -> bool:
        return role in {"Percival", "Loyal Servant", "Merlin"}

    my_role = (players_info.get(my_id) or {}).get("role", "unknown")
    states: List[int] = [get_self_identity_id(my_role)]

    for pid in range(1, 8):
        info = players_info.get(pid, {}) or {}
        role = info.get("role", "unknown")

        if pid == my_id:
            if is_good(my_role):
                states.append(1)
            elif is_bad(my_role):
                states.append(2)
            else:
                states.append(0)
            continue

        # 7 人训练：可能出现的角色集合里，这些规则足够稳定
        if my_role == "Merlin":
            states.append(2 if is_bad(role) else 0)
        elif my_role == "Percival":
            states.append(3 if role in {"Merlin", "Morgana"} else 0)
        elif my_role in {"Assassin", "Morgana", "Mordred", "Minion"}:
            # Oberon 例外：好人玩家不会看到自己身份
            states.append(2 if (is_bad(role) and role != "Oberon") else 0)
        elif my_role == "Oberon":
            states.append(0)
        else:
            states.append(0)

    while len(states) < 8:
        states.append(0)
    return states[:8]


def _build_kg_states_global(players_info: Dict[int, Dict[str, Any]], my_id: int) -> List[int]:
    """Critic global state：无遮罩，使用真实身份桶（7 人局角色集合足够覆盖 0..5）。"""
    my_role = (players_info.get(my_id) or {}).get("role", "unknown")
    states: List[int] = [get_self_identity_id(my_role)]
    for pid in range(1, 8):
        role = (players_info.get(pid) or {}).get("role", "unknown")
        states.append(get_self_identity_id(role))
    while len(states) < 8:
        states.append(0)
    return states[:8]


def _ctx_token(scoreboard: Tuple[int, int, int, int, int], phase_id: int, player: int, players_info: Dict[int, Dict[str, Any]]) -> TokenStruct:
    my_role_name = (players_info.get(player) or {}).get("role", "unknown")
    return TokenStruct(
        token_type=TOKEN_TYPE_CTX,
        actor_id=int(player),
        round_index=0,
        attempt_index=0,
        meta_tuple=scoreboard,
        body={
            "phase": int(phase_id),
            "players_info": players_info,
            "my_id": int(player),
            "my_role_id": get_self_identity_id(my_role_name),
        },
    )


def _build_history_for_decision(history_tokens: List[TokenStruct], phase_id: int, round_number: int, attempt_index: int) -> List[TokenStruct]:
    # DISCUSSION：不包含当前 attempt 内已发生的发言（由调用方分别补入）
    if phase_id == PHASE_DISCUSSION:
        return [
            t
            for t in history_tokens
            if not (t.token_type == TOKEN_TYPE_SPEECH and t.round_index == round_number and t.attempt_index == attempt_index)
        ]
    return list(history_tokens)


def build_decision_tokens_from_log(
    *,
    log: Dict[str, Any],
    phase: Union[str, int],
    player: int,
    round_number: int,
    attempt_index: int,
) -> Tuple[List[TokenStruct], DecisionLabels, Dict[str, Any]]:
    """
    从 game_condition dict 构造单个决策点 token 序列。
    逻辑尽量对齐 `SoDe_Avalon_3/small_model/avalon_infer_api.py` 的实现，
    以保证在线 rollout 与训练/推理特征一致。
    """
    phase_id = _parse_phase(phase)
    player = int(player)
    round_number = int(round_number)
    attempt_index = int(attempt_index)

    if not (1 <= player <= 7):
        raise ValueError(f"player 必须在 1..7 之间，当前为 {player}")

    players_info = _normalize_players_info(log.get("players_config", {}))
    if not players_info:
        raise ValueError("log 中缺少 players_config 或 players_config 为空")
    if player not in players_info:
        raise ValueError(f"player {player} 不在 players_config 中，当前玩家: {list(players_info.keys())}")

    game_timeline: List[Dict[str, Any]] = log.get("game_timeline", log.get("timeline_rounds", [])) or []
    if not game_timeline:
        raise ValueError("log 中缺少 game_timeline 或 game_timeline 为空")

    round_numbers = [int(r.get("round_number", 0)) for r in game_timeline]
    if round_number not in round_numbers:
        raise ValueError(f"round_number {round_number} 不在 game_timeline 中，当前轮次: {round_numbers}")

    success_count = 0
    fail_count = 0
    history_tokens: List[TokenStruct] = []

    def _make_scoreboard() -> Tuple[int, int, int, int, int]:
        return make_meta_tuple(success_count, fail_count, attempt_index)

    for round_data in game_timeline:
        rn = int(round_data.get("round_number", 0))
        attempts = round_data.get("attempts", []) or []

        # 判断 attempt_index 合法性（用于讨论/提议/投票/任务等 token 的选择器）
        if rn == round_number:
            attempt_indices = [int(a.get("attempt_index", 1)) for a in attempts]
            if attempt_index not in attempt_indices:
                raise ValueError(f"attempt_index {attempt_index} 不在 round {round_number} 的 attempts 中，当前: {attempt_indices}")

        for attempt in attempts:
            ai = int(attempt.get("attempt_index", 1))
            steps = attempt.get("steps", {}) or {}
            scoreboard = make_meta_tuple(success_count, fail_count, ai)

            # ========== DISCUSSION ==========
            discussions: List[Dict[str, Any]] = steps.get("discussion", []) or []
            history_discussions: List[Dict[str, Any]] = []
            for disc in discussions:
                agent_info = disc.get("agent_info", {}) or {}
                aid = int(agent_info.get("id", -1))
                answer = disc.get("answer", {}) or {}
                text_raw = answer.get("extracted_result", "")
                if not isinstance(text_raw, str):
                    text_raw = ""

                if phase_id == PHASE_DISCUSSION and rn == round_number and ai == attempt_index and aid == player:
                    global_hist = _build_history_for_decision(history_tokens, PHASE_DISCUSSION, rn, ai)
                    tokens: List[TokenStruct] = [_ctx_token(scoreboard, PHASE_DISCUSSION, player, players_info)]
                    tokens.extend(global_hist)

                    # 将当前讨论轮次中“之前的发言”补入后半段（prev -> SPEECH）
                    for prev in history_discussions:
                        prev_info = prev.get("agent_info", {}) or {}
                        prev_aid = int(prev_info.get("id", -1))
                        prev_answer = prev.get("answer", {}) or {}
                        prev_text = prev_answer.get("extracted_result", "")
                        tokens.append(
                            TokenStruct(
                                token_type=TOKEN_TYPE_SPEECH,
                                actor_id=prev_aid,
                                round_index=rn,
                                attempt_index=ai,
                                meta_tuple=scoreboard,
                                body={"text": prev_text if isinstance(prev_text, str) else ""},
                            )
                        )

                    tokens.append(
                        TokenStruct(
                            token_type=TOKEN_TYPE_TASK,
                            actor_id=int(player),
                            round_index=rn,
                            attempt_index=ai,
                            meta_tuple=scoreboard,
                            body={"phase": PHASE_DISCUSSION},
                        )
                    )

                    labels = DecisionLabels(phase_id=PHASE_DISCUSSION)
                    extra = {
                        "players_info": players_info,
                        "model_name": (players_info.get(int(player)) or {}).get("model"),
                        "phase_id": PHASE_DISCUSSION,
                        "role": (players_info.get(int(player)) or {}).get("role"),
                    }
                    _validate_token_sequence(tokens, phase_id, player)
                    return tokens, labels, extra

                history_discussions.append(disc)
                history_tokens.append(
                    TokenStruct(
                        token_type=TOKEN_TYPE_SPEECH,
                        actor_id=aid,
                        round_index=rn,
                        attempt_index=ai,
                        meta_tuple=scoreboard,
                        body={"text": text_raw},
                    )
                )

            # 讨论轮次中“当前玩家的发言缺失”：退回到“已经发生的历史”+ TASK
            if phase_id == PHASE_DISCUSSION and rn == round_number and ai == attempt_index:
                found_in_discussion = any(int(d.get("agent_info", {}).get("id", -1)) == player for d in discussions)
                if not found_in_discussion:
                    global_hist = _build_history_for_decision(history_tokens, PHASE_DISCUSSION, rn, ai)
                    tokens = [_ctx_token(scoreboard, PHASE_DISCUSSION, player, players_info)]
                    tokens.extend(global_hist)

                    for prev in history_discussions:
                        prev_info = prev.get("agent_info", {}) or {}
                        prev_aid = int(prev_info.get("id", -1))
                        prev_answer = prev.get("answer", {}) or {}
                        prev_text = prev_answer.get("extracted_result", "")
                        tokens.append(
                            TokenStruct(
                                token_type=TOKEN_TYPE_SPEECH,
                                actor_id=prev_aid,
                                round_index=rn,
                                attempt_index=ai,
                                meta_tuple=scoreboard,
                                body={"text": prev_text if isinstance(prev_text, str) else ""},
                            )
                        )

                    tokens.append(
                        TokenStruct(
                            token_type=TOKEN_TYPE_TASK,
                            actor_id=int(player),
                            round_index=rn,
                            attempt_index=ai,
                            meta_tuple=scoreboard,
                            body={"phase": PHASE_DISCUSSION},
                        )
                    )
                    labels = DecisionLabels(phase_id=PHASE_DISCUSSION)
                    extra = {
                        "players_info": players_info,
                        "model_name": (players_info.get(int(player)) or {}).get("model"),
                        "phase_id": PHASE_DISCUSSION,
                        "role": (players_info.get(int(player)) or {}).get("role"),
                    }
                    _validate_token_sequence(tokens, phase_id, player)
                    return tokens, labels, extra

            # ========== PROPOSING ==========
            proposal = steps.get("proposal")
            leader_id = int(attempt.get("leader_id", -1))
            if proposal:
                prop_agent = proposal.get("agent_info", {}) or {}
                leader_id = int(prop_agent.get("id", leader_id))
            if leader_id <= 0:
                leader_id = int(attempt.get("leader_id", -1))

            if phase_id == PHASE_PROPOSING and rn == round_number and ai == attempt_index and leader_id == player:
                global_hist = _build_history_for_decision(history_tokens, PHASE_PROPOSING, rn, ai)
                tokens: List[TokenStruct] = [_ctx_token(scoreboard, PHASE_PROPOSING, player, players_info)]
                tokens.extend(global_hist)
                tokens.append(
                    TokenStruct(
                        token_type=TOKEN_TYPE_TASK,
                        actor_id=int(player),
                        round_index=rn,
                        attempt_index=ai,
                        meta_tuple=scoreboard,
                        body={"phase": PHASE_PROPOSING},
                    )
                )
                labels = DecisionLabels(phase_id=PHASE_PROPOSING)
                extra = {
                    "players_info": players_info,
                    "model_name": (players_info.get(int(player)) or {}).get("model"),
                    "phase_id": PHASE_PROPOSING,
                    "role": (players_info.get(int(player)) or {}).get("role"),
                }
                _validate_token_sequence(tokens, phase_id, player)
                return tokens, labels, extra

            if proposal:
                team_list = (proposal.get("answer", {}) or {}).get("extracted_result", [])
                team_mask = [0.0] * 7
                if isinstance(team_list, list):
                    for pid in team_list:
                        if isinstance(pid, int) and 1 <= pid <= 7:
                            team_mask[pid - 1] = 1.0
                history_tokens.append(
                    TokenStruct(
                        token_type=TOKEN_TYPE_PROPOSE,
                        actor_id=leader_id,
                        round_index=rn,
                        attempt_index=ai,
                        meta_tuple=scoreboard,
                        body={"team_mask": team_mask},
                    )
                )

            # ========== VOTING ==========
            voting = steps.get("voting", {}) or {}
            votes_details: List[Dict[str, Any]] = voting.get("votes_details", []) or []
            if phase_id == PHASE_VOTING and rn == round_number and ai == attempt_index and not proposal:
                raise ValueError("phase voting 需要 steps.proposal 存在以构造 token 序列")

            if votes_details and proposal:
                aggregated_vote_vec = [0.0] * 7
                for vote in votes_details:
                    v_agent = vote.get("agent_info", {}) or {}
                    voter_id = int(v_agent.get("id", -1))
                    v_answer = vote.get("answer", {}) or {}
                    vote_res = v_answer.get("extracted_result", None)
                    if isinstance(vote_res, bool) and 1 <= voter_id <= 7:
                        aggregated_vote_vec[voter_id - 1] = 1.0 if vote_res else 0.0

                    if phase_id == PHASE_VOTING and rn == round_number and ai == attempt_index and voter_id == player:
                        global_hist = _build_history_for_decision(history_tokens, PHASE_VOTING, rn, ai)
                        tokens = [_ctx_token(scoreboard, PHASE_VOTING, player, players_info)]
                        tokens.extend(global_hist)
                        tokens.append(
                            TokenStruct(
                                token_type=TOKEN_TYPE_TASK,
                                actor_id=int(player),
                                round_index=rn,
                                attempt_index=ai,
                                meta_tuple=scoreboard,
                                body={"phase": PHASE_VOTING},
                            )
                        )
                        labels = DecisionLabels(phase_id=PHASE_VOTING)
                        extra = {
                            "players_info": players_info,
                            "model_name": (players_info.get(int(player)) or {}).get("model"),
                            "phase_id": PHASE_VOTING,
                            "role": (players_info.get(int(player)) or {}).get("role"),
                        }
                        _validate_token_sequence(tokens, phase_id, player)
                        return tokens, labels, extra

                voter_ids = [int(v.get("agent_info", {}).get("id", -1)) for v in votes_details]
                if phase_id == PHASE_VOTING and rn == round_number and ai == attempt_index and player not in voter_ids:
                    global_hist = _build_history_for_decision(history_tokens, PHASE_VOTING, rn, ai)
                    tokens = [_ctx_token(scoreboard, PHASE_VOTING, player, players_info)]
                    tokens.extend(global_hist)
                    tokens.append(
                        TokenStruct(
                            token_type=TOKEN_TYPE_TASK,
                            actor_id=int(player),
                            round_index=rn,
                            attempt_index=ai,
                            meta_tuple=scoreboard,
                            body={"phase": PHASE_VOTING},
                        )
                    )
                    labels = DecisionLabels(phase_id=PHASE_VOTING)
                    extra = {
                        "players_info": players_info,
                        "model_name": (players_info.get(int(player)) or {}).get("model"),
                        "phase_id": PHASE_VOTING,
                        "role": (players_info.get(int(player)) or {}).get("role"),
                    }
                    _validate_token_sequence(tokens, phase_id, player)
                    return tokens, labels, extra

                history_tokens.append(
                    TokenStruct(
                        token_type=TOKEN_TYPE_VOTE,
                        actor_id=0,
                        round_index=rn,
                        attempt_index=ai,
                        meta_tuple=scoreboard,
                        body={"vote_vec": aggregated_vote_vec},
                    )
                )

            # ========== MISSION（按 mission_result） ==========
            mission_result = round_data.get("mission_result") or {}
            outcome = mission_result.get("outcome")
            actions = mission_result.get("player_actions", mission_result.get("details", [])) or []

            if actions:
                mission_team_ids = [
                    int(a.get("agent_info", {}).get("id", -1))
                    for a in actions
                    if isinstance(a.get("agent_info"), dict)
                ]
                if phase_id == PHASE_MISSION and rn == round_number and player not in mission_team_ids:
                    raise ValueError(f"phase mission 需要 player {player} 在 mission_result.player_actions 中")

                good_cards = 0
                bad_cards = 0
                for act in actions:
                    a_answer_tmp = act.get("answer", {}) or {}
                    res_tmp = a_answer_tmp.get("extracted_result", "")
                    if res_tmp == "Success":
                        good_cards += 1
                    elif res_tmp == "Fail":
                        bad_cards += 1

                last_attempt_index_in_round = ai
                scoreboard_mission = make_meta_tuple(success_count, fail_count, int(last_attempt_index_in_round))

                for act in actions:
                    a_agent = act.get("agent_info", {}) or {}
                    pid = int(a_agent.get("id", -1))

                    if phase_id == PHASE_MISSION and rn == round_number and pid == player:
                        global_hist = _build_history_for_decision(history_tokens, PHASE_MISSION, rn, int(last_attempt_index_in_round))
                        tokens: List[TokenStruct] = [
                            TokenStruct(
                                token_type=TOKEN_TYPE_CTX,
                                actor_id=int(player),
                                round_index=0,
                                attempt_index=0,
                                meta_tuple=scoreboard_mission,
                                body={
                                    "phase": PHASE_MISSION,
                                    "players_info": players_info,
                                    "my_id": int(player),
                                    "my_role_id": get_self_identity_id((players_info.get(int(player)) or {}).get("role", "unknown")),
                                },
                            )
                        ]
                        tokens.extend(global_hist)
                        tokens.append(
                            TokenStruct(
                                token_type=TOKEN_TYPE_TASK,
                                actor_id=int(player),
                                round_index=rn,
                                attempt_index=int(last_attempt_index_in_round),
                                meta_tuple=scoreboard_mission,
                                body={"phase": PHASE_MISSION},
                            )
                        )
                        labels = DecisionLabels(phase_id=PHASE_MISSION)
                        extra = {
                            "players_info": players_info,
                            "model_name": (players_info.get(int(player)) or {}).get("model"),
                            "phase_id": PHASE_MISSION,
                            "role": (players_info.get(int(player)) or {}).get("role"),
                        }
                        _validate_token_sequence(tokens, phase_id, player)
                        return tokens, labels, extra

                history_tokens.append(
                    TokenStruct(
                        token_type=TOKEN_TYPE_MISSION,
                        actor_id=0,
                        round_index=rn,
                        attempt_index=int(last_attempt_index_in_round),
                        meta_tuple=scoreboard_mission,
                        body={"mission_vec": [float(good_cards), float(bad_cards)]},
                    )
                )

            if outcome == "good_point":
                success_count += 1
            elif outcome == "evil_point":
                fail_count += 1

    # ========== ASSASSINATION ==========
    assassination = log.get("assassination")
    if phase_id == PHASE_ASSASSINATION and not assassination:
        raise ValueError("phase assassination 需要 log 中存在 assassination 数据")

    if assassination:
        a_info = assassination.get("agent_info", {}) or {}
        assassin_id = int(a_info.get("id", -1))
        if phase_id == PHASE_ASSASSINATION and assassin_id == player:
            # 刺杀时：success_count/fail_count 只取已完成任务的计数（用于 meta）
            scoreboard = make_meta_tuple(success_count, fail_count, 0)
            good_roles = {"Merlin", "Percival", "Loyal Servant"}
            good_mask = [0.0] * 7
            for pid in range(1, 8):
                role = (players_info.get(pid) or {}).get("role", "")
                if role in good_roles:
                    good_mask[pid - 1] = 1.0

            global_hist = _build_history_for_decision(history_tokens, PHASE_ASSASSINATION, 0, 0)
            tokens: List[TokenStruct] = [_ctx_token(scoreboard, PHASE_ASSASSINATION, player, players_info)]
            tokens.extend(global_hist)
            tokens.append(
                TokenStruct(
                    token_type=TOKEN_TYPE_TASK,
                    actor_id=int(player),
                    round_index=0,
                    attempt_index=0,
                    meta_tuple=scoreboard,
                    body={"phase": PHASE_ASSASSINATION, "good_mask": good_mask},
                )
            )
            labels = DecisionLabels(phase_id=PHASE_ASSASSINATION)
            extra = {
                "players_info": players_info,
                "model_name": (players_info.get(int(player)) or {}).get("model"),
                "phase_id": PHASE_ASSASSINATION,
                "role": (players_info.get(int(player)) or {}).get("role"),
            }
            _validate_token_sequence(tokens, phase_id, player)
            return tokens, labels, extra

    raise ValueError(
        f"No decision matched selector (phase={phase_id}, player={player}, round_number={round_number}, attempt_index={attempt_index})."
    )


def _validate_token_sequence(tokens: List[TokenStruct], phase_id: int, player: int) -> None:
    if not tokens:
        raise ValueError("Token 序列为空，无法构造 token inputs")
    if tokens[0].token_type != TOKEN_TYPE_CTX:
        raise ValueError(f"Token 序列必须以 CTX 开头，当前首 token 类型为 {tokens[0].token_type}")
    if tokens[-1].token_type != TOKEN_TYPE_TASK:
        raise ValueError(f"Token 序列必须以 TASK 结尾，当前末 token 类型为 {tokens[-1].token_type}")
    if not isinstance(tokens[-1].body, dict):
        raise ValueError("TASK token body 必须是 dict")


def collate_fn(
    batch: List[Tuple[List[TokenStruct], DecisionLabels, Dict[str, Any]]],
    *,
    kg_states_mode: str,
) -> Dict[str, Any]:
    if not batch:
        # 与其它代码保持字段一致（用于安全兜底）
        return {
            "token_type_ids": torch.empty(0, 0, dtype=torch.long),
            "actor_ids": torch.empty(0, 0, dtype=torch.long),
            "round_ids": torch.empty(0, 0, dtype=torch.long),
            "attempt_ids": torch.empty(0, 0, dtype=torch.long),
            "meta_ids": torch.empty(0, 0, dtype=torch.long),
            "task_phase_ids": torch.empty(0, 0, dtype=torch.long),
            "task_good_mask": torch.empty(0, 0, 7, dtype=torch.float32),
            "propose_raw": torch.empty(0, 0, 7, dtype=torch.float32),
            "vote_raw": torch.empty(0, 0, 7, dtype=torch.float32),
            "mission_raw": torch.empty(0, 0, 2, dtype=torch.float32),
            "phase_ids": torch.empty(0, dtype=torch.long),
            "my_ids": torch.empty(0, dtype=torch.long),
            "kg_states": torch.empty(0, 8, dtype=torch.long),
            "padding_mask": torch.empty(0, 0, dtype=torch.bool),
            "speech_texts": [],
            "speech_indices": torch.empty(0, 2, dtype=torch.long),
            "labels_list": [],
            "extra_list": [],
        }

    tokens_batch: List[List[TokenStruct]] = [tokens for tokens, _labels, _extra in batch]
    labels_list: List[DecisionLabels] = [labels for _tokens, labels, _extra in batch]
    extra_list: List[Dict[str, Any]] = [extra for _tokens, _labels, extra in batch]

    bsz = len(tokens_batch)
    lengths = [len(seq) for seq in tokens_batch]
    max_len = max(lengths) if lengths else 0

    token_type_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    actor_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    round_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    attempt_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    meta_ids = torch.zeros(bsz, max_len, dtype=torch.long)

    task_phase_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    task_good_mask = torch.zeros(bsz, max_len, 7, dtype=torch.float32)
    propose_raw = torch.zeros(bsz, max_len, 7, dtype=torch.float32)
    vote_raw = torch.zeros(bsz, max_len, 7, dtype=torch.float32)
    mission_raw = torch.zeros(bsz, max_len, 2, dtype=torch.float32)

    phase_ids = torch.zeros(bsz, dtype=torch.long)
    my_ids = torch.zeros(bsz, dtype=torch.long)
    kg_states = torch.zeros(bsz, 8, dtype=torch.long)
    padding_mask = torch.ones(bsz, max_len, dtype=torch.bool)

    speech_texts: List[str] = []
    speech_indices_list: List[Tuple[int, int]] = []

    for b_idx, tokens in enumerate(tokens_batch):
        seq_len = len(tokens)
        if seq_len == 0:
            continue

        for t_idx, t in enumerate(tokens[:max_len]):
            token_type_ids[b_idx, t_idx] = int(t.token_type)
            actor_ids[b_idx, t_idx] = int(t.actor_id if 0 <= t.actor_id <= MAX_PLAYERS else 0)
            round_ids[b_idx, t_idx] = int(max(0, t.round_index))
            attempt_ids[b_idx, t_idx] = int(max(0, t.attempt_index))

            meta_raw = t.meta_tuple
            if isinstance(meta_raw, (tuple, list)) and len(meta_raw) >= 5:
                # 5 元 meta 压缩（与现有训练实现一致）
                meta_compact = (
                    int(meta_raw[0]) * 96
                    + min(int(meta_raw[1]), 3) * 24
                    + min(int(meta_raw[2]), 3) * 6
                    + (min(max(int(meta_raw[3]), 2), 4) - 2) * 2
                    + min(int(meta_raw[4]), 1)
                )
                meta_compact = max(0, min(int(meta_compact), 575))
            else:
                meta_compact = 0
            meta_ids[b_idx, t_idx] = int(meta_compact)

            if t.token_type == TOKEN_TYPE_TASK:
                phase = int(t.body.get("phase", 0)) if isinstance(t.body, dict) else 0
                phase = max(0, min(4, phase))
                task_phase_ids[b_idx, t_idx] = phase
                good_mask = t.body.get("good_mask", [0.0] * 7) if isinstance(t.body, dict) else [0.0] * 7
                if isinstance(good_mask, list) and len(good_mask) >= 7:
                    for i in range(7):
                        try:
                            task_good_mask[b_idx, t_idx, i] = float(good_mask[i])
                        except Exception:
                            task_good_mask[b_idx, t_idx, i] = 0.0

            elif t.token_type == TOKEN_TYPE_PROPOSE:
                team_mask = t.body.get("team_mask", [0.0] * 7) if isinstance(t.body, dict) else [0.0] * 7
                if isinstance(team_mask, list) and len(team_mask) >= 7:
                    for i in range(7):
                        try:
                            propose_raw[b_idx, t_idx, i] = float(team_mask[i])
                        except Exception:
                            propose_raw[b_idx, t_idx, i] = 0.0

            elif t.token_type == TOKEN_TYPE_VOTE:
                vote_vec = t.body.get("vote_vec", [0.0] * 7) if isinstance(t.body, dict) else [0.0] * 7
                if isinstance(vote_vec, list) and len(vote_vec) >= 7:
                    for i in range(7):
                        try:
                            vote_raw[b_idx, t_idx, i] = float(vote_vec[i])
                        except Exception:
                            vote_raw[b_idx, t_idx, i] = 0.0

            elif t.token_type == TOKEN_TYPE_MISSION:
                mission_vec = t.body.get("mission_vec", [0.0] * 2) if isinstance(t.body, dict) else [0.0] * 2
                if isinstance(mission_vec, list) and len(mission_vec) >= 2:
                    for i in range(2):
                        try:
                            mission_raw[b_idx, t_idx, i] = float(mission_vec[i])
                        except Exception:
                            mission_raw[b_idx, t_idx, i] = 0.0

            elif t.token_type == TOKEN_TYPE_SPEECH:
                # 训练中默认用 use_text_cache=True，不会进入文本编码；这里仍按结构保留
                text = (t.body or {}).get("text", "")
                if isinstance(text, str):
                    text_stripped = text.strip()
                    if text_stripped and not text_stripped.startswith("..."):
                        speech_texts.append(text_stripped)
                        speech_indices_list.append((b_idx, t_idx))

        padding_mask[b_idx, : min(seq_len, max_len)] = False
        phase_ids[b_idx] = int(labels_list[b_idx].phase_id)
        my_id = tokens[0].actor_id if tokens else 0
        my_ids[b_idx] = int(my_id if 0 <= my_id <= MAX_PLAYERS else 0)

        players_info = extra_list[b_idx].get("players_info") or {}
        if kg_states_mode == "local":
            kg_list = _build_kg_states_local(players_info, int(my_ids[b_idx].item()))
        else:
            kg_list = _build_kg_states_global(players_info, int(my_ids[b_idx].item()))

        for i, val in enumerate(kg_list[:8]):
            kg_states[b_idx, i] = int(val)

    speech_indices = (
        torch.tensor(speech_indices_list, dtype=torch.long)
        if speech_indices_list
        else torch.empty(0, 2, dtype=torch.long)
    )

    return {
        "token_type_ids": token_type_ids,
        "actor_ids": actor_ids,
        "round_ids": round_ids,
        "attempt_ids": attempt_ids,
        "meta_ids": meta_ids,
        "task_phase_ids": task_phase_ids,
        "task_good_mask": task_good_mask,
        "propose_raw": propose_raw,
        "vote_raw": vote_raw,
        "mission_raw": mission_raw,
        "phase_ids": phase_ids,
        "my_ids": my_ids,
        "kg_states": kg_states,
        "padding_mask": padding_mask,
        "speech_texts": speech_texts,
        "speech_indices": speech_indices,
        "labels_list": labels_list,
        "extra_list": extra_list,
    }


def _map_game_phase_to_small_phase(phase: str) -> str:
    mapping = {
        "speech": "discussion",
        "proposal": "proposal",
        "voting": "voting",
        "execution": "mission",
        "assassination": "assassination",
    }
    return mapping.get(phase, phase)


def get_round_and_attempt_from_condition(game_condition: Dict[str, Any]) -> Tuple[int, int]:
    """从 game_condition 的 timeline 推导最大 round_number + 该轮最大 attempt_index。"""
    timeline = game_condition.get("game_timeline") or []
    if not timeline:
        return 1, 1
    round_numbers = [r.get("round_number") for r in timeline if r.get("round_number") is not None]
    if not round_numbers:
        return 1, 1
    round_number = max(int(r) for r in round_numbers)
    current_round = next((r for r in timeline if int(r.get("round_number", 0)) == int(round_number)), None)
    if not current_round:
        return round_number, 1
    attempts = current_round.get("attempts") or []
    attempt_indices = [a.get("attempt_index") for a in attempts if a.get("attempt_index") is not None]
    if not attempt_indices:
        return int(round_number), 1
    attempt_index = max(int(a) for a in attempt_indices)
    return int(round_number), int(attempt_index)


def augment_game_condition_for_voting(game_condition: Dict[str, Any], round_number: int, attempt_index: int, player_id: int) -> Dict[str, Any]:
    """
    给 voting 阶段注入 votes_details 占位（对齐 small_model 的 /predict_from_log 构造逻辑）。
    """
    payload = copy.deepcopy(game_condition)
    timeline = payload.get("game_timeline") or []
    current_round = next((r for r in timeline if int(r.get("round_number", 0)) == int(round_number)), None)
    if not current_round:
        return payload
    attempts = current_round.get("attempts") or []
    cur_attempt = next((a for a in attempts if int(a.get("attempt_index", 0)) == int(attempt_index)), None)
    if not cur_attempt:
        return payload
    if not (cur_attempt.get("steps") or {}).get("proposal"):
        return payload

    players_config = payload.get("players_config") or {}
    info = players_config.get(str(player_id)) or {}
    placeholder_vote = {
        "agent_info": {"id": player_id, "role": info.get("role", "unknown"), "model": info.get("model", "unknown")},
        "answer": {"raw_response": "", "extracted_result": None},
    }

    round_idx = next((i for i, r in enumerate(payload["game_timeline"]) if int(r.get("round_number", 0)) == int(round_number)), None)
    if round_idx is None:
        return payload
    attempt_idx = next(
        (
            i
            for i, a in enumerate(payload["game_timeline"][round_idx]["attempts"])
            if int(a.get("attempt_index", 0)) == int(attempt_index)
        ),
        None,
    )
    if attempt_idx is None:
        return payload
    steps = payload["game_timeline"][round_idx]["attempts"][attempt_idx].setdefault("steps", {})
    steps["voting"] = {
        "votes_details": [placeholder_vote],
        "final_outcome": None,
    }
    return payload


def augment_game_condition_for_mission(game_condition: Dict[str, Any], round_number: int, attempt_index: int) -> Dict[str, Any]:
    """
    给 execution(mission) 阶段注入 mission_result.player_actions 占位（对齐 small_model 的 /predict_from_log 构造逻辑）。
    """
    payload = copy.deepcopy(game_condition)
    timeline = payload.get("game_timeline") or []
    if not timeline:
        return payload
    current_round = next((r for r in timeline if int(r.get("round_number", 0)) == int(round_number)), None)
    if not current_round:
        return payload
    attempts = current_round.get("attempts") or []
    last_attempt = next((a for a in attempts if int(a.get("attempt_index", 0)) == int(attempt_index)), None)
    if not last_attempt:
        return payload
    steps = last_attempt.get("steps") or {}
    proposal = steps.get("proposal")
    if not proposal:
        return payload
    team = (proposal.get("answer") or {}).get("extracted_result")
    if not isinstance(team, list) or len(team) == 0:
        return payload

    players_config = payload.get("players_config") or {}
    player_actions = []
    for pid in team:
        pid = int(pid)
        info = players_config.get(str(pid)) or {}
        player_actions.append(
            {
                "agent_info": {"id": pid, "role": info.get("role", "unknown"), "model": info.get("model", "unknown")},
                "answer": {"raw_response": "", "extracted_result": None},
            }
        )

    round_idx = next((i for i, r in enumerate(payload["game_timeline"]) if int(r.get("round_number", 0)) == int(round_number)), None)
    if round_idx is None:
        return payload
    payload["game_timeline"][round_idx]["mission_result"] = {
        "outcome": None,
        "fail_cards_count": 0,
        "player_actions": player_actions,
    }
    return payload


def augment_game_condition_for_assassination(game_condition: Dict[str, Any], player_id: int) -> Dict[str, Any]:
    """
    确保 assassination 字段存在，且 agent_info.id 为当前 player（对齐 small_model 的 /predict_from_log 校验）。
    """
    if game_condition.get("assassination") is not None:
        return game_condition
    payload = copy.deepcopy(game_condition)
    players_config = payload.get("players_config") or {}
    info = players_config.get(str(player_id)) or {}
    payload["assassination"] = {
        "agent_info": {"id": player_id, "role": "Assassin", "model": info.get("model", "unknown")},
        "answer": {"raw_response": "", "extracted_result": None},
        "target_details": {},
    }
    return payload


def build_actor_critic_batches_from_game_condition(
    *,
    game_condition: Dict[str, Any],
    phase: str,
    player_id: int,
    round_number: Optional[int] = None,
    attempt_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Tuple[int, int], Tuple[int, int, int, int, int]]:
    """
    给定当前 game_condition 与决策 selector，构造：
    - actor local obs batch（kg_states local）
    - critic global state batch（kg_states global，非遮罩）
    - extra（主要包含 players_info 等）
    """
    if round_number is None or attempt_index is None:
        round_number2, attempt_index2 = get_round_and_attempt_from_condition(game_condition)
    else:
        round_number2, attempt_index2 = int(round_number), int(attempt_index)

    # 对齐 small_model 的 token 构造：voting/mission/assassination 需要在 log 中补字段
    payload = game_condition
    if phase == "voting":
        payload = augment_game_condition_for_voting(game_condition, round_number2, attempt_index2, int(player_id))
    elif phase == "execution":
        payload = augment_game_condition_for_mission(game_condition, round_number2, attempt_index2)
    elif phase == "assassination":
        payload = augment_game_condition_for_assassination(game_condition, int(player_id))

    small_phase = _map_game_phase_to_small_phase(phase)
    tokens, labels, extra = build_decision_tokens_from_log(
        log=payload,
        phase=small_phase,
        player=int(player_id),
        round_number=int(round_number2),
        attempt_index=int(attempt_index2),
    )
    decision_meta = tokens[0].meta_tuple

    batch = [(tokens, labels, extra)]
    batch_obs_local = collate_fn(batch, kg_states_mode="local")
    batch_state_global = collate_fn(batch, kg_states_mode="global")
    return batch_obs_local, batch_state_global, extra, (round_number2, attempt_index2), decision_meta

