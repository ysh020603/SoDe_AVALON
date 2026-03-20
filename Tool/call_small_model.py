import json
from typing import Any, Dict, List, Optional, Union

import requests
import torch


def decode_labels(logits: Dict[str, Any]) -> Dict[str, Any]:
    labels: Dict[str, Any] = {}

    if "identity_logits" in logits:
        t = torch.tensor(logits["identity_logits"])
        if t.ndim > 1:
            t = t[0]
        pred = int(torch.argmax(t, dim=-1))
        labels["identity_label"] = pred
        identity_names = [
            "None",
            "Good",
            "Evil",
            "Merlin",
            "Percival",
            "Loyal Servant",
            "Morgana",
            "Assassin",
            "Mordred",
            "Oberon",
        ]
        labels["identity_label_text"] = identity_names[pred] if 0 <= pred < len(identity_names) else "Unknown"

    if "intent_logits" in logits:
        t = torch.tensor(logits["intent_logits"])
        if t.ndim > 1:
            t = t[0]
        probs = torch.sigmoid(t)
        bin_mask = (probs > 0.5).long()
        intent_flags = bin_mask.tolist()
        labels["intent_multi_hot"] = intent_flags
        intent_names = [
            "want_to_join",
            "test_others",
            "rely_on_success_record",
            "doubt_successful_team",
            "other",
        ]
        intent_selected = [name for flag, name in zip(intent_flags, intent_names) if flag == 1]
        labels["intent_semantic"] = intent_selected

    if "attitude_logits" in logits:
        t = torch.tensor(logits["attitude_logits"])
        if t.ndim > 2:
            t = t[0]
        preds = torch.argmax(t, dim=-1)
        attitude_flags = preds.tolist()
        labels["attitude_labels"] = attitude_flags
        attitude_names = {0: "none", 1: "pos", 2: "neg"}
        labels["attitude_semantic"] = [attitude_names.get(v, "none") for v in attitude_flags]

    if "propose_logits" in logits:
        t = torch.tensor(logits["propose_logits"])
        if t.ndim > 1:
            t = t[0]
        probs = torch.sigmoid(t)
        team_mask = (probs > 0.5).long()
        labels["propose_targets"] = team_mask.tolist()

    if "vote_logits" in logits:
        t = torch.tensor(logits["vote_logits"])
        if t.ndim > 1:
            t = t[0]
        probs = torch.softmax(t, dim=-1)
        pred = int(torch.argmax(probs, dim=-1))
        labels["vote_label"] = pred
        labels["vote_label_text"] = "Approve" if pred == 1 else "Reject"

    if "mission_logits" in logits:
        t = torch.tensor(logits["mission_logits"])
        if t.ndim > 1:
            t = t[0]
        probs = torch.softmax(t, dim=-1)
        pred = int(torch.argmax(probs, dim=-1))
        labels["mission_label"] = pred
        labels["mission_label_text"] = "Success" if pred == 0 else "Fail"

    if "kill_logits" in logits:
        t = torch.tensor(logits["kill_logits"])
        if t.ndim > 1:
            t = t[0]
        probs = torch.softmax(t, dim=-1)
        pred = int(torch.argmax(probs, dim=-1))
        labels["kill_target"] = pred

    return labels


def build_request_payload(
    log: Dict[str, Any],
    phase: Union[str, int],
    player: int,
    round_number: int,
    attempt_index: int,
) -> Dict[str, Any]:
    return {
        "log": log,
        "phase": phase,
        "player": int(player),
        "round_number": int(round_number),
        "attempt_index": int(attempt_index),
    }


def send_predict_request(
    api_url: str,
    payload: Dict[str, Any],
    timeout: int = 600,
) -> Dict[str, Any]:
    resp = requests.post(api_url, json=payload, timeout=int(timeout))
    resp.raise_for_status()
    return resp.json()


def parse_response_to_output(
    response: Dict[str, Any],
    selector: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logits = response.get("logits", {})
    output_obj: Dict[str, Any] = {
        "response_meta": {
            "phase_id": response.get("phase_id"),
            "phase_name": response.get("phase_name"),
            "round_number": response.get("round_number"),
            "attempt_index": response.get("attempt_index"),
            "actor_id": response.get("actor_id"),
            "role": response.get("role"),
            "model_name": response.get("model_name"),
        },
        "output": decode_labels(logits),
        "raw_response": response,
    }
    if selector is not None:
        output_obj["selector"] = selector
    return output_obj


def predict_from_game_condition(
    game_condition: Dict[str, Any],
    *,
    phase: Union[str, int],
    player: int,
    round_number: int,
    attempt_index: int,
    api_url: str = "http://127.0.0.1:8000/predict_from_log",
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    一站式（dict 版本）：直接使用 game_condition(dict) 调用 /predict_from_log。
    """
    phase_val: Union[str, int] = phase
    try:
        phase_val = int(phase)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        phase_val = str(phase).strip().lower()

    payload = build_request_payload(
        log=game_condition,
        phase=phase_val,
        player=player,
        round_number=round_number,
        attempt_index=attempt_index,
    )
    response = send_predict_request(api_url, payload, timeout)

    selector = {
        "phase": phase_val,
        "player": int(player),
        "round_number": int(round_number),
        "attempt_index": int(attempt_index),
    }
    return parse_response_to_output(response, selector=selector)


def format_suggestion(action: Any) -> str:
    """
    将 action 格式化为可直接注入 prompt 的字符串（原始 JSON，兼容旧逻辑）。
    """
    if isinstance(action, str):
        return action
    try:
        return json.dumps(action, ensure_ascii=False, indent=2)
    except Exception:
        return str(action)


# 意图语义的自然语言描述（与 eval_team / 身份·态度 描述一致）
INTENT_DESCRIPTIONS = {
    "want_to_join": "recommend yourself (the suggested team includes your own ID).",
    "test_others": "Recommending a team to test fresh players and gather information.",
    "rely_on_success_record": "Mentioning the use of players who have successfully executed past missions or have been verified.",
    "doubt_successful_team": "Expressing doubt about a previously successful team or its members (e.g., suspecting a spy might be hiding).",
    "other": "Other intent.",
}


def format_suggestion_natural_language(output: Dict[str, Any], phase: str) -> str:
    """
    将小模型 output（decode_labels 结果）按 phase 转为自然语言 suggestion，便于放入 prompt。
    phase: 游戏阶段 "speech" | "proposal" | "voting" | "execution" | "assassination"
    """
    if not isinstance(output, dict):
        return str(output)
    lines: List[str] = []

    if phase == "speech":
        # 身份：可能声明/呈现的角色
        identity = output.get("identity_label_text") or output.get("identity_label")
        if identity and str(identity) not in ("None", "Unknown"):
            lines.append(f"Identity / role presentation: You may claim or present as: {identity}.")
        # 意图：多条自然语言描述
        intents = output.get("intent_semantic") or []
        if intents:
            descs = [INTENT_DESCRIPTIONS.get(name, name) for name in intents]
            lines.append("You may " + " ".join(descs))
        # 态度：对哪些玩家积极/消极（attitude_semantic 为按玩家顺序的 none/pos/neg 列表，1-based 玩家 ID = index+1）
        att = output.get("attitude_semantic")
        if isinstance(att, list) and len(att) >= 7:
            pos_ids = [i + 1 for i in range(min(7, len(att))) if att[i] == "pos"]
            neg_ids = [i + 1 for i in range(min(7, len(att))) if att[i] == "neg"]
            if pos_ids or neg_ids:
                parts = []
                if pos_ids:
                    parts.append(
                        f"Positive attitude towards players {pos_ids} "
                        "(i.e. expressing trust, belief in their words or proposed team, or support)"
                    )
                if neg_ids:
                    parts.append(
                        f"Negative attitude towards players {neg_ids} "
                        "(i.e. suspecting sabotage, doubting identity, or opposing their participation in the mission)"
                    )
                lines.append("Attitudes: " + "; ".join(parts) + ".")
        if not lines:
            lines.append("No strong identity or intent signal inferred.")

    elif phase == "proposal":
        team_mask = output.get("propose_targets")
        if isinstance(team_mask, list) and len(team_mask) >= 7:
            team = [i + 1 for i in range(7) if team_mask[i] == 1]
            if team:
                lines.append(f"You may propose team: players {team}.")
            else:
                lines.append("You may not propose any specific team (mask empty).")
        else:
            lines.append("You did not output a clear team suggestion.")

    elif phase == "voting":
        vote = output.get("vote_label_text")
        if vote:
            lines.append(f"You may vote: {vote}.")
        else:
            lines.append("You did not output a clear vote suggestion.")

    elif phase == "execution":
        mission = output.get("mission_label_text")
        if mission:
            lines.append(f"You may suggest mission action: {mission}.")
        else:
            lines.append("You did not output a clear mission suggestion.")

    elif phase == "assassination":
        kill = output.get("kill_target")
        if kill is not None:
            # kill_target 为 0-based 座位索引，玩家 ID 常用 1-based
            player_id = int(kill) + 1
            if 1 <= player_id <= 10:
                lines.append(f"You may assassinate: player {player_id}.")
            else:
                lines.append(f"You may assassinate: (index): {kill}.")
        else:
            lines.append("You did not output a clear assassination target.")

    else:
        # 未知 phase 或含 error 时退回简短说明
        if output.get("error"):
            lines.append(f"Suggestion unavailable: {output.get('error')}.")
        else:
            lines.append("Suggestion: see raw output (phase not matched).")
    return "\n".join(lines) if lines else "No suggestion."

