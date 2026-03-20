"""
Avalon Soft Policy API 客户端模块。

提供从 JSON 文件加载对局日志、调用 /predict_from_log 接口、解析响应并保存结果的功能。
"""

import argparse
import json
import os
from typing import Any, Dict, Optional, Union

import requests
import torch


def decode_labels(logits: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 API 返回的 logits 解码为可读的预测标签。

    Args:
        logits: API 响应中的 logits 字典，可能包含 identity_logits、intent_logits、
                attitude_logits、propose_logits、vote_logits、mission_logits、kill_logits。

    Returns:
        解码后的标签字典，可能包含：
        - identity_label, identity_label_text
        - intent_multi_hot, intent_semantic
        - attitude_labels, attitude_semantic
        - propose_targets
        - vote_label, vote_label_text
        - mission_label, mission_label_text
        - kill_target
    """
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
        intent_selected = [
            name for flag, name in zip(intent_flags, intent_names) if flag == 1
        ]
        labels["intent_semantic"] = intent_selected

    if "attitude_logits" in logits:
        t = torch.tensor(logits["attitude_logits"])
        if t.ndim > 2:
            t = t[0]
        preds = torch.argmax(t, dim=-1)
        attitude_flags = preds.tolist()
        labels["attitude_labels"] = attitude_flags
        attitude_names = {0: "none", 1: "pos", 2: "neg"}
        labels["attitude_semantic"] = [
            attitude_names.get(v, "none") for v in attitude_flags
        ]

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


def load_json_from_file(path: str) -> Dict[str, Any]:
    """
    从文件加载 JSON 并返回顶层 dict。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的 JSON 对象（顶层必须为 dict）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件内容不是有效的 JSON 或顶层不是 dict。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"json_path not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("json file must contain a dict/object at top-level.")
    return obj


def build_request_payload(
    log: Dict[str, Any],
    phase: Union[str, int],
    player: int,
    round_number: int,
    attempt_index: int,
) -> Dict[str, Any]:
    """
    构造 /predict_from_log 接口的请求体。
    """
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
    """
    向 /predict_from_log 接口发送 POST 请求并返回响应 JSON。
    """
    resp = requests.post(api_url, json=payload, timeout=int(timeout))
    resp.raise_for_status()
    return resp.json()


def parse_response_to_output(
    response: Dict[str, Any],
    json_path: Optional[str] = None,
    selector: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将 API 响应解析为结构化输出，包含 decoded labels 和 raw_response。
    """
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
    if json_path is not None:
        output_obj["json_path"] = json_path
    if selector is not None:
        output_obj["selector"] = selector
    return output_obj


def predict_from_log(
    json_path: str,
    phase: Union[str, int],
    player: int,
    round_number: int,
    attempt_index: int,
    api_url: str = "http://127.0.0.1:8000/predict_from_log",
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    一站式：从 JSON 文件加载日志、构造请求、发送并解析响应。
    """
    log = load_json_from_file(json_path)
    phase_val: Union[str, int] = phase
    try:
        phase_val = int(phase)
    except (ValueError, TypeError):
        phase_val = str(phase).strip().lower()

    payload = build_request_payload(
        log=log,
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
    return parse_response_to_output(
        response,
        json_path=json_path,
        selector=selector,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Client: load json dict and call /predict_from_log."
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/predict_from_log",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/data1/shy/SoDe_Avalon_2/logs/Avalon/gemini-3-flash-preview_VS_gemini-3-flash-preview_7Players/20260314_130533_1723/Avalon_Multiturn_20260314_130533_1723_7_Players_result_evil_win_assassination_game_condition.json",
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        help='phase: "discussion"|"proposal"|"voting"|"mission"|"assassination" 或 0..4',
    )
    parser.add_argument("--player", type=int, required=True, help="player id/actor id: 1..7")
    parser.add_argument("--round_number", type=int, required=True, help="round number (usually 1..5)")
    parser.add_argument("--attempt_index", type=int, required=True, help="attempt index within round (usually 1..5)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/shy/SoDe_RL/test_result",
    )
    parser.add_argument("--timeout", type=int, default=600, help="request timeout seconds")
    args = parser.parse_args()

    phase_val: Union[str, int] = args.phase
    try:
        phase_val = int(args.phase)
    except (ValueError, TypeError):
        phase_val = str(args.phase).strip().lower()

    result = predict_from_log(
        json_path=args.json_path,
        phase=phase_val,
        player=args.player,
        round_number=args.round_number,
        attempt_index=args.attempt_index,
        api_url=args.api_url,
        timeout=args.timeout,
    )

    raw = result.get("raw_response", {})
    print(
        "fetched decision: "
        f"phase={raw.get('phase_name')}, "
        f"round={raw.get('round_number')}, "
        f"attempt={raw.get('attempt_index')}, "
        f"actor={raw.get('actor_id')}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.basename(args.json_path)
    if base.endswith(".json"):
        base = base[:-5]
    phase_safe = str(phase_val).replace("/", "_")
    out_path = os.path.join(
        args.output_dir,
        f"{base}_phase{phase_safe}_round{args.round_number}_attempt{args.attempt_index}_player{args.player}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    main()

