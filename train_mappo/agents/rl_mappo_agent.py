import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from SoDe_Avalon_3.train_mappo.state.action_space import (
    compute_assassination_log_prob_entropy,
    compute_discussion_log_prob_entropy,
    compute_mission_log_prob_entropy,
    compute_proposal_log_prob_entropy,
    compute_voting_log_prob_entropy,
)
from SoDe_Avalon_3.train_mappo.state.token_builder import (
    PHASE_ASSASSINATION,
    PHASE_DISCUSSION,
    PHASE_MISSION,
    PHASE_PROPOSING,
    PHASE_VOTING,
    build_actor_critic_batches_from_game_condition,
    get_round_and_attempt_from_condition,
)


_GOOD_ROLES = {"Merlin", "Percival", "Loyal Servant"}


def _construct_instruction(phase: str, context: Dict[str, Any]) -> str:
    # 与 Agents.Agent._construct_instruction 保持一致（discussion 才会真正调用 LLM）
    if phase == "speech":
        return """
It is your turn to speak.
Please briefly express your suggestions for the team proposal, or respond to others' views.
Please begin your action now.
You must return the final speech content in JSON format as requested.
""".strip()

    if phase == "proposal":
        team_size = context.get("team_size", 0)
        return f"""
You are the leader. Based on the previous discussion, please select {team_size} players to execute the quest.
Please begin your action now.
You must return the final team content in JSON format as requested.
""".strip()

    if phase == "voting":
        round_id = context.get("round", 0)
        return f"""
It is currently Round {round_id}. Based on the previous discussion, you need to vote on the proposed team.
Please begin your action now.
You must return the final voting content in JSON format as requested.
""".strip()

    if phase == "execution":
        round_id = context.get("round", 0)
        return f"""
It is currently Round {round_id}. You need to execute the quest based on the previous discussion.
Good faction players MUST return True. Evil faction players MAY return False.
Please begin your action now.
You must return the final mission execution result in JSON format as requested.
""".strip()

    if phase == "assassination":
        candidates = context.get("good_candidate_ids", [])
        return f"""
The Good faction has won the quests.
You are the Assassin. You need to assassinate Merlin to steal the victory.
Here is the list of player IDs that are Good-side roles (valid assassination targets): {candidates}.
You MUST choose ONE player ID from this list as your assassination target.
Please begin your action now.
You must return the final assassination target in JSON format as requested.
""".strip()

    return "Please follow the game rules to proceed."


def _format_discussion_suggestion(action: Dict[str, Any]) -> str:
    """
    将讨论阶段 sampled action 转成 suggestion 字符串，供 LLM 使用。
    """
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
    identity_label = int(action["identity_label"])
    identity_text = identity_names[identity_label] if 0 <= identity_label < len(identity_names) else "Unknown"

    intent_names = [
        "want_to_join",
        "test_others",
        "rely_on_success_record",
        "doubt_successful_team",
        "other",
    ]
    intent_multi_hot = action["intent_multi_hot"]
    intent_semantic = [name for flag, name in zip(intent_multi_hot, intent_names) if int(flag) == 1]

    attitude_semantic_map = {0: "none", 1: "pos", 2: "neg"}
    attitude_labels = action["attitude_labels"]
    attitude_semantic = [attitude_semantic_map.get(int(x), "none") for x in attitude_labels]

    # 这里直接复用 existing formatter 的文本风格（避免重新造轮子）
    try:
        from SoDe_Avalon_3.Tool.call_small_model import format_suggestion_natural_language

        out = {
            "identity_label_text": identity_text,
            "intent_semantic": intent_semantic,
            "attitude_semantic": attitude_semantic,
        }
        return format_suggestion_natural_language(out, phase="speech")
    except Exception:
        # fallback：最简文本
        return json.dumps(
            {"identity": identity_text, "intent": intent_semantic, "attitude": attitude_semantic},
            ensure_ascii=False,
        )


def _action_to_env_json(*, phase: str, action: Dict[str, Any]) -> str:
    if phase == "speech":
        # speech phase 环境提取 key=statement
        # 在讨论训练时，该字段来自 LLM，所以这里不直接用 action 构造
        raise RuntimeError("_action_to_env_json for speech should not be called.")

    if phase == "proposal":
        return json.dumps({"team": action["team_order"]}, ensure_ascii=False)

    if phase == "voting":
        return json.dumps({"vote": bool(action["vote"])}, ensure_ascii=False)

    if phase == "execution":
        return json.dumps({"success": bool(action["success"])}, ensure_ascii=False)

    if phase == "assassination":
        return json.dumps({"target": int(action["target_player_id"])}, ensure_ascii=False)

    raise ValueError(f"Unknown phase: {phase}")


class RLSmallModelMAPPOAgent:
    """
    新增 RL Agent（训练专用）。

    - discussion(speech)：
      1) Actor 采样离散/多离散 small-model action
      2) LLM 基于 sampled action 输出 statement JSON
      3) 记录 trajectory：old_log_prob/old_value 来自 Actor/Critic，而不是来自 LLM

    - proposal/voting/execution/assassination：
      1) Actor 直接采样并构造 environment JSON
      2) 记录 trajectory
    """

    def __init__(
        self,
        *,
        player_id: int,
        role: str,
        llm_config: Dict[str, Any],
        llm_func,
        actor,
        critic=None,
        record: bool = False,
        use_llm_in_discussion: bool = True,
    ) -> None:
        self.player_id = int(player_id)
        self.role = role
        self.llm_config = llm_config
        self.llm_func = llm_func

        self.actor = actor
        self.critic = critic
        self.record = bool(record)
        self.use_llm_in_discussion = bool(use_llm_in_discussion)

        # rollout 内收集轨迹
        self.collected_transitions: List[Dict[str, Any]] = []

    def clear_episode(self) -> None:
        self.collected_transitions = []

    def _phase_id_from_game_phase(self, phase: str) -> int:
        if phase == "speech":
            return PHASE_DISCUSSION
        if phase == "proposal":
            return PHASE_PROPOSING
        if phase == "voting":
            return PHASE_VOTING
        if phase == "execution":
            return PHASE_MISSION
        if phase == "assassination":
            return PHASE_ASSASSINATION
        raise ValueError(f"Unknown game phase: {phase}")

    def _force_success_for_execution(self) -> bool:
        # 环境逻辑：good 玩家 can_fail=False -> success 强制 True
        return self.role in _GOOD_ROLES

    def _valid_targets_for_assassination(self, context: Dict[str, Any]) -> List[int]:
        # game passes good_candidate_ids (1..7)
        targets = context.get("good_candidate_ids") or []
        return [int(x) for x in targets]

    def _sample_and_record(
        self,
        *,
        phase: str,
        game_condition: Dict[str, Any],
        round_number: Optional[int],
        attempt_index: Optional[int],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int, float, float, Dict[str, Any], Dict[str, Any]]:
        """
        采样小模型 action，并（在 record mode）记录 old_log_prob/old_value + obs/state tensors。
        返回：
          action, phase_id, old_log_prob, old_value, obs_tensors, state_tensors
        """
        phase_id = self._phase_id_from_game_phase(phase)

        # 构造 actor local obs + critic global state（统一使用 token_builder）
        batch_obs, batch_state, extra, _round_attempt, decision_meta = build_actor_critic_batches_from_game_condition(
            game_condition=game_condition,
            phase=phase,
            player_id=self.player_id,
            round_number=round_number,
            attempt_index=attempt_index,
        )

        logits_dict = self.actor.forward_logits(batch_obs)
        if not logits_dict:
            raise RuntimeError("Actor forward returned empty logits.")

        # 采样 action
        if phase_id == PHASE_DISCUSSION:
            action, old_log_prob, entropy = self.actor.sample_action(
                phase_id=phase_id,
                logits_dict=logits_dict,
                context={},
            )
        elif phase_id == PHASE_PROPOSING:
            team_size = int(context.get("team_size") or decision_meta[3])
            action, old_log_prob, entropy = self.actor.sample_action(
                phase_id=phase_id,
                logits_dict=logits_dict,
                context={"team_size": team_size},
            )
        elif phase_id == PHASE_VOTING:
            action, old_log_prob, entropy = self.actor.sample_action(
                phase_id=phase_id,
                logits_dict=logits_dict,
                context={},
            )
        elif phase_id == PHASE_MISSION:
            action, old_log_prob, entropy = self.actor.sample_action(
                phase_id=phase_id,
                logits_dict=logits_dict,
                context={"force_success": self._force_success_for_execution()},
            )
        elif phase_id == PHASE_ASSASSINATION:
            valid_targets = self._valid_targets_for_assassination(context)
            action, old_log_prob, entropy = self.actor.sample_action(
                phase_id=phase_id,
                logits_dict=logits_dict,
                context={"valid_target_ids": valid_targets},
            )
        else:
            raise ValueError(f"Unsupported phase_id: {phase_id}")

        # critic value（只在 record mode 时需要）
        old_value = 0.0
        if self.record:
            if self.critic is None:
                raise RuntimeError("record mode requires critic.")
            with torch.no_grad():
                old_value_t = self.critic(batch_state)
                old_value = float(old_value_t.squeeze(0).item())

        # 存储 obs/state tensors（去掉 labels_list/extra_list，update 时统一重建）
        def _extract_tensor_dict(d: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            out: Dict[str, torch.Tensor] = {}
            # use_text_cache=True 时 speech_texts/speech_indices 对前向无影响
            # 且 speech_indices 在不同决策点长度不同，会导致 minibatch 堆叠失败
            ignore_keys = {"speech_indices"}
            for k, v in d.items():
                if k in ignore_keys:
                    continue
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu()
            return out

        obs_tensors = _extract_tensor_dict(batch_obs)
        state_tensors = _extract_tensor_dict(batch_state)

        if self.record:
            self.collected_transitions.append(
                {
                    "phase_id": int(phase_id),
                    "action": action,
                    "old_log_prob": float(old_log_prob.detach().cpu().item()),
                    "old_value": float(old_value),
                    "obs_tensors": obs_tensors,
                    "state_tensors": state_tensors,
                    "meta": {"player_id": self.player_id, "role": self.role},
                }
            )

        return action, phase_id, float(old_log_prob.detach().cpu().item()), float(old_value), obs_tensors, state_tensors

    def act(
        self,
        memory: List[Dict[str, str]],
        phase: str,
        observations: List[str],
        context: Dict[str, Any] = None,
        game_condition: Dict[str, Any] = None,
    ) -> str:
        if context is None:
            context = {}
        if game_condition is None:
            raise ValueError("game_condition is required.")

        # round/attempt 用于 token 构造
        round_number, attempt_index = get_round_and_attempt_from_condition(game_condition)
        if phase == "proposal":
            # proposal/voting 需要 attempt_index 正常值，直接使用推导结果即可
            pass
        if phase == "assassination":
            # assassination 不一定依赖 attempt/round，但 token_builder 会自行用推导值兜底
            round_number, attempt_index = None, None

        # 采样 small-model action（并在 record mode 记录 old_log_prob/old_value）
        action, phase_id, _old_logp, _old_value, _obs, _state = self._sample_and_record(
            phase=phase,
            game_condition=game_condition,
            round_number=round_number,
            attempt_index=attempt_index,
            context=context,
        )

        # =========================
        # discussion(speech)：需要 LLM 输出 statement JSON
        # =========================
        if phase == "speech":
            # 构造 suggestion 让 LLM 受 small-model action 约束
            suggestion = _format_discussion_suggestion(action)

            full_user_content = ""
            if observations:
                full_user_content += "== Information you missed/observed ==\n"
                full_user_content += "\n".join(observations) + "\n\n"

            instruction = _construct_instruction(phase=phase, context=context)
            full_user_content += "suggestion:\n"
            full_user_content += suggestion + "\n\n"
            full_user_content += "According to the game rules, execute the instructions in the Suggestion above.\n\n"
            full_user_content += "== Current Task Instruction ==\n"
            full_user_content += instruction

            memory.append({"role": "user", "content": full_user_content})
            response = "{}"
            if self.use_llm_in_discussion:
                try:
                    response = self.llm_func(
                        messages=memory,
                        api_url_config=self.llm_config["api_url_config"],
                        inference_config=self.llm_config["inference_config"],
                    )
                except Exception:
                    response = "{}"
            # 如果 LLM 返回非 JSON，这里返回也会被环境 extractor 容错；为了训练稳定，兜底固定 JSON
            if not isinstance(response, str) or "{" not in response:
                response = json.dumps({"statement": "..."}, ensure_ascii=False)

            memory.append({"role": "assistant", "content": response})
            return response

        # =========================
        # other phases：无需 LLM，直接输出小模型采样动作对应 JSON
        # =========================
        response_json = _action_to_env_json(phase=phase, action=action)
        memory.append({"role": "user", "content": f"Current Task Instruction for {phase}. Please output JSON."})
        memory.append({"role": "assistant", "content": response_json})
        return response_json

