import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from Tool.call_small_model import format_suggestion_natural_language, predict_from_game_condition


class SmallModelAgent:
    """
在基础 Agent prompt 中注入 small-model suggestion 的 Agent。

与 `Agents/Agent.py` 的核心差异：
- 在每次 act 时，额外调用小模型 API（/predict_from_log）拿到 action
- 将 action 以
  suggestion:
  {...}
  的形式拼入 user prompt
    """

    def __init__(
        self,
        player_id: int,
        role: str,
        llm_config: Dict[str, Any],
        llm_func: Callable,
        *,
        small_model_api_url: str = "http://127.0.0.1:8000/predict_from_log",
        small_model_timeout: int = 600,
        enable_small_model: bool = True,
    ):
        self.player_id = player_id
        self.role = role
        self.llm_config = llm_config
        self.llm_func = llm_func

        self.small_model_api_url = small_model_api_url
        self.small_model_timeout = int(small_model_timeout)
        self.enable_small_model = bool(enable_small_model)

    def _construct_instruction(self, phase: str, context: Dict[str, Any]) -> str:
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

    @staticmethod
    def _map_phase_to_small_model(phase: str) -> str:
        mapping = {
            "speech": "discussion",
            "proposal": "proposal",
            "voting": "voting",
            "execution": "mission",
            "assassination": "assassination",
        }
        return mapping.get(phase, phase)

    # 所有环节都请求小模型。proposal/speech 直接用 game_condition；voting/execution/assassination 无数据时需补全 payload。
    _SMALLMODEL_PHASES = ("speech", "proposal", "voting", "execution", "assassination")

    @staticmethod
    def _get_round_and_attempt_from_condition(
        game_condition: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[int]]:
        """从 game_condition 中取当前轮数（最大）及该轮下最大尝试序号。"""
        if not game_condition:
            return None, None
        timeline = game_condition.get("game_timeline") or []
        if not timeline:
            return None, None
        # 最大轮数
        round_numbers = [r.get("round_number") for r in timeline if r.get("round_number") is not None]
        if not round_numbers:
            return None, None
        round_number = max(round_numbers)
        # 该轮对应的 round 对象（取 round_number 最大的那一个）
        current_round = next((r for r in timeline if r.get("round_number") == round_number), None)
        if not current_round:
            return round_number, None
        attempts = current_round.get("attempts") or []
        attempt_indices = [a.get("attempt_index") for a in attempts if a.get("attempt_index") is not None]
        if not attempt_indices:
            return round_number, 1
        attempt_index = max(attempt_indices)
        return round_number, attempt_index

    @staticmethod
    def _game_condition_for_mission(
        game_condition: Dict[str, Any],
        round_number: int,
        last_attempt_index: int,
    ) -> Optional[Dict[str, Any]]:
        """为 execution 阶段构造带 mission_result 的 game_condition（当前轮尚未写入 mission_result 时补全）。"""
        timeline = game_condition.get("game_timeline") or []
        if not timeline:
            return None
        current_round = next(
            (r for r in timeline if int(r.get("round_number", 0)) == int(round_number)),
            None,
        )
        if not current_round:
            return None
        attempts = current_round.get("attempts") or []
        last_attempt = next(
            (a for a in attempts if int(a.get("attempt_index", 0)) == int(last_attempt_index)),
            None,
        )
        if not last_attempt:
            return None
        steps = last_attempt.get("steps") or {}
        proposal = steps.get("proposal")
        if not proposal:
            return None
        team = (proposal.get("answer") or {}).get("extracted_result")
        if not isinstance(team, list) or len(team) == 0:
            return None
        players_config = game_condition.get("players_config") or {}
        player_actions = []
        for pid in team:
            pid = int(pid)
            info = players_config.get(str(pid)) or {}
            player_actions.append({
                "agent_info": {
                    "id": pid,
                    "role": info.get("role", "unknown"),
                    "model": info.get("model", "unknown"),
                },
                "answer": {"raw_response": "", "extracted_result": ""},
            })
        payload = copy.deepcopy(game_condition)
        round_idx = next(
            i for i, r in enumerate(payload["game_timeline"])
            if int(r.get("round_number", 0)) == int(round_number)
        )
        payload["game_timeline"][round_idx]["mission_result"] = {
            "outcome": None,
            "fail_cards_count": 0,
            "player_actions": player_actions,
        }
        return payload

    @staticmethod
    def _game_condition_for_voting(
        game_condition: Dict[str, Any],
        round_number: int,
        attempt_index: int,
        player_id: int,
    ) -> Optional[Dict[str, Any]]:
        """为 voting 阶段构造带 votes_details 的 game_condition（API 要求 votes_details 非空才处理投票）。"""
        timeline = game_condition.get("game_timeline") or []
        if not timeline:
            return None
        current_round = next(
            (r for r in timeline if int(r.get("round_number", 0)) == int(round_number)),
            None,
        )
        if not current_round:
            return None
        attempts = current_round.get("attempts") or []
        cur_attempt = next(
            (a for a in attempts if int(a.get("attempt_index", 0)) == int(attempt_index)),
            None,
        )
        if not cur_attempt or not (cur_attempt.get("steps") or {}).get("proposal"):
            return None
        players_config = game_condition.get("players_config") or {}
        info = players_config.get(str(player_id)) or {}
        placeholder_vote = {
            "agent_info": {
                "id": player_id,
                "role": info.get("role", "unknown"),
                "model": info.get("model", "unknown"),
            },
            "answer": {"raw_response": "", "extracted_result": None},
        }
        payload = copy.deepcopy(game_condition)
        round_idx = next(
            (i for i, r in enumerate(payload["game_timeline"]) if int(r.get("round_number", 0)) == int(round_number)),
            None,
        )
        if round_idx is None:
            return None
        attempt_idx = next(
            (i for i, a in enumerate(payload["game_timeline"][round_idx]["attempts"]) if int(a.get("attempt_index", 0)) == int(attempt_index)),
            None,
        )
        if attempt_idx is None:
            return None
        steps = payload["game_timeline"][round_idx]["attempts"][attempt_idx].setdefault("steps", {})
        steps["voting"] = {
            "votes_details": [placeholder_vote],
            "final_outcome": None,
        }
        return payload

    @staticmethod
    def _game_condition_for_assassination(
        game_condition: Dict[str, Any],
        player_id: int,
    ) -> Dict[str, Any]:
        """为 assassination 阶段补全 log.assassination（API 要求存在且 agent_info.id 为当前玩家）。"""
        if game_condition.get("assassination") is not None:
            return game_condition
        players_config = game_condition.get("players_config") or {}
        info = players_config.get(str(player_id)) or {}
        payload = copy.deepcopy(game_condition)
        payload["assassination"] = {
            "agent_info": {
                "id": player_id,
                "role": "Assassin",
                "model": info.get("model", "unknown"),
            },
            "answer": {"raw_response": "", "extracted_result": None},
        }
        return payload

    def _fetch_small_model_action(
        self,
        *,
        game_condition: Optional[Dict[str, Any]],
        phase: str,
        round_number: Optional[int],
        attempt_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_small_model:
            return None
        if not game_condition:
            return None
        # 仅 vote 与 mission 环节请求小模型（与 /predict_from_log 期望格式一致）
        if phase not in self._SMALLMODEL_PHASES:
            return None
        if round_number is None or attempt_index is None:
            return None

        small_phase = self._map_phase_to_small_model(phase)
        payload: Dict[str, Any] = game_condition
        if phase == "voting":
            payload = self._game_condition_for_voting(
                game_condition, int(round_number), int(attempt_index), int(self.player_id),
            ) or game_condition
        elif phase == "execution":
            payload = self._game_condition_for_mission(
                game_condition, int(round_number), int(attempt_index),
            ) or game_condition
        elif phase == "assassination":
            # API 要求 log.assassination 存在且 agent_info.id 为当前玩家
            payload = self._game_condition_for_assassination(game_condition, int(self.player_id))
        try:
            return predict_from_game_condition(
                payload,
                phase=small_phase,
                player=int(self.player_id),
                round_number=int(round_number),
                attempt_index=int(attempt_index),
                api_url=self.small_model_api_url,
                timeout=self.small_model_timeout,
            )
        except Exception as e:
            return {
                "error": str(e),
                "selector": {
                    "phase": small_phase,
                    "player": int(self.player_id),
                    "round_number": int(round_number),
                    "attempt_index": int(attempt_index),
                },
            }

    def act(
        self,
        memory: List[Dict[str, str]],
        phase: str,
        observations: List[str],
        context: Dict[str, Any] = None,
        game_condition: Dict[str, Any] = None,
        *,
        round_number: Optional[int] = None,
        attempt_index: Optional[int] = None,
    ) -> str:
        if context is None:
            context = {}

        # 小模型 API 的 round/attempt 只从 game_condition 推导（最大轮数 + 该轮最大 attempt_index）
        if game_condition:
            round_number, attempt_index = self._get_round_and_attempt_from_condition(game_condition)

        # 1. 构造完整的 User Content
        full_user_content = ""

        if observations:
            full_user_content += "== Information you missed/observed ==\n"
            full_user_content += "\n".join(observations) + "\n\n"

        # 2. small model suggestion（注入到 prompt）
        sm_result = self._fetch_small_model_action(
            game_condition=game_condition,
            phase=phase,
            round_number=round_number,
            attempt_index=attempt_index,
        )
        if sm_result is not None:
            action = sm_result.get("output", sm_result)
            full_user_content += "suggestion:\n"
            full_user_content += format_suggestion_natural_language(action, phase) + "\n\n"
            full_user_content += "According to the game rules, execute the instructions in the Suggestion above.\n\n"

        # 3. 当前任务指令
        instruction = self._construct_instruction(phase, context)
        full_user_content += "== Current Task Instruction ==\n"
        full_user_content += instruction

        memory.append({"role": "user", "content": full_user_content})
        response = self.call(memory)
        memory.append({"role": "assistant", "content": response})
        return response

    def call(self, messages: List[Dict[str, str]]) -> str:
        api_url_config = self.llm_config.get("api_url_config", {})
        inference_config = self.llm_config.get("inference_config", {})

        response = self.llm_func(
            messages=messages,
            api_url_config=api_url_config,
            inference_config=inference_config,
        )
        return response

