from typing import Callable, Dict, Any, List

class Agent:
    def __init__(
        self,
        player_id: int,
        role: str,  # "civilian" or "spy" / specific role name
        llm_config: Dict[str, Any],
        llm_func: Callable,
    ):
        self.player_id = player_id
        self.role = role
        self.llm_config = llm_config
        self.llm_func = llm_func

    def _construct_instruction(self, phase: str, context: Dict[str, Any]) -> str:
        """
        根据当前游戏阶段构造具体的指令 Prompt
        """
        if phase == "speech":
            return """
It is your turn to speak.
Please briefly express your suggestions for the team proposal, or respond to others' views.
Please begin your action now.
You must return the final speech content in JSON format as requested.
""".strip()

        elif phase == "proposal":
            team_size = context.get("team_size", 0)
            return f"""
You are the leader. Based on the previous discussion, please select {team_size} players to execute the quest.
Please begin your action now.
You must return the final team content in JSON format as requested.
""".strip()

        elif phase == "voting":
            round_id = context.get("round", 0)
            return f"""
It is currently Round {round_id}. Based on the previous discussion, you need to vote on the proposed team.
Please begin your action now.
You must return the final voting content in JSON format as requested.
""".strip()

        elif phase == "execution":
            round_id = context.get("round", 0)
            return f"""
It is currently Round {round_id}. You need to execute the quest based on the previous discussion.
Good faction players MUST return True. Evil faction players MAY return False.
Please begin your action now.
You must return the final mission execution result in JSON format as requested.
""".strip()

        elif phase == "assassination":
            candidates = context.get("good_candidate_ids", [])
            return f"""
The Good faction has won the quests.
You are the Assassin. You need to assassinate Merlin to steal the victory.
Here is the list of player IDs that are Good-side roles (valid assassination targets): {candidates}.
You MUST choose ONE player ID from this list as your assassination target.
Please begin your action now.
You must return the final assassination target in JSON format as requested.
""".strip()
        
        else:
            return "Please follow the game rules to proceed."

    def act(self, memory: List[Dict[str, str]], phase: str, observations: List[str], context: Dict[str, Any] = None, game_condition: Dict[str, Any] = None) -> str:
        """
        Agent 行动的主入口：
        1. 接收环境信息 (Observations)
        2. 接收当前阶段 (Phase) 和 上下文 (Context)
        3. 接收游戏全局状态 (game_condition) - 新增
        4. 内部构造 Prompt
        5. 更新记忆并调用 LLM
        """
        if context is None:
            context = {}

        # 1. 构造完整的 User Content
        full_user_content = ""
        
        # 拼接在此期间错过的/观测到的信息
        if observations:
            full_user_content += "== Information you missed/observed ==\n"
            full_user_content += "\n".join(observations) + "\n\n"
            
        # 拼接当前任务指令
        instruction = self._construct_instruction(phase, context)
        full_user_content += "== Current Task Instruction ==\n"
        full_user_content += instruction
        
        # 2. 更新 Agent 记忆 (User Turn)
        memory.append({"role": "user", "content": full_user_content})
        
        # 3. 调用底层 LLM
        response = self.call(memory)
        
        # 4. 更新 Agent 记忆 (Assistant Turn)
        memory.append({"role": "assistant", "content": response})
        
        return response

    def call(self, messages: List[Dict[str, str]]) -> str:
        """
        统一接口：直接接收符合 OpenAI 格式的 messages 列表
        [{"role": "user", "content": "..."}]
        """
        api_url_config = self.llm_config.get("api_url_config", {})
        inference_config = self.llm_config.get("inference_config", {})

        # 直接透传 messages 到底层函数
        # system_prompt 作为参数传递，底层函数负责插入
        response = self.llm_func(
            messages=messages,
            api_url_config=api_url_config,
            inference_config=inference_config,
        )
        return response