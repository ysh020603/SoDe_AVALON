import json
import re
import random
import os
from datetime import datetime
from collections import Counter
from typing import List, Dict
import ast
from Tool.Json_extractor import extract_json
from prompts.Avalon_system_prompts import get_avalon_prompt

# 引用修改后的 Agent 类
from Agents.Agent import Agent

class Game_Avalon_Multiturn:
    def __init__(self, agents: List[Agent], log_tag: str = ""):
        self.agents = agents
        self.player_count = len(agents)
        self.start_time = datetime.now()
        self.log_tag = log_tag
        
        # 1. 基础人数校验
        if self.player_count < 5 or self.player_count > 10:
            raise ValueError(f"Avalon only supports 5-10 players. Current input: {self.player_count}.")

        # 2. 角色配置校验
        self._validate_roles()

        self.quest_configs = {
            5: [2, 3, 2, 3, 3],
            6: [2, 3, 4, 3, 4],
            7: [2, 3, 3, -4, 4], 
            8: [3, 4, 4, -5, 5],
            9: [3, 4, 4, -5, 5],
            10: [3, 4, 4, -5, 5]
        }
        
        self.good_roles = ["Merlin", "Percival", "Loyal Servant"]
        self.evil_roles = ["Morgana", "Assassin", "Mordred", "Oberon", "Minion"]
        
        self.scores = {"good": 0, "evil": 0}
        self.round = 0 
        self.vote_track = 0 
        self.leader_idx = 0 
        
        # Agent 记忆库
        self.agent_memories = {a.player_id: [] for a in self.agents}
        
        # 待处理观测缓冲区
        self.pending_observations = {a.player_id: [] for a in self.agents}
        
        # 初始化日志结构 (参考 Insider 格式)
        self.game_log = {
            "meta_info": {
                "mode": "Avalon_Multiturn",
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "player_count": self.player_count,
                "roles_config": [a.role for a in agents]
            },
            "agents_info": {},
            "game_process": [],
            "final_result": None
        }

        # === 新增：初始化 Game Condition ===
        self.game_condition = {
            "meta": {
                "game_id": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "final_result": "unfinished",
                "total_players": self.player_count
            },
            "players_config": {},
            "game_timeline": [],
            "assassination": None
        }
        # 填充 players_config
        for a in self.agents:
            # 尝试获取 model 名称，如果没有则默认为 unknown
            model_name = a.llm_config.get("name", "unknown")
            self.game_condition["players_config"][str(a.player_id)] = {
                "role": a.role,
                "model": model_name
            }
        # ==================================
        
        self._assign_roles_and_init_memory()
        
        # 记录 Agent 配置信息
        for agent in self.agents:
            self.game_log["agents_info"][f"player_{agent.player_id}"] = {
                "role": agent.role,
                "config": getattr(agent, "llm_config", {}), 
                "system_prompt": self.agent_memories[agent.player_id][0]["content"]
            }

        self._extract_json = extract_json

    def _validate_roles(self):
        current_roles = Counter([a.role for a in self.agents])
        config_map = {
            5: {"Merlin": 1, "Percival": 1, "Loyal Servant": 1, "Morgana": 1, "Assassin": 1},
            6: {"Merlin": 1, "Percival": 1, "Loyal Servant": 2, "Morgana": 1, "Assassin": 1},
            7: {"Merlin": 1, "Percival": 1, "Loyal Servant": 2, "Morgana": 1, "Assassin": 1, "Oberon": 1},
            8: {"Merlin": 1, "Percival": 1, "Loyal Servant": 3, "Morgana": 1, "Assassin": 1, "Minion": 1},
            9: {"Merlin": 1, "Percival": 1, "Loyal Servant": 4, "Morgana": 1, "Assassin": 1, "Mordred": 1},
            10: {"Merlin": 1, "Percival": 1, "Loyal Servant": 4, "Morgana": 1, "Assassin": 1, "Mordred": 1, "Oberon": 1}
        }
        required = config_map.get(self.player_count)
        if not required: raise ValueError("Configuration undefined")
        for role, count in required.items():
            if current_roles[role] != count:
                raise ValueError(f"Role configuration error: Count for {role} is incorrect")
        if len(self.agents) != sum(required.values()):
             raise ValueError("Player count validation exception")

    def _assign_roles_and_init_memory(self):
        self.role_map = {a.player_id: a.role for a in self.agents}
        self.evils = [pid for pid, role in self.role_map.items() if role in self.evil_roles]
        
        # 1. 动态生成板子配置描述 (board_config_description)
        role_counts = Counter([a.role for a in self.agents])
        board_desc = ", ".join([f"{count} {role}" for role, count in role_counts.items()])
        
        # === 修改处：添加 5 轮任务的具体人数和容错机制说明 ===
        quest_config_list = self.quest_configs[self.player_count]
        quest_details = []
        for i, num in enumerate(quest_config_list):
            count = abs(num)
            fails_needed = 2 if num < 0 else 1
            detail = f"Quest {i+1}: {count} players"
            if fails_needed > 1:
                detail += " (requires 2 fail cars to fail mission)"
            quest_details.append(detail)
        
        board_desc += f". Mission Configuration: {', '.join(quest_details)}."
        # ======================================================

        # 2. 生成所有玩家的编号与身份映射
        visible_evils_for_evil = [pid for pid in self.evils if self.role_map[pid] != "Oberon"]
        merlin_sees = [pid for pid in self.evils if self.role_map[pid] != "Mordred"]
        merlin_morgana = [pid for pid, role in self.role_map.items() if role in ["Merlin", "Morgana"]]
        random.shuffle(merlin_morgana)

        for agent in self.agents:
            pid = agent.player_id
            role = agent.role
            
            # 准备传给 get_avalon_prompt 的参数
            kwargs = {
                "player_id": pid,
            }
            
            # 根据角色分配特定的可见信息
            if role == "Merlin":
                kwargs["seen_evils"] = str(merlin_sees)
            elif role == "Percival":
                kwargs["seen_merlin_morgana"] = str(merlin_morgana)
            elif role in self.evil_roles and role != "Oberon":
                teammates = [p for p in visible_evils_for_evil if p != pid]
                kwargs["teammates"] = str(teammates)
            
            # 3. 调用提示词生成函数
            sys_msg = get_avalon_prompt(
                role_name=role, 
                player_count=self.player_count, 
                board_config=board_desc, 
                **kwargs
            )
            
            self.agent_memories[pid].append({"role": "system", "content": sys_msg})

    def get_agent(self, pid):
        return next((a for a in self.agents if a.player_id == pid), None)

    def _broadcast(self, text, exclude=[]):
        print(f"[Broadcast] {text}")
        for pid in self.agent_memories:
            if pid not in exclude:
                self.pending_observations[pid].append(f"[Game Announcement/Observation]: {text}")


    def call_agent(self, pid, phase, context=None):
        """
        修改后的 call_agent:
        负责准备记忆和观测信息，但将 Prompt 的构造权交给 Agent.act
        新增：传入 game_condition
        """
        if context is None:
            context = {}
            
        agent = self.get_agent(pid)
        memory = self.agent_memories[pid]
        
        # 取出缓冲区中的观测信息并清空
        buffered_obs = self.pending_observations[pid]
        self.pending_observations[pid] = []
        
        # 调用 Agent 的 act 方法，传入阶段、上下文和游戏状态
        response = agent.act(memory, phase, buffered_obs, context, self.game_condition)
        
        # 此时 memory 已经被 agent.act 更新过了（包含 user prompt 和 assistant response）
        # 返回 response 和 除最新的 assistant 回复之外的 history (用于日志记录 input_msg)
        return response, memory[:-1]

    def save_log(self):
        """
        保存日志逻辑
        新增：保存 game_condition 到 _game_condition.json
        """
        base_dir = "logs"

        if self.log_tag:
            game_type_dir = os.path.join("Avalon", self.log_tag)
        else:
            game_type_dir = "Avalon"

        time_folder = self.start_time.strftime("%Y%m%d_%H%M%S") + "_" + str(random.randint(0,9999))
        target_path = os.path.join(base_dir, game_type_dir, time_folder)
        
        if not os.path.exists(target_path): 
            os.makedirs(target_path)

        x = len(self.agents)
        v = self.game_log.get("final_result", "unfinished")
        base_filename = f"Avalon_Multiturn_{time_folder}_{x}_Players_result_{v}"

        # 1. 保存原有详细 JSON
        with open(os.path.join(target_path, f"{base_filename}.json"), "w", encoding="utf-8") as f:
            json.dump(self.game_log, f, ensure_ascii=False, indent=4)

        # 2. 保存 Game Condition JSON (新增)
        with open(os.path.join(target_path, f"{base_filename}_game_condition.json"), "w", encoding="utf-8") as f:
            json.dump(self.game_condition, f, ensure_ascii=False, indent=4)

        # 3. 保存 TXT 战报
        with open(os.path.join(target_path, f"{base_filename}.txt"), "w", encoding="utf-8") as f:
            f.write("="*30 + " Avalon Game Report " + "="*30 + "\n")
            f.write(f"Game Time: {time_folder}\n")
            f.write(f"Final Result: {v}\n\n")
            
            if "assassination" in self.game_log:
                ass = self.game_log["assassination"]
                f.write(f"[Assassination Moment] Assassin(P{ass.get('assassin')}) -> Target(P{ass.get('target')})\n\n")

            f.write("--- Role Assignment ---\n")
            for pid, info in self.game_log["agents_info"].items():
                f.write(f"{pid}: {info['role']}\n")
            f.write("\n")

            f.write("--- Game Process ---\n")
            for turn in self.game_log["game_process"]:
                r = turn.get("round")
                q_config = turn.get("quest_config", {})
                f.write(f"\n>>> Round {r} (Needs {q_config.get('team_size')} players, Fails required: {q_config.get('req_fails')}) <<<\n")
                
                for attempt in turn.get("attempts", []):
                    leader = attempt.get("leader")
                    track = attempt.get("vote_track")
                    f.write(f"\n  [Proposal] Leader: P{leader} (Fail Track: {track})\n")
                    
                    if "discussions" in attempt:
                        f.write(f"    [Discussion]\n")
                        for disc in attempt["discussions"]:
                            f.write(f"      P{disc['player']}: {disc['statement']}\n")
                    
                    team = attempt.get("proposed_team", [])
                    f.write(f"    [Proposed Team]: {team}\n")
                    
                    if "votes" in attempt:
                        votes_str = ", ".join([f"P{v['player']}:{'✅' if v['vote'] else '❌'}" for v in attempt["votes"]])
                        f.write(f"    [Vote Details]: {votes_str}\n")
                    
                    res = attempt.get("result", "unknown")
                    if attempt.get("vote_result") == "forced":
                        f.write(f"    -> Result: Forced Execution (Vote Track Full)\n")
                    else:
                        f.write(f"    -> Result: {'Approved' if res == 'approved' else 'Rejected'}\n")

                if "mission_result" in turn:
                    mr = turn["mission_result"]
                    outcome = turn.get("outcome")
                    icon = "🔴 Evil Score" if outcome == "evil_point" else "🔵 Good Score"
                    f.write(f"\n  [Mission Execution] {mr.get('fail_cards')} Fail Card(s) -> {icon}\n")
                    f.write("-" * 50 + "\n")
        
        print(f"\n[System] Game log saved to {target_path}")

    def run_turn(self):
        self.round += 1
        config = self.quest_configs[self.player_count][self.round - 1]
        team_size = abs(config)
        req_fails = 2 if config < 0 else 1
        
        # === 修改处：播报中增加容错提示 ===
        fail_note = " [Note: This quest requires 2 Fail cards to fail!]" if req_fails > 1 else ""
        self._broadcast(f"====== Round {self.round} Quest (Needs {team_size} players){fail_note} ======")
        # ==================================
        
        turn_log = {
            "round": self.round,
            "quest_config": {"team_size": team_size, "req_fails": req_fails},
            "attempts": []
        }

        # === 新增：Condition Tracker for Current Round ===
        current_round_tracker = {
            "round_number": self.round,
            "attempts": [],
            "mission_result": None
        }
        # 立即加入到总时间线，以便引用更新
        self.game_condition["game_timeline"].append(current_round_tracker)
        # ================================================
        
        task_approved = False
        attempt_count = 0 

        while not task_approved:
            attempt_count += 1
            attempt_log = {"leader": self.agents[self.leader_idx].player_id, "vote_track": self.vote_track}
            leader_agent = self.agents[self.leader_idx]
            leader_id = leader_agent.player_id
            
            # === 新增：Condition Tracker for Current Attempt ===
            current_attempt_tracker = {
                "attempt_index": attempt_count,
                "leader_id": leader_id,
                "leader_role_summary": leader_agent.role,
                "steps": {
                    "discussion": [],
                    "proposal": None,
                    "voting": None
                }
            }
            current_round_tracker["attempts"].append(current_attempt_tracker)
            # ==================================================

            force_execution = (self.vote_track == 4)
            if force_execution:
                self._broadcast(f"!!! Forced Execution Warning: This team proposal will skip voting and execute directly !!!")
            
            self._broadcast(f"Current Leader: Player {leader_id}")

            # --- 发言阶段 ---
            speak_order_indices = []
            for i in range(1, self.player_count + 1):
                idx = (self.leader_idx + i) % self.player_count
                speak_order_indices.append(idx)
            
            discussions_log = []
            
            for idx in speak_order_indices:
                agent = self.agents[idx]
                pid = agent.player_id
                
                # 修改：不再传递 Prompt 字符串，而是传递 Phase
                resp, messages = self.call_agent(pid, phase="speech")
                
                stmt = self._extract_json(resp, "statement")
                if not stmt: stmt = "..."
                
                log_text = f"Player {pid} says: {stmt}"
                self._broadcast(log_text, exclude=[pid])
                
                discussions_log.append({"player": pid, "statement": stmt, "raw": resp, "input_msg": messages})

                # === 新增：Record Discussion Step ===
                current_attempt_tracker["steps"]["discussion"].append({
                    "agent_info": {
                        "id": pid,
                        "role": agent.role,
                        "model": agent.llm_config.get("name", "unknown")
                    },
                    "answer": {
                        "raw_response": resp,
                        "extracted_result": stmt
                    }
                })
                # ====================================

            attempt_log["discussions"] = discussions_log

            # --- 组队阶段 ---
            # 修改：传递 Phase="proposal" 和必要的 Context (team_size)
            resp_leader, messages = self.call_agent(leader_id, phase="proposal", context={"team_size": team_size})
            
            team = self._extract_json(resp_leader, "team")
            
            if not team or not isinstance(team, list) or len(team) != team_size:
                team = [a.player_id for a in random.sample(self.agents, team_size)]
            
            self._broadcast(f"Leader proposes team: {team}")

            attempt_log["proposed_team"] = team
            attempt_log["proposed_team_details"] = {
                "player": leader_id,
                "team_result": team,
                "raw": resp_leader,       
                "input_msg": messages     
            }

            # === 新增：Record Proposal Step ===
            current_attempt_tracker["steps"]["proposal"] = {
                "agent_info": {
                    "id": leader_id,
                    "role": leader_agent.role,
                    "model": leader_agent.llm_config.get("name", "unknown")
                },
                "answer": {
                    "raw_response": resp_leader,
                    "extracted_result": team
                }
            }
            # ==================================

            # --- 投票阶段 ---
            if force_execution:
                task_approved = True
                self.vote_track = 0
                attempt_log["vote_result"] = "forced"
                # Forced execution implicitly approved
                current_attempt_tracker["steps"]["voting"] = {"final_outcome": "forced_approved"}
            else:
                votes = {}
                votes_log = []
                condition_votes_details = [] # 用于记录到 game_condition

                for agent in self.agents:
                    pid = agent.player_id
                    
                    # 修改：传递 Phase="voting" 和 Context (round)
                    resp, messages = self.call_agent(pid, phase="voting", context={"round": self.round})
                    
                    v = self._extract_json(resp, "vote")
                    if v is None: v = True
                    votes[pid] = v
                    votes_log.append({"player": pid, "vote": v, "raw": resp, "input_msg": messages})

                    # === 新增：Prepare Vote Detail ===
                    condition_votes_details.append({
                        "agent_info": {
                            "id": pid,
                            "role": agent.role,
                            "model": agent.llm_config.get("name", "unknown")
                        },
                        "answer": {
                            "raw_response": resp,
                            "extracted_result": v
                        }
                    })
                    # =================================
                
                vote_str = ", ".join([f"{p}:{'✅' if v else '❌'}" for p,v in votes.items()])
                self._broadcast(f"Vote Result: {vote_str}")
                attempt_log["votes"] = votes_log
                
                if sum(votes.values()) > self.player_count / 2:
                    self._broadcast(">>> Vote Approved!")
                    task_approved = True
                    self.vote_track = 0
                    attempt_log["result"] = "approved"
                    voting_outcome = "approved"
                else:
                    self._broadcast(">>> Vote Rejected, quest delayed.")
                    self.vote_track += 1
                    self.leader_idx = (self.leader_idx + 1) % self.player_count
                    attempt_log["result"] = "rejected"
                    voting_outcome = "rejected"

                # === 新增：Record Voting Step ===
                current_attempt_tracker["steps"]["voting"] = {
                    "votes_details": condition_votes_details,
                    "final_outcome": voting_outcome
                }
                # ================================
            
            turn_log["attempts"].append(attempt_log)
        
        # --- 执行任务 ---
        fail_cards = 0
        mission_log = []
        condition_player_actions = [] # 用于 game_condition

        for pid in team:
            agent = self.get_agent(pid)
            role = agent.role
            can_fail = role in self.evil_roles
            
            # 修改：传递 Phase="execution" 和 Context (round)
            resp, messages = self.call_agent(pid, phase="execution", context={"round": self.round})
            
            success = self._extract_json(resp, "success")
            
            if not can_fail: success = True
            if success is None: success = True
            if not success: fail_cards += 1
            mission_log.append({"player": pid, "action": "success" if success else "fail", "raw": resp, "input_msg": messages})

            # === 新增：Prepare Mission Action Detail ===
            condition_player_actions.append({
                "agent_info": {
                    "id": pid,
                    "role": role,
                    "model": agent.llm_config.get("name", "unknown")
                },
                "answer": {
                    "raw_response": resp,
                    "extracted_result": "Success" if success else "Fail"
                }
            })
            # ===========================================
            
        self._broadcast(f"Mission ended. Result contains {fail_cards} fail card(s).")
        turn_log["mission_result"] = {"fail_cards": fail_cards, "details": mission_log}
        
        if fail_cards >= req_fails:
            self.scores["evil"] += 1
            turn_log["outcome"] = "evil_point"
            condition_outcome = "evil_point"
        else:
            self.scores["good"] += 1
            turn_log["outcome"] = "good_point"
            condition_outcome = "good_point"
            
        self.game_log["game_process"].append(turn_log)
        self.leader_idx = (self.leader_idx + 1) % self.player_count

        # === 新增：Record Mission Result in Condition ===
        current_round_tracker["mission_result"] = {
            "outcome": condition_outcome,
            "fail_cards_count": fail_cards,
            "player_actions": condition_player_actions
        }
        # ================================================

    def run_game(self):
        self._broadcast(f"--- Avalon Start ---")
        try:
            while self.scores["good"] < 3 and self.scores["evil"] < 3:
                self.run_turn()
                self._broadcast(f"Score: Good {self.scores['good']} - Evil {self.scores['evil']}")
                
            if self.scores["evil"] >= 3:
                self.game_log["final_result"] = "evil_win_missions"
                self.game_condition["meta"]["final_result"] = "evil_win_missions" # 新增
                return "evil_win"
            else:
                # 刺杀时刻
                assassin = next((a for a in self.agents if a.role == "Assassin"), None)
                if assassin:
                    # 计算当前局中所有好人角色的 id，作为可刺杀对象候选
                    good_ids = [
                        a.player_id
                        for a in self.agents
                        if a.role in self.good_roles
                    ]
                    # 传递 Phase="assassination" 以及可刺杀对象列表
                    resp, messages = self.call_agent(
                        assassin.player_id,
                        phase="assassination",
                        context={"good_candidate_ids": good_ids},
                    )
                    
                    target = self._extract_json(resp, "target")
                    self.game_log["assassination"] = {"assassin": assassin.player_id, "target": target, "raw": resp, "input_msg": messages}
                    
                    # === 新增：Record Assassination in Condition ===
                    self.game_condition["assassination"] = {
                        "agent_info": {
                            "id": assassin.player_id,
                            "role": "Assassin",
                            "model": assassin.llm_config.get("name", "unknown")
                        },
                        "answer": {
                            "raw_response": resp,
                            "extracted_result": target
                        },
                        "target_details": {}
                    }
                    # ===============================================

                    target_role = self.get_agent(target).role if target else ""
                    
                    # 更新 assassination target details
                    if self.game_condition["assassination"]:
                        self.game_condition["assassination"]["target_details"] = {
                            "target_role": target_role
                        }

                    if target_role == "Merlin":
                        self._broadcast("Assassination successful! Merlin is killed! Evil wins!")
                        self.game_log["final_result"] = "evil_win_assassination"
                        self.game_condition["meta"]["final_result"] = "evil_win_assassination" # 新增
                        return "evil_win"
                
                self._broadcast("Assassination failed! Good wins!")
                self.game_log["final_result"] = "good_win"
                self.game_condition["meta"]["final_result"] = "good_win" # 新增
                return "good_win"
        except Exception as e:
            self.game_log["error"] = str(e)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_log()