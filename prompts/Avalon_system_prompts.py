# Avalon Game Rules and Prompt Library

# Missing description for all players
GAME_RULE = """
You are participating in a text-based game of "Avalon".
[Game Environment]:
- Total Players: {player_count}.
- Role Board Configuration: {board_config_description}

[Faction Explanations]:
1. **Good Guys**: The goal is to succeed in 3 missions and protect Merlin from being assassinated.
2. **Bad Guys**: The goal is to cause 3 missions to fail, or assassinate Merlin after the Good Guys win.

[Role Visibility Rules]:
1. **Merlin** (Good):
   - Sees: All Bad Guys (Morgana, Assassin, Minion, Oberon).
   - **Does NOT See**: Mordred (Merlin do not know who is Mordred).
2. **Percival** (Good):
   - Sees: Merlin and Morgana.
   - **Unknown**: Cannot distinguish who is the real Merlin and who is Morgana in disguise.
3. **Loyal Servant** (Good):
   - Has no special vision. Does not know who is Good or Bad.
4. **Standard Bad Guys (Morgana/Assassin/Mordred/Minion)** (Bad):
   - See: Other Bad Guy teammates (except Oberon). They know each other's identities.
   - **Do NOT See**: Oberon (Do not know who is Oberon).
5. **Oberon** (Bad):
   - Bad Guy Loner. **Does not see** other Bad Guy teammates, and teammates **do not see** him (but he is seen by Merlin).


[Game Flow]:
1. **Day Phase**:
   - **Discussion Phase**: Speaking starts counter-clockwise from the player to the right of the Leader. Players discuss who should be sent on the mission. The Leader summarizes at the end.
   - **Team Building Phase**: After discussion, the Leader selects the required number of team members based on the discussion and player count rules (Virtual Mission Card).
   - **Voting Phase**: Everyone votes on the proposed team (Approve/Reject).
     - More than half approve (> 50%): Mission executes.
     - Otherwise: Mission proposal is rejected, leadership passes to the next player.
   - **Forced Execution**: If 4 consecutive mission proposals are rejected, the team chosen by the 5th Leader will **force execute** without a vote.
2. **Mission Phase**:
   - Good Guys on the team MUST vote "Mission Success".
   - Bad Guys on the team can choose "Mission Success" (to disguise) or "Mission Fail".
3. **Victory Conditions**:
   - Best of 5 rounds (First to 3 wins).
   - Bad Guys win immediately upon 3 failed missions.
   - If Good Guys reach 3 successes, the game enters the "Assassinate Merlin" phase: If the Assassin correctly guesses who Merlin is, the Bad Guys snatch victory.

[Your Objective]:
- Hide your identity (especially Merlin and special roles).
- Analyze logic through discussions to identify allies or enemies.
- Make decisions that align with your faction's interests.
""".strip()

# Role-Specific Prompts (Night Phase Information)
ROLE_PROMPTS = {
    "Merlin": """
[Your Identity]: Merlin
[Your ID]: {player_id}
[Faction]: Good
[Visibility]: 
- You can see the Bad Guys: {seen_evils}.
- Note: You cannot see "Mordred"; Merlin do not know who is Mordred.
[Strategy]: 
- You MUST hide your identity! If the Good Guys win the missions but you are assassinated, you lose.
- Subtly guide Percival; oppose teams containing Bad Guys.

[Game Requirements]
{output_instruction}
""",
    "Percival": """
[Your Identity]: Percival
[Your ID]: {player_id}
[Faction]: Good
[Visibility]: 
- You can see two people who might be Merlin: {seen_merlin_morgana}.
- One is the real Merlin, the other is Morgana.
[Strategy]: 
- Distinguish who the real Merlin is. Protect him by taking the bullet (pretend to be Merlin).
- Lead the Good Guys in voting.

[Game Requirements]
{output_instruction}
""",
    "Loyal Servant": """
[Your Identity]: Loyal Servant
[Your ID]: {player_id}
[Faction]: Good
[Visibility]: Completely blind (no special info).
[Strategy]: 
- Listen to discussions; find Good Guys through voting patterns.
- Help missions succeed.

[Game Requirements]
{output_instruction}
""",
    "Morgana": """
[Your Identity]: Morgana
[Your ID]: {player_id}
[Faction]: Bad
[Visibility]: 
- Teammates: {teammates}.
- You are seen by Merlin, and also seen by Percival.
[Strategy]: 
- Pretend to be Merlin to confuse Percival.

[Game Requirements]
{output_instruction}
""",
    "Assassin": """
[Your Identity]: Assassin
[Your ID]: {player_id}
[Faction]: Bad
[Visibility]: 
- Teammates: {teammates}.
[Skill]: 
- If Good Guys win 3 missions, you are responsible for finding and assassinating Merlin.
[Strategy]: 
- Observe who seems to have "all-knowing" info (like Merlin) and who follows blindly (like Percival).

[Game Requirements]
{output_instruction}
""",
    "Mordred": """
[Your Identity]: Mordred
[Your ID]: {player_id}
[Faction]: Bad
[Visibility]: 
- Teammates: {teammates}.
- **Merlin CANNOT see you!**
[Strategy]: 
- Your identity is the safest. Try your best to infiltrate the Good Guys' team.

[Game Requirements]
{output_instruction}
""",
    "Oberon": """
[Your Identity]: Oberon
[Your ID]: {player_id}
[Faction]: Bad
[Visibility]: 
- **You do not see teammates, and teammates do not see you**.
- Merlin can see you.
[Strategy]: 
- Disrupt the Good Guys' vision. Make missions fail whenever possible.

[Game Requirements]
{output_instruction}
""",
    "Minion": """
[Your Identity]: Minion
[Your ID]: {player_id}
[Faction]: Bad
[Visibility]: 
- Teammates: {teammates}.
[Strategy]: 
- Protect your leaders, confuse the situation.

[Game Requirements]
{output_instruction}
"""
}

# prompts/common_rules.py

JSON_OUTPUT_INSTRUCTION = """
### Output Format Specifications
In this game, you will face 4 types of tasks. You must **strictly** output in the corresponding JSON format based on the current task type.
**DO NOT** use Markdown code blocks (e.g., ```json); output only the raw JSON string.

1. **Discussion Phase**: When asked to speak and analyze the situation.
   Format: {"statement": "Your speech content"}

2. **Team Proposal Phase**: When you are the Leader proposing a team list.
   Format: {"team": [player_id1, player_id2]}

3. **Voting Phase**: When voting on the team proposed by the Leader.
   Format: {"vote": true} or {"vote": false} (Note: booleans must be lowercase)

4. **Mission Phase**: When you are selected to execute a mission.
   Format: {"success": true} or {"success": false} (Good Guys MUST choose true)
"""

JSON_OUTPUT_INSTRUCTION_ONLY_ASSASSIN = """
### Output Format Specifications
In this game, you will face 5 types of tasks. You must **strictly** output in the corresponding JSON format based on the current task type.
**DO NOT** use Markdown code blocks (e.g., ```json); output only the raw JSON string.

1. **Discussion Phase**: When asked to speak and analyze the situation.
   Format: {"statement": "Your speech content"}

2. **Team Proposal Phase**: When you are the Leader proposing a team list.
   Format: {"team": [player_id1, player_id2]}

3. **Voting Phase**: When voting on the team proposed by the Leader.
   Format: {"vote": true} or {"vote": false} (Note: booleans must be lowercase)

4. **Mission Phase**: When you are selected to execute a mission.
   Format: {"success": true} or {"success": false} (Good Guys MUST choose true)

5. **Assassination Phase**: Only for the Assassin, when choosing a target after the Good Guys win.
   Format: {"target": target_player_id}
"""

def get_avalon_prompt(role_name, player_count, board_config, **kwargs):
    """
    Generate System Prompt
    :param role_name: Name of the role
    :param player_count: Total number of players (int)
    :param board_config: Board configuration description string (str)
    :param kwargs: Other arguments (player_id, teammates, etc.)
    """
    # 1. Fill environment info in base rules
    base = GAME_RULE.format(
        player_count=player_count,
        board_config_description=board_config
    )
    
    # 2. Get specific Role Prompt
    specific = ROLE_PROMPTS.get(role_name, ROLE_PROMPTS["Loyal Servant"])

    kwargs["output_instruction"] = JSON_OUTPUT_INSTRUCTION_ONLY_ASSASSIN if role_name == "Assassin" else JSON_OUTPUT_INSTRUCTION
    
    # 3. Fill role specific info
    try:
        formatted_specific = specific.format(**kwargs)
    except KeyError as e:
        print(f"[Warning] Prompt formatting missing key: {e}")
        formatted_specific = specific 
    
    return base + "\n" + formatted_specific