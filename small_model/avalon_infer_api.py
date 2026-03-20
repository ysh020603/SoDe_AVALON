import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None
    AutoModel = None


# =========================
# 常量与结构（从训练脚本抽取）
# =========================

PHASE_DISCUSSION = 0
PHASE_PROPOSING = 1
PHASE_VOTING = 2
PHASE_MISSION = 3
PHASE_ASSASSINATION = 4

# phase 与 JSON steps 中 key 的对应关系
PHASE_STR_TO_ID = {
    "discussion": PHASE_DISCUSSION,
    "proposal": PHASE_PROPOSING,
    "voting": PHASE_VOTING,
    "mission": PHASE_MISSION,
    "assassination": PHASE_ASSASSINATION,
}

TOKEN_TYPE_CTX = 0
TOKEN_TYPE_SPEECH = 1
TOKEN_TYPE_PROPOSE = 2
TOKEN_TYPE_VOTE = 3
TOKEN_TYPE_MISSION = 4
TOKEN_TYPE_ASSASSIN = 5
TOKEN_TYPE_TASK = 6

MAX_PLAYERS = 10

# 每轮任务的人数要求与坏票容忍：正数=人数，负数=人数且可容忍 1 张坏票
MISSION_SIZE_RULES = [2, 3, 3, -4, 4]


def make_meta_tuple(
    success_count: int, fail_count: int, attempt_index: int
) -> Tuple[int, int, int, int, int]:
    round_idx = min(success_count + fail_count, len(MISSION_SIZE_RULES) - 1)
    rule = MISSION_SIZE_RULES[round_idx]
    mission_required_size = abs(rule)
    bad_vote_tolerance = 1 if rule < 0 else 0
    return (
        int(success_count),
        int(fail_count),
        int(attempt_index),
        int(mission_required_size),
        int(bad_vote_tolerance),
    )


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


class SimpleTokenEncoder(nn.Module):
    """
    将 TokenStruct 序列编码为 (B, L, d_model)。
    """

    def __init__(
        self,
        d_model: int = 1024,
        max_round_attempt_id: int = 64,
        qwen_model_path: Optional[str] = None,
        finetune_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.type_emb = nn.Embedding(8, 128)
        self.actor_emb = nn.Embedding(MAX_PLAYERS + 1, 128)
        self.time_emb = nn.Embedding(max_round_attempt_id, 128)
        self.meta_emb = nn.Embedding(576, 128)

        # kg_emb: 第一维是自己的身份 id（0-5），后面 7 维是每个座位的可见状态
        self.kg_emb = nn.Embedding(6, 32)
        self.kg_proj = nn.Linear(8 * 32, 512)

        self.speech_proj = nn.Linear(1024, 512)
        self.propose_proj = nn.Linear(7, 512)
        self.vote_proj = nn.Linear(7, 512)
        self.mission_proj = nn.Linear(2, 512)
        # TASK：phase 5 维 one-hot + 刺杀时 7 维好人 mask
        self.task_proj = nn.Linear(5 + 7, 512)

        if qwen_model_path and AutoTokenizer is not None and AutoModel is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
            self.text_model = AutoModel.from_pretrained(qwen_model_path)
            for p in self.text_model.parameters():
                p.requires_grad = finetune_text_encoder
        else:
            self.tokenizer = None
            self.text_model = None

    def encode_batch(
        self,
        *,
        token_type_ids: torch.Tensor,
        actor_ids: torch.Tensor,
        round_ids: torch.Tensor,
        attempt_ids: torch.Tensor,
        meta_ids: torch.Tensor,
        kg_states: torch.Tensor,
        task_phase_ids: torch.Tensor,
        task_good_mask: torch.Tensor,
        propose_raw: torch.Tensor,
        vote_raw: torch.Tensor,
        mission_raw: torch.Tensor,
        speech_texts: Optional[List[str]],
        speech_indices: Optional[torch.Tensor],
        max_text_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if token_type_ids.numel() == 0:
            return torch.empty(0, 0, self.d_model, device=device)

        token_type_ids = token_type_ids.to(device)
        actor_ids = actor_ids.to(device)
        round_ids = round_ids.to(device)
        attempt_ids = attempt_ids.to(device)
        meta_ids = meta_ids.to(device)
        kg_states = kg_states.to(device)
        task_phase_ids = task_phase_ids.to(device)
        task_good_mask = task_good_mask.to(device)
        propose_raw = propose_raw.to(device)
        vote_raw = vote_raw.to(device)
        mission_raw = mission_raw.to(device)

        bsz, seqlen = token_type_ids.shape

        actor_ids = actor_ids.clamp(min=0, max=MAX_PLAYERS)
        time_ids = (round_ids * 8 + attempt_ids).clamp(min=0, max=self.time_emb.num_embeddings - 1)
        meta_ids = meta_ids.clamp(min=0, max=self.meta_emb.num_embeddings - 1)

        header = torch.cat(
            [
                self.type_emb(token_type_ids),
                self.actor_emb(actor_ids),
                self.time_emb(time_ids),
                self.meta_emb(meta_ids),
            ],
            dim=-1,
        )

        body = torch.zeros(bsz, seqlen, 512, device=device, dtype=self.speech_proj.weight.dtype)

        is_ctx = token_type_ids == TOKEN_TYPE_CTX
        if is_ctx.any():
            kg_emb = self.kg_emb(kg_states).view(bsz, -1)
            kg_proj = self.kg_proj(kg_emb).unsqueeze(1).expand(-1, seqlen, -1)
            body = torch.where(is_ctx.unsqueeze(-1), kg_proj, body)

        is_task = token_type_ids == TOKEN_TYPE_TASK
        if is_task.any():
            phase_ids = task_phase_ids.clamp(min=0, max=4)
            one_hot = F.one_hot(phase_ids, num_classes=5).to(dtype=body.dtype, device=device)
            good_mask = task_good_mask.to(dtype=one_hot.dtype, device=device)
            task_raw = torch.cat([one_hot, good_mask], dim=-1)
            task_body = self.task_proj(task_raw)
            body = torch.where(is_task.unsqueeze(-1), task_body, body)

        is_propose = token_type_ids == TOKEN_TYPE_PROPOSE
        if is_propose.any():
            propose_body = self.propose_proj(propose_raw.view(-1, 7)).view(bsz, seqlen, -1)
            body = torch.where(is_propose.unsqueeze(-1), propose_body, body)

        is_vote = token_type_ids == TOKEN_TYPE_VOTE
        if is_vote.any():
            vote_body = self.vote_proj(vote_raw.view(-1, 7)).view(bsz, seqlen, -1)
            body = torch.where(is_vote.unsqueeze(-1), vote_body, body)

        is_mission = token_type_ids == TOKEN_TYPE_MISSION
        if is_mission.any():
            mission_body = self.mission_proj(mission_raw.view(-1, 2)).view(bsz, seqlen, -1)
            body = torch.where(is_mission.unsqueeze(-1), mission_body, body)

        is_speech = token_type_ids == TOKEN_TYPE_SPEECH
        if (
            is_speech.any()
            and speech_texts is not None
            and speech_indices is not None
            and len(speech_texts) == speech_indices.shape[0]
            and self.tokenizer is not None
            and self.text_model is not None
        ):
            encoded = self.tokenizer(
                speech_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_text_len,
            ).to(device)
            outputs = self.text_model(**encoded)
            hidden = outputs.last_hidden_state
            attn_mask = encoded["attention_mask"]
            lengths = (attn_mask.sum(dim=1) - 1).clamp(min=0, max=hidden.size(1) - 1)
            idx = torch.arange(hidden.size(0), device=device)
            pooled = hidden[idx, lengths, :].to(dtype=body.dtype)
            speech_vecs = self.speech_proj(pooled)
            for (b_idx, t_idx), vec in zip(speech_indices.tolist(), speech_vecs):
                if 0 <= b_idx < bsz and 0 <= t_idx < seqlen:
                    body[b_idx, t_idx, :] = vec

        return torch.cat([header.to(dtype=body.dtype), body], dim=-1)

    @staticmethod
    def _build_kg_states(players_info: Dict[int, Dict[str, Any]], my_id: int) -> List[int]:
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

            if my_role == "Merlin":
                states.append(2 if is_bad(role) else 0)
            elif my_role == "Percival":
                states.append(3 if role in {"Merlin", "Morgana"} else 0)
            elif my_role in {"Assassin", "Morgana", "Mordred", "Minion"}:
                states.append(2 if (is_bad(role) and role != "Oberon") else 0)
            elif my_role == "Oberon":
                states.append(0)
            else:
                states.append(0)

        while len(states) < 8:
            states.append(0)
        return states[:8]


class AvalonPolicyModel(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        max_round_attempt_id: int = 64,
        qwen_model_path: Optional[str] = None,
        finetune_text_encoder: bool = False,
        max_seq_len: int = 512,
        max_text_len: int = 256,
        use_causal_mask: bool = True,
        use_text_cache: bool = False,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.use_causal_mask = use_causal_mask
        self.use_text_cache = use_text_cache

        self.token_encoder = SimpleTokenEncoder(
            d_model=d_model,
            max_round_attempt_id=max_round_attempt_id,
            qwen_model_path=qwen_model_path,
            finetune_text_encoder=finetune_text_encoder,
        )

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.identity_head = nn.Linear(d_model, 10)
        self.intent_head = nn.Linear(d_model, 5)
        self.attitude_head = nn.Linear(d_model, MAX_PLAYERS * 3)
        self.propose_head = nn.Linear(d_model, 7)
        self.vote_head = nn.Linear(d_model, 2)
        self.mission_head = nn.Linear(d_model, 2)
        self.kill_head = nn.Linear(d_model, MAX_PLAYERS)

    def forward(self, batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device

        token_emb = self.token_encoder.encode_batch(
            token_type_ids=batch["token_type_ids"],
            actor_ids=batch["actor_ids"],
            round_ids=batch["round_ids"],
            attempt_ids=batch["attempt_ids"],
            meta_ids=batch["meta_ids"],
            kg_states=batch["kg_states"],
            task_phase_ids=batch["task_phase_ids"],
            task_good_mask=batch["task_good_mask"],
            propose_raw=batch["propose_raw"],
            vote_raw=batch["vote_raw"],
            mission_raw=batch["mission_raw"],
            speech_texts=None if self.use_text_cache else batch.get("speech_texts"),
            speech_indices=None if self.use_text_cache else batch.get("speech_indices"),
            max_text_len=self.max_text_len,
            device=device,
        )

        bsz, seqlen, _ = token_emb.shape
        pos_ids = torch.arange(seqlen, device=device).clamp(max=self.max_seq_len - 1)
        token_emb = token_emb + self.pos_emb(pos_ids.unsqueeze(0).expand(bsz, -1))

        padding_mask = batch["padding_mask"].to(device=device, dtype=torch.bool)
        src_mask = None
        if self.use_causal_mask:
            src_mask = torch.triu(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool), diagonal=1)

        encoded = self.encoder(token_emb, mask=src_mask, src_key_padding_mask=padding_mask)

        valid_counts = (~padding_mask).sum(dim=1)
        last_indices = (valid_counts - 1).clamp(min=0)
        batch_indices = torch.arange(bsz, device=device)
        task_vec_batch = encoded[batch_indices, last_indices, :]

        outputs_list: List[Dict[str, torch.Tensor]] = []
        labels_list: List[DecisionLabels] = batch["labels_list"]
        for i, labels in enumerate(labels_list):
            task_vec = task_vec_batch[i : i + 1, :]
            outputs: Dict[str, torch.Tensor] = {}
            if labels.phase_id == PHASE_DISCUSSION:
                outputs["identity_logits"] = self.identity_head(task_vec)
                outputs["intent_logits"] = self.intent_head(task_vec)
                outputs["attitude_logits"] = self.attitude_head(task_vec).view(-1, MAX_PLAYERS, 3)
            if labels.phase_id == PHASE_PROPOSING:
                outputs["propose_logits"] = self.propose_head(task_vec)
            if labels.phase_id == PHASE_VOTING:
                outputs["vote_logits"] = self.vote_head(task_vec)
            if labels.phase_id == PHASE_MISSION:
                outputs["mission_logits"] = self.mission_head(task_vec)
            if labels.phase_id == PHASE_ASSASSINATION:
                outputs["kill_logits"] = self.kill_head(task_vec)
            outputs_list.append(outputs)
        return outputs_list


def collate_fn(batch: List[Tuple[List[TokenStruct], DecisionLabels, Dict[str, Any]]]) -> Dict[str, Any]:
    if not batch:
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

    batch_size = len(tokens_batch)
    lengths = [len(seq) for seq in tokens_batch]
    max_len = max(lengths) if lengths else 0

    token_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    actor_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    round_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attempt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    meta_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    task_phase_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    task_good_mask = torch.zeros(batch_size, max_len, 7, dtype=torch.float32)
    propose_raw = torch.zeros(batch_size, max_len, 7, dtype=torch.float32)
    vote_raw = torch.zeros(batch_size, max_len, 7, dtype=torch.float32)
    mission_raw = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)

    phase_ids = torch.zeros(batch_size, dtype=torch.long)
    my_ids = torch.zeros(batch_size, dtype=torch.long)
    kg_states = torch.zeros(batch_size, 8, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

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
            if len(meta_raw) >= 5:
                meta_compact = (
                    meta_raw[0] * 96
                    + min(meta_raw[1], 3) * 24
                    + min(meta_raw[2], 3) * 6
                    + (min(max(meta_raw[3], 2), 4) - 2) * 2
                    + min(meta_raw[4], 1)
                )
                meta_compact = max(0, min(int(meta_compact), 575))
            else:
                meta_compact = 0
            meta_ids[b_idx, t_idx] = int(meta_compact)

            if t.token_type == TOKEN_TYPE_TASK:
                phase = int(t.body.get("phase", 0))
                phase = max(0, min(4, phase))
                task_phase_ids[b_idx, t_idx] = phase
                good_mask = t.body.get("good_mask", [0.0] * 7)
                if isinstance(good_mask, list) and len(good_mask) >= 7:
                    for i in range(7):
                        try:
                            task_good_mask[b_idx, t_idx, i] = float(good_mask[i])
                        except Exception:
                            task_good_mask[b_idx, t_idx, i] = 0.0
            elif t.token_type == TOKEN_TYPE_PROPOSE:
                team_mask = t.body.get("team_mask", [0.0] * 7)
                if isinstance(team_mask, list) and len(team_mask) >= 7:
                    for i in range(7):
                        try:
                            propose_raw[b_idx, t_idx, i] = float(team_mask[i])
                        except Exception:
                            propose_raw[b_idx, t_idx, i] = 0.0
            elif t.token_type == TOKEN_TYPE_VOTE:
                vote_vec = t.body.get("vote_vec", [0.0] * 7)
                if isinstance(vote_vec, list) and len(vote_vec) >= 7:
                    for i in range(7):
                        try:
                            vote_raw[b_idx, t_idx, i] = float(vote_vec[i])
                        except Exception:
                            vote_raw[b_idx, t_idx, i] = 0.0
            elif t.token_type == TOKEN_TYPE_MISSION:
                mission_vec = t.body.get("mission_vec", [0.0] * 2)
                if isinstance(mission_vec, list) and len(mission_vec) >= 2:
                    for i in range(2):
                        try:
                            mission_raw[b_idx, t_idx, i] = float(mission_vec[i])
                        except Exception:
                            mission_raw[b_idx, t_idx, i] = 0.0
            elif t.token_type == TOKEN_TYPE_SPEECH:
                text = t.body.get("text", "")
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
        kg_list = SimpleTokenEncoder._build_kg_states(players_info, int(my_ids[b_idx].item()))
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


class PredictFromLogRequest(BaseModel):
    """
    主推荐接口：发送原始对局日志(dict) + selector。
    phase: 对应 JSON 中的 key，取 "discussion"|"proposal"|"voting"|"mission"|"assassination"，或 0..4
    """

    log: Dict[str, Any]
    phase: Union[str, int]
    player: int
    round_number: int
    attempt_index: int


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_policy_model(
    weights_path: str,
    device: torch.device,
) -> "AvalonPolicyModel":
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # 兼容 DistributedDataParallel 保存的 "module." 前缀
    if isinstance(state, dict):
        keys = list(state.keys())
        if keys and keys[0].startswith("module."):
            state = {k[len("module.") :]: v for k, v in state.items()}

    # 根据 checkpoint 中的 pos_emb 大小自动对齐 max_seq_len
    max_seq_len = 512
    if isinstance(state, dict):
        pos_key = None
        if "pos_emb.weight" in state:
            pos_key = "pos_emb.weight"
        elif "module.pos_emb.weight" in state:
            pos_key = "module.pos_emb.weight"
        if pos_key is not None:
            try:
                max_seq_len = int(state[pos_key].shape[0])
            except Exception:
                max_seq_len = 512

    qwen_model_path = os.getenv(
        "QWEN_EMB_PATH",
        "/data/shy/model/Qwen/Qwen3-Embedding-0___6B",
    )
    model = AvalonPolicyModel(
        d_model=1024,
        n_heads=8,
        n_layers=4,
        max_round_attempt_id=64,
        qwen_model_path=qwen_model_path,
        finetune_text_encoder=False,
        max_seq_len=max_seq_len,
        max_text_len=256,
        use_causal_mask=True,
        use_text_cache=False,
    )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


_TOKEN_TYPE_NAME = {
    TOKEN_TYPE_CTX: "CTX",
    TOKEN_TYPE_SPEECH: "SPEECH",
    TOKEN_TYPE_PROPOSE: "PROPOSE",
    TOKEN_TYPE_VOTE: "VOTE",
    TOKEN_TYPE_MISSION: "MISSION",
    TOKEN_TYPE_ASSASSIN: "ASSASSIN",
    TOKEN_TYPE_TASK: "TASK",
}

_PHASE_NAME = {
    PHASE_DISCUSSION: "DISCUSSION",
    PHASE_PROPOSING: "PROPOSING",
    PHASE_VOTING: "VOTING",
    PHASE_MISSION: "MISSION",
    PHASE_ASSASSINATION: "ASSASSINATION",
}


def _summarize_tokens(tokens: List[TokenStruct]) -> List[Dict[str, Any]]:
    seq: List[Dict[str, Any]] = []
    for idx, t in enumerate(tokens):
        seq.append(
            {
                "idx": idx,
                "token_type": _TOKEN_TYPE_NAME.get(t.token_type, str(t.token_type)),
                "actor_id": t.actor_id,
                "round": t.round_index,
                "attempt": t.attempt_index,
            }
        )
    return seq


def _tokens_detail(tokens: List[TokenStruct]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for idx, t in enumerate(tokens):
        base: Dict[str, Any] = {
            "idx": idx,
            "token_type": _TOKEN_TYPE_NAME.get(t.token_type, str(t.token_type)),
            "actor_id": t.actor_id,
            "round": t.round_index,
            "attempt": t.attempt_index,
            "meta_tuple": list(t.meta_tuple),
        }
        body: Dict[str, Any] = {}
        if t.token_type == TOKEN_TYPE_SPEECH:
            body["text"] = t.body.get("text", "")
        elif t.token_type == TOKEN_TYPE_PROPOSE:
            body["team_mask"] = t.body.get("team_mask", [])
        elif t.token_type == TOKEN_TYPE_VOTE:
            body["vote_vec"] = t.body.get("vote_vec", [])
        elif t.token_type == TOKEN_TYPE_MISSION:
            body["mission_vec"] = t.body.get("mission_vec", [])
        elif t.token_type == TOKEN_TYPE_TASK:
            body["phase"] = t.body.get("phase")
            body["good_mask"] = t.body.get("good_mask", [])
        elif t.token_type == TOKEN_TYPE_CTX:
            body["phase"] = t.body.get("phase")
            body["players_info"] = t.body.get("players_info")
            body["my_id"] = t.body.get("my_id")
            body["my_role_id"] = t.body.get("my_role_id")
        else:
            body = dict(t.body)
        base["body"] = body
        details.append(base)
    return details


def run_inference_on_samples(
    model: AvalonPolicyModel,
    samples: List[Tuple[List[TokenStruct], DecisionLabels, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    file_role_cache: Dict[str, Dict[int, str]] = {}

    batch = collate_fn(samples)
    # forward 内部会自行将张量移动到正确 device
    outputs_list = model(batch)
    labels_list: List[DecisionLabels] = batch["labels_list"]
    extra_list: List[Dict[str, Any]] = batch.get("extra_list", [])

    results: List[Dict[str, Any]] = []
    for i, (sample, labels, outputs) in enumerate(
        zip(samples, labels_list, outputs_list)
    ):
        tokens, _labels_from_sample, extra = sample
        last_token = tokens[-1] if tokens else TokenStruct(
            token_type=TOKEN_TYPE_TASK,
            actor_id=0,
            round_index=0,
            attempt_index=0,
            meta_tuple=(0, 0, 0, 0, 0),
            body={},
        )
        actor_id = tokens[0].actor_id if tokens else 0
        meta_extra = extra_list[i] if i < len(extra_list) else extra

        logits: Dict[str, Any] = {}
        if "identity_logits" in outputs:
            logits["identity_logits"] = outputs["identity_logits"].detach().cpu().tolist()
        if "intent_logits" in outputs:
            logits["intent_logits"] = outputs["intent_logits"].detach().cpu().tolist()
        if "attitude_logits" in outputs:
            logits["attitude_logits"] = outputs["attitude_logits"].detach().cpu().tolist()
        if "propose_logits" in outputs:
            logits["propose_logits"] = outputs["propose_logits"].detach().cpu().tolist()
        if "vote_logits" in outputs:
            logits["vote_logits"] = outputs["vote_logits"].detach().cpu().tolist()
        if "mission_logits" in outputs:
            logits["mission_logits"] = outputs["mission_logits"].detach().cpu().tolist()
        if "kill_logits" in outputs:
            logits["kill_logits"] = outputs["kill_logits"].detach().cpu().tolist()

        # === 解析玩家角色信息（更鲁棒） ===
        players_info = meta_extra.get("players_info") or extra.get("players_info") or {}
        role_name: Optional[str] = None

        if isinstance(players_info, dict):
            try:
                # 先按 int key 查
                info = players_info.get(actor_id)
                # 如果 key 是字符串，再按 str(actor_id) 查一次
                if info is None:
                    info = players_info.get(str(actor_id))
                if isinstance(info, dict):
                    role_name = info.get("role") or info.get("identity") or info.get("role_name")
            except Exception:
                # players_info 结构异常时直接跳过，走后备方案
                pass

        if role_name is None:
            # 从原始日志 / *_game_condition.json 中解析 players_config 作为后备
            file_path = extra.get("file") or meta_extra.get("file")
            if isinstance(file_path, str):
                if file_path not in file_role_cache:
                    role_map: Dict[int, str] = {}
                    try:
                        # 1) 先尝试当前文件
                        with open(file_path, "r", encoding="utf-8") as f:
                            raw = json.load(f)
                        cfg = raw.get("players_config")

                        # 2) 如果当前文件没有 players_config，尝试同名 *_game_condition.json
                        if not isinstance(cfg, dict):
                            alt_path = None
                            if file_path.endswith(".json"):
                                alt_path = file_path[:-5] + "_game_condition.json"
                            if alt_path and os.path.isfile(alt_path):
                                try:
                                    with open(alt_path, "r", encoding="utf-8") as f:
                                        raw_alt = json.load(f)
                                    cfg = raw_alt.get("players_config")
                                    file_path = alt_path  # 以后缓存直接用 condition 文件
                                except Exception:
                                    cfg = None

                        if isinstance(cfg, dict):
                            for pid_str, info in cfg.items():
                                try:
                                    pid_int = int(pid_str)
                                except Exception:
                                    continue
                                if isinstance(info, dict):
                                    role_map[pid_int] = info.get("role") or info.get("identity") or info.get("role_name") or "unknown"
                    except Exception:
                        role_map = {}

                    file_role_cache[file_path] = role_map

                role_name = file_role_cache.get(file_path, {}).get(actor_id)

        result = {
            "phase_id": labels.phase_id,
            "phase_name": _PHASE_NAME.get(labels.phase_id, str(labels.phase_id)),
            "round_number": last_token.round_index,
            "attempt_index": last_token.attempt_index,
            "actor_id": actor_id,
            "model_name": extra.get("model_name") or meta_extra.get("model_name"),
            "role": role_name or extra.get("role"),
            "logits": logits,
            "token_count": len(tokens),
            "token_sequence": _summarize_tokens(tokens),
            "tokens_detail": _tokens_detail(tokens),
        }
        results.append(result)

    return results


app = FastAPI(title="Avalon Soft Policy Inference API")

_DEVICE = _select_device()
_MODEL_WEIGHTS = "/data1/shy/SoDe_RL/output/avalon_soft_policy.pt"
_MODEL_WEIGHTS = os.getenv("AVALON_POLICY_WEIGHTS", _MODEL_WEIGHTS)
if not os.path.isfile(_MODEL_WEIGHTS):
    raise RuntimeError(f"Model weights not found at `{_MODEL_WEIGHTS}`")
_MODEL = load_policy_model(_MODEL_WEIGHTS, _DEVICE)


def _parse_phase(phase: Union[str, int]) -> int:
    """将 phase 字符串或整数解析为 0..4。"""
    if isinstance(phase, int):
        if 0 <= phase <= 4:
            return phase
        raise ValueError(f"phase 整数必须在 0..4 之间，当前为 {phase}")
    if isinstance(phase, str):
        s = phase.strip().lower()
        if s in PHASE_STR_TO_ID:
            return PHASE_STR_TO_ID[s]
        raise ValueError(
            f"phase 字符串必须为 {list(PHASE_STR_TO_ID.keys())} 之一，当前为 '{phase}'"
        )
    raise ValueError(f"phase 必须为 str 或 int，当前类型为 {type(phase)}")


def _validate_token_sequence(
    tokens: List[TokenStruct],
    phase_id: int,
    player: int,
) -> None:
    """
    校验构造出的 token 序列：缺失、结构不符时抛出 ValueError。
    """
    if not tokens:
        raise ValueError("Token 序列为空，无法进行推理")

    # 必须有且仅有一个 CTX 在开头
    if tokens[0].token_type != TOKEN_TYPE_CTX:
        raise ValueError(
            f"Token 序列必须以 CTX 开头，当前首 token 类型为 {tokens[0].token_type}"
        )
    ctx = tokens[0]
    if ctx.actor_id != player:
        raise ValueError(
            f"CTX 的 actor_id 应与 player 一致，期望 {player}，实际 {ctx.actor_id}"
        )
    body = ctx.body or {}
    if "players_info" not in body or not body["players_info"]:
        raise ValueError("CTX body 中缺少 players_info，无法构造知识图谱状态")
    if "my_id" not in body:
        raise ValueError("CTX body 中缺少 my_id")
    if "my_role_id" not in body:
        raise ValueError("CTX body 中缺少 my_role_id")

    # 必须以 TASK 结尾
    if tokens[-1].token_type != TOKEN_TYPE_TASK:
        raise ValueError(
            f"Token 序列必须以 TASK 结尾，当前末 token 类型为 {tokens[-1].token_type}"
        )
    task_body = tokens[-1].body or {}
    if "phase" not in task_body:
        raise ValueError("TASK body 中缺少 phase 字段")

    # SPEECH token 需有 text
    for i, t in enumerate(tokens):
        if t.token_type == TOKEN_TYPE_SPEECH:
            text = (t.body or {}).get("text")
            if not isinstance(text, str):
                raise ValueError(f"Token[{i}] SPEECH 的 body.text 必须为字符串，当前为 {type(text)}")

    # meta_tuple 必须为 5 元组
    for i, t in enumerate(tokens):
        mt = t.meta_tuple
        if not isinstance(mt, (tuple, list)) or len(mt) != 5:
            raise ValueError(
                f"Token[{i}] meta_tuple 必须为长度 5 的元组，当前为 {type(mt)} len={len(mt) if hasattr(mt, '__len__') else 'N/A'}"
            )


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


def _build_history_for_decision(
    history_tokens: List[TokenStruct],
    phase_id: int,
    round_number: int,
    attempt_index: int,
) -> List[TokenStruct]:
    if phase_id == PHASE_DISCUSSION:
        return [
            t
            for t in history_tokens
            if not (
                t.token_type == TOKEN_TYPE_SPEECH
                and t.round_index == round_number
                and t.attempt_index == attempt_index
            )
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
    从原始对局日志 dict 中，构造与训练时一致的单个决策点 token 序列。
    支持决策点不在 log 中的情况（仅需包含构造 token 序列所需的全部历史数据）。
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
        raise ValueError(
            f"round_number {round_number} 不在 game_timeline 中，当前轮次: {round_numbers}"
        )

    success_count = 0
    fail_count = 0
    history_tokens: List[TokenStruct] = []

    def _ctx_token(scoreboard: Tuple[int, int, int, int, int]) -> TokenStruct:
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
                "my_role_id": get_self_identity_id((players_info.get(int(player)) or {}).get("role", "unknown")),
            },
        )

    for round_data in game_timeline:
        rn = int(round_data.get("round_number", 0))
        attempts = round_data.get("attempts", []) or []
        last_attempt_index_in_round = 1

        if rn == round_number:
            attempt_indices = [int(a.get("attempt_index", 1)) for a in attempts]
            if attempt_index not in attempt_indices:
                raise ValueError(
                    f"attempt_index {attempt_index} 不在 round {round_number} 的 attempts 中，当前: {attempt_indices}"
                )

        for attempt in attempts:
            ai = int(attempt.get("attempt_index", 1))
            last_attempt_index_in_round = ai
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
                    tokens: List[TokenStruct] = [_ctx_token(scoreboard)]
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

            if phase_id == PHASE_DISCUSSION and rn == round_number and ai == attempt_index:
                found_in_discussion = any(
                    int(d.get("agent_info", {}).get("id", -1)) == player for d in discussions
                )
                if not found_in_discussion:
                    global_hist = _build_history_for_decision(
                        history_tokens, PHASE_DISCUSSION, rn, ai
                    )
                    tokens: List[TokenStruct] = [_ctx_token(scoreboard)]
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
                tokens = [_ctx_token(scoreboard)]
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
            if phase_id == PHASE_VOTING and rn == round_number and ai == attempt_index and not proposal:
                raise ValueError(
                    "phase voting 需要 steps.proposal 存在以构造 token 序列，当前 attempt 中 proposal 缺失"
                )
            voting = steps.get("voting", {}) or {}
            votes_details: List[Dict[str, Any]] = voting.get("votes_details", []) or []
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
                        tokens = [_ctx_token(scoreboard)]
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
                    tokens = [_ctx_token(scoreboard)]
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

        # ========== MISSION（按 round_data.mission_result） ==========
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
                raise ValueError(
                    f"phase mission 需要 player {player} 在 mission_result.player_actions 中，"
                    f"当前任务队伍: {mission_team_ids}"
                )
            good_cards = 0
            bad_cards = 0
            for act in actions:
                a_answer_tmp = act.get("answer", {}) or {}
                res_tmp = a_answer_tmp.get("extracted_result", "")
                if res_tmp == "Success":
                    good_cards += 1
                elif res_tmp == "Fail":
                    bad_cards += 1

            scoreboard_mission = make_meta_tuple(
                success_count, fail_count, int(last_attempt_index_in_round)
            )

            for act in actions:
                a_agent = act.get("agent_info", {}) or {}
                pid = int(a_agent.get("id", -1))

                if phase_id == PHASE_MISSION and rn == round_number and pid == player:
                    global_hist = _build_history_for_decision(
                        history_tokens, PHASE_MISSION, rn, int(last_attempt_index_in_round)
                    )
                    tokens = [
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
        raise ValueError(
            "phase assassination 需要 log 中存在 assassination 数据，当前缺失"
        )
    if assassination:
        a_info = assassination.get("agent_info", {}) or {}
        assassin_id = int(a_info.get("id", -1))
        if phase_id == PHASE_ASSASSINATION and assassin_id == player:
            scoreboard = make_meta_tuple(success_count, fail_count, 0)
            good_roles = {"Merlin", "Percival", "Loyal Servant"}
            good_mask = [0.0] * 7
            for pid in range(1, 8):
                role = (players_info.get(pid) or {}).get("role", "")
                if role in good_roles:
                    good_mask[pid - 1] = 1.0

            global_hist = _build_history_for_decision(history_tokens, PHASE_ASSASSINATION, 0, 0)
            tokens = [_ctx_token(scoreboard)]
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
    if phase_id == PHASE_ASSASSINATION and assassination:
        a_info = assassination.get("agent_info", {}) or {}
        assassin_id = int(a_info.get("id", -1))
        raise ValueError(
            f"phase assassination 需要 player 为刺客(id={assassin_id})，当前 player={player}"
        )

    raise ValueError(
        "No decision matched selector "
        f"(phase={phase_id}, player={player}, round_number={round_number}, attempt_index={attempt_index})."
        " 请确认 log 中包含构造该决策点 token 序列所需的全部历史数据。"
    )


@app.post("/predict_from_log")
def predict_from_log(req: PredictFromLogRequest) -> Dict[str, Any]:
    if not isinstance(req.log, dict):
        raise HTTPException(status_code=400, detail="log must be a JSON object/dict.")
    try:
        tokens, labels, extra = build_decision_tokens_from_log(
            log=req.log,
            phase=req.phase,
            player=int(req.player),
            round_number=int(req.round_number),
            attempt_index=int(req.attempt_index),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build decision tokens from log: {e}")

    results = run_inference_on_samples(_MODEL, [(tokens, labels, extra)])
    if not results:
        raise HTTPException(status_code=500, detail="Inference returned empty result.")
    result = results[0]
    result["decision_index"] = 0
    result["total_decisions"] = 1
    return result

