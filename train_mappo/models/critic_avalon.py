from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from SoDe_RL.avalon_soft_policy_train import SimpleTokenEncoder


class AvalonCritic(nn.Module):
    """
    集中 Critic：复用与 Actor 相同的 token encoder + Transformer。
    输出标量 value：取最后一个有效 token 的表示后接线性层。
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_round_attempt_id: int = 64,
        max_seq_len: int = 512,
        max_text_len: int = 256,
        use_causal_mask: bool = True,
        use_text_cache: bool = True,
        qwen_model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
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
            finetune_text_encoder=False,
        )

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.value_head = nn.Linear(d_model, 1)

        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch_state: Dict[str, Any]) -> torch.Tensor:
        device = self.device

        token_type_ids: torch.Tensor = batch_state["token_type_ids"]
        actor_ids: torch.Tensor = batch_state["actor_ids"]
        round_ids: torch.Tensor = batch_state["round_ids"]
        attempt_ids: torch.Tensor = batch_state["attempt_ids"]
        meta_ids: torch.Tensor = batch_state["meta_ids"]
        task_phase_ids: torch.Tensor = batch_state["task_phase_ids"]
        task_good_mask: torch.Tensor = batch_state["task_good_mask"]
        propose_raw: torch.Tensor = batch_state["propose_raw"]
        vote_raw: torch.Tensor = batch_state["vote_raw"]
        mission_raw: torch.Tensor = batch_state["mission_raw"]
        kg_states: torch.Tensor = batch_state["kg_states"]
        padding_mask: torch.Tensor = batch_state["padding_mask"]
        speech_texts = batch_state.get("speech_texts")
        speech_indices = batch_state.get("speech_indices")

        token_emb = self.token_encoder.encode_batch(
            token_type_ids=token_type_ids,
            actor_ids=actor_ids,
            round_ids=round_ids,
            attempt_ids=attempt_ids,
            meta_ids=meta_ids,
            kg_states=kg_states,
            task_phase_ids=task_phase_ids,
            task_good_mask=task_good_mask,
            propose_raw=propose_raw,
            vote_raw=vote_raw,
            mission_raw=mission_raw,
            speech_texts=None if self.use_text_cache else speech_texts,
            speech_indices=None if self.use_text_cache else speech_indices,
            max_text_len=self.max_text_len,
            device=device,
        )

        bsz, seqlen, _ = token_emb.shape
        pos_ids = torch.arange(seqlen, device=device).clamp(max=self.max_seq_len - 1)
        pos_ids = pos_ids.unsqueeze(0).expand(bsz, -1)
        token_emb = token_emb + self.pos_emb(pos_ids)

        padding_mask = padding_mask.to(device=device, dtype=torch.bool)
        src_mask = None
        if self.use_causal_mask:
            src_mask = torch.triu(
                torch.ones(seqlen, seqlen, device=device, dtype=torch.bool),
                diagonal=1,
            )

        encoded = self.encoder(token_emb, mask=src_mask, src_key_padding_mask=padding_mask)
        valid_counts = (~padding_mask).sum(dim=1)
        last_indices = (valid_counts - 1).clamp(min=0)
        batch_indices = torch.arange(bsz, device=device)
        task_vec_batch = encoded[batch_indices, last_indices, :]
        value = self.value_head(task_vec_batch).squeeze(-1)
        return value

