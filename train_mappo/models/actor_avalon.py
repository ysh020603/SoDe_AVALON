from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from SoDe_RL.avalon_soft_policy_train import AvalonPolicyModel

from SoDe_Avalon_3.train_mappo.state.action_space import (
    sample_assassination_action,
    sample_discussion_action,
    sample_mission_action,
    sample_proposal_action,
    sample_voting_action,
)


class AvalonActor(nn.Module):
    """
    Actor 包装器：底层复用 `SoDe_RL.AvalonPolicyModel`，输出各 phase 的 logits。
    我们只负责：
    - forward 得到 logits
    - 可选：在给定 phase_id 时直接采样 action/log_prob/entropy（约束在 context 内提供）
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
        self.model = AvalonPolicyModel(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_round_attempt_id=max_round_attempt_id,
            qwen_model_path=qwen_model_path,
            finetune_text_encoder=False,
            max_seq_len=max_seq_len,
            max_text_len=max_text_len,
            use_causal_mask=use_causal_mask,
            use_text_cache=use_text_cache,
        )
        if device is not None:
            self.model.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward_logits(self, batch_obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs_list = self.model(batch_obs)
        if not outputs_list:
            return {}
        return outputs_list[0]

    def sample_action(
        self,
        *,
        phase_id: int,
        logits_dict: Dict[str, torch.Tensor],
        context: Dict[str, Any],
        rng: Optional[Any] = None,
    ):
        from SoDe_Avalon_3.train_mappo.state.token_builder import (
            PHASE_ASSASSINATION,
            PHASE_DISCUSSION,
            PHASE_MISSION,
            PHASE_PROPOSING,
            PHASE_VOTING,
        )

        if phase_id == PHASE_DISCUSSION:
            return sample_discussion_action(
                identity_logits=logits_dict["identity_logits"].squeeze(0),
                intent_logits=logits_dict["intent_logits"].squeeze(0),
                attitude_logits=logits_dict["attitude_logits"].squeeze(0),
                rng=rng,
            )

        if phase_id == PHASE_PROPOSING:
            return sample_proposal_action(
                propose_logits=logits_dict["propose_logits"].squeeze(0),
                team_size=int(context["team_size"]),
                rng=rng,
            )

        if phase_id == PHASE_VOTING:
            return sample_voting_action(vote_logits=logits_dict["vote_logits"].squeeze(0))

        if phase_id == PHASE_MISSION:
            return sample_mission_action(
                mission_logits=logits_dict["mission_logits"].squeeze(0),
                force_success=bool(context.get("force_success", False)),
            )

        if phase_id == PHASE_ASSASSINATION:
            return sample_assassination_action(
                kill_logits=logits_dict["kill_logits"].squeeze(0),
                valid_target_ids=context["valid_target_ids"],
            )

        raise ValueError(f"Unsupported phase_id for sampling: {phase_id}")

