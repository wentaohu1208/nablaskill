"""Sequential skill token state management.

Manages the past (committed) and ahead (optimizable) token states
for sequential token-by-token skill optimization.
"""

from typing import Optional

import torch
import torch.nn as nn


class SequentialSkillStates:
    """Manages past (committed) and ahead (optimizable) skill token states.

    Args:
        skill_token_ids: Full skill token IDs of shape ``[1, N]``.
        lm_model: Frozen language model (for embedding lookup).
        init_logit_scale: Scale for one-hot logit initialization.
        device: Computation device.
    """

    def __init__(
        self,
        skill_token_ids: torch.Tensor,
        lm_model: nn.Module,
        init_logit_scale: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or skill_token_ids.device
        self.embed_table = lm_model.get_input_embeddings().weight
        self.vocab_size = self.embed_table.shape[0]
        self.init_logit_scale = init_logit_scale

        # Full original skill token IDs [N]
        self.original_ids = skill_token_ids.squeeze(0).to(self.device)
        self.num_tokens = self.original_ids.shape[0]

        # Past: committed token IDs (grows from left)
        self.past_ids: list[int] = []

        # Current position index
        self.position = 0

    @property
    def num_past(self) -> int:
        return len(self.past_ids)

    @property
    def num_ahead(self) -> int:
        return self.num_tokens - self.num_past

    @property
    def is_done(self) -> bool:
        return self.num_past >= self.num_tokens

    def get_past_embeds(self) -> Optional[torch.Tensor]:
        """Get frozen embeddings for committed tokens. Shape ``[1, past_len, dim]``."""
        if not self.past_ids:
            return None
        ids = torch.tensor(self.past_ids, device=self.device)
        return self.embed_table[ids].unsqueeze(0).detach()

    def init_ahead_logits(self) -> nn.Parameter:
        """Initialize logits for the remaining ahead tokens.

        Returns:
            ``nn.Parameter`` of shape ``[1, ahead_len, vocab_size]``.
        """
        ahead_ids = self.original_ids[self.num_past:]  # [ahead_len]
        logits = torch.zeros(
            1, self.num_ahead, self.vocab_size,
            device=self.device, dtype=torch.float32,
        )
        logits.scatter_(2, ahead_ids.unsqueeze(0).unsqueeze(-1), self.init_logit_scale)
        return nn.Parameter(logits)

    def commit(self, n: int = 1, ahead_logits: Optional[torch.Tensor] = None) -> None:
        """Commit the first ``n`` ahead tokens via argmax.

        Args:
            n: Number of tokens to commit.
            ahead_logits: Current ahead logits ``[1, ahead_len, vocab_size]``.
        """
        if ahead_logits is not None:
            committed_ids = torch.argmax(ahead_logits[:, :n, :], dim=-1).squeeze(0)
            if committed_ids.dim() == 0:
                committed_ids = committed_ids.unsqueeze(0)
            for i in range(n):
                self.past_ids.append(committed_ids[i].item())
        else:
            # Fallback: commit original tokens
            start = self.num_past
            for i in range(n):
                idx = start + i
                if idx < self.num_tokens:
                    self.past_ids.append(self.original_ids[idx].item())
        self.position = self.num_past
        assert self.num_past + self.num_ahead == self.num_tokens

    def commit_token_ids(self, token_ids: list[int]) -> None:
        """Commit specific token IDs (used by rejection sampling).

        Args:
            token_ids: Token IDs to commit.

        Raises:
            ValueError: If committing would exceed total token count.
        """
        if len(token_ids) + self.num_past > self.num_tokens:
            raise ValueError(
                f"Cannot commit {len(token_ids)} tokens: "
                f"{self.num_past} past + {len(token_ids)} new > {self.num_tokens} total"
            )
        for tid in token_ids:
            self.past_ids.append(tid)
        self.position = self.num_past
        assert self.num_past + self.num_ahead == self.num_tokens

    def get_committed_text(self, tokenizer) -> str:
        """Decode committed (past) tokens to text."""
        if not self.past_ids:
            return ""
        return tokenizer.decode(self.past_ids, skip_special_tokens=True)

    def get_full_skill_ids(
        self, ahead_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get full skill token IDs (past committed + ahead argmax). Shape ``[1, N]``."""
        if ahead_logits is not None:
            ahead_ids = torch.argmax(ahead_logits, dim=-1).squeeze(0)  # [ahead_len]
        else:
            ahead_ids = self.original_ids[self.num_past:]

        past_tensor = torch.tensor(self.past_ids, device=self.device)
        return torch.cat([past_tensor, ahead_ids]).unsqueeze(0)
