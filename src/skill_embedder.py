"""Differentiable Skill-to-Embedding module.

Adapted from Nabla-Reasoner's DiffLogitsToEmbedding. Optimizes skill token
logits via Straight-Through Estimator (STE) and produces soft embeddings
for both LM and RM forward passes.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

logger = logging.getLogger(__name__)


def straight_through_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    gumbel_noise: float = -1.0,
    dim: int = -1,
) -> torch.Tensor:
    """Softmax with optional Gumbel noise and straight-through estimator.

    Args:
        logits: Raw logits tensor.
        tau: Temperature for softmax.
        hard: If True, use STE (hard forward, soft backward).
        gumbel_noise: Scale of Gumbel noise. <= 0 disables noise.
        dim: Dimension to apply softmax.

    Returns:
        Soft (or hard via STE) probability tensor.
    """
    if gumbel_noise > 0:
        u = torch.rand_like(logits, memory_format=torch.legacy_contiguous_format)
        u = u.clamp_(min=1e-6, max=1.0 - 1e-6)
        gumbels = -torch.log(-torch.log(u)) * gumbel_noise
        y_soft = F.softmax((logits + gumbels) / tau, dim=dim)
    else:
        y_soft = F.softmax(logits / tau, dim=dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


class DiffSkillLogitsToEmbedding(nn.Module):
    """Differentiable skill logits -> soft embeddings for LM and RM.

    Maintains a trainable ``nn.Parameter`` of shape ``[1, num_skill_tokens, vocab_size]``
    that is optimized via gradient descent. The forward pass converts logits
    into soft one-hot vectors (via STE) and multiplies with embedding tables.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        hard: bool = True,
        temperature: float = 1.0,
        gumbel_noise: float = -1.0,
    ):
        super().__init__()
        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer

        self.hard = hard
        self.tau = temperature
        self.gumbel_noise = gumbel_noise

        self.skill_logits: Optional[nn.Parameter] = None

        # LM embedding table (frozen)
        self.register_buffer("lm_embed_in", lm_model.get_input_embeddings().weight)

        # RM embedding table (frozen, vocab-aligned to LM)
        if rm_model is not None and rm_tokenizer is not None:
            aligned_rm_embed = utils.align_vocab(
                rm_model.get_input_embeddings().weight,
                rm_tokenizer,
                lm_tokenizer,
            )
            self.register_buffer("rm_embed_in", aligned_rm_embed)
        else:
            self.rm_embed_in = None

    def is_initialized(self) -> bool:
        return self.skill_logits is not None

    def initialize(self, init_logits: torch.Tensor) -> None:
        """Initialize skill logits for optimization.

        Args:
            init_logits: Tensor of shape ``[1, num_skill_tokens, vocab_size]``.
        """
        if self.skill_logits is not None:
            del self.skill_logits
        self.skill_logits = nn.Parameter(init_logits.clone())

    def deconstruct(self) -> None:
        """Release optimizable parameters."""
        if self.skill_logits is not None:
            del self.skill_logits
        self.skill_logits = None

    def forward(self, onehot_only: bool = False) -> Dict[str, torch.Tensor]:
        """Produce soft embeddings from current skill logits.

        Args:
            onehot_only: If True, skip embedding multiplication.

        Returns:
            Dict with keys ``soft_onehot``, and optionally ``lm_embeds``, ``rm_embeds``.
        """
        soft_onehot = straight_through_softmax(
            self.skill_logits,
            tau=self.tau,
            hard=self.hard,
            gumbel_noise=self.gumbel_noise,
            dim=-1,
        )

        if onehot_only:
            return {"soft_onehot": soft_onehot}

        lm_embeds = torch.matmul(
            soft_onehot.to(self.lm_embed_in.dtype), self.lm_embed_in
        )
        result: Dict[str, torch.Tensor] = {
            "soft_onehot": soft_onehot,
            "lm_embeds": lm_embeds,
        }
        if self.rm_embed_in is not None:
            rm_embeds = torch.matmul(
                soft_onehot.to(self.rm_embed_in.dtype), self.rm_embed_in
            )
            result["rm_embeds"] = rm_embeds
        return result

    @torch.no_grad()
    def get_logits(self) -> torch.Tensor:
        return self.skill_logits

    @torch.no_grad()
    def argmax_decode(self) -> torch.Tensor:
        """Decode skill logits to token IDs via argmax."""
        return torch.argmax(self.skill_logits, dim=-1)

    @torch.no_grad()
    def decode_text(self) -> str:
        """Decode current skill logits to text."""
        token_ids = self.argmax_decode().squeeze(0)
        return self.lm_tokenizer.decode(token_ids, skip_special_tokens=True)
