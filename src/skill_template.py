"""Prompt templates for skill-aware LM and RM forward passes.

Key difference from Nabla-Reasoner templates:
- Nabla: [prefix (fixed)] [response tokens (optimizable)]
- TTSO:  [prefix (fixed)] [skill tokens (optimizable)] [suffix (fixed)]

The skill sits in the MIDDLE of the prompt, with fixed context on both sides.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SKILL_PLACEHOLDER = "<<<__TTSO_SKILL_PLACEHOLDER__>>>"


def format_skill_prompt(
    tokenizer,
    query: str,
    skill_placeholder: str = SKILL_PLACEHOLDER,
    system_prompt: Optional[str] = None,
) -> str:
    """Build a chat-template string with a placeholder where the skill goes.

    The caller splits the result on ``skill_placeholder`` to get prefix/suffix.
    """
    skill_instruction = (
        f"Use the following skill to solve the problem:\n"
        f"{skill_placeholder}\n\n"
        f"Problem: {query}"
    )
    messages = [{"role": "user", "content": skill_instruction}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class SkillGenerationTemplate:
    """Template for LM forward pass with optimizable skill tokens.

    Layout::

        [prefix_embeds] [soft_skill_embeds] [suffix_embeds]
         ^-- fixed --^   ^-- optimizable --^  ^-- fixed --^

    Where:
    - prefix  = system prompt + "Use the following skill:" tokens
    - skill   = soft token embeddings (from SkillEmbedder)
    - suffix  = query continuation + generation prompt + response tokens

    The LM loss is computed over the **response** portion of the output logits.
    """

    def __init__(
        self,
        query: str,
        response_token_ids: Optional[torch.Tensor],
        skill_token_ids: torch.Tensor,
        lm_model: nn.Module,
        lm_tokenizer,
        system_prompt: Optional[str] = None,
    ):
        self.embed_in = lm_model.get_input_embeddings()
        self.tokenizer = lm_tokenizer
        device = self.embed_in.weight.device

        # Build full prompt text with placeholder
        full_text = format_skill_prompt(
            lm_tokenizer, query, SKILL_PLACEHOLDER, system_prompt
        )
        parts = full_text.split(SKILL_PLACEHOLDER)
        if len(parts) != 2:
            raise ValueError(
                f"Skill placeholder not found or duplicated in template. "
                f"Got {len(parts)} parts."
            )

        # Tokenize prefix and suffix separately.
        # NOTE: Piecewise BPE encoding can differ from full-text encoding at
        # boundaries. This matches Nabla-Reasoner's approach (which has the
        # same property) and works well in practice because chat templates
        # typically insert special tokens at boundaries that act as natural
        # BPE barriers. We validate with a length check below.
        self.prefix_ids = lm_tokenizer.encode(
            parts[0], add_special_tokens=False, return_tensors="pt"
        ).to(device)
        query_suffix_ids = lm_tokenizer.encode(
            parts[1], add_special_tokens=False, return_tensors="pt"
        ).to(device)

        # Track lengths for loss computation
        self.prefix_len = self.prefix_ids.shape[1]
        self.skill_len = (
            skill_token_ids.shape[1] if skill_token_ids.dim() == 2
            else skill_token_ids.shape[0]
        )
        self.query_suffix_len = query_suffix_ids.shape[1]

        # Append response tokens to suffix if provided
        if response_token_ids is not None:
            self.suffix_ids = torch.cat(
                [query_suffix_ids, response_token_ids.to(device)], dim=-1
            )
            self.response_len = response_token_ids.shape[1]
        else:
            self.suffix_ids = query_suffix_ids
            self.response_len = 0

        self.suffix_len = self.suffix_ids.shape[1]

        # Validate: compare piecewise encoding with full-text encoding
        skill_text_for_check = lm_tokenizer.decode(
            skill_token_ids[0] if skill_token_ids.dim() == 2 else skill_token_ids,
            skip_special_tokens=False,
        )
        full_text_check = parts[0] + skill_text_for_check + parts[1]
        full_ids_check = lm_tokenizer.encode(
            full_text_check, add_special_tokens=False, return_tensors="pt"
        )
        piecewise_len = self.prefix_len + self.skill_len + self.query_suffix_len
        if full_ids_check.shape[1] != piecewise_len:
            logger.warning(
                "Tokenization boundary mismatch: full=%d vs piecewise=%d "
                "(diff=%d tokens). This may cause minor loss misalignment.",
                full_ids_check.shape[1], piecewise_len,
                abs(full_ids_check.shape[1] - piecewise_len),
            )

        # Pre-compute fixed embeddings
        self.prefix_embeds = self.embed_in(self.prefix_ids)
        self.suffix_embeds = self.embed_in(self.suffix_ids)

        # Position where response tokens start in the full sequence
        self.response_start = self.prefix_len + self.skill_len + self.query_suffix_len

    def apply(self, soft_skill_embeds: torch.Tensor) -> torch.Tensor:
        """Concatenate [prefix, soft_skill, suffix] embeddings.

        Args:
            soft_skill_embeds: Shape ``[1, skill_len, embed_dim]``.

        Returns:
            Full input embeddings for LM forward pass.
        """
        return torch.cat(
            [
                self.prefix_embeds.to(soft_skill_embeds.device),
                soft_skill_embeds,
                self.suffix_embeds.to(soft_skill_embeds.device),
            ],
            dim=-2,
        )

    def apply_to_token_ids(self, skill_token_ids: torch.Tensor) -> torch.Tensor:
        """Concatenate [prefix, skill_ids, suffix] as token IDs."""
        return torch.cat(
            [
                self.prefix_ids.to(skill_token_ids.device),
                skill_token_ids,
                self.suffix_ids.to(skill_token_ids.device),
            ],
            dim=-1,
        )


class SkillRewardTemplate:
    """Template for RM scoring with optimizable skill tokens.

    Layout::

        [rm_prefix_embeds] [soft_skill_rm_embeds] [rm_suffix_embeds]

    Uses the RM's chat template with a placeholder trick (similar to
    Nabla-Reasoner's ORMTemplate) to locate where the skill goes.
    """

    def __init__(
        self,
        query: str,
        response_text: str,
        skill_token_ids: torch.Tensor,
        rm_model: Optional[nn.Module],
        rm_tokenizer,
        system_prompt: Optional[str] = None,
    ):
        if rm_model is None or rm_tokenizer is None:
            self.prefix_embeds = None
            self.suffix_embeds = None
            return

        self.embed_in = rm_model.get_input_embeddings()
        self.tokenizer = rm_tokenizer
        device = self.embed_in.weight.device

        # Build RM input: [user: skill_instruction + query] [assistant: response]
        skill_instruction = (
            f"Use the following skill to solve the problem:\n"
            f"{SKILL_PLACEHOLDER}\n\n"
            f"Problem: {query}"
        )
        messages = [
            {"role": "user", "content": skill_instruction},
            {"role": "assistant", "content": response_text},
        ]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        full_text = rm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        parts = full_text.split(SKILL_PLACEHOLDER)
        if len(parts) < 2:
            raise ValueError("Skill placeholder not found in RM template.")

        prefix_text = parts[0]
        suffix_text = SKILL_PLACEHOLDER.join(parts[1:])  # handle edge case

        self.prefix_ids = rm_tokenizer.encode(
            prefix_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        self.suffix_ids = rm_tokenizer.encode(
            suffix_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)

        self.prefix_embeds = self.embed_in(self.prefix_ids)
        self.suffix_embeds = self.embed_in(self.suffix_ids)

        self.skill_len = skill_token_ids.shape[1] if skill_token_ids.dim() == 2 else skill_token_ids.shape[0]

    def apply(self, soft_skill_embeds: torch.Tensor) -> torch.Tensor:
        """Concatenate [rm_prefix, soft_skill, rm_suffix] embeddings."""
        if self.prefix_embeds is None:
            raise RuntimeError("RM template not initialized (no RM model provided).")
        return torch.cat(
            [
                self.prefix_embeds.to(soft_skill_embeds.device),
                soft_skill_embeds,
                self.suffix_embeds.to(soft_skill_embeds.device),
            ],
            dim=-2,
        )

    def apply_to_token_ids(self, skill_token_ids: torch.Tensor) -> torch.Tensor:
        """Concatenate [rm_prefix, skill_ids, rm_suffix] as token IDs."""
        return torch.cat(
            [
                self.prefix_ids.to(skill_token_ids.device),
                skill_token_ids,
                self.suffix_ids.to(skill_token_ids.device),
            ],
            dim=-1,
        )
