"""Soft Prompt Optimization for skill tokens (Approach A).

Instead of optimizing discrete token logits via STE (which suffers from
gradient bottleneck with large vocabularies), this module directly optimizes
continuous embedding vectors. After optimization, the embeddings are projected
back to the nearest tokens in the vocabulary via cosine similarity.

Key advantage over DTO: gradients flow directly to embeddings without
passing through a softmax/STE bottleneck, enabling effective optimization
even with vocab_size > 100K.
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .skill_template import SkillGenerationTemplate, SkillRewardTemplate
from . import utils

logger = logging.getLogger(__name__)


class SoftPromptEmbedding(nn.Module):
    """Optimizable continuous skill embeddings.

    Unlike ``DiffSkillLogitsToEmbedding`` which maintains logits over the
    vocabulary, this module directly optimizes embedding vectors in the LM's
    hidden dimension space.

    Args:
        lm_model: Frozen language model (for embedding table).
        lm_tokenizer: LM tokenizer.
        rm_model: Optional frozen reward model.
        rm_tokenizer: Optional RM tokenizer.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        rm_projection_temperature: float = 0.1,
    ):
        super().__init__()
        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.rm_projection_temperature = rm_projection_temperature

        self.skill_embeds: Optional[nn.Parameter] = None
        self.init_embeds: Optional[torch.Tensor] = None

        # LM embedding table (frozen)
        self.register_buffer("lm_embed_table", lm_model.get_input_embeddings().weight)

        # RM embedding table (frozen, vocab-aligned to LM)
        if rm_model is not None and rm_tokenizer is not None:
            aligned_rm_embed = utils.align_vocab(
                rm_model.get_input_embeddings().weight,
                rm_tokenizer,
                lm_tokenizer,
            )
            self.register_buffer("rm_embed_table", aligned_rm_embed)
        else:
            self.rm_embed_table = None

    def initialize(self, skill_token_ids: torch.Tensor) -> None:
        """Initialize soft embeddings from token IDs.

        Args:
            skill_token_ids: Token IDs of shape ``[1, N]``.
        """
        if self.skill_embeds is not None:
            del self.skill_embeds
        # Look up initial embeddings from the LM embedding table
        ids = skill_token_ids.squeeze(0)  # [N]
        init = self.lm_embed_table[ids].unsqueeze(0).clone()  # [1, N, hidden_dim]
        self.init_embeds = init.detach().clone()
        self.skill_embeds = nn.Parameter(init)

    def deconstruct(self) -> None:
        """Release optimizable parameters."""
        if self.skill_embeds is not None:
            del self.skill_embeds
        self.skill_embeds = None
        self.init_embeds = None

    def forward(self) -> Dict[str, torch.Tensor]:
        """Return current soft embeddings for LM and RM.

        RM embeddings are computed via differentiable soft weights to maintain
        gradient flow from RM reward back to skill_embeds.

        Returns:
            Dict with ``lm_embeds`` and optionally ``rm_embeds``.
        """
        result: Dict[str, torch.Tensor] = {
            "lm_embeds": self.skill_embeds,
        }
        if self.rm_embed_table is not None:
            # Differentiable projection: cosine similarity → soft weights → RM embeds
            embeds = self.skill_embeds.squeeze(0)  # [N, hidden_dim]
            embeds_norm = F.normalize(embeds.float(), dim=-1)
            table_norm = F.normalize(self.lm_embed_table.float(), dim=-1)
            soft_weights = F.softmax(
                torch.matmul(embeds_norm, table_norm.T) / self.rm_projection_temperature,
                dim=-1,
            )  # [N, vocab_size]
            rm_embeds = torch.matmul(
                soft_weights.to(self.rm_embed_table.dtype), self.rm_embed_table
            ).unsqueeze(0)  # [1, N, rm_hidden_dim]
            result["rm_embeds"] = rm_embeds
        return result

    @torch.no_grad()
    def project_to_token_ids(self) -> torch.Tensor:
        """Project current embeddings to nearest vocabulary tokens.

        Uses cosine similarity for nearest neighbor lookup.

        Returns:
            Token IDs of shape ``[N]``.
        """
        embeds = self.skill_embeds.squeeze(0)  # [N, hidden_dim]
        # Normalize for cosine similarity
        embeds_norm = F.normalize(embeds.float(), dim=-1)
        table_norm = F.normalize(self.lm_embed_table.float(), dim=-1)
        # [N, vocab_size]
        similarity = torch.matmul(embeds_norm, table_norm.T)
        return similarity.argmax(dim=-1)  # [N]

    @torch.no_grad()
    def decode_text(self) -> str:
        """Decode current embeddings to text via nearest neighbor projection."""
        token_ids = self.project_to_token_ids()
        return self.lm_tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def drift_loss(self) -> torch.Tensor:
        """L2 distance between current and initial embeddings."""
        return F.mse_loss(self.skill_embeds, self.init_embeds)


class SoftPromptTrainer:
    """Gradient-based skill optimization in continuous embedding space.

    Same interface as ``SkillTrainer`` for drop-in replacement.

    Args:
        lm_model: Frozen language model.
        lm_tokenizer: LM tokenizer with chat template.
        rm_model: Frozen reward model.
        rm_tokenizer: RM tokenizer.
        max_iters: Maximum optimization steps.
        learning_rate: Adam learning rate.
        min_lr_ratio: Minimum LR as fraction of initial (cosine decay).
        weight_decay: L2 regularization on embeddings.
        warmup_iters_ratio: Fraction of max_iters for LR warmup.
        reward_coeff: Weight for RM reward in the loss.
        response_nll_coeff: Weight for response NLL term.
        embed_drift_coeff: Weight for embedding drift regularization.
        mixed_precision: Torch dtype for autocast.
        show_train_pbar: Show tqdm progress bar.
        show_train_logs: Print per-iteration logs.
        device: Computation device.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        max_iters: int = 100,
        learning_rate: float = 0.1,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.0,
        warmup_iters_ratio: float = 0.0,
        reward_coeff: float = 1.0,
        response_nll_coeff: float = 1e-3,
        embed_drift_coeff: float = 0.01,
        rm_projection_temperature: float = 0.1,
        mixed_precision: torch.dtype = torch.bfloat16,
        show_train_pbar: bool = False,
        show_train_logs: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.lm_model = lm_model
        self.rm_model = rm_model
        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer

        self.max_iters = max_iters
        self.warmup_iters = math.floor(max_iters * warmup_iters_ratio)
        self.lr = learning_rate
        self.min_lr_ratio = min_lr_ratio
        self.wd = weight_decay
        self.reward_coeff = reward_coeff
        self.response_nll_coeff = response_nll_coeff
        self.embed_drift_coeff = embed_drift_coeff

        self.mixed_precision = mixed_precision
        self.device = device or utils.infer_device_from_model(lm_model)

        self.show_train_pbar = show_train_pbar
        self.show_train_logs = show_train_logs

        self.embedder = SoftPromptEmbedding(
            lm_model, lm_tokenizer, rm_model, rm_tokenizer,
            rm_projection_temperature=rm_projection_temperature,
        )
        self.embedder = self.embedder.to(self.device)

    def _log(self, msg: str, *args, **kwargs) -> None:
        if self.show_train_logs:
            logger.info(msg, *args, **kwargs)

    def optimize(
        self,
        query: str,
        response_text: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Run soft prompt optimization on skill embeddings.

        Args:
            query: User query text.
            response_text: Current response text.
            skill_text: Original skill text to optimize.
            system_prompt: Optional system prompt.

        Returns:
            Dict with ``optimized_skill_text``, ``optimized_logits``,
            ``num_llm_calls``, ``num_grad_steps``, ``final_loss``.
        """
        num_llm_calls = 0
        num_grad_steps = 0

        # Tokenize
        skill_token_ids = self.lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        response_token_ids = self.lm_tokenizer.encode(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        # Build templates (reuse existing template system)
        lm_template = SkillGenerationTemplate(
            query, response_token_ids, skill_token_ids,
            self.lm_model, self.lm_tokenizer, system_prompt,
        )

        use_reward = (
            self.rm_model is not None
            and self.reward_coeff is not None
            and self.reward_coeff != 0
        )
        rm_template = None
        if use_reward:
            rm_template = SkillRewardTemplate(
                query, response_text, skill_token_ids,
                self.rm_model, self.rm_tokenizer, system_prompt,
            )

        # Initialize embedder from skill tokens
        self.embedder.initialize(skill_token_ids)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.embedder.parameters(), lr=self.lr, weight_decay=self.wd
        )
        lr_scheduler = utils.get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.warmup_iters,
            num_training_steps=self.max_iters,
            min_lr_ratio=self.min_lr_ratio,
        )

        prev_text = self.embedder.decode_text()
        self._log("Initial skill: %s", prev_text)
        final_loss_val = float("nan")

        # Training loop
        if self.show_train_pbar and not self.show_train_logs:
            train_iter = tqdm.tqdm(range(self.max_iters), desc="SoftPrompt Opt")
        else:
            train_iter = range(self.max_iters)

        for it in train_iter:
            optimizer.zero_grad()

            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type, dtype=self.mixed_precision):
                outputs = self.embedder()
                lm_embeds = outputs["lm_embeds"]

                # Response NLL
                full_embeds = lm_template.apply(lm_embeds)
                lm_outputs = self.lm_model(inputs_embeds=full_embeds)
                all_logits = lm_outputs.logits

                resp_start = lm_template.response_start
                resp_len = lm_template.response_len
                if resp_len > 0:
                    pred_logits = all_logits[
                        :, resp_start - 1: resp_start + resp_len - 1, :
                    ]
                    response_nll = -F.cross_entropy(
                        pred_logits.reshape(-1, pred_logits.shape[-1]),
                        response_token_ids.reshape(-1),
                        reduction="mean",
                    )
                    response_nll = response_nll * self.response_nll_coeff
                else:
                    response_nll = torch.tensor(0.0, device=self.device)

                # Embedding drift regularization
                drift = self.embedder.drift_loss() * self.embed_drift_coeff

                # Reward
                reward = None
                if use_reward and "rm_embeds" in outputs:
                    rm_full_embeds = rm_template.apply(outputs["rm_embeds"])
                    reward = self.rm_model(
                        inputs_embeds=rm_full_embeds
                    ).logits[0][0]

                # Combined loss: maximize NLL + reward, minimize drift
                loss = -(response_nll) + drift
                if reward is not None:
                    loss = loss - self.reward_coeff * reward

                final_loss_val = loss.item()
                num_llm_calls += 2 if use_reward else 1
                num_grad_steps += 1

            loss.backward()

            # Debug logging
            if it % 10 == 0:
                grad = self.embedder.skill_embeds.grad
                if grad is not None:
                    logger.info(
                        "SoftPrompt iter %d | loss=%.4f | grad_norm=%.6f | "
                        "drift=%.6f",
                        it, final_loss_val, grad.norm().item(),
                        drift.item(),
                    )

            optimizer.step()
            lr_scheduler.step()

            # Log when decoded text changes
            new_text = self.embedder.decode_text()
            if new_text != prev_text:
                self._log(
                    "Iter %d/%d | Loss: %.4f | Skill: %s...",
                    it + 1, self.max_iters, final_loss_val, new_text[:80],
                )
                prev_text = new_text

            if self.show_train_pbar and not self.show_train_logs:
                train_iter.set_description(f"Loss: {final_loss_val:.4f}")

        # Extract results
        optimized_skill_text = self.embedder.decode_text()

        # Cleanup
        self.embedder.deconstruct()
        del optimizer, lr_scheduler
        torch.cuda.empty_cache()

        return {
            "optimized_skill_text": optimized_skill_text,
            "optimized_logits": None,
            "num_llm_calls": num_llm_calls,
            "num_grad_steps": num_grad_steps,
            "final_loss": final_loss_val,
        }

    @torch.no_grad()
    def get_reward_for_text(
        self, query: str, skill_text: str, response: str
    ) -> float:
        """Score a (query+skill, response) pair with the RM."""
        if self.rm_model is None:
            return 0.0
        prompt = f"Use the following skill to solve the problem:\n{skill_text}\n\nProblem: {query}"
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted = self.rm_tokenizer.apply_chat_template(conv, tokenize=False)
        tokenized = self.rm_tokenizer(formatted, return_tensors="pt").to(self.device)
        score = self.rm_model(**tokenized).logits[0][0].item()
        return score
