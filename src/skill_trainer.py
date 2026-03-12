"""Skill optimization loop via Differentiable Token Optimization (DTO).

Adapted from Nabla-Reasoner's LatentTrainer. Key difference: optimizes
**skill tokens** (in the prompt prefix) rather than response tokens.

The loss combines:
1. Response NLL: log P_LM(response | prefix + skill + query)
2. Skill fluency: log P_LM(skill_tokens | prefix) -- keeps skill readable
3. Reward model: R_RM(query_with_skill, response)
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .skill_embedder import DiffSkillLogitsToEmbedding
from .skill_template import SkillGenerationTemplate, SkillRewardTemplate
from . import utils

logger = logging.getLogger(__name__)


class SkillTrainer:
    """Gradient-based skill optimization engine.

    Args:
        lm_model: Frozen language model for likelihood computation.
        lm_tokenizer: LM tokenizer with chat template.
        rm_model: Frozen reward model for quality scoring.
        rm_tokenizer: RM tokenizer.
        max_iters: Maximum optimization steps.
        learning_rate: Adam learning rate.
        min_lr_ratio: Minimum LR as fraction of initial (cosine decay).
        weight_decay: L2 regularization on skill logits.
        warmup_iters_ratio: Fraction of max_iters for LR warmup.
        reward_coeff: Weight for RM reward in the loss.
        response_nll_coeff: Weight for response NLL term.
        skill_fluency_coeff: Weight for skill fluency term.
        lr_scheduler_type: LR schedule type.
        device: Computation device.
        mixed_precision: Torch dtype for autocast.
        grad_caching: Enable gradient caching when tokens are stable.
        show_train_pbar: Show tqdm progress bar during optimization.
        show_train_logs: Print detailed per-iteration logs.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        max_iters: int = 100,
        learning_rate: float = 0.01,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.0,
        warmup_iters_ratio: float = 0.0,
        reward_coeff: float = 0.1,
        response_nll_coeff: float = 1e-3,
        skill_fluency_coeff: float = 1e-3,
        lr_scheduler_type: str = "cosine",
        device: Optional[torch.device] = None,
        mixed_precision: torch.dtype = torch.bfloat16,
        grad_caching: bool = True,
        cache_refresh_interval: int = 1,
        show_train_pbar: bool = False,
        show_train_logs: bool = False,
    ):
        self.lm_model = lm_model
        self.rm_model = rm_model
        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer

        self.max_iters = max_iters
        self.warmup_iters = math.floor(self.max_iters * warmup_iters_ratio)
        self.lr = learning_rate
        self.min_lr_ratio = min_lr_ratio
        self.wd = weight_decay
        self.reward_coeff = reward_coeff
        self.response_nll_coeff = response_nll_coeff
        self.skill_fluency_coeff = skill_fluency_coeff
        self.lr_scheduler_type = lr_scheduler_type

        self.mixed_precision = mixed_precision
        self.device = device or utils.infer_device_from_model(self.lm_model)

        self.grad_caching = grad_caching
        self.cache_refresh_interval = cache_refresh_interval
        self.show_train_pbar = show_train_pbar
        self.show_train_logs = show_train_logs

        # Skill embedder (reused across optimize calls)
        self.skill_embedder = DiffSkillLogitsToEmbedding(
            lm_model, lm_tokenizer, rm_model, rm_tokenizer
        )
        self.skill_embedder = self.skill_embedder.to(self.device)

    def _log(self, msg: str, *args, **kwargs) -> None:
        if self.show_train_logs:
            logger.info(msg, *args, **kwargs)

    def optimize(
        self,
        query: str,
        response_text: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """Run DTO to optimize skill tokens.

        Args:
            query: User query text.
            response_text: Current response text (from initial generation).
            skill_text: Original skill text to optimize.
            system_prompt: Optional system prompt.

        Returns:
            Dict with ``optimized_skill_text``, ``optimized_logits``,
            ``num_llm_calls``, ``num_grad_steps``, ``final_loss``.
        """
        num_llm_calls = 0
        num_grad_steps = 0

        # Tokenize skill text -> initial logits
        skill_token_ids = self.lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        vocab_size = len(self.lm_tokenizer)

        # Initialize logits as scaled one-hot of current skill tokens
        init_logits = torch.zeros(
            1, skill_token_ids.shape[1], vocab_size,
            device=self.device, dtype=torch.float32,
        )
        init_logits.scatter_(
            2, skill_token_ids.unsqueeze(-1), 1  # moderate init scale
        )

        # Tokenize response for the LM template
        response_token_ids = self.lm_tokenizer.encode(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        # Build templates
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

        # Initialize embedder
        self.skill_embedder.initialize(init_logits)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.skill_embedder.parameters(), lr=self.lr, weight_decay=self.wd
        )
        lr_scheduler = utils.get_scheduler(
            self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_iters,
            num_training_steps=self.max_iters,
            min_lr_ratio=self.min_lr_ratio,
        )

        # Initial state
        current_skill_text = self.skill_embedder.decode_text()
        self._log(f"Initial skill: {current_skill_text}")

        cached_grad = None
        final_loss_val = float("nan")

        # Training loop
        if self.show_train_pbar and not self.show_train_logs:
            train_iter = tqdm.tqdm(range(self.max_iters), desc="Skill Opt")
        else:
            train_iter = range(self.max_iters)

        for it in train_iter:
            optimizer.zero_grad()
            # Force full forward every cache_refresh_interval steps
            force_refresh = (
                self.grad_caching
                and self.cache_refresh_interval > 0
                and it % self.cache_refresh_interval == 0
            )
            skip_grad = (
                self.grad_caching
                and cached_grad is not None
                and not force_refresh
            )

            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type, dtype=self.mixed_precision):
                if skip_grad:
                    soft_onehot = self.skill_embedder(onehot_only=True)["soft_onehot"]
                    loss = torch.dot(cached_grad.view(-1), soft_onehot.view(-1))
                else:
                    outputs = self.skill_embedder(onehot_only=False)
                    soft_onehot = outputs["soft_onehot"]

                    if self.grad_caching and soft_onehot.requires_grad:
                        soft_onehot.retain_grad()

                    loss_dict = self.compute_loss(
                        outputs, lm_template, rm_template, response_token_ids
                    )
                    loss = loss_dict["loss"]
                    final_loss_val = loss.item()

                    num_llm_calls += 2 if use_reward else 1
                    num_grad_steps += 1

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Cache gradient if enabled
            if not skip_grad and self.grad_caching:
                cached_grad = soft_onehot.grad.detach().clone()

            # Check if tokens changed -> invalidate cache
            new_skill_text = self.skill_embedder.decode_text()
            if new_skill_text != current_skill_text:
                cached_grad = None
                self._log(
                    f"Iter {it + 1}/{self.max_iters} | "
                    f"Loss: {final_loss_val:.4f} | "
                    f"Skill: {new_skill_text[:80]}..."
                )
                current_skill_text = new_skill_text

            if self.show_train_pbar and not self.show_train_logs:
                train_iter.set_description(f"Loss: {final_loss_val:.4f}")

        # Extract results
        optimized_logits = self.skill_embedder.get_logits().detach().clone()
        optimized_skill_text = self.skill_embedder.decode_text()

        # Cleanup
        self.skill_embedder.deconstruct()
        del optimizer, lr_scheduler
        torch.cuda.empty_cache()

        return {
            "optimized_skill_text": optimized_skill_text,
            "optimized_logits": optimized_logits,
            "num_llm_calls": num_llm_calls,
            "num_grad_steps": num_grad_steps,
            "final_loss": final_loss_val,
        }

    def compute_loss(
        self,
        soft_embed_outputs: Dict[str, torch.Tensor],
        lm_template: SkillGenerationTemplate,
        rm_template: Optional[SkillRewardTemplate],
        response_token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss for skill optimization.

        Loss = -(response_nll + skill_fluency + reward_coeff * reward)

        Args:
            soft_embed_outputs: From skill_embedder.forward().
            lm_template: LM prompt template.
            rm_template: RM prompt template (optional).
            response_token_ids: Ground truth response token IDs [1, resp_len].

        Returns:
            Dict with ``loss``, ``response_nll``, ``skill_fluency``, ``reward``.
        """
        soft_onehot = soft_embed_outputs["soft_onehot"]
        lm_soft_embeds = soft_embed_outputs["lm_embeds"]

        # Full LM forward: [prefix, soft_skill, suffix(query+response)]
        full_embeds = lm_template.apply(lm_soft_embeds)
        lm_outputs = self.lm_model(inputs_embeds=full_embeds)
        all_logits = lm_outputs.logits  # [1, total_len, vocab]

        # --- Response NLL ---
        # Extract logits that predict response tokens
        resp_start = lm_template.response_start
        resp_len = lm_template.response_len
        if resp_len > 0:
            # Logits at positions [resp_start-1 : resp_start+resp_len-1]
            # predict tokens at positions [resp_start : resp_start+resp_len]
            pred_logits = all_logits[:, resp_start - 1 : resp_start + resp_len - 1, :]
            response_nll = -F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                response_token_ids.reshape(-1),
                reduction="mean",
            )
            response_nll = response_nll * self.response_nll_coeff
        else:
            response_nll = torch.tensor(0.0, device=lm_soft_embeds.device)

        # --- Skill fluency ---
        # How well the LM predicts skill tokens given prefix
        prefix_len = lm_template.prefix_len
        skill_len = lm_template.skill_len
        if skill_len > 0 and self.skill_fluency_coeff > 0:
            skill_pred_logits = all_logits[
                :, prefix_len - 1 : prefix_len + skill_len - 1, :
            ]
            skill_log_probs = F.log_softmax(skill_pred_logits, dim=-1)
            # Detach soft_onehot as target to avoid moving-target gradient bias
            skill_fluency = (skill_log_probs * soft_onehot.detach()).sum() * self.skill_fluency_coeff
        else:
            skill_fluency = torch.tensor(0.0, device=lm_soft_embeds.device)

        # --- Reward model ---
        reward = None
        if rm_template is not None and "rm_embeds" in soft_embed_outputs:
            rm_full_embeds = rm_template.apply(soft_embed_outputs["rm_embeds"])
            reward = self.rm_model(inputs_embeds=rm_full_embeds).logits[0][0]

        # --- Combined loss ---
        loss = -(response_nll + skill_fluency)
        if reward is not None:
            loss = loss - self.reward_coeff * reward

        return {
            "loss": loss,
            "response_nll": response_nll,
            "skill_fluency": skill_fluency,
            "reward": reward,
        }

    @torch.no_grad()
    def get_reward_for_text(
        self, query: str, skill_text: str, response: str
    ) -> float:
        """Score a (query+skill, response) pair with the RM."""
        if self.rm_model is None:
            return 0.0
        prompt = f"Use the following skill:\n{skill_text}\n\nProblem: {query}"
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted = self.rm_tokenizer.apply_chat_template(conv, tokenize=False)
        tokenized = self.rm_tokenizer(formatted, return_tensors="pt").to(self.device)
        score = self.rm_model(**tokenized).logits[0][0].item()
        return score
