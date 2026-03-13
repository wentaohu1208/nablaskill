"""Sequential token-by-token skill optimization via DTO.

Inspired by Nabla-Reasoner's sequential decoding: instead of optimizing all
skill tokens simultaneously, this module optimizes them one (or a few) at a
time, committing each before moving to the next.

For each position i (0 to N-1):
  1. Split skill into past (committed, frozen) and ahead (optimizable via logits)
  2. Build full sequence: [prefix] [past_embeds] [ahead_soft_embeds] [suffix+response]
  3. Forward through LM → compute response NLL + skill fluency + RM reward
  4. Optimize ahead logits via Adam for max_iters steps
  5. Commit: argmax the first ahead position → move to past
  6. Advance to next position

The response is NOT an optimization variable but the forward pass rolls out
through it to compute loss, matching Nabla-Reasoner's approach.
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .skill_embedder import DiffSkillLogitsToEmbedding, straight_through_softmax
from .skill_template import SkillGenerationTemplate, SkillRewardTemplate
from . import utils

logger = logging.getLogger(__name__)


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

    def get_full_skill_ids(self, ahead_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get full skill token IDs (past committed + ahead argmax). Shape ``[1, N]``."""
        if ahead_logits is not None:
            ahead_ids = torch.argmax(ahead_logits, dim=-1).squeeze(0)  # [ahead_len]
        else:
            ahead_ids = self.original_ids[self.num_past:]

        past_tensor = torch.tensor(self.past_ids, device=self.device)
        return torch.cat([past_tensor, ahead_ids]).unsqueeze(0)


class SequentialSkillTrainer:
    """Sequential token-by-token skill optimization.

    Same ``optimize()`` / ``get_reward_for_text()`` interface as
    ``SkillTrainer`` for drop-in replacement.

    Args:
        lm_model: Frozen language model.
        lm_tokenizer: LM tokenizer with chat template.
        rm_model: Frozen reward model.
        rm_tokenizer: RM tokenizer.
        response_generator: ResponseGenerator for rejection sampling.
        max_iters: Optimization steps per token position.
        learning_rate: Adam learning rate.
        min_lr_ratio: Minimum LR ratio for cosine decay.
        weight_decay: L2 regularization.
        warmup_iters_ratio: Fraction of max_iters for LR warmup.
        reward_coeff: Weight for RM reward.
        response_nll_coeff: Weight for response NLL.
        skill_fluency_coeff: Weight for skill fluency.
        init_logit_scale: Initial one-hot logit scale.
        commit_every: Commit N tokens per step.
        mixed_precision: Torch dtype for autocast.
        show_train_pbar: Show progress bar.
        show_train_logs: Print per-iteration logs.
        device: Computation device.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        response_generator=None,
        max_iters: int = 20,
        learning_rate: float = 0.01,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.0,
        warmup_iters_ratio: float = 0.0,
        reward_coeff: float = 1.0,
        response_nll_coeff: float = 1e-3,
        skill_fluency_coeff: float = 1e-4,
        init_logit_scale: float = 3.0,
        commit_every: int = 1,
        mixed_precision: torch.dtype = torch.bfloat16,
        show_train_pbar: bool = False,
        show_train_logs: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.lm_model = lm_model
        self.rm_model = rm_model
        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.response_generator = response_generator

        self.max_iters = max_iters
        self.warmup_iters = math.floor(max_iters * warmup_iters_ratio)
        self.lr = learning_rate
        self.min_lr_ratio = min_lr_ratio
        self.wd = weight_decay
        self.reward_coeff = reward_coeff
        self.response_nll_coeff = response_nll_coeff
        self.skill_fluency_coeff = skill_fluency_coeff
        self.init_logit_scale = init_logit_scale
        self.commit_every = commit_every

        self.mixed_precision = mixed_precision
        self.device = device or utils.infer_device_from_model(lm_model)

        self.show_train_pbar = show_train_pbar
        self.show_train_logs = show_train_logs

        # Skill embedder (for STE and embedding lookup)
        self.skill_embedder = DiffSkillLogitsToEmbedding(
            lm_model, lm_tokenizer, rm_model, rm_tokenizer,
        )
        self.skill_embedder = self.skill_embedder.to(self.device)

    def _log(self, msg: str, *args, **kwargs) -> None:
        if self.show_train_logs:
            logger.info(msg, *args, **kwargs)

    def _build_full_embeds(
        self,
        past_embeds: Optional[torch.Tensor],
        ahead_soft_embeds: torch.Tensor,
        lm_template: SkillGenerationTemplate,
    ) -> torch.Tensor:
        """Build full LM input: [prefix] [past_skill] [ahead_skill] [suffix].

        Args:
            past_embeds: Frozen past skill embeddings ``[1, past_len, dim]`` or None.
            ahead_soft_embeds: Optimizable ahead embeddings ``[1, ahead_len, dim]``.
            lm_template: Template with prefix/suffix embeddings.

        Returns:
            Full input embeddings for LM forward.
        """
        dtype = ahead_soft_embeds.dtype
        dev = ahead_soft_embeds.device
        parts = [lm_template.prefix_embeds.to(device=dev, dtype=dtype)]
        if past_embeds is not None:
            parts.append(past_embeds.to(device=dev, dtype=dtype))
        parts.append(ahead_soft_embeds)
        parts.append(lm_template.suffix_embeds.to(device=dev, dtype=dtype))
        return torch.cat(parts, dim=-2)

    def _build_rm_full_embeds(
        self,
        past_ids: list[int],
        ahead_soft_onehot: torch.Tensor,
        rm_template: SkillRewardTemplate,
    ) -> torch.Tensor:
        """Build full RM input with past token embeds + ahead soft embeds.

        Args:
            past_ids: Committed token IDs.
            ahead_soft_onehot: STE output ``[1, ahead_len, vocab_size]``.
            rm_template: RM template with prefix/suffix embeddings.

        Returns:
            Full RM input embeddings.
        """
        rm_embed_table = self.skill_embedder.rm_embed_in
        assert rm_embed_table is not None, "RM embed table required for _build_rm_full_embeds"
        parts = [rm_template.prefix_embeds.to(ahead_soft_onehot.device)]

        if past_ids:
            past_tensor = torch.tensor(past_ids, device=self.device)
            past_rm_embeds = rm_embed_table[past_tensor].unsqueeze(0)
            parts.append(past_rm_embeds)

        ahead_rm_embeds = torch.matmul(
            ahead_soft_onehot.to(rm_embed_table.dtype), rm_embed_table,
        )
        parts.append(ahead_rm_embeds)
        parts.append(rm_template.suffix_embeds.to(ahead_soft_onehot.device))
        return torch.cat(parts, dim=-2)

    def _optimize_position(
        self,
        states: SequentialSkillStates,
        lm_template: SkillGenerationTemplate,
        rm_template: Optional[SkillRewardTemplate],
        response_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, float]:
        """Optimize ahead tokens at current position.

        Args:
            states: Sequential skill states.
            lm_template: LM template.
            rm_template: RM template (optional).
            response_token_ids: Response token IDs for NLL loss.

        Returns:
            Tuple of (optimized ahead_logits, num_llm_calls, num_grad_steps, final_loss).
        """
        num_llm_calls = 0
        num_grad_steps = 0
        last_loss = float("nan")

        use_reward = (
            self.rm_model is not None
            and self.reward_coeff is not None
            and self.reward_coeff != 0
        )

        # Initialize ahead logits (optimized directly, not via skill_embedder)
        ahead_logits = states.init_ahead_logits()

        optimizer = torch.optim.Adam(
            [ahead_logits], lr=self.lr, weight_decay=self.wd,
        )
        lr_scheduler = utils.get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.warmup_iters,
            num_training_steps=self.max_iters,
            min_lr_ratio=self.min_lr_ratio,
        )

        past_embeds = states.get_past_embeds()

        for it in range(self.max_iters):
            optimizer.zero_grad()

            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type, dtype=self.mixed_precision):
                # STE: logits → soft one-hot → embeddings
                soft_onehot = straight_through_softmax(
                    ahead_logits, hard=True, dim=-1,
                )
                lm_embeds = torch.matmul(
                    soft_onehot.to(self.skill_embedder.lm_embed_in.dtype),
                    self.skill_embedder.lm_embed_in,
                )

                # Build full LM input and forward
                full_embeds = self._build_full_embeds(
                    past_embeds, lm_embeds, lm_template,
                )
                lm_outputs = self.lm_model(inputs_embeds=full_embeds)
                all_logits = lm_outputs.logits

                # Response NLL
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

                # Skill fluency (over ahead tokens only)
                # Predict ahead tokens given prefix + past tokens
                ahead_start = lm_template.prefix_len + states.num_past
                ahead_len = states.num_ahead
                if ahead_len > 0 and self.skill_fluency_coeff > 0:
                    skill_pred_logits = all_logits[
                        :, ahead_start - 1: ahead_start + ahead_len - 1, :
                    ]
                    skill_log_probs = F.log_softmax(skill_pred_logits, dim=-1)
                    skill_fluency = (
                        skill_log_probs * soft_onehot.detach()
                    ).sum() * self.skill_fluency_coeff
                else:
                    skill_fluency = torch.tensor(0.0, device=self.device)

                # RM reward
                reward = None
                if use_reward and rm_template is not None:
                    rm_full_embeds = self._build_rm_full_embeds(
                        states.past_ids, soft_onehot, rm_template,
                    )
                    reward = self.rm_model(
                        inputs_embeds=rm_full_embeds,
                    ).logits[0][0]

                # Combined loss
                loss = -(response_nll + skill_fluency)
                if reward is not None:
                    loss = loss - self.reward_coeff * reward

                last_loss = loss.item()
                num_llm_calls += 2 if use_reward else 1
                num_grad_steps += 1

            loss.backward()

            if it % 10 == 0:
                grad = ahead_logits.grad
                if grad is not None:
                    logger.info(
                        "Pos %d/%d | Iter %d | loss=%.4f | grad_norm=%.6f",
                        states.num_past, states.num_tokens,
                        it, loss.item(), grad.norm().item(),
                    )

            optimizer.step()
            lr_scheduler.step()

        del optimizer, lr_scheduler
        return ahead_logits.data, num_llm_calls, num_grad_steps, last_loss

    @torch.no_grad()
    def _evaluate_trajectory_reward(
        self,
        states: SequentialSkillStates,
        candidate_token_id: int,
        ahead_logits: torch.Tensor,
        commit_offset: int,
        query: str,
        system_prompt: Optional[str],
        seed: Optional[int],
        prior_commit_ids: Optional[list[int]] = None,
    ) -> float:
        """Evaluate RM reward via full trajectory: decode skill → generate → RM.

        Constructs the full skill by placing ``candidate_token_id`` at the
        current commit position, argmax-decoding the remaining ahead logits,
        then generates a response and scores it with the RM.

        Args:
            states: Current sequential states (not modified).
            candidate_token_id: Token ID to evaluate.
            ahead_logits: Full ahead logits ``[1, ahead_len, V]``.
            commit_offset: Offset within ahead_logits for the candidate.
            query: User query for response generation.
            system_prompt: Optional system prompt.
            seed: Random seed for generation.
            prior_commit_ids: Already-decided tokens in this commit batch
                (positions 0..commit_offset-1). Used when ``commit_every > 1``.

        Returns:
            RM reward score (higher is better).
        """
        if self.response_generator is None:
            raise RuntimeError(
                "response_generator required for trajectory-based rejection"
            )
        if self.rm_model is None:
            raise RuntimeError("RM model required for trajectory-based rejection")

        # Build full skill token IDs: past + prior_commit + candidate + remaining ahead argmax
        remaining_ahead = ahead_logits[:, commit_offset + 1:, :]
        if remaining_ahead.shape[1] > 0:
            remaining_ids = torch.argmax(remaining_ahead, dim=-1).squeeze(0)
            if remaining_ids.dim() == 0:
                remaining_ids = remaining_ids.unsqueeze(0)
            remaining_list = remaining_ids.tolist()
        else:
            remaining_list = []

        all_ids = (
            states.past_ids
            + (prior_commit_ids or [])
            + [candidate_token_id]
            + remaining_list
        )
        skill_text = self.lm_tokenizer.decode(all_ids, skip_special_tokens=True)

        # Generate response with this skill
        response = self.response_generator.generate(
            query, skill_text, system_prompt, seed,
        )

        # Score with RM
        reward = self.get_reward_for_text(query, skill_text, response)
        return reward

    def optimize(
        self,
        query: str,
        response_text: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
        reward_old: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """Run sequential token-by-token skill optimization.

        For each position: optimize all ahead logits, check if the first token
        changed. If changed and RM is available, run trajectory-based rejection
        sampling (decode full skill → generate response → RM score → compare
        against ``reward_old``). Accept only if reward improves; dynamically
        update ``reward_old`` after each accept.

        Args:
            query: User query text.
            response_text: Current response text (fixed during inner loop).
            skill_text: Original skill text to optimize.
            system_prompt: Optional system prompt.
            reward_old: Initial RM reward baseline for rejection sampling.
                If None and RM exists, computed from the original trajectory.
            seed: Random seed for rejection sampling generation.

        Returns:
            Dict with ``optimized_skill_text``, ``optimized_logits``,
            ``num_llm_calls``, ``num_grad_steps``, ``final_loss``,
            ``rejection_accepted``, ``rejection_rejected``, ``reward_old``.
        """
        total_llm_calls = 0
        total_grad_steps = 0
        final_loss = float("nan")

        use_rejection = (
            self.rm_model is not None
            and self.response_generator is not None
        )

        # Compute initial reward_old if not provided
        if use_rejection and reward_old is None:
            reward_old = self.get_reward_for_text(query, skill_text, response_text)
            total_llm_calls += 1
            logger.info("Initial reward_old: %.4f", reward_old)

        # Tokenize
        skill_token_ids = self.lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt",
        ).to(self.device)
        response_token_ids = self.lm_tokenizer.encode(
            response_text, add_special_tokens=False, return_tensors="pt",
        ).to(self.device)

        # Build templates (once, positions stay the same)
        lm_template = SkillGenerationTemplate(
            query, response_token_ids, skill_token_ids,
            self.lm_model, self.lm_tokenizer, system_prompt,
        )

        use_reward_in_loss = (
            self.rm_model is not None
            and self.reward_coeff is not None
            and self.reward_coeff != 0
        )
        rm_template = None
        if use_reward_in_loss:
            rm_template = SkillRewardTemplate(
                query, response_text, skill_token_ids,
                self.rm_model, self.rm_tokenizer, system_prompt,
            )

        # Initialize sequential states
        states = SequentialSkillStates(
            skill_token_ids, self.lm_model,
            init_logit_scale=self.init_logit_scale,
            device=self.device,
        )

        num_positions = states.num_tokens
        logger.info(
            "Sequential DTO: %d skill tokens, commit_every=%d, max_iters=%d/pos",
            num_positions, self.commit_every, self.max_iters,
        )

        # Outer loop: iterate over token positions
        if self.show_train_pbar and not self.show_train_logs:
            pos_iter = tqdm.tqdm(
                range(0, num_positions, self.commit_every),
                desc="Sequential DTO",
            )
        else:
            pos_iter = range(0, num_positions, self.commit_every)

        num_accepted = 0
        num_rejected = 0

        for pos in pos_iter:
            if states.is_done:
                break

            # Optimize ahead tokens at current position
            ahead_logits, calls, steps, pos_loss = self._optimize_position(
                states, lm_template, rm_template, response_token_ids,
            )
            total_llm_calls += calls
            total_grad_steps += steps
            final_loss = pos_loss

            # Per-token rejection sampling: compare optimized vs original
            n_commit = min(self.commit_every, states.num_ahead)
            commit_ids: list[int] = []

            for c in range(n_commit):
                opt_token = torch.argmax(ahead_logits[:, c, :], dim=-1).item()
                orig_token = states.original_ids[states.num_past + c].item()

                if opt_token == orig_token:
                    # Same token — no need to compare, commit directly
                    commit_ids.append(opt_token)
                elif not use_rejection:
                    # No RM or no generator — accept optimized token directly
                    commit_ids.append(opt_token)
                else:
                    # Different token — trajectory-based rejection sampling
                    # Decode full skill with opt_token → generate → RM
                    pos_seed = (seed + states.num_past + c) if seed is not None else None
                    reward_new = self._evaluate_trajectory_reward(
                        states, opt_token, ahead_logits,
                        commit_offset=c, query=query,
                        system_prompt=system_prompt, seed=pos_seed,
                        prior_commit_ids=commit_ids[:c] if c > 0 else None,
                    )
                    # 1 generate call + 1 RM call
                    total_llm_calls += 2

                    accepted = reward_new > reward_old
                    if accepted:
                        commit_ids.append(opt_token)
                        num_accepted += 1
                        self._log(
                            "Pos %d: ACCEPT opt=%d (reward_new=%.4f > reward_old=%.4f)",
                            states.num_past + c, opt_token,
                            reward_new, reward_old,
                        )
                        # Dynamic reward_old update: raise the bar
                        reward_old = reward_new
                    else:
                        commit_ids.append(orig_token)
                        num_rejected += 1
                        self._log(
                            "Pos %d: REJECT opt=%d (reward_new=%.4f <= reward_old=%.4f)",
                            states.num_past + c, opt_token,
                            reward_new, reward_old,
                        )

            states.commit_token_ids(commit_ids)

            self._log(
                "Committed %d token(s) at pos %d/%d: '%s'",
                n_commit, pos, num_positions,
                states.get_committed_text(self.lm_tokenizer)[-40:],
            )

        logger.info(
            "Rejection sampling: %d accepted, %d rejected (out of %d changed tokens)",
            num_accepted, num_rejected, num_accepted + num_rejected,
        )

        # Final optimized skill text
        full_ids = states.get_full_skill_ids()
        optimized_skill_text = self.lm_tokenizer.decode(
            full_ids.squeeze(0), skip_special_tokens=True,
        )

        # Cleanup
        if self.skill_embedder.is_initialized():
            self.skill_embedder.deconstruct()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "optimized_skill_text": optimized_skill_text,
            "optimized_logits": None,
            "num_llm_calls": total_llm_calls,
            "num_grad_steps": total_grad_steps,
            "final_loss": final_loss,
            "rejection_accepted": num_accepted,
            "rejection_rejected": num_rejected,
            "reward_old": reward_old,
        }

    @torch.no_grad()
    def get_reward_for_text(
        self, query: str, skill_text: str, response: str,
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
