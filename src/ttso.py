"""Test-Time Skill Optimization (TTSO) orchestrator.

High-level flow:
1. Retrieve skill(s) for the given query
2. Generate initial response using the original skill
3. Evaluate whether skill optimization is needed (selection criteria)
4. If needed, optimize skill via DTO (SkillTrainer)
5. Regenerate response with the optimized skill
6. Apply rejection sampling: accept optimized version only if RM score improves

This module ties together SkillTrainer, skill retrieval, and the
generate-optimize-accept loop.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from .generation import ResponseGenerator
from .skill_trainer import SkillTrainer
from . import utils

logger = logging.getLogger(__name__)


@dataclass
class TTSOResult:
    """Result of a single TTSO decoding run."""

    query: str = ""
    original_skill: str = ""
    optimized_skill: str = ""
    original_response: str = ""
    final_response: str = ""
    original_reward: float = 0.0
    final_reward: float = 0.0
    skill_was_optimized: bool = False
    optimization_accepted: bool = False
    num_llm_calls: int = 0
    num_grad_steps: int = 0
    num_outer_rounds: int = 1
    round_history: List = field(default_factory=list)
    skill_id: Optional[str] = None


@dataclass
class TTSOConfig:
    """Configuration for TTSO decoding."""

    # Optimization
    max_iters: int = 20
    learning_rate: float = 0.01
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    warmup_iters_ratio: float = 0.0
    reward_coeff: float = 1.0
    response_nll_coeff: float = 1e-3
    skill_fluency_coeff: float = 1e-4
    mixed_precision: torch.dtype = torch.bfloat16
    grad_caching: bool = True
    cache_refresh_interval: int = 10  # force full forward every N steps

    # Selection criteria
    min_reward_threshold: Optional[float] = None
    reward_improvement_threshold: float = 0.0

    # Iterative optimization (outer loop)
    max_outer_rounds: int = 1  # 1 = single-round (original), >1 = iterative

    # Rejection sampling
    rejection_sampling: bool = True

    # Generation
    max_generation_len: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95

    # Verbosity
    verbose: int = 1


class TTSODecoding:
    """Main TTSO orchestrator: optimize skills at test time.

    Args:
        lm_model: Frozen language model.
        lm_tokenizer: LM tokenizer with chat template.
        rm_model: Frozen reward model.
        rm_tokenizer: RM tokenizer.
        config: TTSOConfig with hyperparameters.
        device: Computation device.
        vllm_url: Optional vLLM server URL for fast generation.
        vllm_model_name: Model name for vLLM API calls.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        config: Optional[TTSOConfig] = None,
        device: Optional[torch.device] = None,
        vllm_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
    ):
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        self.rm_model = rm_model
        self.rm_tokenizer = rm_tokenizer

        self.cfg = config or TTSOConfig()
        self.device = device or utils.infer_device_from_model(lm_model)

        # Response generator (vLLM or HF)
        self._generator = ResponseGenerator(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            device=self.device,
            vllm_url=vllm_url,
            vllm_model_name=vllm_model_name,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_generation_len=self.cfg.max_generation_len,
        )

        # Initialize the skill trainer (inner optimization loop)
        self.skill_trainer = SkillTrainer(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            max_iters=self.cfg.max_iters,
            learning_rate=self.cfg.learning_rate,
            min_lr_ratio=self.cfg.min_lr_ratio,
            weight_decay=self.cfg.weight_decay,
            warmup_iters_ratio=self.cfg.warmup_iters_ratio,
            reward_coeff=self.cfg.reward_coeff,
            response_nll_coeff=self.cfg.response_nll_coeff,
            skill_fluency_coeff=self.cfg.skill_fluency_coeff,
            mixed_precision=self.cfg.mixed_precision,
            grad_caching=self.cfg.grad_caching,
            cache_refresh_interval=self.cfg.cache_refresh_interval,
            show_train_pbar=(self.cfg.verbose >= 3),
            show_train_logs=(self.cfg.verbose >= 4),
            device=self.device,
        )

    def _log(self, msg: str, *args, level: int = 1) -> None:
        if self.cfg.verbose >= level:
            logger.info(msg, *args)

    def generate_response(
        self,
        query: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a response using the LM with skill context."""
        return self._generator.generate(query, skill_text, system_prompt, seed)

    def should_optimize(self, original_reward: float) -> bool:
        """Decide whether the skill needs optimization.

        Skips optimization if the initial response already scores high enough.

        Args:
            original_reward: Pre-computed RM score for the initial response.
        """
        if self.cfg.min_reward_threshold is None:
            return True
        if self.rm_model is None:
            return True
        return original_reward < self.cfg.min_reward_threshold

    def run(
        self,
        query: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> TTSOResult:
        """Run the full TTSO pipeline.

        Args:
            query: User query.
            skill_text: Retrieved skill text.
            system_prompt: Optional system prompt.
            seed: Random seed for reproducibility.

        Returns:
            TTSOResult with original/optimized skills and responses.
        """
        result = TTSOResult(query=query, original_skill=skill_text)

        # Step 1: Generate initial response with original skill
        self._log("Generating initial response with skill...", level=1)
        original_response = self.generate_response(
            query, skill_text, system_prompt, seed
        )
        result.original_response = original_response
        result.num_llm_calls += 1

        # Step 2: Score initial response
        if self.rm_model is not None:
            original_reward = self.skill_trainer.get_reward_for_text(
                query, skill_text, original_response
            )
            result.original_reward = original_reward
            result.num_llm_calls += 1
            self._log(f"Original RM score: {original_reward:.4f}", level=1)

        # Step 3: Check if optimization is needed
        if not self.should_optimize(result.original_reward):
            self._log("Skill already good enough, skipping optimization.", level=1)
            result.optimized_skill = skill_text
            result.final_response = original_response
            result.final_reward = result.original_reward
            return result

        # Step 4: Optimize skill via DTO
        self._log("Optimizing skill via DTO...", level=1)
        opt_result = self.skill_trainer.optimize(
            query=query,
            response_text=original_response,
            skill_text=skill_text,
            system_prompt=system_prompt,
        )
        optimized_skill = opt_result["optimized_skill_text"]
        result.optimized_skill = optimized_skill
        result.skill_was_optimized = True
        result.num_llm_calls += opt_result["num_llm_calls"]
        result.num_grad_steps = opt_result["num_grad_steps"]

        self._log(f"Optimized skill: {optimized_skill[:100]}...", level=2)

        # Step 5: Generate response with optimized skill
        self._log("Generating response with optimized skill...", level=1)
        optimized_response = self.generate_response(
            query, optimized_skill, system_prompt,
            seed=(seed + 1000) if seed is not None else None,
        )
        result.num_llm_calls += 1

        # Step 6: Rejection sampling
        if self.rm_model is not None:
            optimized_reward = self.skill_trainer.get_reward_for_text(
                query, optimized_skill, optimized_response
            )
            result.num_llm_calls += 1
            self._log(
                f"Optimized RM score: {optimized_reward:.4f} "
                f"(original: {result.original_reward:.4f})",
                level=1,
            )

            if self.cfg.rejection_sampling:
                improvement = optimized_reward - result.original_reward
                accepted = improvement > self.cfg.reward_improvement_threshold
                result.optimization_accepted = accepted

                if accepted:
                    self._log("Optimization ACCEPTED.", level=1)
                    result.final_response = optimized_response
                    result.final_reward = optimized_reward
                else:
                    self._log(
                        f"Optimization REJECTED (improvement={improvement:.4f}).",
                        level=1,
                    )
                    result.final_response = original_response
                    result.final_reward = result.original_reward
                    result.optimized_skill = skill_text  # revert
            else:
                result.optimization_accepted = True
                result.final_response = optimized_response
                result.final_reward = optimized_reward
        else:
            # No RM -> always accept
            result.optimization_accepted = True
            result.final_response = optimized_response
            result.final_reward = 0.0

        return result

    def run_iterative(
        self,
        query: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> TTSOResult:
        """Run iterative TTSO: alternate skill optimization and response generation.

        Each outer round:
          1. Optimize skill given current response
          2. Regenerate response with optimized skill
          3. Score with RM
          4. Accept if improved, otherwise stop early

        This is the main experimental variant. Use ``run()`` for single-round
        baseline comparison.

        Args:
            query: User query.
            skill_text: Initial skill text.
            system_prompt: Optional system prompt.
            seed: Random seed for reproducibility.

        Returns:
            TTSOResult with full round history.
        """
        max_rounds = self.cfg.max_outer_rounds
        result = TTSOResult(query=query, original_skill=skill_text)

        # Step 1: Generate initial response
        self._log("Generating initial response with skill...", level=1)
        current_response = self.generate_response(
            query, skill_text, system_prompt, seed
        )
        result.original_response = current_response
        result.num_llm_calls += 1

        # Step 2: Score initial response
        current_skill = skill_text
        if self.rm_model is not None:
            current_reward = self.skill_trainer.get_reward_for_text(
                query, current_skill, current_response
            )
            result.num_llm_calls += 1
        else:
            current_reward = 0.0
        result.original_reward = current_reward
        best_reward = current_reward
        best_skill = current_skill
        best_response = current_response

        self._log("Round 0 (initial): RM=%.4f", current_reward, level=1)
        result.round_history.append({
            "round": 0,
            "skill": current_skill,
            "response": current_response,
            "reward": current_reward,
        })

        # Step 3: Check if optimization is needed
        if not self.should_optimize(current_reward):
            self._log("Skill already good enough, skipping optimization.", level=1)
            result.optimized_skill = skill_text
            result.final_response = current_response
            result.final_reward = current_reward
            return result

        # Step 4: Iterative optimization loop
        for rnd in range(1, max_rounds + 1):
            self._log("=== Outer round %d/%d ===", rnd, max_rounds, level=1)

            # 4a: Optimize skill given current response
            self._log("Optimizing skill (round %d)...", rnd, level=1)
            opt_result = self.skill_trainer.optimize(
                query=query,
                response_text=current_response,
                skill_text=current_skill,
                system_prompt=system_prompt,
            )
            new_skill = opt_result["optimized_skill_text"]
            result.num_llm_calls += opt_result["num_llm_calls"]
            result.num_grad_steps += opt_result["num_grad_steps"]
            result.skill_was_optimized = True

            # 4b: Regenerate response with new skill
            self._log("Regenerating response (round %d)...", rnd, level=1)
            round_seed = (seed + rnd * 1000) if seed is not None else None
            new_response = self.generate_response(
                query, new_skill, system_prompt, round_seed
            )
            result.num_llm_calls += 1

            # 4c: Score
            if self.rm_model is not None:
                new_reward = self.skill_trainer.get_reward_for_text(
                    query, new_skill, new_response
                )
                result.num_llm_calls += 1
            else:
                new_reward = 0.0

            self._log(
                "Round %d: RM=%.4f (prev=%.4f, delta=%+.4f)",
                rnd, new_reward, current_reward, new_reward - current_reward,
                level=1,
            )
            result.round_history.append({
                "round": rnd,
                "skill": new_skill,
                "response": new_response,
                "reward": new_reward,
            })

            # 4d: Accept or reject this round
            if new_reward > best_reward:
                best_reward = new_reward
                best_skill = new_skill
                best_response = new_response
                self._log("Round %d ACCEPTED (new best=%.4f)", rnd, best_reward, level=1)
            else:
                self._log(
                    "Round %d: no improvement (%.4f <= %.4f), stopping early.",
                    rnd, new_reward, best_reward, level=1,
                )
                break

            # Update for next round
            current_skill = new_skill
            current_response = new_response
            current_reward = new_reward

        # Step 5: Finalize
        result.num_outer_rounds = len(result.round_history) - 1  # exclude round 0
        improvement = best_reward - result.original_reward

        if self.cfg.rejection_sampling and self.rm_model is not None:
            accepted = improvement > self.cfg.reward_improvement_threshold
            result.optimization_accepted = accepted
            if accepted:
                result.optimized_skill = best_skill
                result.final_response = best_response
                result.final_reward = best_reward
                self._log("Iterative TTSO ACCEPTED (delta=%+.4f)", improvement, level=1)
            else:
                result.optimized_skill = skill_text
                result.final_response = result.original_response
                result.final_reward = result.original_reward
                self._log("Iterative TTSO REJECTED (delta=%+.4f)", improvement, level=1)
        else:
            result.optimization_accepted = True
            result.optimized_skill = best_skill
            result.final_response = best_response
            result.final_reward = best_reward

        return result

    def run_batch(
        self,
        queries: List[str],
        skills: List[str],
        system_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> List[TTSOResult]:
        """Run TTSO on a batch of (query, skill) pairs sequentially.

        Args:
            queries: List of user queries.
            skills: List of skill texts (one per query).
            system_prompt: Optional shared system prompt.
            seed: Base random seed.

        Returns:
            List of TTSOResult objects.
        """
        results = []
        for i, (query, skill) in enumerate(zip(queries, skills)):
            self._log(f"\n{'='*60}", level=1)
            self._log(f"Processing query {i+1}/{len(queries)}", level=1)
            result = self.run(query, skill, system_prompt, seed=seed + i)
            results.append(result)
        return results
