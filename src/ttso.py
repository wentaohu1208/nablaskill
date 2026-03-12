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

    # Selection criteria
    min_reward_threshold: Optional[float] = None
    reward_improvement_threshold: float = 0.0

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

        self.vllm_url = vllm_url
        self.vllm_model_name = vllm_model_name

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
        """Generate a response using the LM with skill context.

        Uses vLLM if available, otherwise HuggingFace generate.
        """
        prompt = (
            f"Use the following skill to solve the problem:\n"
            f"{skill_text}\n\n"
            f"Problem: {query}"
        )

        if self.vllm_url:
            return self._generate_vllm(prompt, system_prompt, seed)
        return self._generate_hf(prompt, system_prompt, seed)

    def _generate_vllm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate via vLLM server."""
        import requests

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        url = self.vllm_url.rstrip("/") + "/v1/chat/completions"
        body = {
            "model": self.vllm_model_name or "default",
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_generation_len,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed

        try:
            resp = requests.post(url, json=body, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"vLLM request failed: {e!r}")

    def _generate_hf(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate via HuggingFace model.generate()."""
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        text = self.lm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.lm_tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            outputs = self.lm_model.generate(
                input_ids,
                max_new_tokens=self.cfg.max_generation_len,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
                top_p=self.cfg.top_p,
                pad_token_id=self.lm_tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, input_ids.shape[1] :]
        return self.lm_tokenizer.decode(new_tokens, skip_special_tokens=True)

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
