"""TextGrad-style LLM-based skill rewriting (Approach B).

Instead of gradient-based token/embedding optimization, this module uses
the LM itself to iteratively rewrite skills based on reward feedback.

Flow per rewrite iteration:
1. Generate response with current skill
2. Score with RM
3. Generate natural language feedback from reward signal
4. LLM rewrites the skill based on feedback
5. Accept if improved, otherwise keep current

Key advantage: skill always remains coherent LLM-generated text.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .generation import ResponseGenerator

logger = logging.getLogger(__name__)

FEEDBACK_PROMPT_TEMPLATE = """\
You are an expert at analyzing problem-solving skills. Given the following:

**Skill used:**
{skill}

**Problem:**
{query}

**Response generated using the skill:**
{response}

**Quality score:** {reward:.4f} (higher is better, previous best: {best_reward:.4f})

Analyze what specific aspects of the skill could be improved to produce a \
better response. Focus on:
- Are the workflow steps clear and actionable?
- Is anything missing that would help solve this type of problem?
- Are there unnecessary or confusing steps?

Provide 2-3 specific, actionable suggestions for improvement."""

REWRITE_PROMPT_TEMPLATE = """\
You are a skill optimization expert. Rewrite the following skill to improve \
its effectiveness at guiding problem-solving, based on the feedback provided.

**Original skill:**
{skill}

**Feedback for improvement:**
{feedback}

**Important rules:**
- Keep the same overall structure (Goal + Workflow steps)
- Make the improvements suggested in the feedback
- Keep the skill concise and actionable
- Do NOT add unnecessary complexity

**Rewritten skill:**"""


class TextGradTrainer:
    """LLM-based skill rewriting optimizer.

    Same ``optimize()`` and ``get_reward_for_text()`` interface as
    ``SkillTrainer`` for drop-in replacement.

    Args:
        lm_model: Frozen language model.
        lm_tokenizer: LM tokenizer with chat template.
        rm_model: Frozen reward model.
        rm_tokenizer: RM tokenizer.
        generator: ResponseGenerator instance for LLM calls.
        max_rewrites: Maximum number of rewrite iterations.
        temperature: Sampling temperature for feedback/rewrite generation.
        device: Computation device.
        verbose: Whether to log detailed information.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        generator: Optional[ResponseGenerator] = None,
        max_rewrites: int = 5,
        temperature: float = 0.7,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        self.rm_model = rm_model
        self.rm_tokenizer = rm_tokenizer

        self.max_rewrites = max_rewrites
        self.temperature = temperature
        self.verbose = verbose
        self.device = device or next(lm_model.parameters()).device

        # Reuse existing generator or create one
        if generator is not None:
            self._generator = generator
        else:
            self._generator = ResponseGenerator(
                lm_model=lm_model,
                lm_tokenizer=lm_tokenizer,
                device=self.device,
                temperature=temperature,
            )

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text from the LM given a prompt."""
        messages = [{"role": "user", "content": prompt}]
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
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.95,
                pad_token_id=self.lm_tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, input_ids.shape[1]:]
        return self.lm_tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_feedback(
        self,
        skill: str,
        query: str,
        response: str,
        reward: float,
        best_reward: float,
    ) -> str:
        """Generate natural language feedback on the skill."""
        prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            skill=skill,
            query=query,
            response=response,
            reward=reward,
            best_reward=best_reward,
        )
        return self._generate_text(prompt, max_tokens=512)

    def _rewrite_skill(self, skill: str, feedback: str) -> str:
        """Rewrite the skill based on feedback."""
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            skill=skill,
            feedback=feedback,
        )
        return self._generate_text(prompt, max_tokens=1024)

    def optimize(
        self,
        query: str,
        response_text: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """Run TextGrad skill optimization via iterative LLM rewriting.

        Args:
            query: User query text.
            response_text: Current response text (used for initial scoring).
            skill_text: Original skill text to optimize.
            system_prompt: Optional system prompt (unused, for interface compat).

        Returns:
            Dict with ``optimized_skill_text``, ``optimized_logits``,
            ``num_llm_calls``, ``num_grad_steps``, ``final_loss``.
        """
        num_llm_calls = 0
        current_skill = skill_text
        current_response = response_text

        # Score initial
        if self.rm_model is not None:
            best_reward = self.get_reward_for_text(query, current_skill, current_response)
            num_llm_calls += 1
        else:
            best_reward = 0.0

        best_skill = current_skill
        best_response = current_response

        if self.verbose:
            logger.info("TextGrad initial reward: %.4f", best_reward)

        for i in range(self.max_rewrites):
            # Step 1: Generate feedback (always from best skill, not rejected one)
            feedback = self._generate_feedback(
                current_skill, query, current_response, best_reward, best_reward,
            )
            num_llm_calls += 1

            if self.verbose:
                logger.info(
                    "TextGrad round %d | feedback: %s...", i + 1, feedback[:100]
                )

            # Step 2: Rewrite skill
            new_skill = self._rewrite_skill(current_skill, feedback)
            num_llm_calls += 1

            if self.verbose:
                logger.info(
                    "TextGrad round %d | new skill: %s...", i + 1, new_skill[:100]
                )

            # Step 3: Generate response with new skill
            new_response = self._generator.generate(
                query, new_skill, system_prompt,
                seed=(42 + i * 1000),
            )
            num_llm_calls += 1

            # Step 4: Score
            if self.rm_model is not None:
                new_reward = self.get_reward_for_text(
                    query, new_skill, new_response
                )
                num_llm_calls += 1
            else:
                new_reward = 0.0

            if self.verbose:
                logger.info(
                    "TextGrad round %d | reward: %.4f (best: %.4f)",
                    i + 1, new_reward, best_reward,
                )

            # Step 5: Accept if improved, otherwise revert to best
            if new_reward > best_reward:
                best_reward = new_reward
                best_skill = new_skill
                best_response = new_response
                current_skill = new_skill
                current_response = new_response
                if self.verbose:
                    logger.info("TextGrad round %d ACCEPTED", i + 1)
            else:
                # Revert to best skill for next iteration
                current_skill = best_skill
                current_response = best_response
                if self.verbose:
                    logger.info("TextGrad round %d rejected, reverting to best", i + 1)

        return {
            "optimized_skill_text": best_skill,
            "optimized_logits": None,
            "num_llm_calls": num_llm_calls,
            "num_grad_steps": 0,  # TextGrad uses no gradients
            "final_loss": -best_reward,
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
