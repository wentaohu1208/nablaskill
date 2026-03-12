"""Tests for the iterative TTSO variant (run_iterative)."""

import torch

from src.ttso import TTSOConfig, TTSODecoding


class TestRunIterative:
    """Test the multi-round iterative optimization loop."""

    def _make_ttso(self, lm_model, lm_tokenizer, device, max_outer_rounds=2):
        config = TTSOConfig(
            max_iters=2,
            max_outer_rounds=max_outer_rounds,
            learning_rate=0.1,
            mixed_precision=torch.float32,
            grad_caching=False,
            rejection_sampling=False,
            verbose=0,
        )
        return TTSODecoding(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

    def test_iterative_returns_valid_result(
        self, lm_model, lm_tokenizer, device
    ):
        ttso = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=2)
        result = ttso.run_iterative(
            query="What is 2+2?",
            skill_text="Use basic arithmetic.",
            seed=42,
        )

        assert result.query == "What is 2+2?"
        assert result.skill_was_optimized
        assert result.optimization_accepted
        assert len(result.optimized_skill) > 0
        assert len(result.final_response) > 0

    def test_iterative_has_round_history(
        self, lm_model, lm_tokenizer, device
    ):
        ttso = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=3)
        result = ttso.run_iterative(
            query="What is 2+2?",
            skill_text="Use basic arithmetic.",
            seed=42,
        )

        # Round 0 (initial) + at least 1 optimization round
        assert len(result.round_history) >= 2
        # Round 0 should be the initial state
        assert result.round_history[0]["round"] == 0
        assert result.round_history[0]["skill"] == "Use basic arithmetic."

        # Each round entry has required keys
        for entry in result.round_history:
            assert "round" in entry
            assert "skill" in entry
            assert "response" in entry
            assert "reward" in entry

    def test_iterative_num_outer_rounds_tracked(
        self, lm_model, lm_tokenizer, device
    ):
        ttso = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=2)
        result = ttso.run_iterative(
            query="What is 2+2?",
            skill_text="Use basic arithmetic.",
            seed=42,
        )
        # num_outer_rounds = len(round_history) - 1 (excluding round 0)
        assert result.num_outer_rounds == len(result.round_history) - 1

    def test_single_round_via_run(self, lm_model, lm_tokenizer, device):
        """run() should still work as single-round baseline."""
        ttso = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=1)
        result = ttso.run(
            query="What is 2+2?",
            skill_text="Use basic arithmetic.",
            seed=42,
        )
        # Single-round run() does not populate round_history
        assert result.round_history == []
        assert result.num_outer_rounds == 1

    def test_iterative_accumulates_grad_steps(
        self, lm_model, lm_tokenizer, device
    ):
        ttso = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=2)
        result = ttso.run_iterative(
            query="What is 2+2?",
            skill_text="Use basic arithmetic.",
            seed=42,
        )
        # Each round does max_iters=2 grad steps, should accumulate
        assert result.num_grad_steps >= 2

    def test_iterative_llm_calls_increase_with_rounds(
        self, lm_model, lm_tokenizer, device
    ):
        # Single round
        ttso1 = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=1)
        r1 = ttso1.run_iterative(
            query="What is 2+2?", skill_text="skill", seed=42
        )

        # Two rounds
        ttso2 = self._make_ttso(lm_model, lm_tokenizer, device, max_outer_rounds=2)
        r2 = ttso2.run_iterative(
            query="What is 2+2?", skill_text="skill", seed=42
        )

        # More rounds = more LLM calls
        assert r2.num_llm_calls >= r1.num_llm_calls
