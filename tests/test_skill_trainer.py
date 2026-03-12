"""Tests for SkillTrainer loss computation and optimization loop."""

import pytest
import torch

from src.skill_embedder import DiffSkillLogitsToEmbedding
from src.skill_template import SkillGenerationTemplate, SkillRewardTemplate
from src.skill_trainer import SkillTrainer


class TestComputeLoss:
    """Test the three-component loss function."""

    def _setup(self, lm_model, lm_tokenizer, device):
        """Build trainer, templates, and embedder outputs for loss testing."""
        query = "What is 2+2?"
        skill_text = "arithmetic"
        response_text = "four"

        skill_ids = lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        response_ids = lm_tokenizer.encode(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        vocab_size = len(lm_tokenizer)

        # Build trainer
        trainer = SkillTrainer(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            rm_model=None,
            rm_tokenizer=None,
            max_iters=5,
            learning_rate=0.01,
            response_nll_coeff=1e-3,
            skill_fluency_coeff=1e-4,
            reward_coeff=0.0,
            device=device,
            mixed_precision=torch.float32,
        )

        # Build template
        lm_template = SkillGenerationTemplate(
            query=query,
            response_token_ids=response_ids,
            skill_token_ids=skill_ids,
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
        )

        # Build embedder outputs
        init_logits = torch.zeros(1, skill_ids.shape[1], vocab_size, device=device)
        init_logits.scatter_(2, skill_ids.unsqueeze(-1), 10.0)
        trainer.skill_embedder.initialize(init_logits)
        outputs = trainer.skill_embedder(onehot_only=False)

        return trainer, lm_template, outputs, response_ids

    def test_loss_is_scalar(self, lm_model, lm_tokenizer, device):
        trainer, lm_template, outputs, response_ids = self._setup(
            lm_model, lm_tokenizer, device
        )
        loss_dict = trainer.compute_loss(outputs, lm_template, None, response_ids)
        assert loss_dict["loss"].dim() == 0

    def test_loss_has_all_components(self, lm_model, lm_tokenizer, device):
        trainer, lm_template, outputs, response_ids = self._setup(
            lm_model, lm_tokenizer, device
        )
        loss_dict = trainer.compute_loss(outputs, lm_template, None, response_ids)
        assert "loss" in loss_dict
        assert "response_nll" in loss_dict
        assert "skill_fluency" in loss_dict
        assert "reward" in loss_dict
        assert loss_dict["reward"] is None  # No RM

    def test_loss_is_finite(self, lm_model, lm_tokenizer, device):
        trainer, lm_template, outputs, response_ids = self._setup(
            lm_model, lm_tokenizer, device
        )
        loss_dict = trainer.compute_loss(outputs, lm_template, None, response_ids)
        assert torch.isfinite(loss_dict["loss"])
        assert torch.isfinite(loss_dict["response_nll"])
        assert torch.isfinite(loss_dict["skill_fluency"])

    def test_loss_gradient_flows(self, lm_model, lm_tokenizer, device):
        trainer, lm_template, outputs, response_ids = self._setup(
            lm_model, lm_tokenizer, device
        )
        loss_dict = trainer.compute_loss(outputs, lm_template, None, response_ids)
        loss_dict["loss"].backward()
        assert trainer.skill_embedder.skill_logits.grad is not None

    def test_zero_coefficients_zero_loss_terms(self, lm_model, lm_tokenizer, device):
        """When coefficients are zero, corresponding loss terms should be zero."""
        trainer, lm_template, outputs, response_ids = self._setup(
            lm_model, lm_tokenizer, device
        )
        trainer.response_nll_coeff = 0.0
        trainer.skill_fluency_coeff = 0.0

        loss_dict = trainer.compute_loss(outputs, lm_template, None, response_ids)
        # With zero coefficients, loss should be -0 = 0 (no reward either)
        assert abs(loss_dict["loss"].item()) < 1e-5


class TestOptimizeLoop:
    """Test the full optimization loop (few iterations)."""

    def test_optimize_returns_valid_result(self, lm_model, lm_tokenizer, device):
        trainer = SkillTrainer(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            max_iters=3,
            learning_rate=0.1,
            device=device,
            mixed_precision=torch.float32,
            grad_caching=False,
        )

        result = trainer.optimize(
            query="What is 2+2?",
            response_text="The answer is 4.",
            skill_text="Use basic arithmetic.",
        )

        assert "optimized_skill_text" in result
        assert "optimized_logits" in result
        assert "num_llm_calls" in result
        assert "num_grad_steps" in result
        assert "final_loss" in result
        assert isinstance(result["optimized_skill_text"], str)
        assert len(result["optimized_skill_text"]) > 0
        assert result["num_grad_steps"] > 0

    def test_optimize_logits_shape(self, lm_model, lm_tokenizer, device):
        trainer = SkillTrainer(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            max_iters=2,
            device=device,
            mixed_precision=torch.float32,
            grad_caching=False,
        )

        skill_text = "simple skill"
        skill_ids = lm_tokenizer.encode(skill_text, add_special_tokens=False)
        num_tokens = len(skill_ids)

        result = trainer.optimize(
            query="test", response_text="answer", skill_text=skill_text
        )

        assert result["optimized_logits"].shape == (
            1,
            num_tokens,
            len(lm_tokenizer),
        )

    def test_optimize_cleans_up_embedder(self, lm_model, lm_tokenizer, device):
        trainer = SkillTrainer(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            max_iters=2,
            device=device,
            mixed_precision=torch.float32,
            grad_caching=False,
        )
        trainer.optimize(
            query="test", response_text="answer", skill_text="basic skill"
        )
        # After optimize, embedder should be deconstructed
        assert not trainer.skill_embedder.is_initialized()
