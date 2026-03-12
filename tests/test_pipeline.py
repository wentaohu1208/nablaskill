"""Tests for TTSOPipeline orchestration."""

import pytest
from unittest.mock import MagicMock, patch

import torch

from src.pipeline import PipelineConfig, PipelineResult, TTSOPipeline
from src.skillbank import SkillCandidate
from src.ttso import TTSOConfig


PHYSICS_SKILL = (
    "1. Draw free-body diagram. "
    "2. Apply F=ma. "
    "3. Solve for unknowns."
)


class TestPipelineDirectSkill:
    """Test pipeline with directly provided skill (no SkillBank)."""

    def test_run_with_direct_skill(self, lm_model, lm_tokenizer, device):
        config = PipelineConfig(
            ttso_config=TTSOConfig(
                max_iters=2,
                mixed_precision=torch.float32,
                grad_caching=False,
                rejection_sampling=False,
                verbose=0,
            ),
        )
        pipeline = TTSOPipeline(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

        result = pipeline.run(
            query="What is 2+2?",
            skill_text=PHYSICS_SKILL,
            seed=42,
        )

        assert result.query == "What is 2+2?"
        assert result.selected_candidate is not None
        assert result.selected_candidate.skill_id == "direct"
        assert result.ttso_result is not None
        assert len(result.ttso_result.optimized_skill) > 0

    def test_raises_without_skill_or_skillbank(
        self, lm_model, lm_tokenizer, device
    ):
        config = PipelineConfig(
            ttso_config=TTSOConfig(
                max_iters=1,
                mixed_precision=torch.float32,
                verbose=0,
            ),
        )
        pipeline = TTSOPipeline(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

        with pytest.raises(ValueError, match="skill_text must be provided"):
            pipeline.run(query="What is 2+2?")

    def test_no_writeback_for_direct_skill(
        self, lm_model, lm_tokenizer, device
    ):
        config = PipelineConfig(
            ttso_config=TTSOConfig(
                max_iters=2,
                mixed_precision=torch.float32,
                grad_caching=False,
                rejection_sampling=False,
                verbose=0,
            ),
            writeback_enabled=True,  # Enabled but should not trigger
        )
        pipeline = TTSOPipeline(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

        result = pipeline.run(
            query="What is 2+2?",
            skill_text=PHYSICS_SKILL,
            seed=42,
        )

        assert result.writeback_skill_id is None


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_default_values(self):
        result = PipelineResult()
        assert result.query == ""
        assert result.retrieved_candidates == []
        assert result.selected_candidate is None
        assert result.ttso_result is None
        assert result.writeback_skill_id is None


class TestSkillSelection:
    """Test skill selection strategies."""

    def test_highest_retrieval_score(self, lm_model, lm_tokenizer, device):
        config = PipelineConfig(
            ttso_config=TTSOConfig(
                max_iters=1,
                mixed_precision=torch.float32,
                verbose=0,
            ),
            selection_strategy="highest_retrieval_score",
        )
        pipeline = TTSOPipeline(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

        candidates = [
            SkillCandidate("id1", "low", "desc", "skill A", score=0.3),
            SkillCandidate("id2", "high", "desc", "skill B", score=0.9),
            SkillCandidate("id3", "mid", "desc", "skill C", score=0.6),
        ]

        selected = pipeline._select_skill("test query", candidates)
        assert selected.skill_id == "id2"
        assert selected.score == 0.9

    def test_unknown_strategy_raises(self, lm_model, lm_tokenizer, device):
        config = PipelineConfig(
            ttso_config=TTSOConfig(
                max_iters=1,
                mixed_precision=torch.float32,
                verbose=0,
            ),
            selection_strategy="nonexistent",
        )
        pipeline = TTSOPipeline(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

        candidates = [
            SkillCandidate("id1", "test", "desc", "skill A", score=0.5),
        ]
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            pipeline._select_skill("query", candidates)


class TestWritebackLogic:
    """Test writeback decision logic."""

    def _make_pipeline(self, lm_model, lm_tokenizer, device, **overrides):
        defaults = dict(
            ttso_config=TTSOConfig(
                max_iters=1,
                mixed_precision=torch.float32,
                verbose=0,
            ),
            writeback_enabled=True,
            writeback_min_improvement=0.1,
        )
        defaults.update(overrides)
        config = PipelineConfig(**defaults)
        return TTSOPipeline(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            config=config,
            device=device,
        )

    def _make_ttso_result(self, original_reward=0.5, final_reward=0.8,
                           accepted=True, optimized=True):
        from src.ttso import TTSOResult
        return TTSOResult(
            query="test",
            original_reward=original_reward,
            final_reward=final_reward,
            optimization_accepted=accepted,
            skill_was_optimized=optimized,
        )

    def test_no_writeback_when_disabled(self, lm_model, lm_tokenizer, device):
        pipeline = self._make_pipeline(
            lm_model, lm_tokenizer, device, writeback_enabled=False
        )
        result = self._make_ttso_result()
        candidate = SkillCandidate("id1", "test", "d", "instr", score=0.9)
        assert not pipeline._should_writeback(result, candidate)

    def test_no_writeback_when_rejected(self, lm_model, lm_tokenizer, device):
        pipeline = self._make_pipeline(lm_model, lm_tokenizer, device)
        result = self._make_ttso_result(accepted=False)
        candidate = SkillCandidate("id1", "test", "d", "instr", score=0.9)
        assert not pipeline._should_writeback(result, candidate)

    def test_no_writeback_for_direct_skill(self, lm_model, lm_tokenizer, device):
        pipeline = self._make_pipeline(lm_model, lm_tokenizer, device)
        result = self._make_ttso_result()
        candidate = SkillCandidate("direct", "test", "d", "instr", score=1.0)
        assert not pipeline._should_writeback(result, candidate)

    def test_no_writeback_below_threshold(self, lm_model, lm_tokenizer, device):
        pipeline = self._make_pipeline(
            lm_model, lm_tokenizer, device, writeback_min_improvement=0.5
        )
        result = self._make_ttso_result(original_reward=0.5, final_reward=0.6)
        candidate = SkillCandidate("id1", "test", "d", "instr", score=0.9)
        assert not pipeline._should_writeback(result, candidate)
