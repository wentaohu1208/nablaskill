"""Tests for SkillGenerationTemplate and SkillRewardTemplate.

Validates position indexing, embedding concatenation, and token ID alignment.
"""

import pytest
import torch

from src.skill_template import (
    SKILL_PLACEHOLDER,
    SkillGenerationTemplate,
    SkillRewardTemplate,
    format_skill_prompt,
)


class TestFormatSkillPrompt:
    """Test the prompt formatter."""

    def test_placeholder_present(self, lm_tokenizer):
        result = format_skill_prompt(lm_tokenizer, "What is 2+2?")
        assert SKILL_PLACEHOLDER in result

    def test_placeholder_unique(self, lm_tokenizer):
        result = format_skill_prompt(lm_tokenizer, "What is 2+2?")
        parts = result.split(SKILL_PLACEHOLDER)
        assert len(parts) == 2, "Placeholder should appear exactly once"

    def test_query_in_suffix(self, lm_tokenizer):
        result = format_skill_prompt(lm_tokenizer, "What is 2+2?")
        parts = result.split(SKILL_PLACEHOLDER)
        assert "What is 2+2?" in parts[1]

    def test_system_prompt_in_prefix(self, lm_tokenizer):
        result = format_skill_prompt(
            lm_tokenizer, "question", system_prompt="Be helpful"
        )
        parts = result.split(SKILL_PLACEHOLDER)
        assert "Be helpful" in parts[0]


class TestSkillGenerationTemplate:
    """Test template position tracking and embedding concatenation."""

    def _make_template(self, lm_model, lm_tokenizer, device):
        query = "What is 2+2?"
        skill_text = "Use arithmetic rules"
        response_text = "The answer is 4"

        skill_ids = lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        response_ids = lm_tokenizer.encode(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)

        template = SkillGenerationTemplate(
            query=query,
            response_token_ids=response_ids,
            skill_token_ids=skill_ids,
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
        )
        return template, skill_ids, response_ids

    def test_position_lengths_are_positive(self, lm_model, lm_tokenizer, device):
        template, _, _ = self._make_template(lm_model, lm_tokenizer, device)
        assert template.prefix_len > 0
        assert template.skill_len > 0
        assert template.query_suffix_len > 0
        assert template.response_len > 0

    def test_response_start_position(self, lm_model, lm_tokenizer, device):
        template, _, _ = self._make_template(lm_model, lm_tokenizer, device)
        expected = template.prefix_len + template.skill_len + template.query_suffix_len
        assert template.response_start == expected

    def test_total_sequence_length(self, lm_model, lm_tokenizer, device):
        template, skill_ids, response_ids = self._make_template(
            lm_model, lm_tokenizer, device
        )
        total = template.prefix_len + template.skill_len + template.suffix_len
        # suffix_len = query_suffix_len + response_len
        assert template.suffix_len == template.query_suffix_len + template.response_len

        # apply() should produce embeddings of this total length
        embed_dim = lm_model.get_input_embeddings().weight.shape[1]
        fake_skill_embeds = torch.randn(1, template.skill_len, embed_dim).to(device)
        full_embeds = template.apply(fake_skill_embeds)
        assert full_embeds.shape == (1, total, embed_dim)

    def test_apply_concatenation_order(self, lm_model, lm_tokenizer, device):
        """Verify [prefix, skill, suffix] order in concatenated embeddings."""
        template, _, _ = self._make_template(lm_model, lm_tokenizer, device)
        embed_dim = lm_model.get_input_embeddings().weight.shape[1]

        # Use distinctive skill embeddings
        skill_embeds = torch.ones(1, template.skill_len, embed_dim) * 999.0
        full_embeds = template.apply(skill_embeds.to(device))

        # Check skill portion
        skill_slice = full_embeds[
            :, template.prefix_len : template.prefix_len + template.skill_len, :
        ]
        assert torch.allclose(skill_slice, skill_embeds, atol=1e-5)

    def test_apply_to_token_ids(self, lm_model, lm_tokenizer, device):
        template, skill_ids, _ = self._make_template(lm_model, lm_tokenizer, device)
        full_ids = template.apply_to_token_ids(skill_ids)
        total_len = template.prefix_len + template.skill_len + template.suffix_len
        assert full_ids.shape == (1, total_len)

    def test_no_response_tokens(self, lm_model, lm_tokenizer, device):
        """Template should work without response tokens."""
        query = "What is 2+2?"
        skill_text = "Use arithmetic"
        skill_ids = lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)

        template = SkillGenerationTemplate(
            query=query,
            response_token_ids=None,
            skill_token_ids=skill_ids,
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
        )
        assert template.response_len == 0
        assert template.suffix_len == template.query_suffix_len


class TestSkillRewardTemplate:
    """Test RM template."""

    def test_rm_template_without_model(self, lm_tokenizer):
        """Should handle missing RM gracefully."""
        skill_ids = torch.tensor([[1, 2, 3]])
        template = SkillRewardTemplate(
            query="test",
            response_text="answer",
            skill_token_ids=skill_ids,
            rm_model=None,
            rm_tokenizer=None,
        )
        assert template.prefix_embeds is None
        assert template.suffix_embeds is None

    def test_rm_template_raises_on_apply_without_model(self, lm_tokenizer):
        skill_ids = torch.tensor([[1, 2, 3]])
        template = SkillRewardTemplate(
            query="test",
            response_text="answer",
            skill_token_ids=skill_ids,
            rm_model=None,
            rm_tokenizer=None,
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            template.apply(torch.randn(1, 3, 128))

    def test_rm_template_with_lm_as_rm(self, lm_model, lm_tokenizer, device):
        """Use LM as a stand-in RM to test template construction."""
        skill_text = "Use logic"
        skill_ids = lm_tokenizer.encode(
            skill_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)

        template = SkillRewardTemplate(
            query="What is 1+1?",
            response_text="2",
            skill_token_ids=skill_ids,
            rm_model=lm_model,
            rm_tokenizer=lm_tokenizer,
        )
        assert template.prefix_embeds is not None
        assert template.suffix_embeds is not None

        embed_dim = lm_model.get_input_embeddings().weight.shape[1]
        fake_embeds = torch.randn(1, template.skill_len, embed_dim).to(device)
        full_embeds = template.apply(fake_embeds)
        expected_len = (
            template.prefix_embeds.shape[1]
            + template.skill_len
            + template.suffix_embeds.shape[1]
        )
        assert full_embeds.shape == (1, expected_len, embed_dim)
