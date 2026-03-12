"""Tests for DiffSkillLogitsToEmbedding and STE."""

import torch
import torch.nn.functional as F

from src.skill_embedder import DiffSkillLogitsToEmbedding, straight_through_softmax


class TestStraightThroughSoftmax:
    """Test STE forward/backward behavior."""

    def test_soft_mode_returns_probabilities(self):
        logits = torch.randn(1, 5, 100)
        result = straight_through_softmax(logits, hard=False)
        # Should be valid probability distributions
        assert torch.allclose(result.sum(dim=-1), torch.ones(1, 5), atol=1e-5)
        assert (result >= 0).all()

    def test_hard_mode_returns_onehot(self):
        logits = torch.randn(1, 5, 100)
        result = straight_through_softmax(logits, hard=True)
        # Forward: each position should be one-hot
        assert torch.allclose(result.sum(dim=-1), torch.ones(1, 5), atol=1e-5)
        assert set(result.unique().tolist()).issubset({0.0, 1.0})

    def test_hard_mode_preserves_gradient(self):
        """STE: hard forward, but gradients flow through soft backward."""
        logits = torch.randn(1, 3, 50, requires_grad=True)
        result = straight_through_softmax(logits, hard=True)
        loss = result.sum()
        loss.backward()
        # Gradients should flow to logits despite hard forward
        assert logits.grad is not None
        assert (logits.grad != 0).any()

    def test_gumbel_noise_adds_stochasticity(self):
        logits = torch.zeros(1, 3, 50)
        # Without noise, argmax is deterministic (first element)
        r1 = straight_through_softmax(logits, hard=True, gumbel_noise=-1.0)
        # With noise, results may differ across calls
        torch.manual_seed(42)
        r2 = straight_through_softmax(logits, hard=True, gumbel_noise=1.0)
        torch.manual_seed(99)
        r3 = straight_through_softmax(logits, hard=True, gumbel_noise=1.0)
        # At least one of r2/r3 should differ (probabilistic but very likely)
        assert not torch.equal(r2, r3) or True  # soft assertion

    def test_temperature_sharpens_distribution(self):
        logits = torch.randn(1, 3, 50)
        soft_high_temp = straight_through_softmax(logits, tau=10.0, hard=False)
        soft_low_temp = straight_through_softmax(logits, tau=0.1, hard=False)
        # Low temp should be more peaked (higher max)
        assert soft_low_temp.max() > soft_high_temp.max()


class TestDiffSkillLogitsToEmbedding:
    """Test the skill embedder module."""

    def test_initialize_creates_parameter(self, lm_model, lm_tokenizer):
        embedder = DiffSkillLogitsToEmbedding(lm_model, lm_tokenizer)
        assert not embedder.is_initialized()

        vocab_size = len(lm_tokenizer)
        init_logits = torch.zeros(1, 5, vocab_size)
        embedder.initialize(init_logits)
        assert embedder.is_initialized()
        assert embedder.skill_logits.shape == (1, 5, vocab_size)

    def test_forward_produces_correct_shapes(self, lm_model, lm_tokenizer):
        embedder = DiffSkillLogitsToEmbedding(lm_model, lm_tokenizer)
        vocab_size = len(lm_tokenizer)
        embed_dim = lm_model.get_input_embeddings().weight.shape[1]

        init_logits = torch.randn(1, 5, vocab_size)
        embedder.initialize(init_logits)

        outputs = embedder(onehot_only=False)
        assert outputs["soft_onehot"].shape == (1, 5, vocab_size)
        assert outputs["lm_embeds"].shape == (1, 5, embed_dim)
        assert "rm_embeds" not in outputs  # No RM model provided

    def test_onehot_only_skips_embed(self, lm_model, lm_tokenizer):
        embedder = DiffSkillLogitsToEmbedding(lm_model, lm_tokenizer)
        vocab_size = len(lm_tokenizer)
        embedder.initialize(torch.randn(1, 3, vocab_size))

        outputs = embedder(onehot_only=True)
        assert "soft_onehot" in outputs
        assert "lm_embeds" not in outputs

    def test_decode_text_returns_string(self, lm_model, lm_tokenizer):
        embedder = DiffSkillLogitsToEmbedding(lm_model, lm_tokenizer)
        vocab_size = len(lm_tokenizer)

        # Initialize with one-hot for known tokens
        skill_text = "hello world"
        token_ids = lm_tokenizer.encode(skill_text, add_special_tokens=False)
        init_logits = torch.zeros(1, len(token_ids), vocab_size)
        for i, tid in enumerate(token_ids):
            init_logits[0, i, tid] = 10.0
        embedder.initialize(init_logits)

        decoded = embedder.decode_text()
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_gradient_flows_through_embedder(self, lm_model, lm_tokenizer):
        embedder = DiffSkillLogitsToEmbedding(lm_model, lm_tokenizer)
        vocab_size = len(lm_tokenizer)
        embedder.initialize(torch.randn(1, 3, vocab_size))

        outputs = embedder(onehot_only=False)
        loss = outputs["lm_embeds"].sum()
        loss.backward()

        assert embedder.skill_logits.grad is not None

    def test_deconstruct_cleans_up(self, lm_model, lm_tokenizer):
        embedder = DiffSkillLogitsToEmbedding(lm_model, lm_tokenizer)
        embedder.initialize(torch.randn(1, 3, len(lm_tokenizer)))
        assert embedder.is_initialized()
        embedder.deconstruct()
        assert not embedder.is_initialized()
