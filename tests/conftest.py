"""Shared test fixtures for nablaskill tests.

Uses a tiny GPT-2 model to keep tests fast and CPU-friendly.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


TINY_MODEL_NAME = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def lm_tokenizer():
    """Load a tiny tokenizer (session-scoped for speed)."""
    tok = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Ensure chat template exists for apply_chat_template
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
    return tok


@pytest.fixture(scope="session")
def lm_model():
    """Load a tiny LM (session-scoped for speed)."""
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_NAME)
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def device():
    return torch.device("cpu")
