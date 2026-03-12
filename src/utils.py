"""Shared utilities adapted from Nabla-Reasoner."""

import os
import math
import random
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import transformers


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except AttributeError:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        pass
    transformers.set_seed(seed)


def infer_device_from_model(model: torch.nn.Module) -> torch.device:
    """Infer the device a model resides on."""
    if hasattr(model, "hf_device_map"):
        device_set = list(set(model.hf_device_map.values()))
    else:
        device_set = list({p.device for p in model.parameters()})
    if len(device_set) != 1:
        raise RuntimeError(f"Model is split across multiple devices: {device_set}")
    return device_set[0]


def align_vocab(
    src_embed: torch.Tensor,
    src_tokenizer,
    ref_tokenizer=None,
    vocab_dim: int = 0,
) -> torch.Tensor:
    """Align the vocabulary of src embedding to ref tokenizer order.

    When LM and RM use different tokenizers, this reorders the RM embedding
    table so that row *i* corresponds to the same token as row *i* in the
    LM tokenizer. Tokens missing from the RM vocab are mapped to pad_token.

    Args:
        src_embed: Source embedding table (e.g., RM embeddings).
        src_tokenizer: Tokenizer that matches ``src_embed`` row order.
        ref_tokenizer: Target tokenizer to align to (e.g., LM tokenizer).
            If None or same vocab as src, only trims/validates size.
        vocab_dim: Dimension along which vocab tokens are arranged.

    Returns:
        Aligned embedding tensor with vocab size == ``len(ref_tokenizer)``.
    """
    import logging
    _logger = logging.getLogger(__name__)

    src_vocab_size = src_embed.shape[vocab_dim]

    # Same vocab: just validate size
    if ref_tokenizer is None or ref_tokenizer.get_vocab() == src_tokenizer.get_vocab():
        target_size = len(src_tokenizer)
        if src_vocab_size > target_size:
            _logger.info(
                "Trimming embedding from %d to %d (same vocab, extra rows)",
                src_vocab_size, target_size,
            )
            slice_idx = [slice(None)] * len(src_embed.shape)
            slice_idx[vocab_dim] = slice(0, target_size)
            return src_embed[tuple(slice_idx)]
        if src_vocab_size == target_size:
            return src_embed
        raise ValueError(
            f"Embedding vocab size {src_vocab_size} < tokenizer vocab {target_size}"
        )

    # Different vocabs: reorder rows to match ref_tokenizer order
    src_vocab = src_tokenizer.get_vocab()
    ref_vocab = sorted(ref_tokenizer.get_vocab().items(), key=lambda t: t[1])

    # Validate pad_token_id for fallback mapping
    pad_id = src_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
        _logger.warning(
            "src_tokenizer has no pad_token_id, using 0 as fallback for "
            "unmapped tokens in vocab alignment"
        )

    idx_mapping = []
    num_unmapped = 0
    for token_str, _ref_idx in ref_vocab:
        src_idx = src_vocab.get(token_str)
        if src_idx is not None and src_idx < src_vocab_size:
            idx_mapping.append(src_idx)
        else:
            idx_mapping.append(pad_id)
            num_unmapped += 1

    if num_unmapped > 0:
        _logger.warning(
            "Vocab alignment: %d / %d ref tokens not found in src vocab "
            "(mapped to pad_id=%d)",
            num_unmapped, len(ref_vocab), pad_id,
        )

    aligned = torch.index_select(
        src_embed, vocab_dim,
        torch.as_tensor(idx_mapping, device=src_embed.device),
    )
    _logger.info(
        "Vocab aligned: src=%d -> ref=%d tokens (unmapped=%d)",
        src_vocab_size, len(ref_vocab), num_unmapped,
    )
    return aligned


# ---------------------------------------------------------------------------
# LR Schedulers (from Nabla-Reasoner)
# ---------------------------------------------------------------------------

def _cosine_with_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_ratio: float,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        min_lr_ratio,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    lr_lambda = partial(
        _cosine_with_warmup_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    scheduler_type: str,
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Get an LR scheduler by name."""
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )
    return transformers.get_scheduler(
        transformers.SchedulerType(scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
