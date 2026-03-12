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
    """Align the vocabulary of src embedding to ref tokenizer order."""
    if ref_tokenizer is None or ref_tokenizer.get_vocab() == src_tokenizer.get_vocab():
        if src_embed.shape[vocab_dim] > len(src_tokenizer):
            slice_idx = [slice(None)] * len(src_embed.shape)
            slice_idx[vocab_dim] = slice(0, len(src_tokenizer))
            return src_embed[tuple(slice_idx)]
        if src_embed.shape[vocab_dim] == len(src_tokenizer):
            return src_embed
        raise ValueError(
            f"Embedding vocab size {src_embed.shape[vocab_dim]} < tokenizer vocab {len(src_tokenizer)}"
        )
    src_vocab = src_tokenizer.get_vocab()
    ref_vocab = sorted(ref_tokenizer.get_vocab().items(), key=lambda t: t[1])
    idx_mapping = [src_vocab.get(t[0], src_tokenizer.pad_token_id) for t in ref_vocab]
    return torch.index_select(
        src_embed, vocab_dim, torch.as_tensor(idx_mapping, device=src_embed.device)
    )


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
