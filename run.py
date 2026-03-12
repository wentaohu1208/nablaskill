"""Single-prompt TTSO runner.

Usage:
    python run.py \
        --lm <lm_model_path> \
        --rm <rm_model_path> \
        --skill "1. Identify variables\n2. Set up equations\n3. Solve step by step" \
        --query "If 2x + 3 = 11, what is x?" \
        --max_iters 20 --learning_rate 0.01 --reward_coeff 1.0
"""

import argparse
import json
import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.ttso import TTSODecoding, TTSOConfig
from src.utils import seed_everything

MIXED_PRECISION_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def init_models(
    lm_model_name: str,
    rm_model_name: Optional[str],
    device: str,
    attn_impl: str,
    torch_dtype: torch.dtype,
):
    """Load LM and RM models."""
    print(f"[{device}] Loading LM: {lm_model_name}", flush=True)
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    if lm_tokenizer.chat_template is None:
        raise ValueError("Only supports LM tokenizer with chat template.")
    lm_model = AutoModelForCausalLM.from_pretrained(
        lm_model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    lm_model.resize_token_embeddings(len(lm_tokenizer))
    for p in lm_model.parameters():
        p.requires_grad_(False)

    rm_model, rm_tokenizer = None, None
    if rm_model_name is not None:
        print(f"[{device}] Loading RM: {rm_model_name}", flush=True)
        rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
        if rm_tokenizer.chat_template is None:
            rm_tokenizer.chat_template = lm_tokenizer.chat_template
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_impl,
            num_labels=1,
        )
        rm_model.resize_token_embeddings(len(rm_tokenizer))
        for p in rm_model.parameters():
            p.requires_grad_(False)

    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
    if rm_tokenizer is not None and rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    return lm_model, lm_tokenizer, rm_model, rm_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Single-prompt TTSO runner.")

    # Model arguments
    parser.add_argument("--lm_model_name", "--lm", type=str, required=True)
    parser.add_argument("--rm_model_name", "--rm", type=str, default=None)
    parser.add_argument("--vllm_url", type=str, default="")
    parser.add_argument("--vllm_model_name", type=str, default=None)

    # Input
    parser.add_argument("--query", type=str, required=True, help="User query.")
    parser.add_argument("--skill", type=str, required=True, help="Skill text.")
    parser.add_argument("--skill_file", type=str, default=None, help="Read skill from file.")
    parser.add_argument("--system_prompt", type=str, default=None)

    # Optimization
    parser.add_argument("--optimization_mode", type=str, default="dto",
                        choices=["dto", "soft_prompt", "textgrad"],
                        help="Skill optimization method")
    parser.add_argument("--max_iters", type=int, default=20,
                        help="DTO/soft_prompt gradient steps per outer round")
    parser.add_argument("--max_outer_rounds", type=int, default=1,
                        help="Iterative rounds (1=single-round, >1=iterative)")
    parser.add_argument("--embed_drift_coeff", type=float, default=0.01,
                        help="Soft prompt: embedding drift regularization")
    parser.add_argument("--rm_projection_temperature", type=float, default=0.1,
                        help="Soft prompt: softmax temperature for RM embedding projection")
    parser.add_argument("--init_logit_scale", type=float, default=3.0,
                        help="DTO: initial one-hot logit scale (higher = more peaked)")
    parser.add_argument("--textgrad_max_rewrites", type=int, default=5,
                        help="TextGrad: max rewrite iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reward_coeff", type=float, default=1.0)
    parser.add_argument("--response_nll_coeff", type=float, default=1e-3)
    parser.add_argument("--skill_fluency_coeff", type=float, default=1e-4)
    parser.add_argument("--warmup_iters_ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--no_grad_caching", action="store_true")
    parser.add_argument("--no_rejection_sampling", action="store_true")

    # Selection
    parser.add_argument("--min_reward_threshold", type=float, default=None)

    # Generation
    parser.add_argument("--max_generation_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Load skill from file if specified
    skill_text = args.skill
    if args.skill_file is not None:
        with open(args.skill_file, "r", encoding="utf-8") as f:
            skill_text = f.read().strip()

    # Set vllm_model_name default
    if args.vllm_model_name is None:
        args.vllm_model_name = args.lm_model_name

    # Load models
    lm_model, lm_tokenizer, rm_model, rm_tokenizer = init_models(
        args.lm_model_name,
        args.rm_model_name,
        device=device,
        attn_impl=args.attn_implementation,
        torch_dtype=MIXED_PRECISION_MAP[args.mixed_precision],
    )

    # Build config
    config = TTSOConfig(
        optimization_mode=args.optimization_mode,
        max_iters=args.max_iters,
        max_outer_rounds=args.max_outer_rounds,
        learning_rate=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        warmup_iters_ratio=args.warmup_iters_ratio,
        reward_coeff=args.reward_coeff,
        response_nll_coeff=args.response_nll_coeff,
        skill_fluency_coeff=args.skill_fluency_coeff,
        mixed_precision=MIXED_PRECISION_MAP[args.mixed_precision],
        grad_caching=not args.no_grad_caching,
        embed_drift_coeff=args.embed_drift_coeff,
        rm_projection_temperature=args.rm_projection_temperature,
        init_logit_scale=args.init_logit_scale,
        textgrad_max_rewrites=args.textgrad_max_rewrites,
        min_reward_threshold=args.min_reward_threshold,
        rejection_sampling=not args.no_rejection_sampling,
        max_generation_len=args.max_generation_len,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
    )

    # Build TTSO
    ttso = TTSODecoding(
        lm_model=lm_model,
        lm_tokenizer=lm_tokenizer,
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        config=config,
        device=device,
        vllm_url=args.vllm_url if args.vllm_url else None,
        vllm_model_name=args.vllm_model_name,
    )

    # Run (iterative if max_outer_rounds > 1)
    run_fn = ttso.run_iterative if args.max_outer_rounds > 1 else ttso.run
    result = run_fn(
        query=args.query,
        skill_text=skill_text,
        system_prompt=args.system_prompt,
        seed=args.seed,
    )

    # Output
    print(f"\n{'='*60}")
    print(f"Original Skill:\n{result.original_skill}\n")
    print(f"Optimized Skill:\n{result.optimized_skill}\n")
    print(f"Original Response:\n{result.original_response[:500]}\n")
    print(f"Final Response:\n{result.final_response[:500]}\n")
    print(f"Original Reward: {result.original_reward:.4f}")
    print(f"Final Reward:    {result.final_reward:.4f}")
    print(f"Optimized: {result.skill_was_optimized} | Accepted: {result.optimization_accepted}")
    print(f"LLM Calls: {result.num_llm_calls} | Grad Steps: {result.num_grad_steps}")

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_data = {
            "query": result.query,
            "original_skill": result.original_skill,
            "optimized_skill": result.optimized_skill,
            "original_response": result.original_response,
            "final_response": result.final_response,
            "original_reward": result.original_reward,
            "final_reward": result.final_reward,
            "skill_was_optimized": result.skill_was_optimized,
            "optimization_accepted": result.optimization_accepted,
            "num_llm_calls": result.num_llm_calls,
            "num_grad_steps": result.num_grad_steps,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
