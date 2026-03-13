"""End-to-end TTSO example with a hardcoded physics skill.

Demonstrates the full optimization pipeline:
1. Load a small LM (and optional RM)
2. Define a physics problem-solving skill
3. Run TTSO optimization on the skill
4. Compare original vs optimized skill text and responses

Usage:
    # CPU with tiny model (for testing):
    python scripts/example_physics.py --lm sshleifer/tiny-gpt2 --device cpu --fp32

    # GPU with real model:
    python scripts/example_physics.py --lm Qwen/Qwen2.5-1.5B-Instruct --device cuda:0

    # With reward model:
    python scripts/example_physics.py \
        --lm Qwen/Qwen2.5-1.5B-Instruct \
        --rm Qwen/Qwen2.5-Math-RM-72B \
        --device cuda:0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import PipelineConfig, TTSOPipeline
from src.ttso import TTSOConfig
from src.utils import seed_everything

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded physics skill
# ---------------------------------------------------------------------------

PHYSICS_SKILL = """\
# Newtonian Mechanics Problem-Solving Skill

## Goal
Solve classical mechanics problems systematically using Newton's laws.

## Workflow
1. **Identify the system**: Determine which objects are involved and draw \
a free-body diagram.
2. **List known quantities**: Extract all given values (mass, velocity, \
force, angle, etc.) with units.
3. **Choose a coordinate system**: Pick axes aligned with the dominant \
direction of motion. Decompose vectors.
4. **Apply Newton's second law**: Write F = ma for each axis. For \
rotational problems, use torque = I * alpha.
5. **Incorporate constraints**: Use kinematic equations, conservation laws \
(energy, momentum), or constraint relations as needed.
6. **Solve algebraically**: Isolate the unknown variable before \
substituting numbers. Check dimensional consistency.
7. **Verify**: Sanity-check the answer (correct units, reasonable \
magnitude, limiting cases).\
"""
MATH_SKILL = """\
# Mathematical Problem-Solving Skill

## Goal
Solve mathematical problems systematically using algebraic reasoning and logical steps.

## Workflow
1. **Understand the problem**: Carefully read the question and determine what quantity needs to be found.
2. **List known quantities**: Extract all given numbers, variables, equations, or constraints from the problem.
3. **Define variables**: Introduce symbols for unknown quantities if they are not already defined.
4. **Formulate equations**: Translate the relationships in the problem into mathematical equations or expressions.
5. **Apply mathematical rules**: Use algebraic manipulation, arithmetic operations, identities, or known formulas to simplify the equations.
6. **Solve step-by-step**: Isolate the unknown variable and compute the solution systematically.
7. **Verify the solution**: Check the result by substituting it back into the original equation or by evaluating whether it satisfies the conditions of the problem.\
"""


ENGLISH_SKILL = """\
# English Reading Comprehension Skill

## Goal
Answer reading comprehension questions by extracting and reasoning over information from a given passage.

## Workflow
1. **Read the passage carefully**: Identify the main topic, key entities, \
and the overall argument or narrative.
2. **Understand the question**: Determine what type of answer is expected \
(factual recall, inference, vocabulary, or author intent).
3. **Locate relevant evidence**: Find specific sentences or phrases in the \
passage that relate to the question.
4. **Reason over evidence**: For inference questions, combine multiple pieces \
of evidence and apply logical reasoning. For factual questions, extract the \
answer directly.
5. **Eliminate distractors**: If options are given, rule out choices that \
contradict the passage or are unsupported.
6. **Formulate the answer**: State the answer clearly, citing or paraphrasing \
the supporting evidence from the passage.
7. **Verify consistency**: Re-read the relevant part of the passage to confirm \
the answer does not misrepresent the original text.\
"""

PHYSICS_QUERY = (
    "A 2 kg block slides down a frictionless inclined plane that makes "
    "a 30-degree angle with the horizontal. What is the acceleration "
    "of the block?"
)


def load_models(
    lm_name: str,
    rm_name: Optional[str],
    device: str,
    use_fp32: bool,
):
    """Load LM and optional RM models."""
    dtype = torch.float32 if use_fp32 else torch.bfloat16

    logger.info("Loading LM: %s (device=%s, dtype=%s)", lm_name, device, dtype)
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    # Add chat template if missing (for tiny models)
    if lm_tokenizer.chat_template is None:
        lm_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )

    lm_model = AutoModelForCausalLM.from_pretrained(
        lm_name, torch_dtype=dtype, device_map=device,
    )
    lm_model.resize_token_embeddings(len(lm_tokenizer))
    lm_model.requires_grad_(False)

    rm_model, rm_tokenizer = None, None
    if rm_name:
        from transformers import AutoModelForSequenceClassification

        logger.info("Loading RM: %s", rm_name)
        rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
        if rm_tokenizer.pad_token is None:
            rm_tokenizer.pad_token = rm_tokenizer.eos_token
        if rm_tokenizer.chat_template is None:
            rm_tokenizer.chat_template = lm_tokenizer.chat_template
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_name, torch_dtype=dtype, device_map=device, num_labels=1,
        )
        rm_model.resize_token_embeddings(len(rm_tokenizer))
        rm_model.requires_grad_(False)

    return lm_model, lm_tokenizer, rm_model, rm_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="TTSO Physics Skill Example")
    parser.add_argument("--lm", type=str, default="/data/hwt/hf_ckpt/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm", type=str, default='/data/hwt/hf_ckpt/Skywork-Reward-V2-Qwen3-4B')
    parser.add_argument("--device", type=str, default='cuda:6')
    parser.add_argument("--fp32", action="store_true", help="Use float32")
    parser.add_argument("--optimization_mode", type=str, default="soft_prompt",
                        choices=["dto", "soft_prompt", "sequential_dto"],
                        help="Skill optimization method")
    parser.add_argument("--max_iters", type=int, default=20,
                        help="DTO/soft_prompt gradient steps per outer round")
    parser.add_argument("--max_outer_rounds", type=int, default=4,
                        help="Iterative rounds (1=single-round baseline)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--response_nll_coeff", type=float, default=1e-3,
                        help="Weight for response NLL loss term")
    parser.add_argument("--skill_fluency_coeff", type=float, default=0.05,
                        help="Weight for skill fluency loss term (DTO)")
    parser.add_argument("--reward_coeff", type=float, default=None,
                        help="Weight for RM reward (default: 1.0 if RM loaded, else 0.0)")
    parser.add_argument("--embed_drift_coeff", type=float, default=0.01,
                        help="Soft prompt: embedding drift regularization")
    parser.add_argument("--rm_projection_temperature", type=float, default=0.1,
                        help="Soft prompt: softmax temperature for RM projection")
    parser.add_argument("--init_logit_scale", type=float, default=3.0,
                        help="DTO: initial one-hot logit scale")
    parser.add_argument("--sequential_commit_every", type=int, default=1,
                        help="Sequential DTO: commit N tokens per step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--skill", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Load models
    lm_model, lm_tokenizer, rm_model, rm_tokenizer = load_models(
        args.lm, args.rm, device, args.fp32,
    )

    # Build pipeline config (no SkillBank -- direct skill mode)
    ttso_config = TTSOConfig(
        optimization_mode=args.optimization_mode,
        max_iters=args.max_iters,
        max_outer_rounds=args.max_outer_rounds,
        learning_rate=args.lr,
        response_nll_coeff=args.response_nll_coeff,
        skill_fluency_coeff=args.skill_fluency_coeff,
        reward_coeff=args.reward_coeff if args.reward_coeff is not None else (1.0 if rm_model else 0.0),
        mixed_precision=torch.float32 if args.fp32 else torch.bfloat16,
        grad_caching=True,
        embed_drift_coeff=args.embed_drift_coeff,
        rm_projection_temperature=args.rm_projection_temperature,
        init_logit_scale=args.init_logit_scale,
        sequential_commit_every=args.sequential_commit_every,
        rejection_sampling=(rm_model is not None),
        verbose=args.verbose,
    )
    pipeline_config = PipelineConfig(
        ttso_config=ttso_config,
        skillbank_config=None,  # Direct skill mode
    )

    pipeline = TTSOPipeline(
        lm_model=lm_model,
        lm_tokenizer=lm_tokenizer,
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        config=pipeline_config,
        device=device,
    )
    # import pdb; pdb.set_trace()

    # Run optimization
    query = args.query or PHYSICS_QUERY
    # skill = args.skill or PHYSICS_SKILL
    skill = args.skill or ENGLISH_SKILL
    

    logger.info("=" * 60)
    logger.info("Query: %s", query)
    logger.info("Skill length: %d chars, ~%d tokens",
                len(skill), len(lm_tokenizer.encode(skill)))
    logger.info("=" * 60)

    result = pipeline.run(
        query=query,
        skill_text=skill,
        seed=args.seed,
    )

    # Display results
    ttso = result.ttso_result
    # import pdb; pdb.set_trace()
    if ttso is None:
        logger.error("TTSO optimization failed - no result.")
        return

    print(f"\n{'='*60}")
    print("ORIGINAL SKILL:")
    print(f"{'='*60}")
    print(ttso.original_skill)

    print(f"\n{'='*60}")
    print("OPTIMIZED SKILL (decoded from logits):")
    print(f"{'='*60}")
    print(ttso.optimized_skill)

    print(f"\n{'='*60}")
    print("ORIGINAL RESPONSE:")
    print(f"{'='*60}")
    print(ttso.original_response)

    print(f"\n{'='*60}")
    print("FINAL RESPONSE:")
    print(f"{'='*60}")
    print(ttso.final_response)

    print(f"\n{'='*60}")
    print("STATS:")
    print(f"{'='*60}")
    print(f"  Optimized:     {ttso.skill_was_optimized}")
    print(f"  Accepted:      {ttso.optimization_accepted}")
    print(f"  Outer Rounds:  {ttso.num_outer_rounds}")
    print(f"  RM (orig):     {ttso.original_reward:.4f}")
    print(f"  RM (final):    {ttso.final_reward:.4f}")
    print(f"  Delta:         {ttso.final_reward - ttso.original_reward:+.4f}")
    print(f"  LLM Calls:     {ttso.num_llm_calls}")
    print(f"  Grad Steps:    {ttso.num_grad_steps}")

    # Show per-round skill and reward
    if ttso.round_history:
        for entry in ttso.round_history:
            rnd = entry["round"]
            reward = entry["reward"]
            skill_text = entry["skill"]
            print(f"\n{'='*60}")
            print(f"ROUND {rnd} | RM={reward:.4f}")
            print(f"{'='*60}")
            print(skill_text)
    print(f"{'='*60}")
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
