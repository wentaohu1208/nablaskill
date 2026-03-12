"""TTSO Pipeline: Query -> Skill Retrieval -> Optimization -> Task Solving.

Orchestrates the full flow from user query to optimized skill application,
integrating SkillBank retrieval, skill selection, TTSO optimization, and
optional writeback of improved skills.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .skillbank import SkillBankAdapter, SkillBankConfig, SkillCandidate
from .ttso import TTSOConfig, TTSODecoding, TTSOResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the TTSO pipeline.

    Attributes:
        ttso_config: Inner TTSO optimization config.
        skillbank_config: SkillBank adapter config (None = no retrieval).
        selection_strategy: How to pick from top-k retrieved skills.
            - "highest_retrieval_score": Use top-1 from retrieval (fast).
            - "best_initial_reward": Score each with RM, pick best (slower).
        writeback_enabled: Whether to write optimized skills back to SkillBank.
        writeback_min_improvement: Min reward delta to trigger writeback.
    """

    ttso_config: TTSOConfig = field(default_factory=TTSOConfig)
    skillbank_config: Optional[SkillBankConfig] = None
    selection_strategy: str = "highest_retrieval_score"
    writeback_enabled: bool = False
    writeback_min_improvement: float = 0.0


@dataclass
class PipelineResult:
    """Result from a full TTSO pipeline run.

    Attributes:
        query: Input query.
        retrieved_candidates: All retrieved skill candidates.
        selected_candidate: The skill chosen for optimization.
        ttso_result: Inner TTSO optimization result.
        writeback_skill_id: ID of written-back skill (if any).
    """

    query: str = ""
    retrieved_candidates: List[SkillCandidate] = field(default_factory=list)
    selected_candidate: Optional[SkillCandidate] = None
    ttso_result: Optional[TTSOResult] = None
    writeback_skill_id: Optional[str] = None


class TTSOPipeline:
    """Full TTSO pipeline: retrieve -> select -> optimize -> writeback.

    Supports two modes:
    1. **With SkillBank**: Retrieves skills, selects best, optimizes, writes back.
    2. **Direct skill**: Skips retrieval, optimizes the provided skill text.

    Args:
        lm_model: Frozen language model.
        lm_tokenizer: LM tokenizer with chat template.
        rm_model: Frozen reward model (optional).
        rm_tokenizer: RM tokenizer (optional).
        config: Pipeline configuration.
        device: Computation device.
        vllm_url: Optional vLLM server URL for generation.
        vllm_model_name: Model name for vLLM API.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        rm_model: Optional[nn.Module] = None,
        rm_tokenizer=None,
        config: Optional[PipelineConfig] = None,
        device: Optional[torch.device] = None,
        vllm_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.rm_model = rm_model

        # Initialize inner TTSO engine
        self.ttso = TTSODecoding(
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            config=self.config.ttso_config,
            device=device,
            vllm_url=vllm_url,
            vllm_model_name=vllm_model_name,
        )

        # Initialize SkillBank adapter (optional)
        self.skillbank: Optional[SkillBankAdapter] = None
        if self.config.skillbank_config is not None:
            self.skillbank = SkillBankAdapter(self.config.skillbank_config)

    def run(
        self,
        query: str,
        skill_text: Optional[str] = None,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> PipelineResult:
        """Run the full TTSO pipeline.

        If ``skill_text`` is provided, retrieval is skipped and the given
        skill is optimized directly. Otherwise, skills are retrieved from
        SkillBank and the best candidate is selected.

        Args:
            query: User query.
            skill_text: Direct skill text (skips retrieval).
            user_id: User ID for SkillBank operations.
            system_prompt: Optional system prompt.
            seed: Random seed for reproducibility.

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        result = PipelineResult(query=query)

        # Step 1: Retrieve or use direct skill
        if skill_text is not None:
            candidate = SkillCandidate(
                skill_id="direct",
                name="direct-skill",
                description="Directly provided skill",
                instructions=skill_text,
                score=1.0,
            )
            result.selected_candidate = candidate
            logger.info("Using directly provided skill (%d chars)", len(skill_text))
        elif self.skillbank is not None:
            candidates = self._retrieve_skills(query, user_id)
            result.retrieved_candidates = candidates
            if not candidates:
                logger.warning("No skills retrieved for query: %s", query[:80])
                return result
            candidate = self._select_skill(query, candidates, system_prompt, seed)
            result.selected_candidate = candidate
        else:
            raise ValueError(
                "Either skill_text must be provided or "
                "skillbank_config must be set for retrieval."
            )

        # Step 2: Run TTSO optimization
        logger.info(
            "Running TTSO with skill '%s' (score=%.3f)",
            candidate.name,
            candidate.score,
        )
        ttso_result = self.ttso.run(
            query=query,
            skill_text=candidate.instructions,
            system_prompt=system_prompt,
            seed=seed,
        )
        result.ttso_result = ttso_result

        # Step 3: Writeback if enabled and improvement is sufficient
        if self._should_writeback(ttso_result, candidate):
            result.writeback_skill_id = self._writeback(
                ttso_result, candidate, query, user_id
            )

        return result

    def _retrieve_skills(
        self,
        query: str,
        user_id: Optional[str] = None,
    ) -> List[SkillCandidate]:
        """Retrieve skill candidates from SkillBank."""
        return self.skillbank.retrieve(query, user_id=user_id)

    def _select_skill(
        self,
        query: str,
        candidates: List[SkillCandidate],
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SkillCandidate:
        """Select the best skill from retrieved candidates.

        Args:
            query: User query.
            candidates: Retrieved skill candidates.
            system_prompt: Optional system prompt.
            seed: Random seed.

        Returns:
            The selected SkillCandidate.
        """
        strategy = self.config.selection_strategy

        if strategy == "highest_retrieval_score":
            selected = max(candidates, key=lambda c: c.score)
            logger.info(
                "Selected skill '%s' (retrieval score=%.3f)",
                selected.name,
                selected.score,
            )
            return selected

        if strategy == "best_initial_reward":
            return self._select_by_reward(
                query, candidates, system_prompt, seed
            )

        raise ValueError(f"Unknown selection strategy: {strategy}")

    def _select_by_reward(
        self,
        query: str,
        candidates: List[SkillCandidate],
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SkillCandidate:
        """Select skill by generating responses and scoring with RM.

        For each candidate, generates an initial response and scores it.
        Returns the candidate with the highest RM score.
        """
        if self.rm_model is None:
            logger.warning(
                "best_initial_reward strategy requires RM. "
                "Falling back to highest_retrieval_score."
            )
            return max(candidates, key=lambda c: c.score)

        best_candidate = candidates[0]
        best_score = float("-inf")

        for i, candidate in enumerate(candidates):
            response = self.ttso.generate_response(
                query,
                candidate.instructions,
                system_prompt,
                seed=(seed + i * 100) if seed is not None else None,
            )
            score = self.ttso.skill_trainer.get_reward_for_text(
                query, candidate.instructions, response
            )
            logger.info(
                "Candidate '%s': RM score=%.4f", candidate.name, score
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        logger.info(
            "Selected skill '%s' (RM score=%.4f)",
            best_candidate.name,
            best_score,
        )
        return best_candidate

    def _should_writeback(
        self,
        ttso_result: TTSOResult,
        candidate: SkillCandidate,
    ) -> bool:
        """Decide whether to write back the optimized skill."""
        if not self.config.writeback_enabled:
            return False
        if self.skillbank is None:
            return False
        if not ttso_result.optimization_accepted:
            return False
        if candidate.skill_id == "direct":
            return False

        improvement = ttso_result.final_reward - ttso_result.original_reward
        return improvement >= self.config.writeback_min_improvement

    def _writeback(
        self,
        ttso_result: TTSOResult,
        candidate: SkillCandidate,
        query: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Write optimized skill back to SkillBank."""
        try:
            reward_delta = ttso_result.final_reward - ttso_result.original_reward
            skill_id = self.skillbank.writeback(
                user_id=user_id,
                source_candidate=candidate,
                optimized_instructions=ttso_result.optimized_skill,
                query=query,
                reward_delta=reward_delta,
            )
            return skill_id
        except (ConnectionError, TimeoutError, OSError, RuntimeError):
            logger.exception("Failed to write back optimized skill")
            return None
