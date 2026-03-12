"""SkillBank adapter for AutoSkill SDK integration.

Provides a thin wrapper around AutoSkill's retrieval and persistence API,
keeping the external dependency isolated. Degrades gracefully if AutoSkill
is not installed.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _check_autoskill() -> None:
    """Raise ImportError with a clear message if AutoSkill SDK is missing."""
    try:
        import autoskill  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "AutoSkill SDK is required for SkillBank integration. "
            "Install it with: pip install -e /path/to/AutoSkill"
        ) from exc


@dataclass(frozen=True)
class SkillCandidate:
    """A retrieved skill candidate ready for TTSO optimization.

    Attributes:
        skill_id: Unique skill identifier.
        name: Human-readable skill name.
        description: Short description.
        instructions: The optimizable skill text (prompt content).
        score: Retrieval relevance score.
        raw_skill: Original AutoSkill Skill object (if available).
    """

    skill_id: str
    name: str
    description: str
    instructions: str
    score: float = 0.0
    raw_skill: Optional[Any] = None


@dataclass(frozen=True)
class SkillBankConfig:
    """Configuration for SkillBank adapter.

    Attributes:
        autoskill_config: Dict passed to AutoSkillConfig.from_dict().
        user_id: User namespace for skill retrieval and storage.
        top_k: Number of skills to retrieve per query.
        writeback_enabled: Whether to persist optimized skills.
        writeback_prefix: Name prefix for optimized skills.
    """

    autoskill_config: Optional[Dict[str, Any]] = None
    user_id: str = "ttso"
    top_k: int = 5
    writeback_enabled: bool = True
    writeback_prefix: str = "ttso-optimized"


class SkillBankAdapter:
    """Thin wrapper around AutoSkill SDK for skill retrieval and persistence.

    Args:
        config: SkillBank configuration.
    """

    def __init__(self, config: SkillBankConfig) -> None:
        _check_autoskill()
        from autoskill import AutoSkill, AutoSkillConfig

        self.config = config
        autoskill_cfg = config.autoskill_config or {
            "llm": {"provider": "mock"},
            "embeddings": {"provider": "hashing", "dims": 256},
            "store": {"provider": "local", "path": "SkillBank"},
        }
        self._client = AutoSkill(AutoSkillConfig.from_dict(autoskill_cfg))

    @property
    def client(self) -> Any:
        """Access the underlying AutoSkill client."""
        return self._client

    def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[SkillCandidate]:
        """Retrieve relevant skills from SkillBank.

        Args:
            query: User query for skill retrieval.
            user_id: Override default user_id.
            top_k: Override default top_k.

        Returns:
            List of SkillCandidate objects sorted by relevance.
        """
        uid = user_id if user_id is not None else self.config.user_id
        k = top_k if top_k is not None else self.config.top_k

        hits = self._client.search(query, user_id=uid, limit=k)
        candidates = []
        for hit in hits:
            skill = hit.skill
            candidates.append(
                SkillCandidate(
                    skill_id=skill.id,
                    name=skill.name,
                    description=skill.description,
                    instructions=skill.instructions,
                    score=hit.score,
                    raw_skill=skill,
                )
            )
        logger.info(
            "Retrieved %d skills for query: %s", len(candidates), query[:80]
        )
        return candidates

    def writeback(
        self,
        user_id: Optional[str] = None,
        *,
        source_candidate: SkillCandidate,
        optimized_instructions: str,
        query: str = "",
        reward_delta: float = 0.0,
    ) -> str:
        """Persist an optimized skill back to SkillBank as a new entry.

        Args:
            user_id: Override default user_id.
            source_candidate: The original skill that was optimized.
            optimized_instructions: The optimized skill text.
            query: The query that triggered optimization.
            reward_delta: Reward improvement from optimization.

        Returns:
            The skill_id of the newly created skill.
        """
        uid = user_id if user_id is not None else self.config.user_id
        now = datetime.now(timezone.utc).isoformat()

        name = f"{self.config.writeback_prefix}-{source_candidate.name}"
        description = (
            f"TTSO-optimized version of '{source_candidate.name}'. "
            f"Reward delta: {reward_delta:+.4f}."
        )

        skill = self._client.upsert(
            user_id=uid,
            name=name,
            description=description,
            instructions=optimized_instructions,
            triggers=["ttso-optimized"],
            tags=["ttso", "optimized", source_candidate.name],
            metadata={
                "source_skill_id": source_candidate.skill_id,
                "source_skill_name": source_candidate.name,
                "optimization_query": query[:200],
                "reward_delta": reward_delta,
                "optimized_at": now,
            },
        )
        logger.info(
            "Wrote back optimized skill '%s' (id=%s)", name, skill.id
        )
        return skill.id

    def render_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Render retrieved skills as formatted context text.

        Args:
            query: User query.
            user_id: Override default user_id.
            limit: Max number of skills to render.

        Returns:
            Formatted skill context string.
        """
        uid = user_id if user_id is not None else self.config.user_id
        k = limit if limit is not None else self.config.top_k
        return self._client.render_context(query, user_id=uid, limit=k)
