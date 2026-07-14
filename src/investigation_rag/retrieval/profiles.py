from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    keyword_candidates: int
    vector_candidates: int
    fusion_candidates: int
    rerank_candidates: int
    final_answer_chunks: int
    query_expansion_enabled: bool = False
    context_expansion_enabled: bool = False
    neighbor_before: int = 1
    neighbor_after: int = 1


LEGACY_PROFILE = RetrievalProfile(
    name="legacy",
    keyword_candidates=8,
    vector_candidates=8,
    fusion_candidates=8,
    rerank_candidates=8,
    final_answer_chunks=8,
)

STANDARD_PROFILE = RetrievalProfile(
    name="standard",
    keyword_candidates=64,
    vector_candidates=64,
    fusion_candidates=80,
    rerank_candidates=80,
    final_answer_chunks=8,
)

DEEP_PROFILE = RetrievalProfile(
    name="deep",
    keyword_candidates=150,
    vector_candidates=150,
    fusion_candidates=200,
    rerank_candidates=200,
    final_answer_chunks=12,
    query_expansion_enabled=True,
    context_expansion_enabled=True,
)

PROFILES = {
    profile.name: profile
    for profile in (
        LEGACY_PROFILE,
        STANDARD_PROFILE,
        DEEP_PROFILE,
    )
}


def get_retrieval_profile(name: str | None) -> RetrievalProfile:
    key = (name or STANDARD_PROFILE.name).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown retrieval profile: {name}")
    return PROFILES[key]


def profile_from_config(config: object, name: str | None) -> RetrievalProfile:
    key = (name or STANDARD_PROFILE.name).strip().lower()
    if key == "legacy":
        return LEGACY_PROFILE
    if key == "deep":
        return RetrievalProfile(
            name="deep",
            keyword_candidates=int(getattr(config, "rag_deep_keyword_candidates", 150)),
            vector_candidates=int(getattr(config, "rag_deep_vector_candidates", 150)),
            fusion_candidates=int(getattr(config, "rag_deep_fusion_candidates", 200)),
            rerank_candidates=int(getattr(config, "rag_deep_rerank_candidates", 200)),
            final_answer_chunks=int(getattr(config, "rag_deep_final_answer_chunks", 12)),
            query_expansion_enabled=True,
            context_expansion_enabled=True,
        )
    if key == "standard":
        return RetrievalProfile(
            name="standard",
            keyword_candidates=int(getattr(config, "rag_keyword_candidates", 64)),
            vector_candidates=int(getattr(config, "rag_vector_candidates", 64)),
            fusion_candidates=int(getattr(config, "rag_fusion_candidates", 80)),
            rerank_candidates=int(getattr(config, "rag_rerank_candidates", 80)),
            final_answer_chunks=int(getattr(config, "rag_final_answer_chunks", 8)),
        )
    raise ValueError(f"Unknown retrieval profile: {name}")
