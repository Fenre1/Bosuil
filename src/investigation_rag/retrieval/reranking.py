from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from investigation_rag.retrieval.ranking import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        ...


class LexicalReranker:
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        scored = []
        for result in results:
            text_terms = set(tokenize(result.chunk_text))
            overlap = len(query_terms & text_terms)
            coverage = overlap / max(1, len(query_terms))
            density = overlap / max(1, len(text_terms))
            fusion_bonus = 1.0 / max(1, result.fused_rank or 1)
            source_bonus = 0.05 if result.retrieval_source == "keyword+vector" else 0.0
            score = coverage * 2.0 + density + fusion_bonus + source_bonus
            scored.append(replace(result, rerank_score=score))
        scored.sort(key=lambda item: (item.rerank_score or 0.0, item.score), reverse=True)
        return scored[:limit]


def tokenize(value: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"[\w\-]+", value, flags=re.UNICODE) if len(token) > 1]
