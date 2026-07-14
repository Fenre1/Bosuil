from __future__ import annotations

import re

from investigation_rag.rag.ollama_llm import OllamaLLM


QUERY_EXPANSION_INSTRUCTIONS = """Generate concise retrieval queries for a local investigation archive.
Return one query per line.
Keep the original meaning.
Include useful synonyms, entities, dates, and indirect evidence angles when they are implied by the question.
Do not answer the question.
"""


class QueryExpander:
    def __init__(self, llm: OllamaLLM, max_queries: int = 4):
        self.llm = llm
        self.max_queries = max_queries

    def expand(self, question: str) -> list[str]:
        prompt = (
            f"{QUERY_EXPANSION_INSTRUCTIONS}\n"
            f"Question:\n{question}\n\n"
            f"Return up to {self.max_queries} retrieval queries:"
        )
        try:
            response = self.llm.answer(prompt)
        except Exception:
            return [question]
        queries = [clean_query(line) for line in response.splitlines()]
        queries = [query for query in queries if query]
        return unique_queries([question, *queries])[: self.max_queries]


def clean_query(value: str) -> str:
    text = value.strip().strip("\"'")
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    if ":" in text and text.split(":", 1)[0].lower() in {"query", "search query", "retrieval query"}:
        text = text.split(":", 1)[1].strip()
    return text.strip("\"' ")


def unique_queries(queries: list[str]) -> list[str]:
    unique = []
    seen = set()
    for query in queries:
        key = " ".join(query.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique
