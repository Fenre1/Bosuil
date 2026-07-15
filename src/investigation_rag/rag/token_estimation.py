from __future__ import annotations

from math import ceil

from investigation_rag.retrieval.ranking import SearchResult


CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_prompt_tokens(prompt: str) -> int:
    if not prompt:
        return 0
    return max(1, ceil(len(prompt) / CHARS_PER_TOKEN_ESTIMATE))


def chunk_text_chars(results: list[SearchResult]) -> int:
    return sum(len(result.chunk_text) for result in results)


def format_prompt_token_estimate(
    label: str,
    prompt: str,
    *,
    results: list[SearchResult] | None = None,
    ledger_entries: int | None = None,
) -> str:
    details = [f"{len(prompt):,} prompt chars"]
    if results is not None:
        chunk_count = len(results)
        chunk_label = "chunk" if chunk_count == 1 else "chunks"
        details.append(f"{chunk_count} {chunk_label}")
        details.append(f"{chunk_text_chars(results):,} chunk chars")
    if ledger_entries is not None:
        entry_label = "entry" if ledger_entries == 1 else "entries"
        details.append(f"{ledger_entries} ledger {entry_label}")
    return (
        f"{label} prompt estimate: ~{estimate_prompt_tokens(prompt):,} tokens "
        f"({'; '.join(details)}; rough {CHARS_PER_TOKEN_ESTIMATE} chars/token)."
    )
