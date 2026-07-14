from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Any

from investigation_rag.rag.citations import format_citations
from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.rag.prompt_builder import (
    ConversationTurn,
    build_evidence_ledger_prompt,
    build_ledger_final_answer_prompt,
    build_contextual_query_prompt,
    build_strict_prompt,
    location_label,
)
from investigation_rag.retrieval.ranking import SearchResult


INSUFFICIENT_EVIDENCE = "The retrieved evidence is insufficient to answer that."
LEDGER_INSUFFICIENT_EVIDENCE = "The evidence ledger is insufficient to answer that."
EVIDENCE_LEDGER_CHUNK_LIMIT = 50
EVIDENCE_LEDGER_BATCH_SIZE = 5


@dataclass
class RAGAnswer:
    answer: str
    citations: str
    results: list[SearchResult]
    retrieval_query: str = ""
    evidence_ledger: list[dict[str, object]] | None = None
    process_feedback: list[str] | None = None


class StrictAnswerer:
    def __init__(self, searcher: object, llm: OllamaLLM):
        self.searcher = searcher
        self.llm = llm

    def answer(
        self,
        investigation_id: int,
        question: str,
        top_k: int | None = None,
        history: list[ConversationTurn] | None = None,
        folder_paths: list[str] | None = None,
    ) -> RAGAnswer:
        retrieval_query = self.contextualize_question(question, history or [])
        search_limit = top_k if top_k is not None else getattr(self.searcher, "final_answer_chunks", 8)
        if folder_paths is None:
            results = self.searcher.search(investigation_id, retrieval_query, limit=search_limit)
        else:
            results = self.searcher.search(
                investigation_id,
                retrieval_query,
                limit=search_limit,
                folder_paths=folder_paths,
            )
        if not results:
            return RAGAnswer(
                answer=INSUFFICIENT_EVIDENCE,
                citations="",
                results=[],
                retrieval_query=retrieval_query,
            )
        prompt = build_strict_prompt(question, results, history=history)
        answer = self.llm.answer(prompt).strip() or INSUFFICIENT_EVIDENCE
        return RAGAnswer(
            answer=answer,
            citations=format_citations(results),
            results=results,
            retrieval_query=retrieval_query,
        )

    def answer_with_evidence_ledger(
        self,
        investigation_id: int,
        question: str,
        history: list[ConversationTurn] | None = None,
        folder_paths: list[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RAGAnswer:
        feedback: list[str] = []
        retrieval_query = self.contextualize_question(question, history or [])
        self._report_progress(feedback, progress_callback, "Retrieving up to 50 chunks for evidence ledger RAG.")
        if folder_paths is None:
            results = self.searcher.search(investigation_id, retrieval_query, limit=EVIDENCE_LEDGER_CHUNK_LIMIT)
        else:
            results = self.searcher.search(
                investigation_id,
                retrieval_query,
                limit=EVIDENCE_LEDGER_CHUNK_LIMIT,
                folder_paths=folder_paths,
            )
        if not results:
            return RAGAnswer(
                answer=LEDGER_INSUFFICIENT_EVIDENCE,
                citations="",
                results=[],
                retrieval_query=retrieval_query,
                evidence_ledger=[],
                process_feedback=feedback,
            )

        ledger: list[dict[str, object]] = []
        batches = [
            results[index : index + EVIDENCE_LEDGER_BATCH_SIZE]
            for index in range(0, len(results), EVIDENCE_LEDGER_BATCH_SIZE)
        ]
        for batch_index, batch in enumerate(batches, start=1):
            self._report_progress(
                feedback,
                progress_callback,
                f"Evidence ledger round {batch_index} of {len(batches)}: analysing chunks "
                f"{((batch_index - 1) * EVIDENCE_LEDGER_BATCH_SIZE) + 1}-{((batch_index - 1) * EVIDENCE_LEDGER_BATCH_SIZE) + len(batch)}.",
            )
            prompt = build_evidence_ledger_prompt(
                question,
                ledger,
                batch,
                batch_number=batch_index,
                total_batches=len(batches),
            )
            raw_response = self.llm.answer(prompt)
            new_entries = parse_ledger_entries(raw_response)
            enriched = [self.enrich_ledger_entry(entry, batch) for entry in new_entries]
            added = 0
            for entry in enriched:
                if entry and not is_duplicate_ledger_entry(entry, ledger):
                    ledger.append(entry)
                    added += 1
            self._report_progress(
                feedback,
                progress_callback,
                f"Evidence ledger round {batch_index} of {len(batches)} complete: added {added} new entr{'y' if added == 1 else 'ies'} "
                f"({len(ledger)} total).",
            )

        self._report_progress(feedback, progress_callback, "Generating final answer from the evidence ledger only.")
        if ledger:
            final_prompt = build_ledger_final_answer_prompt(question, ledger)
            answer = self.llm.answer(final_prompt).strip() or LEDGER_INSUFFICIENT_EVIDENCE
        else:
            answer = LEDGER_INSUFFICIENT_EVIDENCE
        return RAGAnswer(
            answer=answer,
            citations=format_citations(results),
            results=results,
            retrieval_query=retrieval_query,
            evidence_ledger=ledger,
            process_feedback=feedback,
        )

    def contextualize_question(self, question: str, history: list[ConversationTurn]) -> str:
        if not history:
            return question
        prompt = build_contextual_query_prompt(question, history)
        rewritten = self.llm.answer(prompt).strip()
        return self.clean_contextual_query(rewritten) or question

    @staticmethod
    def clean_contextual_query(value: str) -> str:
        text = value.strip()
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = text.strip("\"'")
        lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        for line in reversed(lines):
            label_match = re.match(r"^(standalone\s+)?search\s+query\s*:\s*(.+)$", line, re.IGNORECASE)
            if label_match:
                return label_match.group(2).strip("\"' ")
        return lines[0]

    @staticmethod
    def enrich_ledger_entry(entry: dict[str, object], batch: list[SearchResult]) -> dict[str, object]:
        chunk_id = str(entry.get("source_chunk_id", "")).strip()
        result_by_id = {result.chunk_id: result for result in batch}
        source = result_by_id.get(chunk_id)
        if source is None and len(batch) == 1:
            source = batch[0]
            chunk_id = source.chunk_id
        enriched = {
            "claim": normalize_ledger_text(entry.get("claim")),
            "source_chunk_id": chunk_id,
            "support_type": normalize_choice(entry.get("support_type"), {"supports", "contradicts", "qualifies", "background"}, "background"),
            "confidence": normalize_choice(entry.get("confidence"), {"high", "medium", "low"}, "medium"),
            "why_relevant": normalize_ledger_text(entry.get("why_relevant")),
            "quoted_support": normalize_ledger_text(entry.get("quoted_support")),
        }
        if source is not None:
            enriched["source_path"] = source.path
            enriched["source_location"] = location_label(source)
            enriched["source_type"] = source.source_type
        return enriched

    @staticmethod
    def _report_progress(
        feedback: list[str],
        progress_callback: Callable[[str], None] | None,
        message: str,
    ) -> None:
        feedback.append(message)
        if progress_callback is not None:
            progress_callback(message)


def parse_ledger_entries(raw_response: str) -> list[dict[str, object]]:
    payload = extract_json_payload(raw_response)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        items: Any = parsed
    elif isinstance(parsed, dict):
        items = parsed.get("new_evidence", [])
    else:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and normalize_ledger_text(item.get("claim"))]


def extract_json_payload(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def is_duplicate_ledger_entry(entry: dict[str, object], ledger: list[dict[str, object]]) -> bool:
    claim = normalize_ledger_text(entry.get("claim")).casefold()
    chunk_id = normalize_ledger_text(entry.get("source_chunk_id"))
    for existing in ledger:
        if normalize_ledger_text(existing.get("claim")).casefold() == claim:
            return True
        if chunk_id and chunk_id == normalize_ledger_text(existing.get("source_chunk_id")):
            existing_claim = normalize_ledger_text(existing.get("claim")).casefold()
            if existing_claim == claim:
                return True
    return False


def normalize_choice(value: object, allowed: set[str], fallback: str) -> str:
    text = normalize_ledger_text(value).casefold()
    return text if text in allowed else fallback


def normalize_ledger_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
