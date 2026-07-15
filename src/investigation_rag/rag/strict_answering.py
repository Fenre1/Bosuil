from __future__ import annotations

from dataclasses import dataclass
import json
import re
from time import perf_counter
from typing import Callable, Any

from investigation_rag.rag.citations import format_citations
from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.rag.prompt_builder import (
    ConversationTurn,
    build_evidence_ledger_prompt,
    build_ledger_final_answer_prompt,
    build_lenient_ledger_final_answer_prompt,
    build_lenient_prompt,
    build_contextual_query_prompt,
    build_strict_prompt,
    location_label,
)
from investigation_rag.rag.token_estimation import format_prompt_token_estimate
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.utils.timing import TimingStep, format_timing_step, format_timing_steps


INSUFFICIENT_EVIDENCE = "The retrieved evidence is insufficient to answer that."
LEDGER_INSUFFICIENT_EVIDENCE = "The evidence ledger is insufficient to answer that."
EVIDENCE_LEDGER_CHUNK_LIMIT = 50
EVIDENCE_LEDGER_BATCH_SIZE = 5
ANSWER_STYLE_STRICT = "strict"
ANSWER_STYLE_LENIENT = "lenient"
ANSWER_STYLE_BOTH = "both"
ANSWER_STYLE_CHOICES = {ANSWER_STYLE_STRICT, ANSWER_STYLE_LENIENT, ANSWER_STYLE_BOTH}


@dataclass
class RAGAnswer:
    answer: str
    citations: str
    results: list[SearchResult]
    retrieval_query: str = ""
    answer_style: str = ANSWER_STYLE_STRICT
    evidence_ledger: list[dict[str, object]] | None = None
    process_feedback: list[str] | None = None


@dataclass
class EvidenceLedgerRun:
    retrieval_query: str
    results: list[SearchResult]
    ledger: list[dict[str, object]]
    process_feedback: list[str]


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
        source_types: list[str] | None = None,
        answer_style: str = ANSWER_STYLE_STRICT,
    ) -> RAGAnswer:
        feedback: list[str] = []
        normalized_style = normalize_answer_style(answer_style)
        started = perf_counter()
        retrieval_query = self.contextualize_question(question, history or [], prompt_callback=feedback.append)
        if history:
            feedback.append(format_timing_step(TimingStep("Question contextualization", perf_counter() - started)))
        search_limit = top_k if top_k is not None else getattr(self.searcher, "final_answer_chunks", 8)
        started = perf_counter()
        if folder_paths is None and source_types is None:
            results = self.searcher.search(investigation_id, retrieval_query, limit=search_limit)
        elif source_types is None:
            results = self.searcher.search(
                investigation_id,
                retrieval_query,
                limit=search_limit,
                folder_paths=folder_paths,
            )
        else:
            results = self.searcher.search(
                investigation_id,
                retrieval_query,
                limit=search_limit,
                folder_paths=folder_paths,
                source_types=source_types,
            )
        feedback.extend(self.search_timing_feedback(perf_counter() - started))
        if not results and normalized_style == ANSWER_STYLE_STRICT:
            return RAGAnswer(
                answer=INSUFFICIENT_EVIDENCE,
                citations="",
                results=[],
                retrieval_query=retrieval_query,
                answer_style=normalized_style,
                process_feedback=feedback,
            )
        started = perf_counter()
        answer = self.answer_from_results(
            question,
            results,
            history=history,
            answer_style=normalized_style,
            prompt_callback=feedback.append,
        )
        feedback.append(format_timing_step(TimingStep("Final answer", perf_counter() - started)))
        return RAGAnswer(
            answer=answer,
            citations=format_citations(results),
            results=results,
            retrieval_query=retrieval_query,
            answer_style=normalized_style,
            process_feedback=feedback,
        )

    def answer_from_results(
        self,
        question: str,
        results: list[SearchResult],
        history: list[ConversationTurn] | None = None,
        answer_style: str = ANSWER_STYLE_STRICT,
        prompt_callback: Callable[[str], None] | None = None,
    ) -> str:
        normalized_style = normalize_answer_style(answer_style)
        if normalized_style == ANSWER_STYLE_STRICT:
            if not results:
                return INSUFFICIENT_EVIDENCE
            prompt = build_strict_prompt(question, results, history=history)
            self.report_prompt_estimate(prompt_callback, "Final strict answer", prompt, results=results)
            return self.llm.answer(prompt).strip() or INSUFFICIENT_EVIDENCE
        if normalized_style == ANSWER_STYLE_LENIENT:
            prompt = build_lenient_prompt(question, results, history=history)
            self.report_prompt_estimate(prompt_callback, "Final lenient answer", prompt, results=results)
            return self.llm.answer(prompt).strip() or INSUFFICIENT_EVIDENCE

        strict_answer = self.answer_from_results(
            question,
            results,
            history=history,
            answer_style=ANSWER_STYLE_STRICT,
            prompt_callback=prompt_callback,
        )
        lenient_answer = self.answer_from_results(
            question,
            results,
            history=history,
            answer_style=ANSWER_STYLE_LENIENT,
            prompt_callback=prompt_callback,
        )
        return format_dual_answer(strict_answer, lenient_answer)

    def answer_with_evidence_ledger(
        self,
        investigation_id: int,
        question: str,
        history: list[ConversationTurn] | None = None,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
        ledger_callback: Callable[[list[dict[str, object]]], None] | None = None,
        answer_style: str = ANSWER_STYLE_STRICT,
        chunk_limit: int = EVIDENCE_LEDGER_CHUNK_LIMIT,
        batch_size: int = EVIDENCE_LEDGER_BATCH_SIZE,
    ) -> RAGAnswer:
        normalized_style = normalize_answer_style(answer_style)
        started = perf_counter()
        ledger_run = self.build_evidence_ledger(
            investigation_id,
            question,
            history=history,
            folder_paths=folder_paths,
            source_types=source_types,
            progress_callback=progress_callback,
            ledger_callback=ledger_callback,
            chunk_limit=chunk_limit,
            batch_size=batch_size,
        )
        feedback = [
            format_timing_step(TimingStep("Evidence ledger creation", perf_counter() - started)),
            *ledger_run.process_feedback,
        ]
        if progress_callback is not None:
            progress_callback(feedback[-1])
        if not ledger_run.results and normalized_style == ANSWER_STYLE_STRICT:
            return RAGAnswer(
                answer=LEDGER_INSUFFICIENT_EVIDENCE,
                citations="",
                results=[],
                retrieval_query=ledger_run.retrieval_query,
                answer_style=normalized_style,
                evidence_ledger=[],
                process_feedback=feedback,
            )

        self._report_progress(
            [],
            progress_callback,
            final_answer_progress_message(normalized_style),
        )
        started = perf_counter()
        answer = self.answer_from_evidence_ledger(
            question,
            ledger_run.ledger,
            answer_style=normalized_style,
            prompt_callback=self._feedback_and_progress_callback(feedback, progress_callback),
        )
        feedback.append(format_timing_step(TimingStep("Final answer", perf_counter() - started)))
        return RAGAnswer(
            answer=answer,
            citations=format_citations(ledger_run.results),
            results=ledger_run.results,
            retrieval_query=ledger_run.retrieval_query,
            answer_style=normalized_style,
            evidence_ledger=ledger_run.ledger,
            process_feedback=feedback,
        )

    def build_evidence_ledger(
        self,
        investigation_id: int,
        question: str,
        history: list[ConversationTurn] | None = None,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
        ledger_callback: Callable[[list[dict[str, object]]], None] | None = None,
        chunk_limit: int = EVIDENCE_LEDGER_CHUNK_LIMIT,
        batch_size: int = EVIDENCE_LEDGER_BATCH_SIZE,
    ) -> EvidenceLedgerRun:
        progress_feedback: list[str] = []
        retrieval_query = self.contextualize_question(question, history or [], prompt_callback=progress_callback)
        chunk_limit = max(1, int(chunk_limit))
        batch_size = max(1, int(batch_size))
        self._report_progress(
            progress_feedback,
            progress_callback,
            f"Retrieving up to {chunk_limit} chunks for evidence ledger RAG.",
        )
        if folder_paths is None and source_types is None:
            results = self.searcher.search(investigation_id, retrieval_query, limit=chunk_limit)
        elif source_types is None:
            results = self.searcher.search(
                investigation_id,
                retrieval_query,
                limit=chunk_limit,
                folder_paths=folder_paths,
            )
        else:
            results = self.searcher.search(
                investigation_id,
                retrieval_query,
                limit=chunk_limit,
                folder_paths=folder_paths,
                source_types=source_types,
            )
        search_feedback = self.structured_search_feedback()
        for item in search_feedback:
            self._report_progress(progress_feedback, progress_callback, item)
        if not results:
            if ledger_callback is not None:
                ledger_callback([])
            return EvidenceLedgerRun(
                retrieval_query=retrieval_query,
                results=[],
                ledger=[],
                process_feedback=search_feedback,
            )

        ledger: list[dict[str, object]] = []
        batches = [
            results[index : index + batch_size]
            for index in range(0, len(results), batch_size)
        ]
        for batch_index, batch in enumerate(batches, start=1):
            self._report_progress(
                progress_feedback,
                progress_callback,
                f"Evidence ledger round {batch_index} of {len(batches)}: analysing chunks "
                f"{((batch_index - 1) * batch_size) + 1}-{((batch_index - 1) * batch_size) + len(batch)}.",
            )
            prompt = build_evidence_ledger_prompt(
                question,
                ledger,
                batch,
                batch_number=batch_index,
                total_batches=len(batches),
            )
            self._report_progress(
                progress_feedback,
                progress_callback,
                format_prompt_token_estimate(
                    f"Evidence ledger round {batch_index} of {len(batches)}",
                    prompt,
                    results=batch,
                    ledger_entries=len(ledger),
                ),
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
                progress_feedback,
                progress_callback,
                f"Evidence ledger round {batch_index} of {len(batches)} complete: added {added} new entr{'y' if added == 1 else 'ies'} "
                f"({len(ledger)} total).",
            )
            if ledger_callback is not None:
                ledger_callback([dict(entry) for entry in ledger])

        return EvidenceLedgerRun(
            retrieval_query=retrieval_query,
            results=results,
            ledger=ledger,
            process_feedback=search_feedback,
        )

    def answer_from_evidence_ledger(
        self,
        question: str,
        ledger: list[dict[str, object]],
        answer_style: str = ANSWER_STYLE_STRICT,
        prompt_callback: Callable[[str], None] | None = None,
    ) -> str:
        normalized_style = normalize_answer_style(answer_style)
        if normalized_style == ANSWER_STYLE_STRICT:
            if not ledger:
                return LEDGER_INSUFFICIENT_EVIDENCE
            final_prompt = build_ledger_final_answer_prompt(question, ledger)
            self.report_prompt_estimate(
                prompt_callback,
                "Final strict ledger answer",
                final_prompt,
                ledger_entries=len(ledger),
            )
            return self.llm.answer(final_prompt).strip() or LEDGER_INSUFFICIENT_EVIDENCE
        if normalized_style == ANSWER_STYLE_LENIENT:
            final_prompt = build_lenient_ledger_final_answer_prompt(question, ledger)
            self.report_prompt_estimate(
                prompt_callback,
                "Final lenient ledger answer",
                final_prompt,
                ledger_entries=len(ledger),
            )
            return self.llm.answer(final_prompt).strip() or LEDGER_INSUFFICIENT_EVIDENCE

        strict_answer = self.answer_from_evidence_ledger(
            question,
            ledger,
            answer_style=ANSWER_STYLE_STRICT,
            prompt_callback=prompt_callback,
        )
        lenient_answer = self.answer_from_evidence_ledger(
            question,
            ledger,
            answer_style=ANSWER_STYLE_LENIENT,
            prompt_callback=prompt_callback,
        )
        return format_dual_answer(strict_answer, lenient_answer)

    def contextualize_question(
        self,
        question: str,
        history: list[ConversationTurn],
        prompt_callback: Callable[[str], None] | None = None,
    ) -> str:
        if not history:
            return question
        prompt = build_contextual_query_prompt(question, history)
        self.report_prompt_estimate(prompt_callback, "Question contextualization", prompt)
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

    def search_timing_feedback(self, fallback_seconds: float) -> list[str]:
        feedback: list[str] = []
        messages = getattr(self.searcher, "last_process_feedback", None)
        if isinstance(messages, list):
            feedback.extend(str(message) for message in messages if message)
        steps = getattr(self.searcher, "last_timing_steps", None)
        if isinstance(steps, list) and steps:
            feedback.extend(format_timing_steps(steps))
            return feedback
        feedback.append(format_timing_step(TimingStep("Retrieval", fallback_seconds)))
        return feedback

    def structured_search_feedback(self) -> list[str]:
        feedback: list[str] = []
        messages = getattr(self.searcher, "last_process_feedback", None)
        if isinstance(messages, list):
            feedback.extend(str(message) for message in messages if message)
        steps = getattr(self.searcher, "last_timing_steps", None)
        if isinstance(steps, list) and steps:
            feedback.extend(format_timing_steps(steps))
        return feedback

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

    @staticmethod
    def report_prompt_estimate(
        prompt_callback: Callable[[str], None] | None,
        label: str,
        prompt: str,
        *,
        results: list[SearchResult] | None = None,
        ledger_entries: int | None = None,
    ) -> None:
        if prompt_callback is not None:
            prompt_callback(
                format_prompt_token_estimate(
                    label,
                    prompt,
                    results=results,
                    ledger_entries=ledger_entries,
                )
            )

    @staticmethod
    def _feedback_and_progress_callback(
        feedback: list[str],
        progress_callback: Callable[[str], None] | None,
    ) -> Callable[[str], None]:
        def callback(message: str) -> None:
            feedback.append(message)
            if progress_callback is not None:
                progress_callback(message)

        return callback


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


def normalize_answer_style(answer_style: str) -> str:
    normalized = str(answer_style or ANSWER_STYLE_STRICT).strip().casefold()
    if normalized not in ANSWER_STYLE_CHOICES:
        raise ValueError(f"Unknown answer style: {answer_style}")
    return normalized


def format_dual_answer(strict_answer: str, lenient_answer: str) -> str:
    return f"## Strict answer\n\n{strict_answer}\n\n## Lenient answer\n\n{lenient_answer}"


def final_answer_progress_message(answer_style: str) -> str:
    if answer_style == ANSWER_STYLE_BOTH:
        return "Generating strict and lenient final answers from the evidence ledger."
    return f"Generating final {answer_style} answer from the evidence ledger."


def normalize_choice(value: object, allowed: set[str], fallback: str) -> str:
    text = normalize_ledger_text(value).casefold()
    return text if text in allowed else fallback


def normalize_ledger_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
