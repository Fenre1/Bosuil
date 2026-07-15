from __future__ import annotations

import re
from dataclasses import replace
from time import perf_counter
from typing import Any

from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.retrieval.reranking import LexicalReranker, Reranker
from investigation_rag.retrieval.vector_search import VectorSearcher
from investigation_rag.utils.timing import TimingStep


HYDE_PREFERRED_MODEL = "Gemma4:e4b"
HYDE_REGISTERS = (
    "ordinary language report",
    "interview with a witness",
    "government document",
    "log file",
)
HYDE_VECTOR_CANDIDATES_PER_DOCUMENT = 80
HYDE_TOP_CHUNKS_PER_DOCUMENT = 20
HYDE_FINAL_CHUNK_LIMIT = len(HYDE_REGISTERS) * HYDE_TOP_CHUNKS_PER_DOCUMENT
HYDE_LEDGER_BATCH_SIZE = 8


class HydeSearcher:
    def __init__(
        self,
        vector: VectorSearcher,
        llm: OllamaLLM,
        reranker: Reranker | None = None,
        registers: tuple[str, ...] = HYDE_REGISTERS,
        vector_candidates_per_document: int = HYDE_VECTOR_CANDIDATES_PER_DOCUMENT,
        top_chunks_per_document: int = HYDE_TOP_CHUNKS_PER_DOCUMENT,
    ):
        self.vector = vector
        self.llm = llm
        self.reranker = reranker or LexicalReranker()
        self.registers = registers
        self.vector_candidates_per_document = vector_candidates_per_document
        self.top_chunks_per_document = top_chunks_per_document
        self.final_answer_chunks = len(registers) * top_chunks_per_document
        self.last_timing_steps: list[TimingStep] = []
        self.last_process_feedback: list[str] = []

    def search(
        self,
        investigation_id: int,
        query: str,
        limit: int | None = None,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        self.last_timing_steps = []
        self.last_process_feedback = []
        timings: list[TimingStep] = []
        total_started = perf_counter()
        final_limit = limit or self.final_answer_chunks

        started = perf_counter()
        fake_documents = self.generate_fake_documents(query)
        timings.append(TimingStep("Hyde document generation", perf_counter() - started))
        self.last_process_feedback.append(f"Hyde fake-document model: {self.llm.model}.")
        self.last_process_feedback.append(f"Hyde generated {len(fake_documents)} fake retrieval document(s).")

        per_document_hits: list[SearchResult] = []
        vector_seconds = 0.0
        rerank_seconds = 0.0
        for register, fake_document in fake_documents:
            started = perf_counter()
            vector_results = self.vector.search_embedding_text(
                investigation_id,
                fake_document,
                limit=self.vector_candidates_per_document,
                folder_paths=folder_paths,
                source_types=source_types,
            )
            vector_seconds += perf_counter() - started
            tagged_results = tag_hyde_results(vector_results, register)
            started = perf_counter()
            per_document_hits.extend(
                self.reranker.rerank(
                    fake_document,
                    tagged_results,
                    self.top_chunks_per_document,
                )
            )
            rerank_seconds += perf_counter() - started
        timings.append(TimingStep(f"Hyde vector retrieval ({len(fake_documents)} documents)", vector_seconds))
        timings.append(TimingStep(f"Hyde rerank ({len(fake_documents)} documents)", rerank_seconds))

        started = perf_counter()
        merged = merge_hyde_results(per_document_hits, final_limit)
        timings.append(TimingStep("Hyde result merge", perf_counter() - started))
        self.last_process_feedback.append(
            f"Hyde selected {len(merged)} unique chunk(s) from {len(per_document_hits)} per-document hit(s)."
        )
        timings.append(TimingStep("Retrieval total", perf_counter() - total_started))
        self.last_timing_steps = timings
        return merged

    def generate_fake_documents(self, question: str) -> list[tuple[str, str]]:
        fake_documents: list[tuple[str, str]] = []
        for register in self.registers:
            prompt = build_hyde_document_prompt(register, question)
            fake_document = clean_fake_document(self.llm.answer(prompt))
            if fake_document:
                fake_documents.append((register, fake_document))
        return fake_documents


def build_hyde_document_prompt(register: str, question: str) -> str:
    return (
        "You are a document maker that writes a short paragraph in the style of the following register: "
        f"{register}, which contains a made up answer to the question from the user. "
        f"Uuser question: {question}."
    )


def clean_fake_document(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return text


def tag_hyde_results(results: list[SearchResult], register: str) -> list[SearchResult]:
    tagged = []
    for rank, result in enumerate(results, start=1):
        tagged.append(
            replace(
                result,
                retrieval_source="hyde-vector",
                retrieval_queries=(f"Hyde register: {register}",),
                vector_rank=rank,
            )
        )
    return tagged


def merge_hyde_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}
    queries: dict[str, set[str]] = {}
    for result in results:
        queries.setdefault(result.chunk_id, set()).update(result.retrieval_queries)
        existing = merged.get(result.chunk_id)
        if existing is None or hyde_score_tuple(result) > hyde_score_tuple(existing):
            merged[result.chunk_id] = result
    return [
        replace(result, retrieval_queries=tuple(sorted(queries.get(result.chunk_id, set()))))
        for result in list(merged.values())[:limit]
    ]


def hyde_score_tuple(result: SearchResult) -> tuple[float, float]:
    return (float(result.rerank_score or 0.0), float(result.score))


def choose_hyde_model(client: Any, fallback_model: str, preferred_model: str = HYDE_PREFERRED_MODEL) -> str:
    fallback_model = str(fallback_model or "").strip()
    try:
        models = client.list_models()
    except Exception:
        return fallback_model or preferred_model
    preferred = matching_model_name(models, preferred_model)
    return preferred or fallback_model or preferred_model


def matching_model_name(models: list[dict[str, Any]], requested: str) -> str:
    requested_key = requested.strip().casefold()
    for model in models:
        model_name = str(model.get("name") or model.get("model") or "").strip()
        if model_name.casefold() == requested_key:
            return model_name
    return ""
