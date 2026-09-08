from __future__ import annotations

from collections.abc import Callable

from investigation_rag.cli import build_rag_searcher
from investigation_rag.config import AppConfig
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.rag.prompt_builder import ConversationTurn
from investigation_rag.rag.strict_answering import (
    ANSWER_STYLE_STRICT,
    EVIDENCE_LEDGER_BATCH_SIZE,
    EVIDENCE_LEDGER_CHUNK_LIMIT,
    RAGAnswer,
    StrictAnswerer,
)
from investigation_rag.retrieval.hyde import HYDE_FINAL_CHUNK_LIMIT, HYDE_LEDGER_BATCH_SIZE


LEDGER_PROFILE_NAMES = {"ledger", "hyde"}


class RagRunner:
    """Reusable RAG pipeline configured for one investigation and one set of options."""

    def __init__(
        self,
        config: AppConfig,
        investigation_id: int,
        profile_name: str,
        answer_style: str,
        answer_model: str,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ):
        self.config = config
        self.investigation_id = investigation_id
        self.profile_name = profile_name
        self.answer_style = answer_style
        self.folder_paths = folder_paths
        self.source_types = source_types
        self.client = OllamaClient(
            config.ollama_base_url,
            timeout=config.ollama_timeout_seconds,
            allow_remote=config.ollama_allow_remote,
            num_ctx=config.ollama_num_ctx,
        )
        try:
            self.llm = OllamaLLM(self.client, answer_model)
            db = SQLiteDB(config.sqlite_path)
            searcher_profile = "standard" if profile_name == "ledger" else profile_name
            searcher = build_rag_searcher(config, db, searcher_profile, self.llm)
            self.answerer = StrictAnswerer(searcher, self.llm)
        except Exception:
            self.client.close()
            raise

    def answer(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
        progress_callback: Callable[[str], None] | None = None,
        ledger_callback: Callable[[list[dict[str, object]]], None] | None = None,
    ) -> RAGAnswer:
        if self.profile_name in LEDGER_PROFILE_NAMES:
            if self.profile_name == "hyde":
                progress_message = "Building Hyde evidence ledger in the background."
                chunk_limit = HYDE_FINAL_CHUNK_LIMIT
                batch_size = HYDE_LEDGER_BATCH_SIZE
            else:
                progress_message = "Building evidence ledger in the background."
                chunk_limit = EVIDENCE_LEDGER_CHUNK_LIMIT
                batch_size = EVIDENCE_LEDGER_BATCH_SIZE
            if progress_callback is not None:
                progress_callback(progress_message)
            return self.answerer.answer_with_evidence_ledger(
                self.investigation_id,
                question,
                history=history or [],
                folder_paths=self.folder_paths,
                source_types=self.source_types,
                progress_callback=progress_callback,
                ledger_callback=ledger_callback,
                answer_style=self.answer_style,
                chunk_limit=chunk_limit,
                batch_size=batch_size,
            )

        if progress_callback is not None:
            progress_callback("Retrieving evidence and generating the answer in the background.")
        return self.answerer.answer(
            self.investigation_id,
            question,
            history=history or [],
            folder_paths=self.folder_paths,
            source_types=self.source_types,
            answer_style=self.answer_style or ANSWER_STYLE_STRICT,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "RagRunner":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
