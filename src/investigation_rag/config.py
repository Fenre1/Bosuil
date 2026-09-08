from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from investigation_rag.utils.env import load_dotenv_if_available
from investigation_rag.utils.paths import default_app_data_dir


DEFAULT_EMBED_QUERY_INSTRUCTION = "Given a user question, retrieve relevant passages from a local document collection that answer the question."


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _positive_int_env(name: str, default: int) -> int:
    value = _int_env(name, default)
    return value if value > 0 else default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AppConfig:
    app_data_dir: Path
    sqlite_path: Path
    lancedb_dir: Path
    ollama_base_url: str
    ollama_allow_remote: bool
    ollama_timeout_seconds: int
    ollama_num_ctx: int
    ollama_embed_num_ctx: int
    ollama_llm_model: str
    ollama_embed_model: str
    ollama_embed_query_instruction: str
    chunk_target_chars: int
    chunk_max_chars: int
    chunk_overlap_chars: int
    rag_keyword_candidates: int
    rag_vector_candidates: int
    rag_fusion_candidates: int
    rag_rerank_candidates: int
    rag_final_answer_chunks: int
    rag_deep_keyword_candidates: int
    rag_deep_vector_candidates: int
    rag_deep_fusion_candidates: int
    rag_deep_rerank_candidates: int
    rag_deep_final_answer_chunks: int
    gui_lazy_load_file_chunk_text: bool
    pdf_chunk_soft_min_chars: int = 1200
    pdf_chunk_overlap_mode: str = "sentence"
    scan_discovery_workers: int = 4
    scan_hash_workers: int = 2
    scan_verbose_file_limit: int = 100
    chunk_max_tokens: int = 1400

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv_if_available()

        app_data_raw = os.getenv("INVESTIGATION_RAG_DATA_DIR")
        app_data_dir = Path(app_data_raw).expanduser() if app_data_raw else default_app_data_dir()
        app_data_dir = app_data_dir.resolve()

        return cls(
            app_data_dir=app_data_dir,
            sqlite_path=app_data_dir / "catalog.sqlite3",
            lancedb_dir=app_data_dir / "investigations",
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_allow_remote=_bool_env("INVESTIGATION_RAG_ALLOW_REMOTE_OLLAMA", False),
            ollama_timeout_seconds=_positive_int_env("OLLAMA_TIMEOUT_SECONDS", 120),
            ollama_num_ctx=_positive_int_env("OLLAMA_NUM_CTX", 0),
            ollama_embed_num_ctx=_positive_int_env("OLLAMA_EMBED_NUM_CTX", 0),
            ollama_llm_model=os.getenv("OLLAMA_LLM_MODEL", "").strip(),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "").strip(),
            ollama_embed_query_instruction=os.getenv(
                "OLLAMA_EMBED_QUERY_INSTRUCTION",
                DEFAULT_EMBED_QUERY_INSTRUCTION,
            ).strip(),
            chunk_target_chars=_int_env("RAG_CHUNK_TARGET_CHARS", 2800),
            chunk_max_chars=_int_env("RAG_CHUNK_MAX_CHARS", 4800),
            chunk_overlap_chars=_int_env("RAG_CHUNK_OVERLAP_CHARS", 240),
            rag_keyword_candidates=_int_env("RAG_KEYWORD_CANDIDATES", 64),
            rag_vector_candidates=_int_env("RAG_VECTOR_CANDIDATES", 64),
            rag_fusion_candidates=_int_env("RAG_FUSION_CANDIDATES", 80),
            rag_rerank_candidates=_int_env("RAG_RERANK_CANDIDATES", 80),
            rag_final_answer_chunks=_int_env("RAG_FINAL_ANSWER_CHUNKS", 8),
            rag_deep_keyword_candidates=_int_env("RAG_DEEP_KEYWORD_CANDIDATES", 150),
            rag_deep_vector_candidates=_int_env("RAG_DEEP_VECTOR_CANDIDATES", 150),
            rag_deep_fusion_candidates=_int_env("RAG_DEEP_FUSION_CANDIDATES", 200),
            rag_deep_rerank_candidates=_int_env("RAG_DEEP_RERANK_CANDIDATES", 200),
            rag_deep_final_answer_chunks=_int_env("RAG_DEEP_FINAL_ANSWER_CHUNKS", 12),
            gui_lazy_load_file_chunk_text=_bool_env("GUI_LAZY_LOAD_FILE_CHUNK_TEXT", True),
            pdf_chunk_soft_min_chars=_int_env("RAG_PDF_CHUNK_SOFT_MIN_CHARS", 1200),
            pdf_chunk_overlap_mode=os.getenv("RAG_PDF_CHUNK_OVERLAP_MODE", "sentence").strip().lower() or "sentence",
            scan_discovery_workers=_positive_int_env("SCAN_DISCOVERY_WORKERS", 4),
            scan_hash_workers=_positive_int_env("SCAN_HASH_WORKERS", 2),
            scan_verbose_file_limit=_positive_int_env("SCAN_VERBOSE_FILE_LIMIT", 100),
            chunk_max_tokens=_positive_int_env("RAG_CHUNK_MAX_TOKENS", 1400),
        )

    def ensure_dirs(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        (self.app_data_dir / "investigations").mkdir(parents=True, exist_ok=True)

    def for_workspace(self, workspace_path: Path) -> "AppConfig":
        return replace(
            self,
            sqlite_path=workspace_path / "investigation.sqlite3",
            lancedb_dir=workspace_path / "lancedb",
        )
