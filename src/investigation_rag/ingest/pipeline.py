from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

from investigation_rag.chunking.base import Chunk
from investigation_rag.chunking.deterministic import DeterministicChunker
from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.embedding_cache import EmbeddingCache
from investigation_rag.embeddings.ollama_client import OllamaClient, OllamaError
from investigation_rag.extractors import get_extractor_for_path
from investigation_rag.extractors.base import EXTRACTOR_VERSION, ExtractedDocument
from investigation_rag.ingest.disappeared import queue_disappeared_files
from investigation_rag.ingest.extraction_cache import ExtractionCache
from investigation_rag.ingest.hashing import sha256_file, sha256_text
from investigation_rag.ingest.scanner import ScannedFile, iter_supported_files


LogCallback = Callable[[str], None]
PROGRESS_LOG_INTERVAL_SECONDS = 5.0
PROGRESS_LOG_FILE_INTERVAL = 10


@dataclass(frozen=True)
class ManifestDecision:
    reason: str
    needs_hash: bool
    needs_extraction: bool


def classify_manifest_change(existing: dict[str, object] | None, scanned: ScannedFile) -> ManifestDecision:
    if existing is None:
        return ManifestDecision("new", needs_hash=True, needs_extraction=True)
    same_size = int(existing.get("size", -1)) == scanned.size
    same_modified = int(existing.get("modified_ns", -1)) == scanned.modified_ns
    has_hash = bool(existing.get("content_hash"))
    if same_size and same_modified and has_hash:
        return ManifestDecision("unchanged_metadata", needs_hash=False, needs_extraction=False)
    return ManifestDecision("suspicious_or_changed", needs_hash=True, needs_extraction=True)


@dataclass
class ScanSummary:
    total_files: int = 0
    total_bytes: int = 0
    scanned: int = 0
    scanned_bytes: int = 0
    unchanged: int = 0
    metadata_only: int = 0
    extracted: int = 0
    failed: int = 0
    disappeared: int = 0
    embedding_errors: int = 0

    @property
    def ingested(self) -> int:
        return self.metadata_only + self.extracted


class ScanPipeline:
    def __init__(
        self,
        config: AppConfig,
        db: SQLiteDB,
        lancedb_store: LanceDBStore | None = None,
        log: LogCallback | None = None,
    ):
        self.config = config
        self.db = db
        self.lancedb_store = lancedb_store
        self.log = log or (lambda message: None)
        self.chunker = DeterministicChunker(
            target_chars=config.chunk_target_chars,
            max_chars=config.chunk_max_chars,
            overlap_chars=config.chunk_overlap_chars,
        )
        self.extraction_cache = ExtractionCache(db)

    def scan_investigation(self, investigation_id: int) -> ScanSummary:
        roots = self.db.get_roots(investigation_id)
        if not roots:
            raise ValueError("Investigation has no root folders.")
        scan_id = self.db.start_scan(investigation_id)
        summary = ScanSummary()
        seen_file_ids: set[int] = set()
        started_at = monotonic()
        try:
            self.log(f"Scanning {len(roots)} root folder(s).")
            self.log("Discovering supported files.")
            scanned_files, total_bytes = self._discover_supported_files(roots)
            summary.total_files = len(scanned_files)
            summary.total_bytes = total_bytes
            self.log(
                f"Discovered {summary.total_files} supported file(s) "
                f"({self._format_bytes(summary.total_bytes)})."
            )
            last_progress_at = started_at
            for index, scanned in enumerate(scanned_files, start=1):
                summary.scanned += 1
                summary.scanned_bytes += scanned.size
                self.log(
                    f"Processing {index}/{summary.total_files} "
                    f"({self._format_percent(index, summary.total_files)}, "
                    f"{self._format_bytes(scanned.size)}): "
                    f"{scanned.path}"
                )
                try:
                    file_id, outcome = self._process_scanned_file(investigation_id, scanned)
                    seen_file_ids.add(file_id)
                    if outcome == "unchanged":
                        summary.unchanged += 1
                    elif outcome == "metadata":
                        summary.metadata_only += 1
                    elif outcome == "extracted":
                        summary.extracted += 1
                    elif outcome == "embedding_error":
                        summary.extracted += 1
                        summary.embedding_errors += 1
                except Exception as exc:
                    summary.failed += 1
                    self.log(f"Failed: {scanned.path} ({exc})")
                now = monotonic()
                if self._should_log_progress(index, summary.total_files, now, last_progress_at):
                    self.log(self._progress_text(summary, now - started_at))
                    last_progress_at = now
            self.log("Checking for disappeared files.")
            summary.disappeared = queue_disappeared_files(self.db, investigation_id, seen_file_ids)
            if summary.disappeared:
                self.log(f"Queued {summary.disappeared} disappeared file(s) for review.")
            elapsed_seconds = monotonic() - started_at
            self.db.finish_scan(scan_id, "finished", self._summary_text(summary, elapsed_seconds))
            self.log(self._summary_text(summary, elapsed_seconds))
            return summary
        except Exception as exc:
            self.db.finish_scan(scan_id, "failed", str(exc))
            raise

    def _discover_supported_files(self, roots: list[Path]) -> tuple[list[ScannedFile], int]:
        files: list[ScannedFile] = []
        total_bytes = 0
        last_log_at = monotonic()
        for scanned in iter_supported_files(roots):
            files.append(scanned)
            total_bytes += scanned.size
            now = monotonic()
            if now - last_log_at >= PROGRESS_LOG_INTERVAL_SECONDS:
                self.log(
                    f"Discovered {len(files)} supported file(s) so far "
                    f"({self._format_bytes(total_bytes)})."
                )
                last_log_at = now
        return files, total_bytes

    def _process_scanned_file(self, investigation_id: int, scanned: ScannedFile) -> tuple[int, str]:
        existing = self.db.get_file_by_path(investigation_id, scanned.normalized_path)
        if existing is None:
            existing = self.db.get_file_by_stable_id(investigation_id, scanned.stable_id)

        existing_dict = dict(existing) if existing is not None else None
        decision = classify_manifest_change(existing_dict, scanned)

        if existing is not None and not decision.needs_hash:
            file_id = int(existing["id"])
            content_hash = str(existing["content_hash"])
            self.db.upsert_file(
                investigation_id=investigation_id,
                path=scanned.path,
                normalized_path=scanned.normalized_path,
                stable_id=scanned.stable_id,
                size=scanned.size,
                modified_ns=scanned.modified_ns,
                content_hash=content_hash,
                source_type=scanned.suffix.lstrip("."),
                file_id=file_id,
            )
            self.db.update_chunk_paths_for_file(file_id, scanned.path)
            self.db.resolve_disappeared_for_file(file_id)
            version_outcome = self._refresh_versions_or_embeddings(
                investigation_id, file_id, scanned.path, scanned.suffix, content_hash
            )
            if version_outcome is not None:
                return file_id, version_outcome
            return file_id, "unchanged"

        content_hash = sha256_file(scanned.path)
        if existing is None:
            existing = self.db.get_active_file_by_content_hash(investigation_id, content_hash)

        file_id = int(existing["id"]) if existing is not None else None
        if existing is not None and existing["content_hash"] == content_hash:
            file_id = self.db.upsert_file(
                investigation_id=investigation_id,
                path=scanned.path,
                normalized_path=scanned.normalized_path,
                stable_id=scanned.stable_id,
                size=scanned.size,
                modified_ns=scanned.modified_ns,
                content_hash=content_hash,
                source_type=scanned.suffix.lstrip("."),
                file_id=file_id,
            )
            self.db.update_chunk_paths_for_file(file_id, scanned.path)
            self.db.resolve_disappeared_for_file(file_id)
            version_outcome = self._refresh_versions_or_embeddings(
                investigation_id, file_id, scanned.path, scanned.suffix, content_hash
            )
            if version_outcome is not None:
                return file_id, version_outcome
            self.log(f"Metadata updated without re-embedding: {scanned.path}")
            return file_id, "metadata"

        file_id = self.db.upsert_file(
            investigation_id=investigation_id,
            path=scanned.path,
            normalized_path=scanned.normalized_path,
            stable_id=scanned.stable_id,
            size=scanned.size,
            modified_ns=scanned.modified_ns,
            content_hash=content_hash,
            source_type=scanned.suffix.lstrip("."),
            file_id=file_id,
        )
        self.db.resolve_disappeared_for_file(file_id)
        outcome = self._extract_chunk_and_embed(investigation_id, file_id, scanned.path, scanned.suffix, content_hash)
        return file_id, outcome

    def _refresh_versions_or_embeddings(
        self,
        investigation_id: int,
        file_id: int,
        path: Path,
        suffix: str,
        content_hash: str,
    ) -> str | None:
        has_current_chunks = self.db.has_active_chunks_for_versions(
            file_id,
            EXTRACTOR_VERSION,
            self.chunker.version,
        )
        if not has_current_chunks:
            if not self.db.active_chunks_for_file(file_id) and self.db.file_extraction_done_for_version(file_id, EXTRACTOR_VERSION):
                self.log(f"Unchanged with no selectable text chunks: {path}")
                return None
            self.log(f"Reusing extraction content but rebuilding chunks for current versions: {path}")
            return self._extract_chunk_and_embed(investigation_id, file_id, path, suffix, content_hash)

        chunk_count = self.db.active_chunk_count_for_file(file_id)
        self.log(f"Unchanged, using {chunk_count} existing chunk(s): {path}")
        refreshed_metadata = False
        if self.db.active_chunks_missing_metadata(file_id):
            self._refresh_existing_chunk_metadata(investigation_id, file_id, path, suffix, content_hash)
            self.log(f"Refreshed chunk metadata without re-embedding: {path}")
            refreshed_metadata = True
        if self.db.active_chunks_need_embedding_model(file_id, self.config.ollama_embed_model):
            if self.lancedb_store is None:
                self.log(f"Embedding model changed but LanceDB is unavailable: {path}")
                return "metadata" if refreshed_metadata else None
            try:
                self._embed_existing_file_chunks(file_id)
                self.log(f"Re-embedded chunks for embedding model {self.config.ollama_embed_model}: {path}")
                return "metadata"
            except (OllamaError, RuntimeError) as exc:
                self.log(f"Re-embedding failed, existing keyword search remains available for {path}: {exc}")
                return "embedding_error"
        if refreshed_metadata:
            return "metadata"
        return None

    def _extract_chunk_and_embed(
        self,
        investigation_id: int,
        file_id: int,
        path: Path,
        suffix: str,
        content_hash: str,
    ) -> str:
        source_type = suffix.lstrip(".")
        try:
            self.log(f"Extracting text: {path}")
            document = self._extract_document(path, content_hash, source_type)
            chunks = self.chunker.chunk_document(document)
            records = self._chunk_records(investigation_id, file_id, path, content_hash, chunks)
            self.db.set_chunks_inactive_for_file(file_id)
            self.db.insert_chunks(records)
            self.db.mark_file_extraction(file_id, "done", None, EXTRACTOR_VERSION)
        except Exception as exc:
            self.db.mark_file_extraction(file_id, "failed", str(exc), EXTRACTOR_VERSION)
            raise

        if not records:
            self.log(f"No selectable text found: {path}")
            return "extracted"
        if not self.config.ollama_embed_model:
            self.log(f"Embedding skipped because OLLAMA_EMBED_MODEL is not configured: {path}")
            return "extracted"
        if self.lancedb_store is None:
            self.log(f"Embedding skipped because LanceDB is unavailable: {path}")
            return "extracted"

        try:
            self.log(f"Embedding {len(records)} chunk(s): {path}")
            self._embed_records(records)
        except (OllamaError, RuntimeError) as exc:
            self.log(f"Embedding failed, keyword search remains available for {path}: {exc}")
            return "embedding_error"
        return "extracted"

    def _extract_document(self, path: Path, content_hash: str, source_type: str, use_cache: bool = True) -> ExtractedDocument:
        if use_cache:
            document = self.extraction_cache.get(content_hash, EXTRACTOR_VERSION, source_type)
            if document is not None:
                self.log(f"Using cached extraction: {path}")
                return document
        extractor = get_extractor_for_path(path)
        document = extractor.extract(path)
        self.extraction_cache.set(content_hash, EXTRACTOR_VERSION, source_type, document)
        return document

    def _refresh_existing_chunk_metadata(
        self,
        investigation_id: int,
        file_id: int,
        path: Path,
        suffix: str,
        content_hash: str,
    ) -> None:
        source_type = suffix.lstrip(".")
        document = self._extract_document(path, content_hash, source_type, use_cache=False)
        chunks = self.chunker.chunk_document(document)
        records = self._chunk_records(investigation_id, file_id, path, content_hash, chunks)
        self.db.update_chunk_metadata(records)

    def _chunk_records(
        self,
        investigation_id: int,
        file_id: int,
        path: Path,
        content_hash: str,
        chunks: list[Chunk],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for chunk in chunks:
            metadata = chunk.metadata
            chunk_text_hash = sha256_text(chunk.text)
            chunk_id = sha256_text(
                f"{investigation_id}:{file_id}:{content_hash}:{self.chunker.version}:{chunk.chunk_index}:{chunk_text_hash}"
            )
            records.append(
                {
                    "id": chunk_id,
                    "investigation_id": investigation_id,
                    "file_id": file_id,
                    "content_hash": content_hash,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.text,
                    "display_markdown": chunk.display_markdown,
                    "chunk_text_hash": chunk_text_hash,
                    "source_type": chunk.source_type,
                    "page_number": chunk.page_number,
                    "email_subject": metadata.get("subject") or metadata.get("parent_email_subject"),
                    "email_sender": metadata.get("sender") or metadata.get("parent_email_sender"),
                    "email_recipients": metadata.get("recipients"),
                    "email_date": metadata.get("date"),
                    "attachment_name": metadata.get("attachment_name"),
                    "metadata": self._display_metadata(path, chunk.source_type, metadata),
                    "path": str(path),
                    "is_active": True,
                    "extractor_version": EXTRACTOR_VERSION,
                    "chunker_version": self.chunker.version,
                    "embedding_model": self.config.ollama_embed_model,
                    "searchable_text": self._searchable_text(path, chunk.text, metadata),
                }
            )
        return records

    @staticmethod
    def _display_metadata(path: Path, source_type: str, metadata: dict[str, object]) -> dict[str, object]:
        display_metadata: dict[str, object] = {
            "filename": path.name,
            "file_type": source_type,
        }
        for key in (
            "path",
            "size_bytes",
            "modified_ns",
            "page_count",
            "page_number",
            "title",
            "author",
            "subject",
            "keywords",
            "creator",
            "producer",
            "creation_date",
            "modification_date",
            "category",
            "comments",
            "last_modified_by",
            "revision",
            "sender",
            "recipients",
            "cc",
            "date",
            "attachment_name",
            "parent_email_subject",
            "parent_email_sender",
            "error",
        ):
            value = metadata.get(key)
            if value not in (None, ""):
                display_metadata[key] = value
        return display_metadata

    def _embed_records(self, records: list[dict[str, object]]) -> None:
        ollama = OllamaClient(
            self.config.ollama_base_url,
            timeout=self.config.ollama_timeout_seconds,
            allow_remote=self.config.ollama_allow_remote,
        )
        cache = EmbeddingCache(self.db, ollama, self.config.ollama_embed_model)
        texts = [str(record.get("searchable_text") or record["chunk_text"]) for record in records]
        vectors = cache.embed_texts(texts)
        lancedb_records = []
        for record, vector in zip(records, vectors):
            lancedb_records.append(
                {
                    "chunk_id": str(record["id"]),
                    "investigation_id": int(record["investigation_id"]),
                    "file_id": int(record["file_id"]),
                    "path": str(record["path"]),
                    "source_type": str(record["source_type"]),
                    "text": str(record.get("searchable_text") or record["chunk_text"]),
                    "is_active": True,
                    "embedding_model": self.config.ollama_embed_model,
                    "vector": vector,
                }
            )
        self.lancedb_store.upsert_chunks(lancedb_records)
        self.db.update_embedding_model_for_chunks(
            [str(record["id"]) for record in records],
            self.config.ollama_embed_model,
        )

    def _embed_existing_file_chunks(self, file_id: int) -> None:
        rows = self.db.active_chunks_for_file(file_id)
        records = [
            {
                "id": row["id"],
                "investigation_id": row["investigation_id"],
                "file_id": row["file_id"],
                "path": row["path"],
                "source_type": row["source_type"],
                "chunk_text": row["chunk_text"],
                "searchable_text": self._searchable_text(Path(str(row["path"])), row["chunk_text"], dict(row)),
            }
            for row in rows
        ]
        self._embed_records(records)

    @staticmethod
    def _searchable_text(path: Path, chunk_text: str, metadata: dict[str, object]) -> str:
        context_parts = [
            f"Document path: {path}",
            f"Document title: {path.stem}",
        ]
        for label, key in (
            ("Subject", "subject"),
            ("Subject", "email_subject"),
            ("Sender", "sender"),
            ("Sender", "email_sender"),
            ("Recipients", "recipients"),
            ("Recipients", "email_recipients"),
            ("Date", "date"),
            ("Date", "email_date"),
            ("Attachment", "attachment_name"),
            ("Parent subject", "parent_email_subject"),
        ):
            value = metadata.get(key)
            if value:
                context_parts.append(f"{label}: {value}")
        return "\n".join([*context_parts, "", str(chunk_text)])

    @staticmethod
    def _should_log_progress(index: int, total_files: int, now: float, last_progress_at: float) -> bool:
        if total_files == 0 or index == total_files:
            return True
        return index % PROGRESS_LOG_FILE_INTERVAL == 0 or now - last_progress_at >= PROGRESS_LOG_INTERVAL_SECONDS

    @classmethod
    def _progress_text(cls, summary: ScanSummary, elapsed_seconds: float) -> str:
        parts = [
            f"Progress: files={summary.scanned}/{summary.total_files} "
            f"({cls._format_percent(summary.scanned, summary.total_files)})",
        ]
        if summary.total_files:
            parts.append(
                f"bytes={cls._format_bytes(summary.scanned_bytes)}/{cls._format_bytes(summary.total_bytes)}"
            )
        parts.extend(
            [
                f"ingested={summary.ingested}",
                f"unchanged={summary.unchanged}",
                f"failed={summary.failed}",
                f"elapsed={cls._format_duration(elapsed_seconds)}",
            ]
        )
        if summary.total_files > summary.scanned and summary.scanned > 0 and elapsed_seconds > 0:
            rate = summary.scanned / elapsed_seconds
            remaining_seconds = (summary.total_files - summary.scanned) / rate
            parts.append(f"eta~{cls._format_duration(remaining_seconds)}")
            parts.append(f"rate={cls._format_file_rate(rate)}")
        return ", ".join(parts)

    @classmethod
    def _summary_text(cls, summary: ScanSummary, elapsed_seconds: float | None = None) -> str:
        parts = [
            f"Scan complete. scanned={summary.scanned}/{summary.total_files}",
            f"ingested={summary.ingested}",
            f"unchanged={summary.unchanged}",
            f"metadata_only={summary.metadata_only}",
            f"extracted={summary.extracted}",
            f"failed={summary.failed}",
            f"disappeared={summary.disappeared}",
            f"embedding_errors={summary.embedding_errors}",
        ]
        if summary.total_files:
            parts.append(f"bytes={cls._format_bytes(summary.scanned_bytes)}/{cls._format_bytes(summary.total_bytes)}")
        if elapsed_seconds is not None:
            parts.append(f"elapsed={cls._format_duration(elapsed_seconds)}")
            if elapsed_seconds > 0 and summary.scanned > 0:
                parts.append(f"average_rate={cls._format_file_rate(summary.scanned / elapsed_seconds)}")
        return ", ".join(parts)

    @staticmethod
    def _format_percent(count: int, total: int) -> str:
        if total <= 0:
            return "100%"
        return f"{(count / total) * 100:.1f}%"

    @staticmethod
    def _format_bytes(byte_count: int) -> str:
        value = float(max(byte_count, 0))
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(value)} B"
                if value < 10:
                    return f"{value:.1f} {unit}"
                return f"{value:.0f} {unit}"
            value /= 1024
        return f"{int(value)} B"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = int(max(0, round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    @staticmethod
    def _format_file_rate(files_per_second: float) -> str:
        if files_per_second >= 1:
            return f"{files_per_second:.1f} files/s"
        return f"{files_per_second * 60:.1f} files/min"
