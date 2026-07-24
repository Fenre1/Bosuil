from __future__ import annotations

from dataclasses import dataclass, field
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
from investigation_rag.extractors import get_extractor_for_path, get_extractor_version_for_suffix
from investigation_rag.extractors.base import ExtractedDocument
from investigation_rag.ingest.disappeared import queue_disappeared_files
from investigation_rag.ingest.extraction_cache import ExtractionCache
from investigation_rag.ingest.hashing import sha256_file, sha256_text
from investigation_rag.ingest.scanner import ScannedFile, iter_supported_files


LogCallback = Callable[[str], None]
StopCallback = Callable[[], bool]
PROGRESS_LOG_INTERVAL_SECONDS = 5.0
PROGRESS_LOG_FILE_INTERVAL = 10
EMBEDDING_BATCH_MAX_CHUNKS = 64
EMBEDDING_BATCH_MAX_CHARS = 180_000
LANCEDB_OPTIMIZE_AFTER_WRITES = 1_000


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
    stopped: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    timing_counts: dict[str, int] = field(default_factory=dict)

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
        stop_requested: StopCallback | None = None,
    ):
        self.config = config
        self.db = db
        self.lancedb_store = lancedb_store
        self.log = log or (lambda message: None)
        self.stop_requested = stop_requested or (lambda: False)
        self.chunker = DeterministicChunker(
            target_chars=config.chunk_target_chars,
            max_chars=config.chunk_max_chars,
            overlap_chars=config.chunk_overlap_chars,
            pdf_soft_min_chars=config.pdf_chunk_soft_min_chars,
            pdf_overlap_mode=config.pdf_chunk_overlap_mode,
        )
        self.extraction_cache = ExtractionCache(db)
        self._timings: dict[str, float] = {}
        self._timing_counts: dict[str, int] = {}
        self._slow_files: list[tuple[float, str, str]] = []
        self._embedding_queue: list[dict[str, object]] = []
        self._embedding_queue_chars = 0
        self._embedding_error_file_ids: set[int] = set()
        self._ollama_client: OllamaClient | None = None
        self._embedded_chunk_count = 0

    def _record_timing(self, label: str, seconds: float, count: int = 1) -> None:
        self._timings[label] = self._timings.get(label, 0.0) + seconds
        self._timing_counts[label] = self._timing_counts.get(label, 0) + count

    def scan_investigation(self, investigation_id: int) -> ScanSummary:
        self._timings = {}
        self._timing_counts = {}
        self._slow_files = []
        self._embedding_queue = []
        self._embedding_queue_chars = 0
        self._embedding_error_file_ids = set()
        self._ollama_client = None
        self._embedded_chunk_count = 0
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
            phase_started = monotonic()
            scanned_files, total_bytes = self._discover_supported_files(roots)
            self._record_timing("file discovery", monotonic() - phase_started, len(scanned_files))
            summary.total_files = len(scanned_files)
            summary.total_bytes = total_bytes
            if self.stop_requested():
                summary.stopped = True
                self._attach_and_log_timings(summary)
                self.db.finish_scan(scan_id, "stopped", "Scan stopped during file discovery.")
                self.log("Scan stopped. Progress was saved; run the scan again to continue.")
                return summary
            self.log(
                f"Discovered {summary.total_files} supported file(s) "
                f"({self._format_bytes(summary.total_bytes)})."
            )
            last_progress_at = started_at
            for index, scanned in enumerate(scanned_files, start=1):
                if self.stop_requested():
                    summary.stopped = True
                    break
                summary.scanned += 1
                summary.scanned_bytes += scanned.size
                self.log(
                    f"Processing {index}/{summary.total_files} "
                    f"({self._format_percent(index, summary.total_files)}, "
                    f"{self._format_bytes(scanned.size)}): "
                    f"{scanned.path}"
                )
                try:
                    file_started = monotonic()
                    file_id, outcome = self._process_scanned_file(investigation_id, scanned)
                    file_seconds = monotonic() - file_started
                    self._record_timing("file processing", file_seconds)
                    self._slow_files.append((file_seconds, str(scanned.path), outcome))
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
                    file_seconds = monotonic() - file_started
                    self._record_timing("file processing", file_seconds)
                    self._slow_files.append((file_seconds, str(scanned.path), "failed"))
                    summary.failed += 1
                    self.log(f"Failed: {scanned.path} ({exc})")
                now = monotonic()
                if self._should_log_progress(index, summary.total_files, now, last_progress_at):
                    self.log(self._progress_text(summary, now - started_at))
                    last_progress_at = now
            self._flush_embedding_queue()
            self._maybe_optimize_vector_store()
            summary.embedding_errors = len(self._embedding_error_file_ids)
            if summary.stopped or self.stop_requested():
                summary.stopped = True
                elapsed_seconds = monotonic() - started_at
                message = self._summary_text(summary, elapsed_seconds)
                self._attach_and_log_timings(summary)
                self.db.finish_scan(scan_id, "stopped", message)
                self.log("Scan stopped. Progress was saved; run the scan again to continue.")
                self.log(message)
                return summary
            self.log("Checking for disappeared files.")
            phase_started = monotonic()
            summary.disappeared = queue_disappeared_files(self.db, investigation_id, seen_file_ids)
            self._record_timing("disappeared-file check", monotonic() - phase_started)
            if summary.disappeared:
                self.log(f"Queued {summary.disappeared} disappeared file(s) for review.")
            elapsed_seconds = monotonic() - started_at
            self._attach_and_log_timings(summary)
            self.db.finish_scan(scan_id, "finished", self._summary_text(summary, elapsed_seconds))
            self.log(self._summary_text(summary, elapsed_seconds))
            return summary
        except Exception as exc:
            self._attach_and_log_timings(summary)
            self.db.finish_scan(scan_id, "failed", str(exc))
            raise
        finally:
            if self._ollama_client is not None:
                self._ollama_client.close()

    def _attach_and_log_timings(self, summary: ScanSummary) -> None:
        summary.timings = dict(self._timings)
        summary.timing_counts = dict(self._timing_counts)
        ordered = sorted(self._timings.items(), key=lambda item: item[1], reverse=True)
        if ordered:
            self.log(
                "Timing breakdown: "
                + ", ".join(
                    f"{label}={self._format_duration(seconds)}"
                    f" ({self._timing_counts.get(label, 0)}x)"
                    for label, seconds in ordered
                )
            )
        cache_hits = self._timing_counts.get("embedding cache hits", 0)
        cache_misses = self._timing_counts.get("embedding cache misses", 0)
        if cache_hits or cache_misses:
            self.log(f"Embedding cache: hits={cache_hits}, misses={cache_misses}")
        slowest = sorted(self._slow_files, reverse=True)[:10]
        if slowest:
            self.log("Slowest files:")
            for seconds, path, outcome in slowest:
                self.log(f"  {self._format_duration(seconds)} [{outcome}] {path}")

    def _discover_supported_files(self, roots: list[Path]) -> tuple[list[ScannedFile], int]:
        files: list[ScannedFile] = []
        total_bytes = 0
        last_log_at = monotonic()
        for scanned in iter_supported_files(roots):
            if self.stop_requested():
                break
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
        phase_started = monotonic()
        existing = self.db.get_file_by_path(investigation_id, scanned.normalized_path)
        if existing is None:
            existing = self.db.get_file_by_stable_id(investigation_id, scanned.stable_id)

        existing_dict = dict(existing) if existing is not None else None
        decision = classify_manifest_change(existing_dict, scanned)
        self._record_timing("manifest lookup/classification", monotonic() - phase_started)

        if existing is not None and not decision.needs_hash:
            file_id = int(existing["id"])
            content_hash = str(existing["content_hash"])
            path_changed = str(existing["path"]) != str(scanned.path)
            metadata_changed = (
                path_changed
                or existing["stable_id"] != scanned.stable_id
                or str(existing["status"]) != "active"
                or str(existing["source_type"]) != scanned.suffix.lstrip(".")
            )
            if metadata_changed:
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
                if path_changed:
                    self.db.update_chunk_paths_for_file(file_id, scanned.path)
                self.db.resolve_disappeared_for_file(file_id)
            version_outcome = self._refresh_versions_or_embeddings(
                investigation_id, file_id, scanned.path, scanned.suffix, content_hash
            )
            if version_outcome is not None:
                return file_id, version_outcome
            return file_id, "unchanged"

        phase_started = monotonic()
        content_hash = sha256_file(scanned.path)
        self._record_timing("file hashing", monotonic() - phase_started)
        if existing is None:
            existing = self.db.get_active_file_by_content_hash(investigation_id, content_hash)

        file_id = int(existing["id"]) if existing is not None else None
        if existing is not None and existing["content_hash"] == content_hash:
            path_changed = str(existing["path"]) != str(scanned.path)
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
            if path_changed:
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
            self._extractor_version_for_suffix(suffix),
            self._chunker_version_for_source_type(suffix.lstrip(".")),
        )
        if not has_current_chunks:
            extractor_version = self._extractor_version_for_suffix(suffix)
            if not self.db.active_chunks_for_file(file_id) and self.db.file_extraction_done_for_version(file_id, extractor_version):
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
            self._queue_existing_file_chunks(file_id)
            self.log(f"Queued chunks for embedding model {self.config.ollama_embed_model}: {path}")
            return "metadata"
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
        extractor_version = self._extractor_version_for_suffix(suffix)
        try:
            self.log(f"Extracting text: {path}")
            phase_started = monotonic()
            document = self._extract_document(path, content_hash, source_type)
            self._record_timing("text/layout extraction", monotonic() - phase_started)
            phase_started = monotonic()
            chunks = self.chunker.chunk_document(document)
            self._record_timing("chunking", monotonic() - phase_started, len(chunks))
            records = self._chunk_records(investigation_id, file_id, path, content_hash, chunks, extractor_version)
            phase_started = monotonic()
            self.db.set_chunks_inactive_for_file(file_id)
            self.db.insert_chunks(records)
            self.db.mark_file_extraction(file_id, "done", None, extractor_version)
            self._record_timing("SQLite/FTS chunk writes", monotonic() - phase_started, len(records))
        except Exception as exc:
            self.db.mark_file_extraction(file_id, "failed", str(exc), extractor_version)
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

        self.log(f"Queued {len(records)} chunk(s) for embedding: {path}")
        self._queue_embedding_records(records)
        return "extracted"

    def _extract_document(self, path: Path, content_hash: str, source_type: str, use_cache: bool = True) -> ExtractedDocument:
        extractor = get_extractor_for_path(path)
        extractor_version = extractor.version
        if use_cache:
            document = self.extraction_cache.get(content_hash, extractor_version, source_type)
            if document is not None:
                self.log(f"Using cached extraction: {path}")
                return document
        document = extractor.extract(path)
        self.extraction_cache.set(content_hash, extractor_version, source_type, document)
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
        extractor_version = self._extractor_version_for_suffix(suffix)
        document = self._extract_document(path, content_hash, source_type, use_cache=False)
        chunks = self.chunker.chunk_document(document)
        records = self._chunk_records(investigation_id, file_id, path, content_hash, chunks, extractor_version)
        self.db.update_chunk_metadata(records)

    def _chunk_records(
        self,
        investigation_id: int,
        file_id: int,
        path: Path,
        content_hash: str,
        chunks: list[Chunk],
        extractor_version: str,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for chunk in chunks:
            metadata = chunk.metadata
            chunker_version = self._chunker_version_for_source_type(chunk.source_type)
            chunk_text_hash = sha256_text(chunk.text)
            chunk_id = sha256_text(
                f"{investigation_id}:{file_id}:{content_hash}:{chunker_version}:{chunk.chunk_index}:{chunk_text_hash}"
            )
            stored_metadata = dict(metadata)
            stored_metadata.update(self._display_metadata(path, chunk.source_type, metadata))
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
                    "metadata": stored_metadata,
                    "path": str(path),
                    "is_active": True,
                    "extractor_version": extractor_version,
                    "chunker_version": chunker_version,
                    "embedding_model": "",
                    "searchable_text": self._searchable_text(path, chunk.text, metadata),
                }
            )
        return records

    def _extractor_version_for_suffix(self, suffix: str) -> str:
        return get_extractor_version_for_suffix(suffix)

    def _chunker_version_for_source_type(self, source_type: str) -> str:
        return self.chunker.version_for_source_type(source_type)

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
            "page_start",
            "page_end",
            "pages",
            "heading_path_text",
            "content_type",
            "chunk_index_within_section",
            "is_continuation",
            "contains_table",
            "contains_list",
            "contains_caption",
            "contains_footnote",
            "contains_figure",
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
        if self._ollama_client is None:
            self._ollama_client = OllamaClient(
                self.config.ollama_base_url,
                timeout=self.config.ollama_timeout_seconds,
                allow_remote=self.config.ollama_allow_remote,
            )
        ollama = self._ollama_client
        cache = EmbeddingCache(self.db, ollama, self.config.ollama_embed_model)
        texts = [str(record.get("searchable_text") or record["chunk_text"]) for record in records]
        try:
            vectors = cache.embed_texts(texts)
        finally:
            timing = cache.last_timing
            self._record_timing("embedding cache lookups", timing.cache_lookup_seconds, len(texts))
            if timing.cache_write_seconds:
                self._record_timing("embedding cache writes", timing.cache_write_seconds, timing.cache_misses)
            if timing.ollama_seconds:
                self._record_timing("Ollama embedding inference", timing.ollama_seconds, timing.ollama_requests)
            self._timing_counts["embedding cache hits"] = (
                self._timing_counts.get("embedding cache hits", 0) + timing.cache_hits
            )
            self._timing_counts["embedding cache misses"] = (
                self._timing_counts.get("embedding cache misses", 0) + timing.cache_misses
            )
        lancedb_records = []
        for record, vector in zip(records, vectors):
            lancedb_records.append(
                {
                    "chunk_id": str(record["id"]),
                    "investigation_id": int(record["investigation_id"]),
                    "file_id": int(record["file_id"]),
                    "path": str(record["path"]),
                    "source_type": str(record["source_type"]),
                    "is_active": True,
                    "embedding_model": self.config.ollama_embed_model,
                    "vector": vector,
                }
            )
        phase_started = monotonic()
        self.lancedb_store.upsert_chunks(lancedb_records)
        self._record_timing("LanceDB writes", monotonic() - phase_started, len(lancedb_records))
        self._embedded_chunk_count += len(lancedb_records)
        phase_started = monotonic()
        self.db.update_embedding_model_for_chunks(
            [str(record["id"]) for record in records],
            self.config.ollama_embed_model,
        )
        self._record_timing("embedding metadata updates", monotonic() - phase_started, len(records))

    def _queue_existing_file_chunks(self, file_id: int) -> None:
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
        self._queue_embedding_records(records)

    def _queue_embedding_records(self, records: list[dict[str, object]]) -> None:
        for record in records:
            text = str(record.get("searchable_text") or record["chunk_text"])
            if self._embedding_queue and (
                len(self._embedding_queue) >= EMBEDDING_BATCH_MAX_CHUNKS
                or self._embedding_queue_chars + len(text) > EMBEDDING_BATCH_MAX_CHARS
            ):
                self._flush_embedding_queue()
            self._embedding_queue.append(record)
            self._embedding_queue_chars += len(text)
        if len(self._embedding_queue) >= EMBEDDING_BATCH_MAX_CHUNKS:
            self._flush_embedding_queue()

    def _flush_embedding_queue(self) -> None:
        if not self._embedding_queue:
            return
        records = self._embedding_queue
        self._embedding_queue = []
        self._embedding_queue_chars = 0
        file_ids = {int(record["file_id"]) for record in records}
        self.log(f"Embedding batch: {len(records)} chunk(s) from {len(file_ids)} file(s).")
        try:
            self._embed_records(records)
        except (OllamaError, RuntimeError) as exc:
            self._embedding_error_file_ids.update(file_ids)
            self.log(
                "Embedding batch failed; keyword search remains available for "
                f"{len(file_ids)} file(s): {exc}"
            )

    def _maybe_optimize_vector_store(self) -> None:
        if (
            self.lancedb_store is None
            or self._embedded_chunk_count < LANCEDB_OPTIMIZE_AFTER_WRITES
        ):
            return
        phase_started = monotonic()
        try:
            optimized = self.lancedb_store.optimize()
        except Exception as exc:
            self.log(f"LanceDB compaction skipped: {exc}")
            return
        if optimized:
            self._record_timing("LanceDB compaction", monotonic() - phase_started)
            self.log(f"Compacted LanceDB after {self._embedded_chunk_count} vector write(s).")

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
            f"{'Scan stopped' if summary.stopped else 'Scan complete'}. "
            f"scanned={summary.scanned}/{summary.total_files}",
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
