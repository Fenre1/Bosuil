from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import os
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
from investigation_rag.ingest.extraction_cache import ExtractionCache
from investigation_rag.ingest.hashing import sha256_file, sha256_text
from investigation_rag.ingest.pipeline_state import PipelineSignatures, pipeline_signatures
from investigation_rag.ingest.scan_mode import ScanMode
from investigation_rag.ingest.scanner import ScannedFile, scan_supported_files
from investigation_rag.ingest.searchable_text import keyword_searchable_text, stable_embedding_text


LogCallback = Callable[[str], None]
StopCallback = Callable[[], bool]
PROGRESS_LOG_INTERVAL_SECONDS = 0.5
PROGRESS_LOG_FILE_INTERVAL = 250
EMBEDDING_BATCH_MAX_CHUNKS = 64
EMBEDDING_BATCH_MAX_CHARS = 180_000


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
    stored_stable_id = existing.get("stable_id")
    same_identity = not stored_stable_id or not scanned.stable_id or stored_stable_id == scanned.stable_id
    has_hash = bool(existing.get("content_hash"))
    if same_size and same_modified and same_identity and has_hash:
        return ManifestDecision("unchanged_metadata", needs_hash=False, needs_extraction=False)
    return ManifestDecision("suspicious_or_changed", needs_hash=True, needs_extraction=True)


@dataclass
class ScanSummary:
    total_files: int = 0
    total_bytes: int = 0
    entries_visited: int = 0
    directories_visited: int = 0
    inventory_errors: int = 0
    scanned: int = 0
    scanned_bytes: int = 0
    hashed: int = 0
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
        self._scan_session: object | None = None
        self._pending_embedding_signatures: dict[int, PipelineSignatures] = {}
        self._pending_embedding_expected: dict[int, int] = {}
        self._embedded_vectors_by_file: dict[int, int] = {}
        self._verbose_file_logs = True
        self._signature_cache: dict[str, PipelineSignatures] = {}
        self._file_notice_keys: set[str] = set()

    def _record_timing(self, label: str, seconds: float, count: int = 1) -> None:
        self._timings[label] = self._timings.get(label, 0.0) + seconds
        self._timing_counts[label] = self._timing_counts.get(label, 0) + count

    def _signatures(self, suffix: str) -> PipelineSignatures:
        key = suffix.lower()
        signatures = self._signature_cache.get(key)
        if signatures is None:
            signatures = pipeline_signatures(
                key,
                self.chunker,
                self.config.ollama_embed_model,
            )
            self._signature_cache[key] = signatures
        return signatures

    def _log_file_routine(self, message: str) -> None:
        if self._verbose_file_logs:
            self.log(message)

    def _log_file_notice(self, key: str, message: str) -> None:
        if self._verbose_file_logs:
            self.log(message)
            return
        if key in self._file_notice_keys:
            return
        self._file_notice_keys.add(key)
        self.log(f"{message} Further per-file messages of this kind are suppressed.")

    def scan_investigation(
        self,
        investigation_id: int,
        mode: ScanMode | str = ScanMode.FAST,
    ) -> ScanSummary:
        scan_mode = ScanMode.parse(mode)
        self._timings = {}
        self._timing_counts = {}
        self._slow_files = []
        self._embedding_queue = []
        self._embedding_queue_chars = 0
        self._embedding_error_file_ids = set()
        self._ollama_client = None
        self._pending_embedding_signatures = {}
        self._pending_embedding_expected = {}
        self._embedded_vectors_by_file = {}
        self._signature_cache = {}
        self._file_notice_keys = set()
        self._scan_session = None
        summary = ScanSummary()
        seen_file_ids: set[int] = set()
        started_at = monotonic()
        scan_id: int | None = None
        try:
            with self.db.scan_session(investigation_id) as scan_session:
                self._scan_session = scan_session
                roots = self.db.get_roots(investigation_id)
                if not roots:
                    raise ValueError("Investigation has no root folders.")
                scan_id = self.db.start_scan(investigation_id)
                self.log(f"Scanning {len(roots)} root folder(s) in {scan_mode.value} mode.")

                phase_started = monotonic()
                manifest = list(scan_session.load_manifest())
                self._record_timing("manifest preload", monotonic() - phase_started, len(manifest))
                manifest_by_path = {str(row.normalized_path): row for row in manifest}

                self.log("Discovering supported files.")
                last_discovery_log = [monotonic()]

                def discovery_progress(counters: object) -> None:
                    now = monotonic()
                    if now - last_discovery_log[0] < PROGRESS_LOG_INTERVAL_SECONDS:
                        return
                    entries = int(getattr(counters, "entries_visited", 0))
                    supported = int(getattr(counters, "supported_files", 0))
                    directories = int(getattr(counters, "directories_visited", 0))
                    self.log(
                        f"Discovery: entries={entries}, directories={directories}, supported={supported}."
                    )
                    last_discovery_log[0] = now

                phase_started = monotonic()
                inventory = scan_supported_files(
                    roots,
                    stop_requested=self.stop_requested,
                    directory_workers=self.config.scan_discovery_workers,
                    progress=discovery_progress,
                )
                scanned_files = list(inventory.files)
                self._record_timing("file discovery", monotonic() - phase_started, len(scanned_files))
                summary.total_files = len(scanned_files)
                summary.total_bytes = int(inventory.total_bytes)
                summary.entries_visited = int(getattr(inventory.counters, "entries_visited", 0))
                summary.directories_visited = int(getattr(inventory.counters, "directories_visited", 0))
                summary.inventory_errors = len(inventory.errors)
                self._verbose_file_logs = summary.total_files <= self.config.scan_verbose_file_limit
                self.log(
                    f"Discovered {summary.total_files} supported file(s) "
                    f"({self._format_bytes(summary.total_bytes)}) from "
                    f"{summary.entries_visited} filesystem entries."
                )
                if not self._verbose_file_logs:
                    self.log(
                        "Per-file scan messages are suppressed for this large collection; "
                        "aggregate progress and failures will still be reported."
                    )
                if inventory.errors:
                    self.log(
                        f"Inventory encountered {len(inventory.errors)} traversal error(s); "
                        "missing files will not be marked disappeared."
                    )
                if inventory.cancelled or self.stop_requested():
                    summary.stopped = True
                    return self._finish_stopped_scan(scan_id, summary, started_at)

                existing_by_path: dict[str, object] = {}
                matched_old_ids: set[int] = set()
                for scanned in scanned_files:
                    existing = manifest_by_path.get(scanned.normalized_path)
                    if existing is None:
                        continue
                    existing_by_path[scanned.normalized_path] = existing
                    matched_old_ids.add(int(existing.id))
                    seen_file_ids.add(int(existing.id))

                # Portable stable IDs are hints, not proof: every path miss is
                # hashed before an occurrence is reconnected.
                unmatched_by_stable: dict[str, list[object]] = {}
                for row in manifest:
                    if (
                        int(row.id) in matched_old_ids
                        or not row.stable_id
                        or str(row.status) == "inactive"
                    ):
                        continue
                    unmatched_by_stable.setdefault(str(row.stable_id), []).append(row)
                scanned_by_stable: dict[str, list[ScannedFile]] = {}
                for scanned in scanned_files:
                    if scanned.normalized_path in existing_by_path or not scanned.stable_id:
                        continue
                    scanned_by_stable.setdefault(str(scanned.stable_id), []).append(scanned)
                stable_path_candidates: dict[str, object] = {}
                for stable_id, scanned_candidates in scanned_by_stable.items():
                    matches = unmatched_by_stable.get(stable_id, [])
                    if len(matches) != 1 or len(scanned_candidates) != 1:
                        continue
                    existing = matches[0]
                    if int(existing.id) not in matched_old_ids:
                        stable_path_candidates[scanned_candidates[0].normalized_path] = existing

                hash_candidates = self._hash_candidates(scanned_files, existing_by_path, scan_mode)
                hashes, hash_errors = self._hash_files(hash_candidates)
                summary.hashed = len(hashes)
                identity_resolution_complete = not hash_errors

                for normalized_path, existing in stable_path_candidates.items():
                    content_hash = hashes.get(normalized_path)
                    if not content_hash or content_hash != str(existing.content_hash or ""):
                        continue
                    existing_by_path[normalized_path] = existing
                    matched_old_ids.add(int(existing.id))
                    seen_file_ids.add(int(existing.id))

                # A content hash may reconnect one unique missing occurrence, but
                # must never collapse a copy into an already-present occurrence.
                missing_by_hash: dict[str, list[object]] = {}
                for row in manifest:
                    if (
                        int(row.id) in matched_old_ids
                        or not row.content_hash
                        or str(row.status) == "inactive"
                    ):
                        continue
                    missing_by_hash.setdefault(str(row.content_hash), []).append(row)
                new_by_hash: dict[str, list[ScannedFile]] = {}
                for scanned in scanned_files:
                    if scanned.normalized_path in existing_by_path:
                        continue
                    content_hash = hashes.get(scanned.normalized_path)
                    if content_hash:
                        new_by_hash.setdefault(content_hash, []).append(scanned)
                if inventory.complete:
                    for content_hash, candidates in new_by_hash.items():
                        old_candidates = missing_by_hash.get(content_hash, [])
                        if len(candidates) != 1 or len(old_candidates) != 1:
                            continue
                        scanned = candidates[0]
                        existing = old_candidates[0]
                        existing_by_path[scanned.normalized_path] = existing
                        matched_old_ids.add(int(existing.id))
                        seen_file_ids.add(int(existing.id))

                last_progress_at = monotonic()
                for index, scanned in enumerate(scanned_files, start=1):
                    if self.stop_requested():
                        summary.stopped = True
                        break
                    summary.scanned += 1
                    summary.scanned_bytes += scanned.size
                    existing = existing_by_path.get(scanned.normalized_path)
                    if existing is not None:
                        seen_file_ids.add(int(existing.id))

                    if self._verbose_file_logs:
                        self.log(
                            f"Processing {index}/{summary.total_files} "
                            f"({self._format_percent(index, summary.total_files)}, "
                            f"{self._format_bytes(scanned.size)}): {scanned.path}"
                        )

                    hash_error = hash_errors.get(scanned.normalized_path)
                    if hash_error is not None:
                        summary.failed += 1
                        if existing is None:
                            identity_resolution_complete = False
                        self.log(f"Failed to hash: {scanned.path} ({hash_error})")
                        continue

                    known_hash = hashes.get(scanned.normalized_path)
                    if (
                        existing is not None
                        and known_hash is None
                        and scan_mode is ScanMode.FAST
                        and self._file_is_unchanged_and_current(existing, scanned)
                    ):
                        summary.unchanged += 1
                        continue

                    try:
                        file_started = monotonic()
                        file_id, outcome = self._process_scanned_file(
                            investigation_id,
                            scanned,
                            existing=existing,
                            known_hash=known_hash,
                            force_rebuild=scan_mode is ScanMode.REBUILD,
                        )
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
                summary.embedding_errors = len(self._embedding_error_file_ids)
                if summary.stopped or self.stop_requested():
                    summary.stopped = True
                    return self._finish_stopped_scan(scan_id, summary, started_at)

                phase_started = monotonic()
                scan_session.touch_seen_files(seen_file_ids)
                self._record_timing("presence update", monotonic() - phase_started, len(seen_file_ids))

                if scan_mode is ScanMode.REBUILD and self.lancedb_store is not None:
                    phase_started = monotonic()
                    eligible_chunk_ids = scan_session.vector_eligible_chunk_ids()
                    removed_vectors = self.lancedb_store.reconcile_active_chunks(
                        investigation_id,
                        eligible_chunk_ids,
                    )
                    self._record_timing("LanceDB reconciliation", monotonic() - phase_started)
                    if removed_vectors:
                        self.log(f"Removed {removed_vectors} stale vector record(s).")

                phase_started = monotonic()
                if inventory.complete and identity_resolution_complete:
                    active_ids = {int(row.id) for row in manifest if str(row.status) == "active"}
                    missing_ids = active_ids - seen_file_ids
                    summary.disappeared = scan_session.mark_disappeared(sorted(missing_ids))
                    set_vector_activity = getattr(self.lancedb_store, "set_files_active", None)
                    if missing_ids and callable(set_vector_activity):
                        set_vector_activity(
                            sorted(missing_ids),
                            False,
                            investigation_id=investigation_id,
                        )
                    if summary.disappeared:
                        self.log(f"Queued {summary.disappeared} disappeared file(s) for review.")
                else:
                    self.log("Disappeared-file check skipped because inventory or identity matching was incomplete.")
                self._record_timing("disappeared-file check", monotonic() - phase_started)

                elapsed_seconds = monotonic() - started_at
                self._attach_and_log_timings(summary)
                message = self._summary_text(summary, elapsed_seconds)
                self.db.finish_scan(scan_id, "finished", message)
                self.log(message)
                return summary
        except Exception as exc:
            self._attach_and_log_timings(summary)
            if scan_id is not None:
                self.db.finish_scan(scan_id, "failed", str(exc))
            raise
        finally:
            self._scan_session = None
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

    def _finish_stopped_scan(self, scan_id: int, summary: ScanSummary, started_at: float) -> ScanSummary:
        summary.stopped = True
        elapsed_seconds = monotonic() - started_at
        message = self._summary_text(summary, elapsed_seconds)
        self._attach_and_log_timings(summary)
        self.db.finish_scan(scan_id, "stopped", message)
        self.log("Scan stopped. Progress was saved; run the scan again to continue.")
        self.log(message)
        return summary

    def _hash_candidates(
        self,
        scanned_files: list[ScannedFile],
        existing_by_path: dict[str, object],
        mode: ScanMode,
    ) -> list[ScannedFile]:
        candidates: list[ScannedFile] = []
        for scanned in scanned_files:
            existing = existing_by_path.get(scanned.normalized_path)
            if existing is None or mode in (ScanMode.VERIFY, ScanMode.REBUILD):
                candidates.append(scanned)
                continue
            decision = classify_manifest_change(self._manifest_dict(existing), scanned)
            if decision.needs_hash:
                candidates.append(scanned)
        return candidates

    def _hash_files(
        self,
        candidates: list[ScannedFile],
    ) -> tuple[dict[str, str], dict[str, Exception]]:
        if not candidates:
            return {}, {}
        hashes: dict[str, str] = {}
        errors: dict[str, Exception] = {}
        phase_started = monotonic()
        workers = max(1, min(self.config.scan_hash_workers, len(candidates)))

        def hash_one(scanned: ScannedFile) -> tuple[str, float]:
            started = monotonic()
            before = os.stat(scanned.path, follow_symlinks=False)
            self._assert_stat_matches_inventory(scanned, before, "before hashing")
            content_hash = sha256_file(scanned.path)
            after = os.stat(scanned.path, follow_symlinks=False)
            self._assert_stat_matches_inventory(scanned, after, "after hashing")
            if self._stat_fingerprint(before) != self._stat_fingerprint(after):
                raise RuntimeError("file metadata changed while it was being hashed")
            return content_hash, monotonic() - started

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scan-hash") as executor:
            pending_candidates = iter(candidates)
            futures: dict[Future[tuple[str, float]], ScannedFile] = {}
            inflight_limit = max(workers, workers * 2)

            def fill_inflight() -> None:
                while len(futures) < inflight_limit:
                    try:
                        scanned = next(pending_candidates)
                    except StopIteration:
                        return
                    futures[executor.submit(hash_one, scanned)] = scanned

            fill_inflight()
            while futures:
                if self.stop_requested():
                    for future in futures:
                        future.cancel()
                completed, _ = wait(
                    tuple(futures),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    continue
                for future in completed:
                    scanned = futures.pop(future)
                    if future.cancelled():
                        continue
                    try:
                        content_hash, seconds = future.result()
                        hashes[scanned.normalized_path] = content_hash
                        self._slow_files.append((seconds, str(scanned.path), "hashed"))
                    except Exception as exc:
                        errors[scanned.normalized_path] = exc
                if not self.stop_requested():
                    fill_inflight()
        self._record_timing("file hashing", monotonic() - phase_started, len(hashes))
        return hashes, errors

    @classmethod
    def _assert_stat_matches_inventory(
        cls,
        scanned: ScannedFile,
        stat_result: os.stat_result,
        phase: str,
    ) -> None:
        size, modified_ns, stable_id = cls._stat_fingerprint(stat_result)
        if size != scanned.size or modified_ns != scanned.modified_ns:
            raise RuntimeError(f"file metadata changed {phase}")
        if scanned.stable_id and stable_id and scanned.stable_id != stable_id:
            raise RuntimeError(f"file identity changed {phase}")

    @staticmethod
    def _stat_fingerprint(stat_result: os.stat_result) -> tuple[int, int, str | None]:
        modified_ns = getattr(stat_result, "st_mtime_ns", None)
        if modified_ns is None:
            modified_ns = int(stat_result.st_mtime * 1_000_000_000)
        device = int(getattr(stat_result, "st_dev", 0) or 0)
        inode = int(getattr(stat_result, "st_ino", 0) or 0)
        stable_id = f"{device}:{inode}" if device and inode else None
        return int(stat_result.st_size), int(modified_ns), stable_id

    def _file_is_unchanged_and_current(self, existing: object, scanned: ScannedFile) -> bool:
        decision = classify_manifest_change(self._manifest_dict(existing), scanned)
        if decision.needs_hash:
            return False
        if str(getattr(existing, "status", "")) != "active":
            return False
        if str(getattr(existing, "path", "")) != str(scanned.path):
            return False
        if str(getattr(existing, "source_type", "")) != scanned.suffix.lstrip("."):
            return False
        return self._pipeline_action(existing, scanned.suffix) == "current"

    @staticmethod
    def _manifest_dict(existing: object) -> dict[str, object]:
        keys = ("size", "modified_ns", "stable_id", "content_hash")
        return {key: getattr(existing, key, None) for key in keys}

    def _pipeline_action(self, existing: object, suffix: str) -> str:
        signatures = self._signatures(suffix)
        if not bool(getattr(existing, "pipeline_state_exists", False)):
            return "rebuild"
        chunks_current = (
            str(getattr(existing, "source_content_hash", ""))
            == str(getattr(existing, "content_hash", ""))
            and str(getattr(existing, "extraction_signature", "")) == signatures.extraction
            and str(getattr(existing, "chunking_signature", "")) == signatures.chunking
            and str(getattr(existing, "metadata_signature", "")) == signatures.metadata
            and str(getattr(existing, "extraction_status", "")) == "done"
            and str(getattr(existing, "chunk_status", "")) == "done"
            and str(getattr(existing, "metadata_status", "")) == "done"
        )
        if not chunks_current:
            return "rebuild"
        if not signatures.embedding_model:
            return "current"
        vectors_current = (
            str(getattr(existing, "embedding_input_signature", "")) == signatures.embedding_input
            and str(getattr(existing, "embedding_model", "")) == signatures.embedding_model
            and str(getattr(existing, "vector_status", "")) == "done"
            and int(getattr(existing, "active_vector_count", 0))
            == int(getattr(existing, "active_chunk_count", 0))
        )
        return "current" if vectors_current else "embed"

    def _set_pipeline_state(
        self,
        file_id: int,
        content_hash: str,
        signatures: PipelineSignatures,
        *,
        vector_status: str,
        active_vector_count: int,
    ) -> None:
        if self._scan_session is None:
            return
        self._scan_session.update_pipeline_state(
            file_id,
            source_content_hash=content_hash,
            extraction_signature=signatures.extraction,
            chunking_signature=signatures.chunking,
            metadata_signature=signatures.metadata,
            embedding_input_signature=signatures.embedding_input,
            embedding_model=signatures.embedding_model,
            extraction_status="done",
            chunk_status="done",
            metadata_status="done",
            vector_status=vector_status,
            active_vector_count=active_vector_count,
        )

    def _set_pipeline_failure(
        self,
        file_id: int,
        content_hash: str,
        signatures: PipelineSignatures,
        error: str,
    ) -> None:
        del error  # The detailed extractor error remains on the files row.
        if self._scan_session is None:
            return
        self._scan_session.update_pipeline_state(
            file_id,
            source_content_hash=content_hash,
            extraction_signature=signatures.extraction,
            chunking_signature=signatures.chunking,
            metadata_signature=signatures.metadata,
            embedding_input_signature=signatures.embedding_input,
            embedding_model=signatures.embedding_model,
            extraction_status="failed",
            chunk_status="pending",
            metadata_status="pending",
            vector_status="pending",
            active_vector_count=0,
        )

    def _set_vector_state(
        self,
        file_id: int,
        suffix: str,
        status: str,
        *,
        active_vector_count: int,
    ) -> None:
        if self._scan_session is None:
            return
        signatures = self._signatures(suffix)
        self._scan_session.update_pipeline_state(
            file_id,
            embedding_input_signature=signatures.embedding_input,
            embedding_model=signatures.embedding_model,
            vector_status=status,
            active_vector_count=active_vector_count,
        )

    def _clear_pipeline_state(self, file_id: int) -> None:
        if self._scan_session is not None:
            self._scan_session.clear_pipeline_state(file_id)

    def _process_scanned_file(
        self,
        investigation_id: int,
        scanned: ScannedFile,
        *,
        existing: object | None,
        known_hash: str | None,
        force_rebuild: bool = False,
    ) -> tuple[int, str]:
        phase_started = monotonic()
        existing_hash = str(getattr(existing, "content_hash", "") or "") if existing is not None else ""
        content_hash = known_hash or existing_hash
        if not content_hash:
            raise RuntimeError(f"No content hash is available for {scanned.path}.")
        file_id = int(getattr(existing, "id")) if existing is not None else None
        path_changed = existing is not None and str(getattr(existing, "path", "")) != str(scanned.path)
        source_type = scanned.suffix.lstrip(".")
        stable_id = scanned.stable_id or (getattr(existing, "stable_id", None) if existing is not None else None)
        metadata_changed = existing is not None and (
            path_changed
            or int(getattr(existing, "size", -1)) != scanned.size
            or int(getattr(existing, "modified_ns", -1)) != scanned.modified_ns
            or getattr(existing, "stable_id", None) != stable_id
            or str(getattr(existing, "status", "")) != "active"
            or str(getattr(existing, "source_type", "")) != source_type
        )
        same_content = existing is not None and existing_hash == content_hash
        self._record_timing("manifest classification", monotonic() - phase_started)

        if same_content:
            if metadata_changed:
                if path_changed:
                    update_vector_path = getattr(self.lancedb_store, "update_file_path", None)
                    if callable(update_vector_path):
                        update_vector_path(file_id, scanned.path, investigation_id=investigation_id)
                if str(getattr(existing, "status", "")) != "active":
                    set_vector_activity = getattr(self.lancedb_store, "set_files_active", None)
                    if callable(set_vector_activity):
                        set_vector_activity([file_id], True, investigation_id=investigation_id)
                file_id = self.db.upsert_file(
                    investigation_id=investigation_id,
                    path=scanned.path,
                    normalized_path=scanned.normalized_path,
                    stable_id=stable_id,
                    size=scanned.size,
                    modified_ns=scanned.modified_ns,
                    content_hash=content_hash,
                    source_type=source_type,
                    file_id=file_id,
                )
                self.db.update_chunk_paths_for_file(
                    file_id,
                    scanned.path,
                    size=scanned.size,
                    modified_ns=scanned.modified_ns,
                )
                self.db.resolve_disappeared_for_file(file_id)
            version_outcome = self._refresh_versions_or_embeddings(
                investigation_id,
                file_id,
                scanned,
                content_hash,
                existing,
                force_rebuild=force_rebuild,
            )
            if version_outcome is not None:
                return file_id, version_outcome
            return file_id, "metadata" if metadata_changed else "unchanged"

        file_id = self.db.upsert_file(
            investigation_id=investigation_id,
            path=scanned.path,
            normalized_path=scanned.normalized_path,
            stable_id=stable_id,
            size=scanned.size,
            modified_ns=scanned.modified_ns,
            content_hash=content_hash,
            source_type=source_type,
            file_id=file_id,
        )
        if existing is not None:
            self.db.resolve_disappeared_for_file(file_id)
            self._clear_pipeline_state(file_id)
            self.db.set_chunks_inactive_for_file(file_id)
            delete_vectors = getattr(self.lancedb_store, "delete_vectors_for_files", None)
            if callable(delete_vectors):
                delete_vectors([file_id], investigation_id=investigation_id)
        outcome = self._extract_chunk_and_embed(
            investigation_id,
            file_id,
            scanned,
            content_hash,
            replace_existing_vectors=False,
        )
        return file_id, outcome

    def _refresh_versions_or_embeddings(
        self,
        investigation_id: int,
        file_id: int,
        scanned: ScannedFile,
        content_hash: str,
        existing: object,
        *,
        force_rebuild: bool = False,
    ) -> str | None:
        path = scanned.path
        suffix = scanned.suffix
        action = "rebuild" if force_rebuild else self._pipeline_action(existing, suffix)
        if action == "current":
            return None
        if action == "rebuild":
            self._log_file_routine(f"Building current extraction/chunk index: {path}")
            return self._extract_chunk_and_embed(
                investigation_id,
                file_id,
                scanned,
                content_hash,
                replace_existing_vectors=True,
            )
        if self.lancedb_store is None:
            self._set_vector_state(file_id, suffix, "unavailable", active_vector_count=0)
            self._log_file_notice(
                "lancedb-unavailable",
                f"Embedding index is outdated but LanceDB is unavailable: {path}",
            )
            return "metadata"
        signatures = self._signatures(suffix)
        chunk_count = self._queue_existing_file_chunks(file_id, signatures)
        if chunk_count == 0:
            self.lancedb_store.delete_vectors_for_files(
                [file_id], investigation_id=investigation_id
            )
            self._set_vector_state(file_id, suffix, "done", active_vector_count=0)
            return "metadata"
        self._set_vector_state(file_id, suffix, "pending", active_vector_count=0)
        self._log_file_routine(
            f"Queued existing chunks for embedding model {self.config.ollama_embed_model}: {path}"
        )
        return "metadata"

    def _extract_chunk_and_embed(
        self,
        investigation_id: int,
        file_id: int,
        scanned: ScannedFile,
        content_hash: str,
        *,
        replace_existing_vectors: bool,
    ) -> str:
        path = scanned.path
        suffix = scanned.suffix
        source_type = suffix.lstrip(".")
        extractor_version = self._extractor_version_for_suffix(suffix)
        signatures = self._signatures(suffix)
        try:
            self._log_file_routine(f"Extracting text: {path}")
            phase_started = monotonic()
            self._assert_stat_matches_inventory(
                scanned,
                os.stat(path, follow_symlinks=False),
                "before extraction",
            )
            document = self._extract_document(path, content_hash, source_type)
            self._assert_stat_matches_inventory(
                scanned,
                os.stat(path, follow_symlinks=False),
                "after extraction",
            )
            self._record_timing("text/layout extraction", monotonic() - phase_started)
            phase_started = monotonic()
            chunks = self.chunker.chunk_document(document)
            self._record_timing("chunking", monotonic() - phase_started, len(chunks))
            records = self._chunk_records(
                investigation_id,
                file_id,
                scanned,
                content_hash,
                chunks,
                extractor_version,
            )
            phase_started = monotonic()
            batch = self._scan_session.batch() if self._scan_session is not None else nullcontext()
            with batch:
                self.db.set_chunks_inactive_for_file(file_id)
                self.db.insert_chunks(records)
                self.db.mark_file_extraction(file_id, "done", None, extractor_version)
                self._set_pipeline_state(
                    file_id,
                    content_hash,
                    signatures,
                    vector_status="pending" if records and signatures.embedding_model else "done",
                    active_vector_count=0,
                )
            self._record_timing("SQLite/FTS chunk writes", monotonic() - phase_started, len(records))
            delete_vectors = getattr(self.lancedb_store, "delete_vectors_for_files", None)
            if replace_existing_vectors and callable(delete_vectors):
                delete_vectors([file_id], investigation_id=investigation_id)
        except Exception as exc:
            self.db.mark_file_extraction(file_id, "failed", str(exc), extractor_version)
            self._set_pipeline_failure(file_id, content_hash, signatures, str(exc))
            raise

        if not records:
            self._log_file_routine(f"No selectable text found: {path}")
            return "extracted"
        if not self.config.ollama_embed_model:
            self._log_file_notice(
                "embedding-model-unconfigured",
                f"Embedding skipped because OLLAMA_EMBED_MODEL is not configured: {path}",
            )
            return "extracted"
        if self.lancedb_store is None:
            self._set_vector_state(file_id, suffix, "unavailable", active_vector_count=0)
            self._log_file_notice(
                "lancedb-unavailable",
                f"Embedding skipped because LanceDB is unavailable: {path}",
            )
            return "extracted"

        self._log_file_routine(f"Queued {len(records)} chunk(s) for embedding: {path}")
        self._pending_embedding_signatures[file_id] = signatures
        self._pending_embedding_expected[file_id] = len(records)
        self._embedded_vectors_by_file[file_id] = 0
        self._queue_embedding_records(records)
        return "extracted"

    def _extract_document(self, path: Path, content_hash: str, source_type: str, use_cache: bool = True) -> ExtractedDocument:
        extractor = get_extractor_for_path(path)
        extractor_version = extractor.version
        if use_cache:
            document = self.extraction_cache.get(content_hash, extractor_version, source_type)
            if document is not None:
                self._log_file_routine(f"Using cached extraction: {path}")
                return document
        document = extractor.extract(path)
        self.extraction_cache.set(content_hash, extractor_version, source_type, document)
        return document

    def _chunk_records(
        self,
        investigation_id: int,
        file_id: int,
        scanned: ScannedFile,
        content_hash: str,
        chunks: list[Chunk],
        extractor_version: str,
    ) -> list[dict[str, object]]:
        path = scanned.path
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
            stored_metadata.update(
                path=str(path),
                filename=path.name,
                size_bytes=scanned.size,
                modified_ns=scanned.modified_ns,
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
                    "metadata": stored_metadata,
                    "path": str(path),
                    "is_active": True,
                    "extractor_version": extractor_version,
                    "chunker_version": chunker_version,
                    "embedding_model": "",
                    "searchable_text": keyword_searchable_text(path, chunk.text, metadata),
                    "embedding_text": stable_embedding_text(chunk.text, metadata),
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
                num_ctx=self.config.ollama_embed_num_ctx,
            )
        ollama = self._ollama_client
        cache = EmbeddingCache(self.db, ollama, self.config.ollama_embed_model)
        texts = [str(record.get("embedding_text") or record["chunk_text"]) for record in records]
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
        phase_started = monotonic()
        self.db.update_embedding_model_for_chunks(
            [str(record["id"]) for record in records],
            self.config.ollama_embed_model,
        )
        self._record_timing("embedding metadata updates", monotonic() - phase_started, len(records))
        completed_in_batch: dict[int, int] = {}
        for record in records:
            file_id = int(record["file_id"])
            completed_in_batch[file_id] = completed_in_batch.get(file_id, 0) + 1
        for file_id, count in completed_in_batch.items():
            completed = self._embedded_vectors_by_file.get(file_id, 0) + count
            self._embedded_vectors_by_file[file_id] = completed
            expected = self._pending_embedding_expected.get(file_id, completed)
            if completed < expected or self._scan_session is None:
                continue
            signatures = self._pending_embedding_signatures.get(file_id)
            fields: dict[str, object] = {
                "vector_status": "done",
                "active_vector_count": completed,
            }
            if signatures is not None:
                fields.update(
                    embedding_input_signature=signatures.embedding_input,
                    embedding_model=signatures.embedding_model,
                )
            self._scan_session.update_pipeline_state(file_id, fields)

    def _queue_existing_file_chunks(self, file_id: int, signatures: PipelineSignatures) -> int:
        rows = self.db.active_chunks_for_file(file_id)
        records = [
            {
                "id": row["id"],
                "investigation_id": row["investigation_id"],
                "file_id": row["file_id"],
                "path": row["path"],
                "source_type": row["source_type"],
                "chunk_text": row["chunk_text"],
                "embedding_text": stable_embedding_text(
                    str(row["chunk_text"]), self._stored_chunk_metadata(row)
                ),
            }
            for row in rows
        ]
        self._pending_embedding_signatures[file_id] = signatures
        self._pending_embedding_expected[file_id] = len(records)
        self._embedded_vectors_by_file[file_id] = 0
        self._queue_embedding_records(records)
        return len(records)

    def _queue_embedding_records(self, records: list[dict[str, object]]) -> None:
        for record in records:
            text = str(record.get("embedding_text") or record["chunk_text"])
            if self._embedding_queue and (
                len(self._embedding_queue) >= EMBEDDING_BATCH_MAX_CHUNKS
                or self._embedding_queue_chars + len(text) > EMBEDDING_BATCH_MAX_CHARS
            ):
                self._flush_embedding_queue()
            self._embedding_queue.append(record)
            self._embedding_queue_chars += len(text)
        if len(self._embedding_queue) >= EMBEDDING_BATCH_MAX_CHUNKS:
            self._flush_embedding_queue()

    @staticmethod
    def _stored_chunk_metadata(row: object) -> dict[str, object]:
        try:
            raw = row["metadata_json"]
        except (KeyError, TypeError, IndexError):
            raw = None
        metadata: dict[str, object]
        try:
            parsed = json.loads(str(raw)) if raw else {}
            metadata = parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            metadata = {}
        return metadata

    def _flush_embedding_queue(self) -> None:
        if not self._embedding_queue:
            return
        records = self._embedding_queue
        self._embedding_queue = []
        self._embedding_queue_chars = 0
        file_ids = {int(record["file_id"]) for record in records}
        self._log_file_routine(
            f"Embedding batch: {len(records)} chunk(s) from {len(file_ids)} file(s)."
        )
        try:
            self._embed_records(records)
        except (OllamaError, RuntimeError) as exc:
            self._embedding_error_file_ids.update(file_ids)
            if self._scan_session is not None:
                for file_id in file_ids:
                    self._scan_session.update_pipeline_state(
                        file_id,
                        vector_status="failed",
                        active_vector_count=self._embedded_vectors_by_file.get(file_id, 0),
                    )
            self.log(
                "Embedding batch failed; keyword search remains available for "
                f"{len(file_ids)} file(s): {exc}"
            )

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
