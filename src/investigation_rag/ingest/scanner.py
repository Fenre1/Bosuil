from __future__ import annotations

import os
import stat
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Iterator


SUPPORTED_EXTENSIONS = frozenset({".txt", ".docx", ".pdf", ".msg"})
DEFAULT_DIRECTORY_WORKERS = 4
_NAME_SURROGATE_REPARSE_BIT = 0x20000000

StopCallback = Callable[[], bool]
ProgressCallback = Callable[["InventoryProgress"], None]


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    normalized_path: str
    suffix: str
    stable_id: str | None
    size: int
    modified_ns: int


@dataclass(frozen=True)
class InventoryError:
    """A path that could not be inventoried completely."""

    root: Path
    path: Path
    operation: str
    message: str
    errno: int | None = None


@dataclass(frozen=True)
class InventoryCounters:
    roots_requested: int = 0
    roots_scanned: int = 0
    roots_deduplicated: int = 0
    directories_scanned: int = 0
    entries_seen: int = 0
    type_checks: int = 0
    metadata_reads: int = 0
    identity_reads: int = 0
    supported_files: int = 0
    supported_bytes: int = 0
    unsupported_entries: int = 0
    special_entries: int = 0
    skipped_links: int = 0
    duplicate_paths: int = 0
    errors: int = 0

    @property
    def entries_visited(self) -> int:
        """Compatibility name used by scan summaries."""

        return self.entries_seen

    @property
    def directories_visited(self) -> int:
        """Compatibility name used by scan summaries."""

        return self.directories_scanned


@dataclass(frozen=True)
class InventoryProgress:
    counters: InventoryCounters
    current_root: Path | None = None

    @property
    def entries_visited(self) -> int:
        return self.counters.entries_visited

    @property
    def directories_visited(self) -> int:
        return self.counters.directories_visited

    @property
    def supported_files(self) -> int:
        return self.counters.supported_files


@dataclass(frozen=True)
class RootScanResult:
    root: Path
    normalized_root: str
    exists: bool | None
    complete: bool
    covered_by: Path | None
    supported_files: int
    supported_bytes: int
    errors: tuple[InventoryError, ...] = ()


@dataclass(frozen=True)
class ScanInventory:
    files: tuple[ScannedFile, ...]
    root_results: tuple[RootScanResult, ...]
    errors: tuple[InventoryError, ...]
    counters: InventoryCounters
    cancelled: bool = False

    @property
    def total_bytes(self) -> int:
        return self.counters.supported_bytes

    @property
    def complete(self) -> bool:
        return not self.cancelled and all(result.complete for result in self.root_results)


@dataclass
class _CounterState:
    roots_requested: int = 0
    roots_scanned: int = 0
    roots_deduplicated: int = 0
    directories_scanned: int = 0
    entries_seen: int = 0
    type_checks: int = 0
    metadata_reads: int = 0
    identity_reads: int = 0
    supported_files: int = 0
    supported_bytes: int = 0
    unsupported_entries: int = 0
    special_entries: int = 0
    skipped_links: int = 0
    duplicate_paths: int = 0
    errors: int = 0

    def snapshot(self) -> InventoryCounters:
        return InventoryCounters(**vars(self))


@dataclass
class _RootState:
    path: Path
    normalized_path: str
    exists: bool | None = None
    kind: str | None = None
    safe: bool = False
    covered_by: Path | None = None
    errors: list[InventoryError] = field(default_factory=list)
    stat_result: os.stat_result | None = None


@dataclass(frozen=True)
class _DirectoryJob:
    path: Path
    root: Path
    root_device: int


@dataclass
class _DirectoryBatch:
    job: _DirectoryJob
    child_directories: list[Path] = field(default_factory=list)
    files: list[ScannedFile] = field(default_factory=list)
    errors: list[InventoryError] = field(default_factory=list)
    directories_scanned: int = 0
    entries_seen: int = 0
    type_checks: int = 0
    metadata_reads: int = 0
    identity_reads: int = 0
    unsupported_entries: int = 0
    special_entries: int = 0
    skipped_links: int = 0
    cancelled: bool = False


def lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute, normalized path without resolving it on the filesystem."""

    expanded = os.path.expanduser(os.fspath(path))
    return Path(os.path.normpath(os.path.abspath(expanded)))


def normalize_inventory_path(path: str | Path) -> str:
    """Build the case-insensitive manifest key without a filesystem lookup."""

    return os.path.normcase(os.fspath(lexical_absolute_path(path))).casefold()


def scan_supported_files(
    roots: Iterable[str | Path],
    *,
    stop_requested: StopCallback | None = None,
    directory_workers: int = DEFAULT_DIRECTORY_WORKERS,
    progress: ProgressCallback | None = None,
) -> ScanInventory:
    """Build a metadata-only inventory of supported files below *roots*.

    Directory symlinks, junctions, and other name-surrogate reparse points are not
    followed. File symlinks are skipped as well, which prevents later content reads
    from escaping an investigation root. Ordinary private-cloud placeholder files
    are still included: a reparse point is skipped only when it is a link/name
    surrogate rather than merely carrying provider metadata.

    ``directory_workers`` bounds concurrent ``os.scandir`` calls. A value of one is
    useful for providers that serialize requests or perform worse under concurrency.
    Cancellation is checked between directory entries and while waiting for slow
    directory operations. An operating-system call already blocked in a provider
    cannot be interrupted by Python.
    """

    if isinstance(directory_workers, bool) or directory_workers < 1:
        raise ValueError("directory_workers must be at least 1")

    should_stop = stop_requested or (lambda: False)
    counters = _CounterState()
    errors: list[InventoryError] = []
    cancel_event = Event()

    requested_paths = [lexical_absolute_path(root) for root in roots]
    counters.roots_requested = len(requested_paths)
    root_states = _prepare_unique_roots(requested_paths, counters)

    for state in root_states:
        if should_stop():
            cancel_event.set()
            break
        try:
            root_stat = os.lstat(state.path)
            counters.metadata_reads += 1
        except OSError as exc:
            error = _inventory_error(state.path, state.path, "root_metadata", exc)
            state.errors.append(error)
            errors.append(error)
            counters.errors += 1
            state.exists = False if isinstance(exc, FileNotFoundError) else None
            continue

        state.exists = True
        state.stat_result = root_stat
        if stat.S_ISLNK(root_stat.st_mode) or _is_name_surrogate(root_stat):
            state.kind = "link"
            counters.skipped_links += 1
            error = InventoryError(
                root=state.path,
                path=state.path,
                operation="unsafe_root_link",
                message="Root is a symlink, junction, or name-surrogate reparse point.",
            )
            state.errors.append(error)
            errors.append(error)
            counters.errors += 1
            continue
        if stat.S_ISDIR(root_stat.st_mode):
            state.kind = "directory"
            state.safe = True
        elif stat.S_ISREG(root_stat.st_mode):
            state.kind = "file"
            state.safe = True
        else:
            state.kind = "special"
            state.safe = True
            counters.special_entries += 1

    _mark_covered_roots(root_states, counters)

    files_by_path: dict[str, ScannedFile] = {}
    directory_queue: deque[_DirectoryJob] = deque()
    known_directories: set[str] = set()

    for state in root_states:
        if cancel_event.is_set() or not state.safe or state.covered_by is not None:
            continue
        counters.roots_scanned += 1
        if state.kind == "directory":
            known_directories.add(state.normalized_path)
            root_device = int(getattr(state.stat_result, "st_dev", 0) or 0)
            directory_queue.append(
                _DirectoryJob(path=state.path, root=state.path, root_device=root_device)
            )
        elif state.kind == "file":
            if state.stat_result is not None:
                scanned = _scanned_file_from_stat(state.path, state.stat_result)
                if scanned is not None:
                    _add_file(scanned, files_by_path, counters)

    if progress is not None:
        progress(InventoryProgress(counters.snapshot()))

    executor = ThreadPoolExecutor(
        max_workers=directory_workers,
        thread_name_prefix="inventory-scandir",
    )
    futures: dict[Future[_DirectoryBatch], _DirectoryJob] = {}
    try:
        while directory_queue or futures:
            if not cancel_event.is_set() and should_stop():
                cancel_event.set()

            if cancel_event.is_set():
                directory_queue.clear()
                for future in futures:
                    future.cancel()
            else:
                while directory_queue and len(futures) < directory_workers:
                    job = directory_queue.popleft()
                    future = executor.submit(_scan_directory, job, cancel_event)
                    futures[future] = job

            if not futures:
                break

            completed, _ = wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
            if not completed:
                continue

            for future in completed:
                job = futures.pop(future)
                try:
                    batch = future.result()
                except CancelledError:
                    continue
                _merge_batch(batch, counters, errors, files_by_path)
                if batch.cancelled:
                    cancel_event.set()
                if not cancel_event.is_set():
                    for child in batch.child_directories:
                        child_key = normalize_inventory_path(child)
                        if child_key in known_directories:
                            counters.duplicate_paths += 1
                            continue
                        known_directories.add(child_key)
                        directory_queue.append(
                            _DirectoryJob(
                                path=child,
                                root=job.root,
                                root_device=job.root_device,
                            )
                        )
                if progress is not None:
                    progress(InventoryProgress(counters.snapshot(), current_root=job.root))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not cancel_event.is_set() and should_stop():
        cancel_event.set()

    sorted_files = tuple(sorted(files_by_path.values(), key=lambda item: item.normalized_path))
    root_results = _build_root_results(root_states, sorted_files, errors, cancel_event.is_set())
    return ScanInventory(
        files=sorted_files,
        root_results=root_results,
        errors=tuple(errors),
        counters=counters.snapshot(),
        cancelled=cancel_event.is_set(),
    )


def iter_supported_files(
    roots: Iterable[str | Path],
    *,
    stop_requested: StopCallback | None = None,
    directory_workers: int = DEFAULT_DIRECTORY_WORKERS,
) -> Iterator[ScannedFile]:
    """Compatibility iterator; new callers should consume :func:`scan_supported_files`."""

    inventory = scan_supported_files(
        roots,
        stop_requested=stop_requested,
        directory_workers=directory_workers,
    )
    yield from inventory.files


def _prepare_unique_roots(paths: list[Path], counters: _CounterState) -> list[_RootState]:
    unique: list[_RootState] = []
    seen: set[str] = set()
    for path in paths:
        normalized = normalize_inventory_path(path)
        if normalized in seen:
            counters.roots_deduplicated += 1
            continue
        seen.add(normalized)
        unique.append(_RootState(path=path, normalized_path=normalized))
    return unique


def _mark_covered_roots(states: list[_RootState], counters: _CounterState) -> None:
    safe_directories = sorted(
        (state for state in states if state.safe and state.kind == "directory"),
        key=lambda state: (len(state.path.parts), state.normalized_path),
    )
    effective_directories: list[_RootState] = []
    for state in safe_directories:
        covering = next(
            (
                candidate
                for candidate in effective_directories
                if _path_is_within(state.normalized_path, candidate.normalized_path)
            ),
            None,
        )
        if covering is None:
            effective_directories.append(state)
            continue
        state.covered_by = covering.path
        counters.roots_deduplicated += 1

    for state in states:
        if not state.safe or state.kind == "directory" or state.covered_by is not None:
            continue
        covering = next(
            (
                candidate
                for candidate in effective_directories
                if _path_is_within(state.normalized_path, candidate.normalized_path)
            ),
            None,
        )
        if covering is not None:
            state.covered_by = covering.path
            counters.roots_deduplicated += 1


def _scan_directory(job: _DirectoryJob, cancel_event: Event) -> _DirectoryBatch:
    batch = _DirectoryBatch(job=job)
    try:
        with os.scandir(job.path) as entries:
            batch.directories_scanned = 1
            for entry in entries:
                if cancel_event.is_set():
                    batch.cancelled = True
                    break
                batch.entries_seen += 1
                try:
                    batch.type_checks += 1
                    if entry.is_symlink():
                        batch.skipped_links += 1
                        continue

                    if entry.is_dir(follow_symlinks=False):
                        if _directory_entry_is_name_surrogate(entry, batch):
                            batch.skipped_links += 1
                        else:
                            batch.child_directories.append(lexical_absolute_path(entry.path))
                        continue

                    suffix = os.path.splitext(entry.name)[1].lower()
                    if suffix not in SUPPORTED_EXTENSIONS:
                        batch.unsupported_entries += 1
                        continue

                    if not entry.is_file(follow_symlinks=False):
                        batch.special_entries += 1
                        continue

                    entry_stat = entry.stat(follow_symlinks=False)
                    batch.metadata_reads += 1
                    batch.identity_reads += 1
                    stable_id = _stable_id_from_entry(
                        entry,
                        entry_stat,
                        root_device=job.root_device,
                    )
                    scanned = _scanned_file_from_stat(
                        Path(entry.path),
                        entry_stat,
                        suffix=suffix,
                        stable_id=stable_id,
                    )
                    if scanned is not None:
                        batch.files.append(scanned)
                except OSError as exc:
                    batch.errors.append(
                        _inventory_error(job.root, Path(entry.path), "entry_metadata", exc)
                    )
    except OSError as exc:
        batch.errors.append(_inventory_error(job.root, job.path, "scandir", exc))
    return batch


def _directory_entry_is_name_surrogate(
    entry: os.DirEntry[str], batch: _DirectoryBatch
) -> bool:
    is_junction = getattr(entry, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    if os.name != "nt":
        return False
    entry_stat = entry.stat(follow_symlinks=False)
    batch.metadata_reads += 1
    return _is_name_surrogate(entry_stat)


def _is_name_surrogate(stat_result: os.stat_result) -> bool:
    reparse_tag = int(getattr(stat_result, "st_reparse_tag", 0) or 0)
    return bool(reparse_tag & _NAME_SURROGATE_REPARSE_BIT)


def _scanned_file_from_stat(
    path: Path,
    stat_result: os.stat_result,
    *,
    suffix: str | None = None,
    stable_id: str | None = None,
) -> ScannedFile | None:
    suffix = suffix if suffix is not None else path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS or not stat.S_ISREG(stat_result.st_mode):
        return None
    if stable_id is None:
        inode = int(getattr(stat_result, "st_ino", 0) or 0)
        device = int(getattr(stat_result, "st_dev", 0) or 0)
        stable_id = f"{device}:{inode}" if inode and device else None
    absolute_path = lexical_absolute_path(path)
    modified_ns = getattr(stat_result, "st_mtime_ns", None)
    if modified_ns is None:
        modified_ns = int(stat_result.st_mtime * 1_000_000_000)
    return ScannedFile(
        path=absolute_path,
        normalized_path=normalize_inventory_path(absolute_path),
        suffix=suffix,
        stable_id=stable_id,
        size=int(stat_result.st_size),
        modified_ns=int(modified_ns),
    )


def _add_file(
    scanned: ScannedFile,
    files_by_path: dict[str, ScannedFile],
    counters: _CounterState,
) -> None:
    if scanned.normalized_path in files_by_path:
        counters.duplicate_paths += 1
        return
    files_by_path[scanned.normalized_path] = scanned
    counters.supported_files += 1
    counters.supported_bytes += scanned.size


def _merge_batch(
    batch: _DirectoryBatch,
    counters: _CounterState,
    errors: list[InventoryError],
    files_by_path: dict[str, ScannedFile],
) -> None:
    counters.directories_scanned += batch.directories_scanned
    counters.entries_seen += batch.entries_seen
    counters.type_checks += batch.type_checks
    counters.metadata_reads += batch.metadata_reads
    counters.identity_reads += batch.identity_reads
    counters.unsupported_entries += batch.unsupported_entries
    counters.special_entries += batch.special_entries
    counters.skipped_links += batch.skipped_links
    counters.errors += len(batch.errors)
    errors.extend(batch.errors)
    for scanned in batch.files:
        _add_file(scanned, files_by_path, counters)


def _build_root_results(
    states: list[_RootState],
    files: tuple[ScannedFile, ...],
    errors: list[InventoryError],
    cancelled: bool,
) -> tuple[RootScanResult, ...]:
    results: list[RootScanResult] = []
    for state in states:
        relevant_files = [
            scanned
            for scanned in files
            if _path_is_within(scanned.normalized_path, state.normalized_path)
        ]
        relevant_errors = list(state.errors)
        for error in errors:
            if error in relevant_errors:
                continue
            error_path = normalize_inventory_path(error.path)
            error_is_below_root = _path_is_within(error_path, state.normalized_path)
            root_is_below_failed_directory = (
                error.operation == "scandir"
                and _path_is_within(state.normalized_path, error_path)
            )
            if error_is_below_root or root_is_below_failed_directory:
                relevant_errors.append(error)
        complete = state.safe and state.exists is True and not cancelled and not relevant_errors
        results.append(
            RootScanResult(
                root=state.path,
                normalized_root=state.normalized_path,
                exists=state.exists,
                complete=complete,
                covered_by=state.covered_by,
                supported_files=len(relevant_files),
                supported_bytes=sum(scanned.size for scanned in relevant_files),
                errors=tuple(relevant_errors),
            )
        )
    return tuple(results)


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _inventory_error(
    root: Path, path: Path, operation: str, exc: OSError
) -> InventoryError:
    return InventoryError(
        root=root,
        path=lexical_absolute_path(path),
        operation=operation,
        message=str(exc),
        errno=exc.errno,
    )


def _stable_id_from_entry(
    entry: os.DirEntry[str],
    stat_result: os.stat_result,
    *,
    root_device: int,
) -> str | None:
    """Use cached directory-enumeration identity where the platform exposes it."""

    inode = int(getattr(stat_result, "st_ino", 0) or 0)
    if not inode:
        try:
            inode = int(entry.inode() or 0)
        except OSError:
            return None
    device = int(getattr(stat_result, "st_dev", 0) or 0) or root_device
    return f"{device}:{inode}" if device and inode else None
