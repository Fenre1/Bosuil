from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.ingest.hashing import sha256_text
from investigation_rag.ingest.searchable_text import stable_embedding_text
from investigation_rag.workspace import (
    WORKSPACE_INVESTIGATION_ID,
    InvestigationWorkspace,
    WorkspaceCatalog,
    WorkspaceEntry,
)

if TYPE_CHECKING:
    from sqlite3 import Row


ARCHIVE_FORMAT = "bosuil-investigation"
ARCHIVE_FORMAT_VERSION = 1
ARCHIVE_EXTENSION = ".bosuil"
ARCHIVE_METADATA = "export.json"
ARCHIVE_MANIFEST = "manifest.json"
ARCHIVE_DATABASE = "investigation.sqlite3"
_ARCHIVE_FILES = frozenset({ARCHIVE_METADATA, ARCHIVE_MANIFEST, ARCHIVE_DATABASE})
_VECTOR_BATCH_SIZE = 250


@dataclass(frozen=True)
class WorkspaceExportResult:
    path: Path
    workspace_id: str
    database_bytes: int


@dataclass(frozen=True)
class VectorRebuildResult:
    active_chunks: int
    restored_vectors: int
    missing_vectors: int
    embedding_model: str
    backup_path: Path | None = None


@dataclass(frozen=True)
class WorkspaceImportResult:
    workspace: InvestigationWorkspace
    vector_rebuild: VectorRebuildResult | None = None
    vector_warning: str = ""


def export_workspace(entry: WorkspaceEntry, destination: Path) -> WorkspaceExportResult:
    """Create a portable archive whose SQLite snapshot is the source of truth."""

    destination = Path(destination).expanduser()
    if destination.suffix.casefold() != ARCHIVE_EXTENSION:
        destination = destination.with_suffix(ARCHIVE_EXTENSION)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = entry.workspace_path / ARCHIVE_MANIFEST
    database_path = entry.workspace_path / ARCHIVE_DATABASE
    if not manifest_path.is_file() or not database_path.is_file():
        raise FileNotFoundError(
            "A portable workspace requires manifest.json and investigation.sqlite3."
        )

    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_bytes, ARCHIVE_MANIFEST)
    if str(manifest.get("id", "")) != entry.id:
        raise ValueError("Workspace manifest ID does not match the catalog entry.")

    temp_dir = (destination.parent / f".bosuil-export-{uuid.uuid4()}").resolve()
    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        snapshot_path = temp_dir / ARCHIVE_DATABASE
        from investigation_rag.db.sqlite_db import SQLiteDB

        SQLiteDB(database_path).backup_to(snapshot_path)
        _validate_investigation_database(snapshot_path, expected_name=str(manifest["name"]))
        database_size = snapshot_path.stat().st_size
        metadata = {
            "format": ARCHIVE_FORMAT,
            "format_version": ARCHIVE_FORMAT_VERSION,
            "workspace_id": entry.id,
            "created_at": _utc_now(),
            "vector_store": "rebuild-from-sqlite-embedding-cache",
            "files": {
                ARCHIVE_MANIFEST: _sha256(manifest_bytes),
                ARCHIVE_DATABASE: _sha256_file(snapshot_path),
            },
        }
        archive_path = temp_dir / f"export{ARCHIVE_EXTENSION}"
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(ARCHIVE_METADATA, json.dumps(metadata, indent=2) + "\n")
            archive.writestr(ARCHIVE_MANIFEST, manifest_bytes)
            archive.write(snapshot_path, ARCHIVE_DATABASE)
        os.replace(archive_path, destination)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return WorkspaceExportResult(destination, entry.id, database_size)


def import_workspace(
    catalog: WorkspaceCatalog,
    archive_path: Path,
    *,
    preferred_embedding_model: str = "",
    rebuild_vectors: bool = True,
) -> WorkspaceImportResult:
    """Validate and import a portable archive, then recreate derived vectors."""

    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Investigation archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path, mode="r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Investigation archive contains duplicate file names.")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {name}")
        missing = _ARCHIVE_FILES - set(names)
        if missing:
            raise ValueError(
                "Investigation archive is incomplete; missing: " + ", ".join(sorted(missing))
            )
        metadata_bytes = archive.read(ARCHIVE_METADATA)
        manifest_bytes = archive.read(ARCHIVE_MANIFEST)

    metadata = _load_json_object(metadata_bytes, ARCHIVE_METADATA)
    manifest = _load_json_object(manifest_bytes, ARCHIVE_MANIFEST)
    if metadata.get("format") != ARCHIVE_FORMAT:
        raise ValueError("This is not a Bosuil investigation archive.")
    if int(metadata.get("format_version", 0)) != ARCHIVE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported investigation archive version: {metadata.get('format_version')!r}."
        )

    workspace_id = str(manifest.get("id", ""))
    try:
        parsed_id = uuid.UUID(workspace_id)
    except ValueError as exc:
        raise ValueError("Workspace manifest contains an invalid UUID.") from exc
    if str(parsed_id) != workspace_id.casefold():
        raise ValueError("Workspace manifest UUID is not in canonical form.")
    if str(metadata.get("workspace_id", "")) != workspace_id:
        raise ValueError("Archive metadata and workspace manifest IDs do not match.")
    name = str(manifest.get("name", "")).strip()
    if not name:
        raise ValueError("Workspace manifest does not contain a name.")

    checksums = metadata.get("files")
    if not isinstance(checksums, dict):
        raise ValueError("Investigation archive does not contain file checksums.")
    if str(checksums.get(ARCHIVE_MANIFEST, "")) != _sha256(manifest_bytes):
        raise ValueError(f"Checksum verification failed for {ARCHIVE_MANIFEST}.")

    destination = (catalog.workspaces_dir / workspace_id).resolve()
    if destination.exists():
        raise FileExistsError(f"Investigation workspace already exists: {destination}")
    existing = catalog.get(workspace_id)
    if existing is not None and existing.workspace_path.exists():
        raise FileExistsError(f"Investigation {workspace_id} is already registered.")
    same_name = catalog.get_by_name(name)
    if same_name is not None and same_name.id != workspace_id:
        raise ValueError(f"An investigation named {name!r} already exists.")

    staging = (catalog.workspaces_dir / f".import-{uuid.uuid4()}").resolve()
    staging.mkdir(parents=False, exist_ok=False)
    try:
        (staging / ARCHIVE_MANIFEST).write_bytes(manifest_bytes)
        database_path = staging / ARCHIVE_DATABASE
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            with archive.open(ARCHIVE_DATABASE, mode="r") as source:
                with database_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        if str(checksums.get(ARCHIVE_DATABASE, "")) != _sha256_file(database_path):
            raise ValueError(f"Checksum verification failed for {ARCHIVE_DATABASE}.")
        _validate_investigation_database(database_path, expected_name=name)
        staging.rename(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    catalog.discover()
    entry = catalog.get(workspace_id)
    if entry is None:
        raise RuntimeError("Imported workspace could not be registered in the local catalog.")
    workspace = catalog.open(entry)

    vector_result: VectorRebuildResult | None = None
    vector_warning = ""
    if rebuild_vectors:
        try:
            vector_result = rebuild_workspace_vectors(
                workspace,
                preferred_embedding_model=preferred_embedding_model,
            )
        except Exception as exc:
            vector_warning = str(exc)
    return WorkspaceImportResult(workspace, vector_result, vector_warning)


def rebuild_workspace_vectors(
    workspace: InvestigationWorkspace,
    *,
    preferred_embedding_model: str = "",
) -> VectorRebuildResult:
    """Replace a workspace's LanceDB with vectors reconstructed from SQLite."""

    rows = workspace.db.active_chunks_for_investigation(workspace.investigation_id)
    active_chunks = len(rows)
    model_counts = Counter(
        str(row["embedding_model"]).strip()
        for row in rows
        if str(row["embedding_model"]).strip()
    )
    preferred = preferred_embedding_model.strip()
    if preferred and preferred in model_counts:
        embedding_model = preferred
    elif model_counts:
        embedding_model = model_counts.most_common(1)[0][0]
    else:
        embedding_model = preferred

    if active_chunks and not embedding_model:
        raise ValueError(
            "SQLite contains active chunks but no embedding model metadata. "
            "Configure OLLAMA_EMBED_MODEL and re-embed the stored chunks."
        )

    workspace_path = workspace.entry.workspace_path.resolve()
    current_store_path = (workspace_path / "lancedb").resolve()
    staging_path = (workspace_path / f".lancedb-rebuild-{uuid.uuid4()}").resolve()
    staging_path.mkdir(parents=False, exist_ok=False)
    fresh_store = LanceDBStore(staging_path)
    restored = 0
    vector_size: int | None = None
    try:
        for start in range(0, len(rows), _VECTOR_BATCH_SIZE):
            batch = rows[start : start + _VECTOR_BATCH_SIZE]
            contextual_hashes: list[str] = []
            raw_hashes: list[str] = []
            for row in batch:
                metadata = _chunk_metadata(row)
                embedding_text = stable_embedding_text(str(row["chunk_text"]), metadata)
                contextual_hashes.append(sha256_text(embedding_text))
                raw_hashes.append(str(row["chunk_text_hash"]))
            cached = workspace.db.get_embeddings(
                [*contextual_hashes, *raw_hashes], embedding_model
            )
            records: list[dict[str, object]] = []
            for row, contextual_hash, raw_hash in zip(batch, contextual_hashes, raw_hashes):
                vector = cached.get(contextual_hash) or cached.get(raw_hash)
                if vector is None:
                    continue
                if vector_size is None:
                    vector_size = len(vector)
                elif len(vector) != vector_size:
                    raise ValueError(
                        "Cached embeddings have inconsistent dimensions; the vector index "
                        "cannot be reconstructed safely."
                    )
                records.append(
                    {
                        "chunk_id": str(row["id"]),
                        "investigation_id": int(row["investigation_id"]),
                        "file_id": int(row["file_id"]),
                        "path": str(row["path"]),
                        "source_type": str(row["source_type"]),
                        "is_active": True,
                        "embedding_model": embedding_model,
                        "vector": vector,
                    }
                )
            if records:
                fresh_store.upsert_chunks(records)
                restored += len(records)

        if active_chunks and restored == 0:
            raise ValueError(
                f"No cached {embedding_model!r} embeddings matched the {active_chunks} active "
                "SQLite chunks. The chunks remain available for keyword search, but Ollama "
                "must re-embed them before semantic search can be restored."
            )

        backup_path: Path | None = None
        if current_store_path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = (
                workspace_path / f"lancedb.backup-{stamp}-{uuid.uuid4().hex[:8]}"
            ).resolve()
            current_store_path.rename(backup_path)
        try:
            staging_path.rename(current_store_path)
        except Exception:
            if backup_path is not None and backup_path.exists() and not current_store_path.exists():
                backup_path.rename(current_store_path)
            raise
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise

    return VectorRebuildResult(
        active_chunks=active_chunks,
        restored_vectors=restored,
        missing_vectors=active_chunks - restored,
        embedding_model=embedding_model,
        backup_path=backup_path,
    )


def _validate_investigation_database(database_path: Path, expected_name: str) -> None:
    connection = sqlite3.connect(database_path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]).casefold() != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity!r}")
        row = connection.execute(
            "SELECT name FROM investigations WHERE id = ?",
            (WORKSPACE_INVESTIGATION_ID,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Investigation archive does not contain investigation ID "
                f"{WORKSPACE_INVESTIGATION_ID}."
            )
        if str(row[0]) != expected_name:
            raise ValueError("Workspace manifest and SQLite investigation names do not match.")
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid investigation SQLite database: {exc}") from exc
    finally:
        connection.close()


def _chunk_metadata(row: Row) -> dict[str, object]:
    raw = row["metadata_json"]
    try:
        parsed = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_json_object(content: bytes, filename: str) -> dict[str, object]:
    try:
        parsed = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {filename}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{filename} must contain a JSON object.")
    return parsed


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
