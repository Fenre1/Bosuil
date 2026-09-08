from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB


WORKSPACE_INVESTIGATION_ID = 1
WORKSPACE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class WorkspaceEntry:
    id: str
    name: str
    workspace_path: Path


@dataclass(frozen=True)
class InvestigationWorkspace:
    entry: WorkspaceEntry
    db: SQLiteDB
    store: LanceDBStore

    @property
    def investigation_id(self) -> int:
        return WORKSPACE_INVESTIGATION_ID


class WorkspaceCatalog:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.path = self.data_dir / "catalog.sqlite3"
        self.workspaces_dir = self.data_dir / "investigations"

    def initialize(self) -> None:
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces(
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    workspace_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        self.discover()

    def discover(self) -> None:
        """Register self-describing workspaces copied into the investigations directory."""
        for manifest_path in self.workspaces_dir.glob("*/manifest.json"):
            workspace_path = manifest_path.parent.resolve()
            if not (workspace_path / "investigation.sqlite3").is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                workspace_id = str(manifest["id"])
                name = str(manifest["name"])
                created_at = str(manifest.get("created_at") or _utc_now())
                updated_at = str(manifest.get("updated_at") or created_at)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            if workspace_path.name != workspace_id:
                continue
            try:
                with self._connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO workspaces(
                            id, name, workspace_path, created_at, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            name = excluded.name,
                            workspace_path = excluded.workspace_path,
                            updated_at = excluded.updated_at
                        """,
                        (workspace_id, name, str(workspace_path), created_at, updated_at),
                    )
            except sqlite3.IntegrityError:
                # A separately created workspace may already own the same name.
                # Leave that conflict for an explicit import instead of making
                # every otherwise valid workspace undiscoverable.
                continue

    def list(self) -> list[WorkspaceEntry]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, name, workspace_path FROM workspaces ORDER BY name"
            ).fetchall()
        return [
            WorkspaceEntry(str(row["id"]), str(row["name"]), Path(str(row["workspace_path"])))
            for row in rows
        ]

    def get(self, workspace_id: str) -> WorkspaceEntry | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, name, workspace_path FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkspaceEntry(str(row["id"]), str(row["name"]), Path(str(row["workspace_path"])))

    def get_by_name(self, name: str) -> WorkspaceEntry | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, name, workspace_path FROM workspaces WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return WorkspaceEntry(str(row["id"]), str(row["name"]), Path(str(row["workspace_path"])))

    def create(self, name: str, roots: list[Path], notes: str = "") -> InvestigationWorkspace:
        if self.get_by_name(name) is not None:
            raise ValueError(f"An investigation named {name!r} already exists.")
        workspace_id = str(uuid.uuid4())
        workspace_path = (self.workspaces_dir / workspace_id).resolve()
        workspace_path.mkdir(parents=True, exist_ok=False)
        db = SQLiteDB(workspace_path / "investigation.sqlite3")
        db.initialize()
        investigation_id = db.create_investigation(name, roots, notes)
        if investigation_id != WORKSPACE_INVESTIGATION_ID:
            raise RuntimeError("A new investigation workspace must contain investigation ID 1.")
        now = _utc_now()
        manifest = {
            "id": workspace_id,
            "name": name,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
        }
        (workspace_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO workspaces(id, name, workspace_path, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (workspace_id, name, str(workspace_path), now, now),
                )
        except Exception:
            # The workspace remains self-describing and can be registered manually.
            raise
        return self.open(WorkspaceEntry(workspace_id, name, workspace_path))

    def update(self, entry: WorkspaceEntry, name: str, notes: str, roots: list[Path]) -> WorkspaceEntry:
        existing = self.get_by_name(name)
        if existing is not None and existing.id != entry.id:
            raise ValueError(f"An investigation named {name!r} already exists.")
        workspace = self.open(entry)
        workspace.db.update_investigation(WORKSPACE_INVESTIGATION_ID, name, notes, roots)
        now = _utc_now()
        manifest_path = entry.workspace_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"name": name, "updated_at": now})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with self._connection() as conn:
            conn.execute(
                "UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, entry.id),
            )
        return WorkspaceEntry(entry.id, name, entry.workspace_path)

    def open(self, entry: WorkspaceEntry) -> InvestigationWorkspace:
        database_path = entry.workspace_path / "investigation.sqlite3"
        if not database_path.is_file():
            raise FileNotFoundError(
                "Investigation database not found at the catalogued workspace path: "
                f"{database_path}. Refresh the catalog or import the workspace again."
            )
        db = SQLiteDB(database_path)
        db.initialize()
        if db.get_investigation(WORKSPACE_INVESTIGATION_ID) is None:
            raise RuntimeError(
                f"Investigation database {database_path} does not contain investigation ID "
                f"{WORKSPACE_INVESTIGATION_ID}."
            )
        return InvestigationWorkspace(entry, db, LanceDBStore(entry.workspace_path / "lancedb"))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
