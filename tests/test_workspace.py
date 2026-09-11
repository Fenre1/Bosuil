import json
import shutil

import pytest

from investigation_rag.workspace import WORKSPACE_INVESTIGATION_ID, WorkspaceCatalog


def test_catalog_creates_isolated_investigation_workspaces(tmp_path):
    catalog = WorkspaceCatalog(tmp_path / "data")
    catalog.initialize()

    alpha = catalog.create("Alpha", [tmp_path / "source-alpha"], "alpha notes")
    beta = catalog.create("Beta", [tmp_path / "source-beta"], "beta notes")

    assert alpha.entry.id != beta.entry.id
    assert alpha.entry.workspace_path.parent == catalog.workspaces_dir
    assert alpha.db.path == alpha.entry.workspace_path / "investigation.sqlite3"
    assert alpha.store.db_dir == alpha.entry.workspace_path / "lancedb"
    assert alpha.investigation_id == WORKSPACE_INVESTIGATION_ID
    assert beta.investigation_id == WORKSPACE_INVESTIGATION_ID
    assert alpha.db.get_investigation(WORKSPACE_INVESTIGATION_ID)["name"] == "Alpha"
    assert beta.db.get_investigation(WORKSPACE_INVESTIGATION_ID)["name"] == "Beta"


def test_catalog_discovers_a_copied_workspace(tmp_path):
    source_catalog = WorkspaceCatalog(tmp_path / "shared")
    source_catalog.initialize()
    workspace = source_catalog.create("Shared case", [tmp_path / "evidence"])

    # Simulate a fresh local catalog alongside an already copied workspace.
    source_catalog.path.unlink()
    fresh_catalog = WorkspaceCatalog(tmp_path / "shared")
    fresh_catalog.initialize()

    entries = fresh_catalog.list()
    assert [(entry.id, entry.name) for entry in entries] == [
        (workspace.entry.id, "Shared case")
    ]
    manifest = json.loads(
        (workspace.entry.workspace_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["id"] == workspace.entry.id


def test_catalog_repairs_absolute_paths_after_data_directory_move(tmp_path):
    original_dir = tmp_path / "original"
    original = WorkspaceCatalog(original_dir)
    original.initialize()
    workspace = original.create("Moved case", [tmp_path / "unavailable-evidence"])

    moved_dir = tmp_path / "moved"
    shutil.copytree(original_dir, moved_dir)
    moved = WorkspaceCatalog(moved_dir)
    moved.initialize()

    entry = moved.get(workspace.entry.id)
    assert entry is not None
    assert entry.workspace_path == (moved.workspaces_dir / workspace.entry.id).resolve()
    assert moved.open(entry).db.get_investigation(WORKSPACE_INVESTIGATION_ID)["name"] == "Moved case"


def test_open_does_not_silently_create_a_database_at_a_stale_path(tmp_path):
    catalog = WorkspaceCatalog(tmp_path / "data")
    catalog.initialize()
    workspace = catalog.create("Case", [tmp_path / "evidence"])
    workspace.db.path.unlink()

    with pytest.raises(FileNotFoundError, match="Investigation database not found"):
        catalog.open(workspace.entry)

    assert not workspace.db.path.exists()
