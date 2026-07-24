import json

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
