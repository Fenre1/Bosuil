from copy import deepcopy
from threading import Event

import pytest

from test_knowledge_graph import Model, corpus
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.knowledge_graph import KGCancelled, KGSettings, KGStore, build_graph


def seed_export(corpus, *, legacy=False):
    store, inv, file_id = corpus
    build_graph(store, inv, [file_id], KGSettings(), Model(), Event(), lambda msg: None)
    claim = store.claims(inv)[0]
    store.review(inv, claim["id"], "accepted", "Reviewed original source")
    payload = store.export_graph(inv)
    if legacy:
        payload["format"] = "bosuil-knowledge-graph-v1"
        payload.pop("graph_id")
        for node in payload["nodes"]:
            node.pop("portable_ids")
        for claim in payload["claims"]:
            claim.pop("portable_id")
    return payload


def empty_store(path):
    db = SQLiteDB(path / "target.sqlite3")
    db.initialize()
    return KGStore(db), db.create_investigation("Imported case", [], "")


@pytest.mark.parametrize("legacy", [True, False])
def test_import_without_sources_preserves_identity_evidence_and_review(corpus, tmp_path, legacy):
    payload = seed_export(corpus, legacy=legacy)
    target, inv = empty_store(tmp_path / "target")
    result = target.import_graph(inv, payload)
    assert result["nodes"] == 2 and result["claims"] == 1 and result["unlinked"] == 1
    claim = target.claims(inv)[0]
    assert claim["id"] < 0 and claim["imported"] == 1 and claim["source_current"] == 0
    for key in ("quote", "polarity", "attribution", "time_text", "attribute", "source_hash", "model", "verification", "review_status", "review_note"):
        assert claim[key] == payload["claims"][0][key]
    with target.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM kg_batches").fetchone()[0] == 0
    assert target.import_graph(inv, payload)["existing"] == 1
    assert len(target.claims(inv)) == 1 and len(target.nodes(inv)) == 2
    third, third_inv = empty_store(tmp_path / "third")
    third.import_graph(third_inv, target.export_graph(inv))
    assert third.claims(third_inv)[0]["quote"] == claim["quote"]
    assert third.claims(third_inv)[0]["portable_key"] == claim["portable_key"]
    assert third.import_graph(third_inv, payload)["existing"] == 1
    assert len(third.claims(third_inv)) == 1 and len(third.nodes(third_inv)) == 2


@pytest.mark.parametrize("legacy", [True, False])
def test_import_into_original_graph_never_duplicates_or_overwrites_edits(corpus, legacy):
    payload = seed_export(corpus, legacy=legacy)
    store, inv, _ = corpus
    claim = store.claims(inv)[0]
    store.edit_claim(inv, claim["id"], "contacted", "uncertain", "", "", "Latest correction")
    store.edit_node(inv, claim["subject_id"], "Alice Smith", "person")
    store.review(inv, claim["id"], "rejected", "Local review takes precedence")
    result = store.import_graph(inv, payload)
    assert result["nodes"] == result["claims"] == 0 and result["existing"] == 1
    assert len(store.claims(inv)) == 1 and len(store.nodes(inv)) == 2
    current = store.claims(inv)[0]
    assert current["id"] == claim["id"] and current["subject"] == "Alice Smith"
    assert current["predicate"] == "contacted" and current["review_status"] == "rejected"
    assert current["review_note"] == "Local review takes precedence"


def test_import_remaps_ids_and_preserves_namesakes_and_isolated_nodes(corpus, tmp_path):
    payload = seed_export(corpus)
    payload["nodes"].append({"id": 999, "name": "Alice", "node_type": "person"})
    target, inv = empty_store(tmp_path / "target")
    with target.db.transaction() as conn:
        conn.execute("INSERT INTO kg_nodes(investigation_id,name,node_type,created_at) VALUES(?,'Unrelated','person','now')", (inv,))
    target.import_graph(inv, payload)
    claim = target.claims(inv)[0]
    assert claim["subject_id"] != payload["claims"][0]["subject_id"]
    assert len([n for n in target.nodes(inv) if n["name"] == "Alice"]) == 2
    assert len(target.orphan_nodes(inv)) == 2  # Unrelated and the isolated exported namesake.


def test_reimport_preserves_imported_corrections_reviews_and_merges(corpus, tmp_path):
    payload = seed_export(corpus)
    target, inv = empty_store(tmp_path / "target")
    target.import_graph(inv, payload)
    claim = target.claims(inv)[0]
    target.edit_claim(inv, claim["id"], "contacted", "uncertain", "", "", "Changed locally", "")
    target.edit_node(inv, claim["subject_id"], "Alice Smith", "person")
    target.review(inv, claim["id"], "rejected", "Keep this review")
    target.merge_nodes(inv, claim["object_id"], claim["subject_id"])
    target.import_graph(inv, payload)
    current = target.claims(inv)[0]
    assert current["subject"] == current["object"] == "Alice Smith"
    assert current["predicate"] == "contacted" and current["review_status"] == "rejected"
    assert current["review_note"] == "Keep this review" and len(target.nodes(inv)) == 1
    third, third_inv = empty_store(tmp_path / "third")
    third.import_graph(third_inv, target.export_graph(inv))
    assert len(third.nodes(third_inv)) == 1 and third.claims(third_inv)[0]["subject_id"] == third.claims(third_inv)[0]["object_id"]


def test_import_links_matching_passages_even_with_new_chunk_ids(corpus, tmp_path):
    payload = seed_export(corpus)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target, inv, _ = corpus_fixture(target_dir)
    with target.db.transaction() as conn:
        conn.execute("UPDATE chunks SET id='new-c0' WHERE id='c0'")
    result = target.import_graph(inv, payload)
    current = target.claims(inv)[0]
    assert result["linked"] == 1 and current["source_current"] == 1
    assert current["chunk_id"] == "new-c0" and current["imported"] == 1
    with target.db.transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE id='new-c0'")
    current = target.claims(inv)[0]
    assert current["source_current"] == 0 and current["quote"] == payload["claims"][0]["quote"]


def test_wrong_hash_or_quote_does_not_claim_local_evidence_is_current(corpus, tmp_path):
    payload = seed_export(corpus)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target, inv, _ = corpus_fixture(target_dir)
    with target.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text='Different text with reused ID.',chunk_text_hash='different' WHERE id='c0'")
    result = target.import_graph(inv, payload)
    assert result["unlinked"] == 1 and target.claims(inv)[0]["source_current"] == 0


def test_reimport_attaches_newly_available_evidence_without_overwriting_review(corpus, tmp_path):
    payload = seed_export(corpus)
    target, inv = empty_store(tmp_path / "target")
    target.import_graph(inv, payload)
    current = target.claims(inv)[0]
    target.review(inv, current["id"], "rejected", "Retain note")
    source, _, _ = corpus
    with source.db.connection() as conn:
        file = dict(conn.execute("SELECT * FROM files LIMIT 1").fetchone())
        chunk = dict(conn.execute("SELECT * FROM chunks WHERE id='c0'").fetchone())
    with target.db.transaction() as conn:
        file["investigation_id"], chunk["investigation_id"] = inv, inv
        for table, row in (("files", file), ("chunks", chunk)):
            conn.execute(f"INSERT INTO {table}(" + ",".join(row) + ") VALUES(" + ",".join("?" for _ in row) + ")", list(row.values()))
    result = target.import_graph(inv, payload)
    assert result["claims"] == 0 and result["linked"] == 1
    current = target.claims(inv)[0]
    assert current["source_current"] == 1 and current["review_note"] == "Retain note"
    assert current["review_status"] == "rejected"


@pytest.mark.parametrize("change", ["format", "node_type", "node_reference", "duplicate_nodes", "polarity", "review", "flag", "quote", "portable_conflict"])
def test_invalid_import_is_atomic_and_leaves_existing_graph_untouched(corpus, tmp_path, change):
    payload = seed_export(corpus)
    if change == "format":
        payload["format"] = "unknown"
    elif change == "node_type":
        payload["nodes"][0]["node_type"] = "anything"
    elif change == "node_reference":
        payload["claims"][0]["subject_id"] = 500
    elif change == "duplicate_nodes":
        payload["nodes"].append(deepcopy(payload["nodes"][0]))
    elif change == "polarity":
        payload["claims"][0]["polarity"] = "certain"
    elif change == "review":
        payload["claims"][0]["review_status"] = "approved"
    elif change == "flag":
        payload["claims"][0]["source_current"] = "true"
    elif change == "quote":
        payload["claims"][0]["quote"] = ""
    else:
        payload["nodes"][1]["portable_ids"] = payload["nodes"][0]["portable_ids"]
    target, inv = empty_store(tmp_path / "target")
    with pytest.raises(ValueError):
        target.import_graph(inv, payload)
    assert target.nodes(inv) == target.claims(inv) == []


def test_cancel_after_inserts_rolls_back_entire_import(corpus, tmp_path):
    payload = seed_export(corpus)
    target, inv = empty_store(tmp_path / "target")
    cancel = Event()
    def progress(message):
        if "Importing relationships" in message:
            cancel.set()
    with pytest.raises(KGCancelled):
        target.import_graph(inv, payload, cancel=cancel, progress=progress)
    assert target.nodes(inv) == target.claims(inv) == []
    with target.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kg_import_nodes").fetchone()[0] == 0


def test_imported_graph_survives_bosuil_archive(corpus, tmp_path):
    from investigation_rag.portability import export_workspace, import_workspace
    from investigation_rag.workspace import WorkspaceCatalog
    payload = seed_export(corpus)
    catalog = WorkspaceCatalog(tmp_path / "workspaces")
    catalog.initialize()
    workspace = catalog.create("Portable", [])
    with workspace.db.connection() as conn:
        inv = conn.execute("SELECT id FROM investigations LIMIT 1").fetchone()[0]
    store = KGStore(workspace.db)
    store.import_graph(inv, payload)
    archive = tmp_path / "imported.bosuil"
    export_workspace(workspace.entry, archive)
    destination = WorkspaceCatalog(tmp_path / "restored")
    destination.initialize()
    restored = import_workspace(destination, archive)
    imported = KGStore(restored.workspace.db)
    claim = imported.claims(inv)[0]
    assert claim["quote"] == payload["claims"][0]["quote"] and claim["review_status"] == "accepted"
    assert imported.import_graph(inv, payload)["existing"] == 1 and len(imported.nodes(inv)) == 2


def test_original_review_and_correction_survive_source_loss_after_import(corpus):
    payload = seed_export(corpus)
    store, inv, _ = corpus
    original = store.claims(inv)[0]
    store.import_graph(inv, payload)
    store.edit_claim(inv, original["id"], "contacted", "uncertain", "", "", "Corrected", "")
    store.review(inv, original["id"], "rejected", "Latest local review")
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text_hash='changed' WHERE id='c0'")
    snapshot = store.claims(inv)[0]
    assert snapshot["id"] == original["id"] and snapshot["imported"] == 1
    assert snapshot["predicate"] == "contacted" and snapshot["review_note"] == "Latest local review"
    store.review(inv, snapshot["id"], "suggested", "Reviewed imported evidence")
    with store.db.transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE id='c0'")
    retained = store.claims(inv)[0]
    assert retained["id"] < 0 and retained["review_status"] == "suggested"
    assert retained["review_note"] == "Reviewed imported evidence" and retained["source_current"] == 0


def test_conflicting_portable_claim_identity_does_not_create_duplicate(corpus):
    payload = seed_export(corpus)
    store, inv, _ = corpus
    payload["claims"][0]["quote"] = "Different evidence for the same identity."
    with pytest.raises(ValueError, match="identity conflicts"):
        store.import_graph(inv, payload)
    assert len(store.claims(inv)) == 1 and len(store.nodes(inv)) == 2


def test_two_legacy_node_only_exports_do_not_conflate_unrelated_nodes(tmp_path):
    target, inv = empty_store(tmp_path / "target")
    def node_only(name):
        return {"format": "bosuil-knowledge-graph-v1", "investigation_id": 1,
                "nodes": [{"id": 1, "name": name, "node_type": "concept"}], "claims": []}
    target.import_graph(inv, node_only("Concept A"))
    target.import_graph(inv, node_only("Concept B"))
    assert {n["name"] for n in target.orphan_nodes(inv)} == {"Concept A", "Concept B"}


# Keep the pytest fixture available and also use its constructor for a second corpus.
corpus_fixture = corpus.__wrapped__
