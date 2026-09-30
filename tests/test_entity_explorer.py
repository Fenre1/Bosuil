import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from investigation_rag.db.sqlite_db import SQLiteDB, utc_now
from investigation_rag.entity_network import (
    EntityNetwork, ExplorationRepository, NetworkChunk, NetworkDocument, NetworkEntity,
    connection_layout, load_network, rank_connections,
)
from investigation_rag.gui.entity_explorer import EntityGraphPanel


def corpus(tmp_path):
    db = SQLiteDB(tmp_path / "graph.sqlite3")
    db.initialize()
    case = db.create_investigation("Case", [tmp_path])
    now = utc_now()
    # The second file is an exact copy. The third contains both entities but in
    # separate passages, to distinguish document links from passage links.
    sources = [
        ("original.txt", "same-file", [("ab1", "Alice checks Pump P-204.", "same-text", [1, 2])]),
        ("copy.txt", "same-file", [("ab2", "Alice checks Pump P-204.", "same-text", [1, 2])]),
        ("report.txt", "report", [("a", "Alice reports.", "text-a", [1]), ("b", "Pump P-204 stopped.", "text-b", [2]),
                                   ("bc", "Pump P-204 supplied by ACME.", "text-bc", [2, 3])]),
        ("rare.txt", "rare", [("rare", "Witness Z observed Valve V-1.", "text-rare", [4, 5])]),
    ]
    with db.transaction() as conn:
        for i, name, entity_type in [(1, "Alice", "person"), (2, "Pump P-204", "asset"),
                                      (3, "ACME", "organization"), (4, "Witness Z", "person"), (5, "Valve V-1", "asset")]:
            conn.execute("""INSERT INTO entities(id, investigation_id, name, normalized_name, entity_type, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?)""", (i, case, name, name.casefold(), entity_type, now, now))
        for path, content_hash, passages in sources:
            file_id = conn.execute("""INSERT INTO files(investigation_id,path,normalized_path,status,source_type,content_hash,first_seen_at,last_seen_at,updated_at)
                VALUES(?,?,?,'active','txt',?,?,?,?)""", (case, path, path, content_hash, now, now, now)).lastrowid
            for index, (chunk_id, text, text_hash, entities) in enumerate(passages):
                conn.execute("""INSERT INTO chunks(id,investigation_id,file_id,content_hash,chunk_index,chunk_text,chunk_text_hash,source_type,path,extractor_version,chunker_version,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,'txt',?,'1','1',?,?)""", (chunk_id,case,file_id,content_hash,index,text,text_hash,path,now,now))
                for i in entities:
                    name = conn.execute("SELECT name FROM entities WHERE id=?", (i,)).fetchone()[0]
                    start = text.index(name)
                    conn.execute("""INSERT INTO entity_mentions(entity_id,file_id,chunk_id,mention_text,start_offset,end_offset,confidence,created_at)
                        VALUES(?,?,?,?,?,?,1,?)""", (i,file_id,chunk_id,name,start,start+len(name),now))
    return db, case


def test_complete_connections_discount_copies_and_distinguish_document_evidence(tmp_path):
    db, case = corpus(tmp_path)
    network = load_network(db, case)
    passage = network.connection(1, 2)
    assert passage.chunks == {"ab1", "ab2"}
    assert len(passage.documents) == 2
    assert passage.unique_documents == passage.unique_passages == 1
    document = network.connection(1, 2, "document")
    assert document.chunks == {"ab1", "ab2", "a", "b", "bc"}
    assert len(document.documents) == 3
    assert document.unique_documents == 2
    assert network.connection(1, 3) is None
    assert network.connection(1, 3, "document").chunks == {"a", "bc"}


def test_scope_rare_entities_and_ranking_use_full_memberships(tmp_path):
    db, case = corpus(tmp_path)
    network = load_network(db, case)
    scope = frozenset([network.chunks["a"].document])
    assert network.connection(1, 2, scope=scope) is None
    assert network.connection(1, 2, "document", scope).unique_documents == 1
    assert network.eligible(scope) == {1, 2, 3}
    assert network.eligible(types={"person"}) == {1, 4}
    assert network.connection(4, 5).unique_passages == 1
    assert {frozenset(g) for g in network.groups()} == {frozenset([1, 2, 3]), frozenset([4, 5])}
    assert {c.second for c in network.neighbors(1, "document")} == {2, 3}
    assert rank_connections(network.neighbors(2), "specificity")


def test_inactive_sources_and_other_cases_are_excluded(tmp_path):
    db, case = corpus(tmp_path)
    other = db.create_investigation("Other", [tmp_path])
    assert not load_network(db, other).entities
    with db.transaction() as conn:
        conn.execute("UPDATE chunks SET is_active=0 WHERE id='ab1'")
        conn.execute("UPDATE files SET status='inactive' WHERE path='copy.txt'")
    network = load_network(db, case)
    assert network.connection(1, 2) is None
    assert "ab1" not in network.chunks and "ab2" not in network.chunks


def test_no_entity_or_membership_cap_and_stable_incremental_layout():
    count = 6540
    entities = {i: NetworkEntity(i, f"Entity {i}", "person") for i in range(count)}
    chunks, docs = {}, {}
    for i in range(count - 1):
        c, d = f"c{i}", f"d{i}"
        docs[d] = NetworkDocument(d, i, d, "root", d)
        chunks[c] = NetworkChunk(c, d, i, d, i, None, c, "")
        entities[i].chunks.add(c)
        entities[i + 1].chunks.add(c)
    network = EntityNetwork(entities, chunks, docs)
    edges = network.edges(entities)
    assert len(edges) == count - 1
    assert len(network.groups()[0]) == count
    positions = connection_layout(entities, edges)
    assert len(positions) == count
    assert len(set(positions.values())) == count
    earlier = connection_layout({0, 1, 2}, edges[:2])
    expanded = connection_layout({0, 1, 2, 3}, edges[:3], earlier)
    assert all(expanded[i] == p for i, p in earlier.items())


def test_cancelled_calculations_do_not_return_partial_results(tmp_path):
    db, case = corpus(tmp_path)
    with pytest.raises(InterruptedError):
        load_network(db, case, lambda: True)
    network = load_network(db, case)
    with pytest.raises(InterruptedError):
        network.edges(network.entities, cancelled=lambda: True)


def test_alias_merge_type_correction_and_undo_preserve_mentions(tmp_path):
    db, case = corpus(tmp_path)
    repo = ExplorationRepository(db)
    original = load_network(db, case)
    target = repo.edit(case, [1, 4], "Alice", "person")
    merged = load_network(db, case)
    assert target == 1
    assert 4 not in merged.entities
    assert "Witness Z" in merged.entities[1].aliases
    assert merged.entities[1].chunks == original.entities[1].chunks | original.entities[4].chunks
    repo.edit(case, [1], "Alice", "witness")
    assert load_network(db, case).entities[1].entity_type == "witness"
    repo.undo_edit(case)
    assert load_network(db, case).entities[1].entity_type == "person"
    repo.undo_edit(case)
    restored = load_network(db, case)
    assert restored.entities == original.entities
    assert restored.connection(4, 5) == original.connection(4, 5)


def test_type_can_return_to_old_alias_type_and_conflicts_rollback(tmp_path):
    db, case = corpus(tmp_path)
    repo = ExplorationRepository(db)
    repo.edit(case, [1], "Alice", "witness")
    repo.edit(case, [1], "Alice", "person")
    assert load_network(db, case).entities[1].entity_type == "person"
    repo.undo_edit(case)
    assert load_network(db, case).entities[1].entity_type == "witness"
    with db.transaction() as conn:
        conn.execute("UPDATE entities SET updated_at='newer-work' WHERE id=1")
    with pytest.raises(ValueError, match="newer work"):
        repo.undo_edit(case)
    assert load_network(db, case).entities[1].entity_type == "witness"


def test_saved_explorations_are_scoped_and_survive_database_reopen(tmp_path):
    db, case = corpus(tmp_path)
    repo = ExplorationRepository(db)
    state = {"visible": [1, 2], "positions": {"1": [35, 70]}, "scope": None}
    repo.save(case, "Pump investigation", state)
    reopened = ExplorationRepository(SQLiteDB(db.path))
    assert reopened.saved(case) == {"Pump investigation": state}
    other = db.create_investigation("Other", [tmp_path])
    assert not reopened.saved(other)


@pytest.fixture
def panel(tmp_path):
    app = QApplication.instance() or QApplication([])
    db, case = corpus(tmp_path)
    widget = EntityGraphPanel(db)
    widget.resize(1450, 950)
    widget.set_context(case)
    widget.show()
    widget.set_active(True)
    wait_for(app, lambda: widget.network is not None and not widget._jobs)
    yield app, widget
    widget.shutdown()
    wait_for(app, lambda: not widget._jobs)
    widget.close()


def wait_for(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate(), "Graph worker did not finish"


def test_search_finds_seeds_without_filtering_neighbors_and_expands_all(panel):
    app, widget = panel
    widget.search.setText("Alice")
    assert widget.search_results.count() == 1
    widget._open_search()
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == {1, 2}
    assert len(widget.view.edges) == 1
    widget.expand(2)
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == {1, 2, 3}
    widget.show_all()
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == {1, 2, 3, 4, 5}
    assert len(widget.view.edges) == 3
    # Results filtering/table order must not invalidate graph scope.
    widget.set_context(widget.investigation_id, [1])
    assert widget.visible == {1, 2, 3, 4, 5}


def test_edge_evidence_highlights_both_names_and_selection_preserves_camera(panel):
    app, widget = panel
    widget.explore_entities([1])
    wait_for(app, lambda: not widget._jobs)
    widget.view.scale(1.7, 1.7)
    before = widget.view.viewport_state()
    widget.select_entity(2)
    assert widget.view.viewport_state() == before
    widget.select_edge(widget.network.connection(1, 2))
    assert set(widget._preview_ids) == {"ab1", "ab2"}
    widget.show_chunk("ab1")
    assert widget.preview.toPlainText() == "Alice checks Pump P-204."
    doc = widget.preview.viewer.document()
    first = doc.find("Alice").charFormat().background().color().name()
    second = doc.find("Pump P-204").charFormat().background().color().name()
    assert first == "#fff1a8" and second == "#bce8ff"
    assert widget.view.viewport_state() == before
    state = widget._state()
    widget._restore(state)
    wait_for(app, lambda: not widget._jobs)
    assert widget.preview.toPlainText() == "Alice checks Pump P-204."
    assert widget._preview_ids[widget._preview_index] == "ab1"
    widget.follow_connection()
    wait_for(app, lambda: not widget._jobs)
    assert widget.selected == 1
    assert widget.visible == {1, 2}


def test_history_hide_pin_and_saved_positions_restore(panel):
    app, widget = panel
    widget.explore_entities([1])
    wait_for(app, lambda: not widget._jobs)
    widget.view.nodes[1].setPos(280, 125)
    widget.select_entity(1)
    widget.pin_selected()
    wait_for(app, lambda: not widget._jobs)
    assert widget.pinned == {1}
    assert widget.view.positions()[1] == (280, 125)
    state = widget._state()
    widget.hide_selected()
    wait_for(app, lambda: not widget._jobs)
    assert 1 not in widget.visible
    widget.navigate(-1)
    wait_for(app, lambda: not widget._jobs)
    assert 1 in widget.visible and 1 in widget.pinned
    assert widget.view.positions()[1] == (280, 125)
    widget.show_all()
    wait_for(app, lambda: not widget._jobs)
    widget._restore(state)
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == {1, 2}
    assert widget.view.positions()[1] == (280, 125)


def test_shared_connections_and_scope_do_not_claim_missing_evidence(panel):
    app, widget = panel
    widget.show_all()
    wait_for(app, lambda: not widget._jobs)
    widget.view.nodes[1].setSelected(True)
    widget.view.nodes[3].setSelected(True)
    widget.shared_connections()
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == {1, 2, 3}
    widget.scope = frozenset([widget.network.chunks["rare"].document])
    widget._scope_changed()
    wait_for(app, lambda: not widget._jobs)
    assert not widget.visible
    widget.show_all()
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == {4, 5}


def test_cancel_keeps_last_complete_view_and_disables_saving_pending_state(panel):
    app, widget = panel
    widget.explore_entities([1])
    wait_for(app, lambda: not widget._jobs)
    previous = set(widget.visible)
    widget._capture_current()
    widget.visible = set(widget.network.entities)
    def work(cancelled):
        while not cancelled():
            time.sleep(0.005)
        raise InterruptedError()
    widget._run(work, lambda result: pytest.fail("Cancelled result was applied"), "Building…")
    assert not widget.search.isEnabled()
    assert widget.cancel_button.isEnabled()
    widget._cancel_requested()
    wait_for(app, lambda: not widget._jobs)
    assert widget.visible == previous == set(widget.view.nodes)
    assert widget.search.isEnabled()
    assert not widget.preview.previous_button.isEnabled()


def test_switching_database_with_same_investigation_id_clears_exploration(panel, tmp_path):
    app, widget = panel
    widget.explore_entities([1])
    wait_for(app, lambda: not widget._jobs)
    other_db = SQLiteDB(tmp_path / "other.sqlite3")
    other_db.initialize()
    other_id = other_db.create_investigation("Other", [tmp_path])
    assert other_id == widget.investigation_id
    widget.set_data_sources(other_db, None)
    widget.set_context(other_id)
    wait_for(app, lambda: widget.network is not None and not widget._jobs)
    assert not widget.visible and not widget.network.entities
    assert widget.selected is None
    assert not widget.pinned


def test_topic_map_falls_back_without_embeddings_and_restores_pinned_positions(panel):
    app, widget = panel
    widget.explore_entities([1])
    wait_for(app, lambda: not widget._jobs)
    widget.view.nodes[1].setPos(500, 700)
    widget.pin_selected()
    wait_for(app, lambda: not widget._jobs)
    widget.layout_mode.setCurrentText("Topic map")
    wait_for(app, lambda: not widget._jobs)
    assert widget.view.positions()[1] == (500, 700)
    assert "No stored embeddings" in widget.status.text()
    assert len(widget.view.edges) == 1


def test_topic_map_separates_identical_contexts_without_dropping_entities(panel):
    app, widget = panel
    class Vectors:
        def get_vectors_for_chunks(self, ids, **kwargs):
            return {i: [1.0, 0.0, 0.0] for i in ids}
    widget.store = Vectors()
    with widget.db.transaction() as conn:
        conn.execute("UPDATE chunks SET embedding_model='test'")
    widget.refresh()
    wait_for(app, lambda: not widget._dirty and not widget._jobs)
    widget.explore_entities([1])
    wait_for(app, lambda: not widget._jobs)
    widget.layout_mode.setCurrentText("Topic map")
    wait_for(app, lambda: not widget._jobs)
    assert len(set(widget.view.positions().values())) == 2
    assert "Embeddings cover 2 of 2" in widget.status.text()
