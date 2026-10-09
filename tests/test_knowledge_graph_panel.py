import os
from dataclasses import replace
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from investigation_rag.config import AppConfig
from investigation_rag.db.sqlite_db import SQLiteDB, utc_now
from investigation_rag.gui.knowledge_graph_panel import KnowledgeGraphPanel
from investigation_rag.gui.knowledge_graph_view import KGEdgeItem, KGNodeItem
from investigation_rag.knowledge_graph import KGSettings, KGStore


@pytest.fixture
def panel(tmp_path):
    app = QApplication.instance() or QApplication([])
    db = SQLiteDB(tmp_path / "panel.sqlite3")
    db.initialize()
    inv = db.create_investigation("Test", [], "")
    now = utc_now()
    with db.transaction() as conn:
        file_id = conn.execute("INSERT INTO files(investigation_id,path,normalized_path,status,source_type,first_seen_at,last_seen_at,updated_at) "
                               "VALUES(?,'sample.txt','sample.txt','active','txt',?,?,?)", (inv, now, now, now)).lastrowid
        conn.execute("INSERT INTO chunks(id,investigation_id,file_id,content_hash,chunk_index,chunk_text,chunk_text_hash,source_type,path,"
                     "extractor_version,chunker_version,created_at,updated_at) VALUES('c',?,?,'h',0,'Alice works for Acme.','h','txt','sample.txt','1','1',?,?)",
                     (inv, file_id, now, now))
    store = KGStore(db)
    settings = KGSettings()
    run = store.start_run(inv, settings)
    store.save_batch(inv, run, "batch", store.chunks(inv, [file_id]), [
        {"subject": "Alice", "subject_type": "person", "predicate": "works_for", "object": "Acme", "object_type": "organization",
         "chunk_id": "c", "quote": "Alice works for Acme.", "polarity": "affirmed", "attribution": "", "time_text": ""}], settings, "", "", {})
    panel = KnowledgeGraphPanel(replace(AppConfig.from_env(), ollama_llm_model="current-model", app_data_dir=tmp_path), db)
    panel.set_investigation_id(inv)
    panel.resize(1400, 820)
    panel.show()
    app.processEvents()
    yield panel, app
    panel.shutdown()
    panel.close()


def test_independent_graph_selection_evidence_and_review(panel):
    widget, app = panel
    assert widget.model.currentText() == "current-model"
    assert widget.output_tokens == 4096
    assert widget.documents.count() == 1
    assert not widget.build_button.isEnabled()
    widget.documents.item(0).setSelected(True)
    assert widget.build_button.isEnabled()
    assert widget.table.rowCount() == 1
    assert len([i for i in widget.graph.scene().items() if isinstance(i, KGNodeItem)]) == 2
    edges = [i for i in widget.graph.scene().items() if isinstance(i, KGEdgeItem)]
    assert len(edges) == 1
    widget.select_claim(edges[0].claim["id"])
    assert "Alice works for Acme." in widget.claim_details.toPlainText()
    assert "Alice works for Acme." in widget.preview.toPlainText()
    widget.reject_button.click()
    assert widget.table.rowCount() == 0
    widget.review_filter.setCurrentIndex(3)
    assert widget.table.rowCount() == 1
    widget.table.selectRow(0)
    widget.reset_button.click()
    assert widget.table.rowCount() == 0


def test_saved_token_default_is_upgraded_when_opening_investigation(panel):
    import json
    from dataclasses import asdict
    widget, app = panel
    old = asdict(KGSettings(model="gemma4:12b", num_ctx=16000, output_tokens=2048))
    old.pop("efficiency_version")
    with widget.db.transaction() as conn:
        conn.execute("INSERT INTO settings(key,value,updated_at) VALUES('knowledge_graph_settings',?,?)", (json.dumps(old), utc_now()))
    widget.set_investigation_id(widget.investigation_id)
    assert widget.output_tokens == 4096 and widget.num_ctx == 16000
    assert widget.model.currentText() == "gemma4:12b"


def test_filter_and_focus_preserve_complete_relationship_table(panel):
    widget, app = panel
    widget.search.setText("unmentioned")
    assert widget.table.rowCount() == 0
    widget.search.clear()
    widget.focus_node(widget._claims[0]["subject_id"])
    assert widget.table.rowCount() == 1
    widget.select_claim(widget._claims[0]["id"])
    assert widget.accept_button.isEnabled()


def test_workspace_change_drops_old_results_and_ignores_late_signals(panel, tmp_path):
    widget, app = panel
    old_generation = widget._generation
    other = SQLiteDB(tmp_path / "other.sqlite3")
    other.initialize()
    inv = other.create_investigation("Other", [], "")
    widget.set_data_sources(widget.config, other)
    widget.set_investigation_id(inv)
    assert widget.table.rowCount() == 0
    assert widget.documents.count() == 0
    widget.apply_progress(old_generation, "Old workspace result")
    assert "Old workspace" not in widget.log.toPlainText()
    widget.worker_finished(old_generation, False, None, "Old failure")
    assert "Old failure" not in widget.status.text()


def test_close_requests_cancellation_without_destroying_running_thread(panel):
    widget, app = panel
    class Worker:
        cancel_event = Event()
        import_path = None
        def cancel(self):
            self.cancel_event.set()
    widget.thread = object()
    widget.worker = Worker()
    assert widget.shutdown() is False
    assert widget.worker.cancel_event.is_set()
    widget.thread = widget.worker = None


def test_export_keeps_review_and_evidence_statuses(panel, tmp_path, monkeypatch):
    import json
    from PyQt5.QtWidgets import QFileDialog
    widget, app = panel
    path = tmp_path / "graph.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    widget.export_graph()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["claims"][0]["quote"] == "Alice works for Acme."
    assert data["claims"][0]["review_status"] == "suggested"
    assert data["claims"][0]["source_current"] == 1


def test_graph_view_retains_dragged_positions_and_handles_empty_scope(panel):
    widget, app = panel
    node = next(i for i in widget.graph.scene().items() if isinstance(i, KGNodeItem))
    node.setPos(150, 270)
    node_id = node.node_id
    widget.graph.draw_claims(widget._claims)
    restored = next(i for i in widget.graph.scene().items() if isinstance(i, KGNodeItem) and i.node_id == node_id)
    assert restored.pos().x() == 150 and restored.pos().y() == 270
    assert widget.graph.draw_claims([]) == (0, 0)


def test_build_worker_publishes_completed_batch_and_restores_controls(panel, monkeypatch):
    import json
    from time import monotonic
    from PyQt5.QtTest import QTest
    from investigation_rag.embeddings.ollama_client import OllamaClient
    widget, app = panel
    class Client(OllamaClient):
        def show_model(self, model):
            return {"capabilities": ["completion"]}
        def generate_kg(self, settings, prompt, schema, cancel, progress):
            if "claims" in schema["properties"]:
                value = {"claims": [{"s": "Alice", "st": "person", "r": "works_for", "o": "Acme",
                                      "ot": "organization", "p": "affirmed", "e": [1]}]}
            else:
                value = {"reasons": ["Explicit employment statement"], "supported": [True]}
            return json.dumps(value), {"seconds": 0.01}
    monkeypatch.setattr("investigation_rag.gui.knowledge_graph_panel.KGModelClient", Client)
    widget.documents.item(0).setSelected(True)
    widget.start_build()
    assert widget.thread is not None and not widget.build_button.isEnabled()
    deadline = monotonic() + 5
    while widget.thread is not None and monotonic() < deadline:
        QTest.qWait(10)
        app.processEvents()
    assert widget.thread is None
    assert "Build complete" in widget.status.text()
    assert widget.build_button.isEnabled()
    assert not widget.stop_button.isEnabled()
    assert widget.views.currentWidget() is widget.graph.parentWidget()
    assert widget.store.diagnostics(widget.investigation_id)["runs"][0]["settings"]["model"] == "current-model"
    run = widget.store.diagnostics(widget.investigation_id)["runs"][0]
    assert run["summary"]["claims"] == 1 and len(run["summary"]["requests"]) == 2
    assert run["settings"]["output_tokens"] == 4096


def test_invalid_local_endpoint_finishes_worker_with_visible_error(panel):
    from time import monotonic
    from PyQt5.QtTest import QTest
    widget, app = panel
    widget.config = replace(widget.config, ollama_base_url="http://untrusted.example", ollama_allow_remote=False)
    widget.documents.item(0).setSelected(True)
    widget.start_build()
    deadline = monotonic() + 5
    while widget.thread is not None and monotonic() < deadline:
        QTest.qWait(10)
        app.processEvents()
    assert widget.thread is None
    assert "Remote Ollama endpoint blocked" in widget.status.text()
    assert widget.build_button.isEnabled()


def test_literal_properties_remain_distinct_in_table_graph_and_export(panel, tmp_path, monkeypatch):
    import json
    from PyQt5.QtWidgets import QFileDialog
    widget, app = panel
    text = "Bridge C has a height of 15 m. Bridge C has a width of 15 m."
    with widget.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=?,chunk_text_hash='measurement' WHERE id='c'", (text,))
    settings = KGSettings()
    run = widget.store.start_run(widget.investigation_id, settings)
    window = widget.store.chunks(widget.investigation_id, [widget.documents.item(0).data(Qt.UserRole)])
    claims = [{"subject": "Bridge C", "subject_type": "object", "predicate": "has_measurement", "object": "15 m", "object_type": "quantity",
               "chunk_id": "c", "quote": f"Bridge C has a {name} of 15 m.", "polarity": "affirmed", "attribution": "", "time_text": "", "attribute": name} for name in ("height", "width")]
    widget.store.save_batch(widget.investigation_id, run, "measurements", window, claims, settings, "", "", {})
    widget.reload_claims()
    assert widget.table.rowCount() == 2
    assert {widget.table.item(i, 3).text() for i in range(2)} == {"height", "width"}
    assert len([i for i in widget.graph.scene().items() if isinstance(i, KGEdgeItem)]) == 2
    widget.search.setText("height")
    assert widget.table.rowCount() == 1
    widget.select_claim(widget._visible[0]["id"])
    assert "Property: height" in widget.claim_details.toPlainText()
    path = tmp_path / "literal-graph.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    widget.export_graph()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {c["attribute"] for c in data["claims"] if c["source_current"]} == {"height", "width"}


def test_display_limit_all_option_layout_and_preferences(panel):
    from test_knowledge_graph_layout import claims
    widget, app = panel
    widget._claims = [{**widget._claims[0], **claim} for claim in claims(80)]
    widget.apply_filters()
    assert widget.node_limit.value() == 0 and "81 of 81" in widget.graph_status.text()
    assert widget.table.rowCount() == 80
    widget.node_limit.setValue(12)
    assert "12 of 81" in widget.graph_status.text() and widget.table.rowCount() == 80
    widget.node_limit.setValue(0)
    assert "81 of 81" in widget.graph_status.text()
    node = next(i for i in widget.graph.scene().items() if isinstance(i, KGNodeItem))
    node.setPos(99999, 99999)
    widget.arrange_button.click()
    assert not widget.graph.positions
    assert all(i.pos().x() < 10000 for i in widget.graph.scene().items() if isinstance(i, KGNodeItem))
    widget.graph_layout_choice.setCurrentIndex(1)
    widget.node_limit.setValue(70)
    widget.set_investigation_id(widget.investigation_id)
    assert widget.node_limit.value() == 70 and widget.graph_layout_choice.currentData() == "grid"


def wait_for_worker(widget, app):
    from time import monotonic
    from PyQt5.QtTest import QTest
    deadline = monotonic() + 5
    while widget.thread is not None and monotonic() < deadline:
        QTest.qWait(10)
        app.processEvents()
    assert widget.thread is None


def test_import_worker_shows_unlinked_quote_and_needs_no_model(panel, tmp_path, monkeypatch):
    import json
    from PyQt5.QtWidgets import QFileDialog
    widget, app = panel
    claim = {**widget._claims[0], "id": 100, "subject_id": 90, "object_id": 91,
             "chunk_id": "foreign", "source_hash": "foreign-hash", "path": "original-case.txt",
             "claim_key": "foreign-claim", "review_status": "accepted", "review_note": "Review from export"}
    payload = {"format": "bosuil-knowledge-graph-v1", "investigation_id": 999,
               "nodes": [{"id": 90, "name": "Alice", "node_type": "person"},
                         {"id": 91, "name": "Acme", "node_type": "organization"}], "claims": [claim]}
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    def forbidden_model(*args, **kwargs):
        raise AssertionError("Graph imports must not contact an LLM")
    monkeypatch.setattr("investigation_rag.gui.knowledge_graph_panel.KGModelClient", forbidden_model)
    widget.import_button.click()
    assert not widget.import_button.isEnabled()
    wait_for_worker(widget, app)
    assert widget.import_button.isEnabled() and "Imported 2 new nodes and 1 new claims" in widget.status.text()
    assert widget.table.rowCount() == 2
    imported = next(c for c in widget._claims if c["imported"])
    widget.select_claim(imported["id"])
    assert "Imported quote" in widget.claim_details.toPlainText()
    assert "Alice works for Acme." in widget.preview.toPlainText()
    assert not widget.preview.source_button.isEnabled()
    widget.reject_button.click()
    assert widget.table.rowCount() == 1
    widget.import_button.click()
    wait_for_worker(widget, app)
    assert len(widget._claims) == 2 and widget.table.rowCount() == 1
    assert next(c for c in widget._claims if c["imported"])["review_status"] == "rejected"


def test_invalid_import_restores_controls_and_keeps_graph(panel, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    widget, app = panel
    path = tmp_path / "bad.json"
    path.write_text('{"format":"unknown"}', encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    widget.import_button.click()
    wait_for_worker(widget, app)
    assert "Choose a Bosuil knowledge graph" in widget.status.text()
    assert widget.import_button.isEnabled() and len(widget._claims) == 1
