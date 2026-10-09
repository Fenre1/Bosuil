import json
from dataclasses import replace
from threading import Event

import pytest

from investigation_rag.db.sqlite_db import SQLiteDB, utc_now
from investigation_rag.knowledge_graph import (
    KGCancelled, KGModelClient, KGSettings, KGStore, build_graph, extraction_prompt,
    LEGACY_RELATIONS, RELATIONS, RELATION_GROUPS, claim_fingerprint,
    make_windows, settings_from_saved, validate_candidates, verified_indices,
)


@pytest.fixture
def corpus(tmp_path):
    db = SQLiteDB(tmp_path / "kg.sqlite3")
    db.initialize()
    investigation = db.create_investigation("Case", [], "")
    now = utc_now()
    with db.transaction() as conn:
        file_id = conn.execute("INSERT INTO files(investigation_id,path,normalized_path,status,source_type,first_seen_at,last_seen_at,updated_at) "
                               "VALUES(?, 'source.txt','source.txt','active','txt',?,?,?)", (investigation, now, now, now)).lastrowid
        for index, text in enumerate(["Alice works for Acme.", "Bob alleges that Acme paid Delta in May. Acme did not pay Echo."]):
            conn.execute("INSERT INTO chunks(id,investigation_id,file_id,content_hash,chunk_index,chunk_text,chunk_text_hash,source_type,path,"
                         "extractor_version,chunker_version,created_at,updated_at) VALUES(?,?,?,'content',?,?,?,'txt','source.txt','v1','v1',?,?)",
                         (f"c{index}", investigation, file_id, index, text, f"hash{index}", now, now))
    return KGStore(db), investigation, file_id


def candidate(**updates):
    return {"subject": "Alice", "subject_type": "person", "predicate": "works_for", "object": "Acme",
            "object_type": "organization", "polarity": "affirmed", "attribution": "", "time_text": "",
            "chunk_id": "c0", "quote": "Alice works for Acme.", **updates}


class Model:
    def __init__(self, extraction=None, verifier=None):
        self.extraction = extraction if extraction is not None else {"claims": [candidate()]}
        self.verifier = verifier if verifier is not None else {"verdicts": [{"index": 0, "supported": True}]}
        self.calls = []

    def show_model(self, model):
        return {"model_info": {"llama.context_length": 8192}, "capabilities": ["completion"], "thinking": {"values": [False]}}

    def generate_kg(self, settings, prompt, schema, cancel, progress):
        self.calls.append(prompt)
        if "claims" in schema["properties"]:
            from investigation_rag.knowledge_graph_protocol import FIELDS
            units = json.loads(prompt.split("(JSON DATA):\n")[-1])
            claims = []
            for item in self.extraction["claims"]:
                wire = {k: item[v] for k, v in FIELDS.items() if item.get(v)}
                wire["e"] = [i for i, part, text in units if text in item["quote"]] or [999]
                claims.append(wire)
            response = {"claims": claims}
        else:
            response = {"supported": [v["supported"] for v in self.verifier["verdicts"]],
                        "reasons": [v.get("reason", "Explicit qualified source claim") for v in self.verifier["verdicts"]]}
        return json.dumps(response), {"seconds": 0.1}


def test_grounding_rejects_invented_mentions_quotes_and_qualifiers(corpus):
    store, inv, file_id = corpus
    window = store.chunks(inv, [file_id])
    items = [candidate(), candidate(subject="Ali"), candidate(quote="Alice owns Acme."), candidate(chunk_id="outside"),
             candidate(time_text="yesterday"), candidate(predicate="associated_with"), candidate(subject="Al")]
    good, rejected = validate_candidates(json.dumps({"claims": items}), window, KGSettings())
    assert good == [candidate()]
    assert len(rejected) == 6


def test_preserves_allegations_negation_and_dates(corpus):
    store, inv, file_id = corpus
    alleged = candidate(subject="Acme", subject_type="organization", object="Delta", predicate="paid", polarity="uncertain",
                        attribution="Bob", time_text="May", chunk_id="c1", quote="Bob alleges that Acme paid Delta in May.")
    denied = candidate(subject="Acme", subject_type="organization", object="Echo", predicate="paid", polarity="negated",
                       chunk_id="c1", quote="Acme did not pay Echo.")
    model = Model({"claims": [alleged, denied]}, {"verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": True}]})
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    assert result["status"] == "complete"
    claims = store.claims(inv)
    assert {c["polarity"] for c in claims} == {"negated", "uncertain"}
    assert next(c for c in claims if c["object"] == "Delta")["attribution"] == "Bob"
    assert all(c["review_status"] == "suggested" for c in claims)


def test_verifier_rejection_and_missing_verdict_never_publish(corpus):
    store, inv, file_id = corpus
    model = Model(verifier={"verdicts": []})
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    assert result["status"] == "partial"
    assert store.claims(inv) == []
    assert store.diagnostics(inv)["batches"] == []
    model.verifier = {"verdicts": [{"index": 0, "supported": False}]}
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    assert result["status"] == "complete" and result["rejected"] == 1
    assert store.claims(inv) == []
    assert len(model.calls) == 3  # Successful extraction survives failed verification.
    assert result["extraction_cached"] == 1


def test_incremental_cache_settings_and_review_preservation(corpus):
    store, inv, file_id = corpus
    model = Model()
    first = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    claim = store.claims(inv)[0]
    store.review(inv, claim["id"], "rejected", "Employment not established")
    cached = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    assert cached["cached"] == 1 and len(model.calls) == 2
    changed = build_graph(store, inv, [file_id], replace(KGSettings(), output_tokens=1024), model, Event(), lambda message: None)
    assert changed["cached"] == 1 and len(model.calls) == 2
    assert len(store.claims(inv)) == 1
    assert store.claims(inv)[0]["review_status"] == "rejected"
    assert store.claims(inv)[0]["review_note"] == "Employment not established"
    assert first["run_id"] != changed["run_id"]


def test_cancel_during_verification_keeps_unfinished_batch_out_of_graph(corpus):
    store, inv, file_id = corpus
    cancel = Event()
    class CancellingModel(Model):
        def generate_kg(self, settings, prompt, schema, cancel, progress):
            result = super().generate_kg(settings, prompt, schema, cancel, progress)
            if "supported" in schema["properties"]:
                cancel.set()
            return result
    result = build_graph(store, inv, [file_id], KGSettings(), CancellingModel(), cancel, lambda message: None)
    assert result["status"] == "cancelled"
    assert store.claims(inv) == [] and store.nodes(inv) == []
    assert store.diagnostics(inv)["runs"][0]["status"] == "cancelled"


def test_inactive_changed_or_disappeared_sources_hidden(corpus):
    store, inv, file_id = corpus
    build_graph(store, inv, [file_id], KGSettings(), Model(), Event(), lambda message: None)
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text_hash='changed' WHERE id='c0'")
    assert store.claims(inv) == []
    assert not store.claims(inv, include_stale=True)[0]["source_current"]
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text_hash='hash0' WHERE id='c0'")
        conn.execute("UPDATE files SET status='disappeared' WHERE id=?", (file_id,))
    assert store.claims(inv) == []


def test_source_mutation_before_commit_rejects_batch(corpus):
    store, inv, file_id = corpus
    class MutatingModel(Model):
        def generate_kg(self, *args):
            result = super().generate_kg(*args)
            with store.db.transaction() as conn:
                conn.execute("UPDATE chunks SET is_active=0 WHERE id='c0'")
            return result
    result = build_graph(store, inv, [file_id], KGSettings(), MutatingModel(), Event(), lambda message: None)
    assert result["status"] == "partial"
    assert store.claims(inv, include_stale=True) == []


def test_document_and_attachment_identity_are_separate_and_merge_persists(corpus):
    store, inv, file_id = corpus
    chunks = store.chunks(inv, [file_id])
    other = {**chunks[0], "id": "attachment", "metadata_json": json.dumps({"source_document_id": "attachment-one"})}
    with store.db.transaction() as conn:
        row = dict(other)
        row.pop("fts_rowid")
        conn.execute("INSERT INTO chunks(" + ",".join(row) + ") VALUES(" + ",".join("?" for _ in row) + ")", list(row.values()))
    assert len(make_windows([chunks[0], other])) == 2
    run = store.start_run(inv, KGSettings())
    store.save_batch(inv, run, "root", [chunks[0]], [candidate()], KGSettings(), "", "", {})
    store.save_batch(inv, run, "attachment", [other], [candidate(chunk_id="attachment")], KGSettings(), "", "", {})
    people = [n for n in store.nodes(inv) if n["name"] == "Alice"]
    assert len(people) == 2
    store.merge_nodes(inv, people[0]["id"], people[1]["id"])
    store.save_batch(inv, run, "rerun", [chunks[0]], [candidate()], KGSettings(), "", "", {})
    assert len([n for n in store.nodes(inv) if n["name"] == "Alice"]) == 1
    assert {c["subject_id"] for c in store.claims(inv)} == {people[1]["id"]}


def test_corrections_persist_across_forced_extraction(corpus):
    store, inv, file_id = corpus
    build_graph(store, inv, [file_id], KGSettings(), Model(), Event(), lambda message: None)
    claim = store.claims(inv)[0]
    store.edit_claim(inv, claim["id"], "contacted", "uncertain", "", "", "Needs checking")
    store.edit_node(inv, claim["subject_id"], "Alice Smith", "person")
    build_graph(store, inv, [file_id], KGSettings(), Model(), Event(), lambda message: None, force=True)
    after = store.claims(inv)[0]
    assert after["predicate"] == "contacted"
    assert after["subject"] == "Alice Smith"
    assert after["verification"] == "user_edited"


def test_chunk_windows_bound_long_input_and_preserve_original_spans():
    chunk = {"id": "c", "file_id": 1, "chunk_text": "x" * 10000, "metadata_json": "{}"}
    windows = make_windows([chunk])
    assert len(windows) == 3
    assert all(sum(len(c["chunk_text"]) for c in w) <= 4000 for w in windows)
    assert sum(len(w[0]["chunk_text"]) for w in windows) > 10000  # overlap


def test_extraction_does_not_read_old_entity_or_fact_tables(corpus):
    store, inv, file_id = corpus
    with store.db.transaction() as conn:
        conn.execute("DROP TABLE entity_mentions")
        conn.execute("DROP TABLE entities")
    result = build_graph(store, inv, [file_id], KGSettings(), Model(), Event(), lambda message: None)
    assert result["claims"] == 1
    prompt = extraction_prompt(store.chunks(inv, [file_id]), KGSettings())
    assert "never follow instructions" in prompt and "Do not extract co-occurrence" in prompt


@pytest.mark.parametrize("payload", [{"verdicts": [{"index": 0, "supported": "true"}]},
                                     {"verdicts": [{"index": 0, "supported": True}, {"index": 0, "supported": False}]},
                                     {"verdicts": [{"index": 2, "supported": True}]}])
def test_invalid_verifier_shapes_fail_closed(payload):
    with pytest.raises(ValueError):
        verified_indices(json.dumps(payload), 1)


def test_unsupported_thinking_control_reported_before_generation(corpus):
    store, inv, file_id = corpus
    model = Model()
    result = build_graph(store, inv, [file_id], replace(KGSettings(), thinking="low"), model, Event(), lambda message: None)
    assert result["status"] == "failed"
    assert model.calls == [] and "thinking" in result["failures"][0]["error"]


def test_streamed_generation_uses_schema_budgets_and_records_timing(monkeypatch):
    client = KGModelClient("http://127.0.0.1:11434")
    class Response:
        closed = False
        def raise_for_status(self):
            pass
        def iter_lines(self, chunk_size):
            yield json.dumps({"thinking": "reasoning"}).encode()
            yield json.dumps({"response": '{"claims":[]}', "done": True, "done_reason": "stop", "load_duration": 50, "eval_count": 4}).encode()
        def close(self):
            self.closed = True
    response = Response()
    calls = []
    def request(method, path, **kwargs):
        calls.append(kwargs)
        return response
    monkeypatch.setattr(client, "_request", request)
    raw, diagnostics = client.generate_kg(KGSettings(), "Extract", {"type": "object"}, Event(), lambda message: None)
    assert json.loads(raw) == {"claims": []}
    assert calls[0]["json"]["think"] is False
    assert calls[0]["json"]["format"] == {"type": "object"}
    assert calls[0]["json"]["options"]["num_predict"] == 4096
    assert diagnostics["thinking_characters"] == 9 and diagnostics["load_duration"] == 50
    assert response.closed


@pytest.mark.parametrize("failure", ["length", "incomplete", "cancel"])
def test_stream_truncation_disconnect_and_cancellation_discard_output(monkeypatch, failure):
    client = KGModelClient("http://127.0.0.1:11434")
    cancel = Event()
    class Response:
        closed = False
        def raise_for_status(self):
            pass
        def iter_lines(self, chunk_size):
            if failure == "cancel":
                cancel.set()
            yield json.dumps({"response": "{}", "done": failure == "length", "done_reason": "length"}).encode()
        def close(self):
            self.closed = True
    response = Response()
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises((ValueError, KGCancelled)):
        client.generate_kg(KGSettings(), "Extract", {}, cancel, lambda message: None)
    assert response.closed


def test_graph_survives_investigation_archive_transfer(corpus, tmp_path):
    from investigation_rag.portability import export_workspace, import_workspace
    from investigation_rag.workspace import WorkspaceCatalog
    store, inv, file_id = corpus
    source = WorkspaceCatalog(tmp_path / "source")
    source.initialize()
    workspace = source.create("Case", [])
    # Use the seeded independent graph corpus as the workspace database.
    store.db.backup_to(workspace.db.path)
    measurement = candidate(subject="Bridge C", subject_type="object", predicate="has_measurement", object="15 m", object_type="quantity",
                            attribute="height", quote="Bridge C has a height of 15 m.", chunk_id="c1")
    with workspace.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=?,chunk_text_hash='height' WHERE id='c1'", (measurement["quote"],))
    model = Model({"claims": [candidate(), measurement]}, {"verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": True}]})
    build_graph(KGStore(workspace.db), inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    graph_store = KGStore(workspace.db)
    claim = next(c for c in graph_store.claims(inv) if c["predicate"] == "works_for")
    graph_store.review(inv, claim["id"], "accepted", "Checked source")
    archive = tmp_path / "graph.bosuil"
    export_workspace(workspace.entry, archive)
    target = WorkspaceCatalog(tmp_path / "target")
    target.initialize()
    restored = import_workspace(target, archive)
    imported = KGStore(restored.workspace.db)
    restored_claim = next(c for c in imported.claims(inv) if c["predicate"] == "works_for")
    assert restored_claim["quote"] == "Alice works for Acme."
    assert restored_claim["review_status"] == "accepted"
    assert restored_claim["review_note"] == "Checked source"
    assert next(c for c in imported.claims(inv) if c["predicate"] == "has_measurement")["attribute"] == "height"
    assert len(imported.nodes(inv)) == 4
    assert len(imported.diagnostics(inv)["batches"]) == 1
    assert {r["stage"]: r["entries"] for r in imported.diagnostics(inv)["stage_cache"]} == {"extraction": 1, "verification": 1}
    resumed_model = Model()
    resumed = build_graph(imported, inv, [file_id], KGSettings(), resumed_model, Event(), lambda message: None)
    assert resumed["cached"] == 1 and not resumed_model.calls


def test_temporal_spatial_factual_and_literal_claims_are_grounded(corpus):
    store, inv, file_id = corpus
    text = ("Inspection A occurred on 12 May 2025 in Utrecht before Repair B. "
            "Bridge C has a height of 15 m and a width of 15 m. Pump D is inactive. "
            "Canal E is north of Village F. Repair B lasted 3 days.")
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=?,chunk_text_hash='general' WHERE id='c0'", (text,))
    quote = "Bridge C has a height of 15 m and a width of 15 m."
    claims = [
        candidate(subject="Inspection A", subject_type="event", predicate="occurred_on", object="12 May 2025", object_type="date",
                  quote="Inspection A occurred on 12 May 2025 in Utrecht before Repair B."),
        candidate(subject="Inspection A", subject_type="event", predicate="occurs_at", object="Utrecht", object_type="location",
                  quote="Inspection A occurred on 12 May 2025 in Utrecht before Repair B."),
        candidate(subject="Inspection A", subject_type="event", predicate="before", object="Repair B", object_type="event",
                  quote="Inspection A occurred on 12 May 2025 in Utrecht before Repair B."),
        candidate(subject="Bridge C", subject_type="object", predicate="has_measurement", object="15 m", object_type="quantity", attribute="height", quote=quote),
        candidate(subject="Bridge C", subject_type="object", predicate="has_measurement", object="15 m", object_type="quantity", attribute="width", quote=quote),
        candidate(subject="Pump D", subject_type="object", predicate="has_attribute", object="inactive", object_type="value", quote="Pump D is inactive."),
        candidate(subject="Canal E", subject_type="location", predicate="north_of", object="Village F", object_type="location", quote="Canal E is north of Village F."),
        candidate(subject="Repair B", subject_type="event", predicate="has_duration", object="3 days", object_type="duration", quote="Repair B lasted 3 days."),
    ]
    model = Model({"claims": claims}, {"verdicts": [{"index": i, "supported": True} for i in range(len(claims))]})
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    assert result["status"] == "complete" and result["claims"] == 8
    stored = store.claims(inv)
    measurements = [c for c in stored if c["predicate"] == "has_measurement"]
    assert {c["attribute"] for c in measurements} == {"height", "width"}
    assert measurements[0]["object_id"] == measurements[1]["object_id"]
    assert measurements[0]["id"] != measurements[1]["id"]
    assert next(c for c in stored if c["predicate"] == "has_duration")["object_type"] == "duration"


def test_properties_and_literal_values_cannot_be_borrowed_from_other_passages(corpus):
    store, inv, file_id = corpus
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text='Bridge C is inspected. Its height is not stated.' WHERE id='c0'")
        conn.execute("UPDATE chunks SET chunk_text='Bridge Q has a width of 15 m on 12 May.' WHERE id='c1'")
    claims = [
        candidate(subject="Bridge C", subject_type="object", predicate="has_measurement", object="15 m", object_type="quantity", quote="Bridge C is inspected."),
        candidate(subject="Bridge Q", subject_type="object", predicate="has_measurement", object="15 m", object_type="quantity", attribute="height",
                  chunk_id="c1", quote="Bridge Q has a width of 15 m on 12 May."),
        candidate(subject="Bridge Q", subject_type="object", predicate="occurred_on", object="12 May", object_type="organization",
                  chunk_id="c1", quote="Bridge Q has a width of 15 m on 12 May."),
    ]
    good, rejected = validate_candidates(json.dumps({"claims": claims}), store.chunks(inv, [file_id]), KGSettings())
    assert not good and len(rejected) == 3


def test_literal_property_corrections_survive_reruns(corpus):
    store, inv, file_id = corpus
    text = "Bridge C has a height of 15 m."
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=?,chunk_text_hash='measurement' WHERE id='c0'", (text,))
    item = candidate(subject="Bridge C", subject_type="object", predicate="has_measurement", object="15 m", object_type="quantity", attribute="height", quote=text)
    run = store.start_run(inv, KGSettings())
    window = store.chunks(inv, [file_id])
    store.save_batch(inv, run, "measurement", window, [item], KGSettings(), "", "", {})
    claim = store.claims(inv)[0]
    store.edit_claim(inv, claim["id"], "has_measurement", "uncertain", "", "", "Needs checking", "")
    store.save_batch(inv, run, "measurement-again", window, [item], KGSettings(), "", "", {})
    after = store.claims(inv)[0]
    assert after["attribute"] == "" and after["verification"] == "user_edited"
    assert len(store.claims(inv)) == 1


def test_saved_defaults_upgrade_but_custom_and_current_selections_remain():
    from dataclasses import asdict
    old = asdict(KGSettings(relations=LEGACY_RELATIONS))
    old.pop("vocabulary_version")
    assert settings_from_saved(old).relations == tuple(RELATIONS)
    old["relations"] = ["paid", "works_for"]
    assert settings_from_saved(old).relations == ("paid", "works_for")
    current = asdict(KGSettings(relations=LEGACY_RELATIONS))
    assert settings_from_saved(current).relations == LEGACY_RELATIONS
    assert claim_fingerprint(candidate()) == claim_fingerprint(candidate(attribute=""))
    grouped = [r for names in RELATION_GROUPS.values() for r in names]
    assert len(grouped) == len(set(grouped)) and set(grouped) == set(RELATIONS)


def test_general_prompts_do_not_prioritize_actors_and_preserve_literals(corpus):
    from investigation_rag.knowledge_graph import verification_prompt
    store, inv, file_id = corpus
    prompt = extraction_prompt(store.chunks(inv, [file_id]), KGSettings())
    assert "do not prioritize human actions" in prompt
    assert "height" in prompt and "15 m/quantity" in prompt and "12 May 2025/date" in prompt
    verification = verification_prompt(store.chunks(inv, [file_id]), [candidate(predicate="before")])
    assert "causal relationships inferred solely" in verification


def test_clock_times_and_unnamed_concept_facts_are_valid_source_claims(corpus):
    store, inv, file_id = corpus
    text = "Inspection starts at 09:30. Heavy rainfall causes flooding. A pump is a machine."
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=? WHERE id='c0'", (text,))
    claims = [
        candidate(subject="Inspection", subject_type="event", predicate="starts_on", object="09:30", object_type="time", quote="Inspection starts at 09:30."),
        candidate(subject="Heavy rainfall", subject_type="concept", predicate="causes", object="flooding", object_type="event", quote="Heavy rainfall causes flooding."),
        candidate(subject="pump", subject_type="object", predicate="instance_of", object="machine", object_type="concept", quote="A pump is a machine."),
    ]
    good, rejected = validate_candidates(json.dumps({"claims": claims}), store.chunks(inv, [file_id]), KGSettings())
    assert good == claims and not rejected


def test_quantities_with_case_sensitive_units_do_not_merge(corpus):
    store, inv, file_id = corpus
    text = "Device A has a power of 5 mW. Device B has a power of 5 MW."
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=?,chunk_text_hash='power' WHERE id='c0'", (text,))
    claims = [candidate(subject=name, subject_type="object", predicate="has_measurement", object=value, object_type="quantity",
                        attribute="power", quote=f"{name} has a power of {value}.") for name, value in (("Device A", "5 mW"), ("Device B", "5 MW"))]
    model = Model({"claims": claims}, {"verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": True}]})
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda message: None)
    assert result["status"] == "complete"
    stored = store.claims(inv)
    assert {c["object"] for c in stored} == {"5 mW", "5 MW"}
    assert len({c["object_id"] for c in stored}) == 2
