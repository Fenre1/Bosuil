import json
from dataclasses import asdict, replace
from threading import Event

import pytest

from test_knowledge_graph import Model, candidate, corpus
from investigation_rag.knowledge_graph import (
    KGContextLimit, KGModelClient, KGOutputLimit, KGSettings, LEGACY_PROMPT_VERSION, build_graph,
    digest, make_windows, settings_from_saved, source_key,
)
from investigation_rag.knowledge_graph_protocol import (
    compact_schema, compact_verification_prompt, decode_claims, decode_verdicts, source_units, split_window,
)


def wire(**updates):
    return {"s": "Alice", "st": "person", "r": "works_for", "o": "Acme",
            "ot": "organization", "p": "affirmed", "e": [1], **updates}


def test_output_default_and_saved_settings_upgrade_once():
    assert KGSettings().output_tokens == 4096
    old = asdict(KGSettings(output_tokens=2048))
    old.pop("efficiency_version")
    assert settings_from_saved(old).output_tokens == 4096
    old["output_tokens"] = 3072
    assert settings_from_saved(old).output_tokens == 3072
    assert settings_from_saved(asdict(KGSettings(output_tokens=2048))).output_tokens == 2048
    old.update(output_tokens=2048, num_ctx=4096)
    assert settings_from_saved(old).num_ctx == 8192


def test_compact_evidence_preserves_original_whitespace_and_qualifiers(corpus):
    store, inv, file_id = corpus
    chunks = store.chunks(inv, [file_id])
    chunks[0]["chunk_text"] = "Bob alleges that Alice works for Acme.\r\n\tAcme denies this."
    raw = json.dumps({"claims": [wire(p="uncertain", a="Bob", e=[1, 2])]})
    good, rejected, count = decode_claims(raw, chunks, KGSettings())
    assert not rejected and count == 1
    assert good[0]["quote"] == chunks[0]["chunk_text"]
    assert good[0]["attribution"] == "Bob"
    assert good[0]["chunk_id"] == "c0"


@pytest.mark.parametrize("evidence", [[True], [], [999], [2, 1], [1, 1], [1, 3], [1, 2]])
def test_compact_evidence_rejects_invalid_ids_and_cross_chunk_spans(corpus, evidence):
    store, inv, file_id = corpus
    good, rejected, _ = decode_claims(json.dumps({"claims": [wire(e=evidence)]}), store.chunks(inv, [file_id]), KGSettings())
    assert not good and rejected


def test_compact_protocol_keeps_grounding_and_omits_repeated_quote(corpus):
    store, inv, file_id = corpus
    window = store.chunks(inv, [file_id])
    good, rejected, _ = decode_claims(json.dumps({"claims": [wire(), wire(s="Ali"), wire(k="height"), wire(o="Ghost")]}), window, KGSettings())
    assert good == [candidate()] and len(rejected) == 3
    assert "quote" not in compact_schema(KGSettings())["properties"]["claims"]["items"]["properties"]
    prompt = compact_verification_prompt(window, good)
    assert prompt.count(candidate()["quote"]) == 1
    assert '"e":[1]' in prompt


@pytest.mark.parametrize("payload", [{"supported": []}, {"supported": ["true"]}, {"supported": [1]},
                                     {"supported": [True, False]}, {"supported": [True], "extra": True},
                                     {"supported": [True], "reasons": []},
                                     {"supported": [True], "reasons": [""]},
                                     {"supported": [True], "reasons": ["x" * 121]}])
def test_compact_verification_fails_closed(payload):
    with pytest.raises(ValueError):
        decode_verdicts(json.dumps(payload), 1)


def test_verification_resume_reuses_extraction_with_changed_budget_and_timeout(corpus):
    store, inv, file_id = corpus
    model = Model(verifier={"verdicts": []})
    first = build_graph(store, inv, [file_id], KGSettings(output_tokens=2048), model, Event(), lambda msg: None)
    assert first["status"] == "partial"
    assert json.loads(first["failures"][0]["verification_output"]) == {"supported": [], "reasons": []}
    model.verifier = {"verdicts": [{"index": 0, "supported": True}]}
    resumed = build_graph(store, inv, [file_id], KGSettings(output_tokens=4096, request_seconds=300), model, Event(), lambda msg: None)
    assert resumed["status"] == "complete" and resumed["extraction_cached"] == 1
    assert len(model.calls) == 3 and len(store.claims(inv)) == 1


def test_toggle_verification_reuses_extraction_but_still_verifies(corpus):
    store, inv, file_id = corpus
    model = Model()
    build_graph(store, inv, [file_id], KGSettings(verify=False), model, Event(), lambda msg: None)
    checked = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert checked["extraction_cached"] == 1 and len(model.calls) == 2
    assert checked["completed"] == 1


def test_failed_commit_reuses_both_stages_and_preserves_human_review(corpus, monkeypatch):
    store, inv, file_id = corpus
    model = Model()
    save = store.save_batch
    def fail(*args, **kwargs):
        raise RuntimeError("Transient commit failure")
    monkeypatch.setattr(store, "save_batch", fail)
    first = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert first["status"] == "partial" and not store.claims(inv)
    monkeypatch.setattr(store, "save_batch", save)
    resumed = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert resumed["extraction_cached"] == resumed["verification_cached"] == 1
    assert len(model.calls) == 2
    claim = store.claims(inv)[0]
    store.review(inv, claim["id"], "accepted", "Checked")
    build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None, force=True)
    assert len(model.calls) == 4 and store.claims(inv)[0]["review_note"] == "Checked"


def large_window(corpus):
    store, inv, file_id = corpus
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text=?,chunk_text_hash='large' WHERE id='c0'",
                     ("Alice works for Acme. " + "Background context without a relationship. " * 22,))
        conn.execute("UPDATE chunks SET is_active=0 WHERE id='c1'")
    return store, inv, file_id


class TruncatingModel(Model):
    def __init__(self, error=KGOutputLimit):
        super().__init__()
        self.error = error
        self.parent_calls = 0

    def generate_kg(self, settings, prompt, schema, cancel, progress):
        if "claims" in schema["properties"]:
            units = json.loads(prompt.split("(JSON DATA):\n")[-1])
            if sum(len(text) for _, _, text in units) > 700:
                self.parent_calls += 1
                self.calls.append(prompt)
                if self.error is KGOutputLimit:
                    raise KGOutputLimit('{"claims":[', {"eval_count": settings.output_tokens, "done_reason": "length"})
                raise KGContextLimit("Input context exceeded")
            self.extraction = {"claims": [candidate()]} if any(candidate()["quote"] in text for _, _, text in units) else {"claims": []}
        return super().generate_kg(settings, prompt, schema, cancel, progress)


@pytest.mark.parametrize("error", [KGOutputLimit, KGContextLimit])
def test_truncated_or_oversize_window_recovers_and_caches_children(corpus, error):
    store, inv, file_id = large_window(corpus)
    model = TruncatingModel(error)
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert result["status"] == "complete" and result["splits"] == 1
    assert result["sections_saved"] == 2 and len(store.claims(inv)) == 1
    if error is KGOutputLimit:
        assert result["requests"][0]["eval_count"] == 4096
        assert result["requests"][0]["discarded_output"] == '{"claims":['
    calls = len(model.calls)
    cached = build_graph(store, inv, [file_id], KGSettings(output_tokens=2048), model, Event(), lambda msg: None)
    assert cached["cached"] == 1 and len(model.calls) == calls


def test_cancellation_preserves_recovered_sections_for_resume(corpus):
    store, inv, file_id = large_window(corpus)
    cancel = Event()
    def progress(message):
        if "saved 1 suggestions" in message:
            cancel.set()
    model = TruncatingModel()
    first = build_graph(store, inv, [file_id], KGSettings(), model, cancel, progress)
    assert first["status"] == "cancelled" and len(store.claims(inv)) == 1
    resumed = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert resumed["status"] == "complete" and model.parent_calls == 1
    assert resumed["sections_saved"] == 1 and resumed["extraction_cached"] == 1


def test_terminal_truncation_is_retried_after_raising_budget(corpus):
    store, inv, file_id = corpus
    class BudgetModel(Model):
        def generate_kg(self, settings, *args):
            if settings.output_tokens < 4096:
                self.calls.append(args[0])
                raise KGOutputLimit("partial", {"eval_count": settings.output_tokens})
            return super().generate_kg(settings, *args)
    model = BudgetModel()
    first = build_graph(store, inv, [file_id], KGSettings(output_tokens=2048), model, Event(), lambda msg: None)
    assert first["status"] == "partial" and not store.claims(inv)
    resumed = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert resumed["status"] == "complete" and len(store.claims(inv)) == 1


def test_verifier_truncation_retries_smaller_candidate_groups(corpus):
    store, inv, file_id = corpus
    denied = candidate(subject="Acme", subject_type="organization", predicate="paid", object="Echo", polarity="negated",
                       chunk_id="c1", quote="Acme did not pay Echo.")
    class VerifierLimit(Model):
        def generate_kg(self, settings, prompt, schema, cancel, progress):
            if "supported" in schema["properties"]:
                items = json.loads(prompt.split("CANDIDATES (JSON DATA):\n")[1])
                if len(items) > 1:
                    self.calls.append(prompt)
                    raise KGOutputLimit("partial verdicts", {"eval_count": 4096})
            return super().generate_kg(settings, prompt, schema, cancel, progress)
    model = VerifierLimit({"claims": [candidate(), denied]})
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert result["status"] == "complete" and result["claims"] == 2 and len(model.calls) == 4


def test_source_model_or_relation_changes_invalidate_semantic_cache(corpus):
    store, inv, file_id = corpus
    model = Model()
    build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    for settings in (KGSettings(model="different-model"), KGSettings(relations=("works_for",))):
        result = build_graph(store, inv, [file_id], settings, model, Event(), lambda msg: None)
        assert not result["cached"] and not result["extraction_cached"]
    with store.db.transaction() as conn:
        conn.execute("UPDATE chunks SET chunk_text_hash='changed-source' WHERE id='c0'")
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert not result["cached"] and not result["extraction_cached"]


def test_legacy_completed_windows_survive_default_budget_upgrade(corpus):
    store, inv, file_id = corpus
    settings = KGSettings(output_tokens=2048)
    old = asdict(settings)
    old.pop("efficiency_version")
    run = store.start_run(inv, settings)
    with store.db.transaction() as conn:
        conn.execute("UPDATE kg_runs SET settings_json=? WHERE id=?", (json.dumps(old), run))
    window = make_windows(store.chunks(inv, [file_id]))[0]
    key = digest([LEGACY_PROMPT_VERSION, None, None, old,
                  [(c["id"], c["chunk_text_hash"], source_key(c), c["chunk_text"]) for c in window]])
    store.save_batch(inv, run, key, window, [candidate()], settings, '{"claims":[]}', "", {})
    store.review(inv, store.claims(inv)[0]["id"], "rejected", "Do not resurrect")
    model = Model()
    result = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None)
    assert result["cached"] == result["legacy_cached"] == 1
    assert model.calls == [] and store.claims(inv)[0]["review_note"] == "Do not resurrect"
    forced = build_graph(store, inv, [file_id], KGSettings(), model, Event(), lambda msg: None, force=True)
    assert forced["completed"] == 1 and len(model.calls) == 2


def test_split_sections_overlap_without_changing_source_identity(corpus):
    store, inv, file_id = large_window(corpus)
    window = store.chunks(inv, [file_id])
    left, right = split_window(window)
    original = window[0]
    assert all(part["chunk_text"] in original["chunk_text"] and part["id"] == original["id"]
               and part["chunk_text_hash"] == original["chunk_text_hash"] for child in (left, right) for part in child)
    assert len(left[0]["chunk_text"]) + len(right[0]["chunk_text"]) > len(original["chunk_text"])
    assert len(left[0]["chunk_text"]) < len(original["chunk_text"])


def test_compact_schema_mismatch_fails_instead_of_caching_empty_success(corpus):
    store, inv, file_id = corpus
    class WrongSchema(Model):
        def generate_kg(self, *args):
            return json.dumps({"claims": [candidate()]}), {}
    result = build_graph(store, inv, [file_id], KGSettings(), WrongSchema(), Event(), lambda msg: None)
    assert result["status"] == "partial" and not store.claims(inv)
    assert store.diagnostics(inv)["stage_cache"] == []


def test_claim_cap_recovers_instead_of_silently_dropping_dense_facts(corpus):
    store, inv, file_id = large_window(corpus)
    class DenseModel(TruncatingModel):
        def generate_kg(self, settings, prompt, schema, cancel, progress):
            if "claims" in schema["properties"]:
                units = json.loads(prompt.split("(JSON DATA):\n")[-1])
                if sum(len(text) for _, _, text in units) > 700:
                    self.calls.append(prompt)
                    return json.dumps({"claims": [wire() for _ in range(24)]}), {"eval_count": 1500}
            return super().generate_kg(settings, prompt, schema, cancel, progress)
    result = build_graph(store, inv, [file_id], KGSettings(), DenseModel(), Event(), lambda msg: None)
    assert result["status"] == "complete" and result["splits"] == 1
    assert len(store.claims(inv)) == 1 and result["sections_saved"] == 2


def test_literal_property_names_remain_in_original_language(corpus):
    store, inv, file_id = corpus
    window = store.chunks(inv, [file_id])[:1]
    window[0]["chunk_text"] = "Brug C heeft een hoogte van 15 m en een breedte van 15 m."
    base = wire(s="Brug C", st="object", r="has_measurement", o="15 m", ot="quantity")
    raw = json.dumps({"claims": [{**base, "k": name} for name in ("hoogte", "breedte", "height")]})
    good, rejected, _ = decode_claims(raw, window, KGSettings())
    assert {c["attribute"] for c in good} == {"hoogte", "breedte"} and len(rejected) == 1
    assert all(c["quote"] == window[0]["chunk_text"] for c in good)


def test_stream_limit_preserves_discarded_output_and_generation_metrics(monkeypatch):
    client = KGModelClient("http://127.0.0.1:11434")
    class Response:
        closed = False
        def raise_for_status(self):
            pass
        def iter_lines(self, chunk_size):
            yield json.dumps({"response": '{"claims":[', "done": True, "done_reason": "length",
                              "eval_count": 4096, "eval_duration": 123}).encode()
        def close(self):
            self.closed = True
    response = Response()
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises(KGOutputLimit) as error:
        client.generate_kg(KGSettings(), "Extract", {"type": "object"}, Event(), lambda msg: None)
    assert error.value.raw == '{"claims":['
    assert error.value.diagnostics["eval_count"] == 4096 and error.value.diagnostics["eval_duration"] == 123
    assert response.closed
    client.close()
