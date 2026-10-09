"""Portable KG JSON, preserving imported evidence without manufacturing source chunks."""
from __future__ import annotations

import ntpath
from threading import Event
from uuid import uuid4

from investigation_rag.db.sqlite_db import utc_now
from investigation_rag.knowledge_graph import KGCancelled, NODE_TYPES, RELATIONS, digest

FORMAT = "bosuil-knowledge-graph-v2"


def export_graph(store, investigation_id: int) -> dict:
    nodes = store.nodes(investigation_id)
    claims = store.claims(investigation_id, include_stale=True)
    with store.db.transaction() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='knowledge_graph_identity'").fetchone()
        identity = row[0] if row else str(uuid4())
        if not row:
            conn.execute("INSERT INTO settings VALUES('knowledge_graph_identity',?,?)", (identity, utc_now()))
        origins = {}
        for node in nodes:
            key = f"{identity}/node/{node['id']}"
            conn.execute("INSERT OR IGNORE INTO kg_import_nodes VALUES(?,?,?)", (investigation_id, key, node["id"]))
        for row in conn.execute("SELECT portable_key,node_id FROM kg_import_nodes WHERE investigation_id=?", (investigation_id,)):
            origins.setdefault(row["node_id"], []).append(row["portable_key"])
        imported = {row["native_claim_id"]: row["portable_key"] for row in conn.execute(
            "SELECT native_claim_id,portable_key FROM kg_imported_claims WHERE investigation_id=? AND native_claim_id IS NOT NULL",
            (investigation_id,))}
    for node in nodes:
        node["portable_ids"] = sorted(origins.get(node["id"], []))
    for claim in claims:
        claim["portable_id"] = claim.get("portable_key") or imported.get(claim["id"]) or f"{identity}/claim/{claim['id']}"
    return {"format": FORMAT, "graph_id": identity, "investigation_id": investigation_id,
            "exported_at": utc_now(), "nodes": nodes, "claims": claims,
            "note": "Source claims retain review and evidence status. Imported quotes may have no linked local source. Acceptance is review, not proof."}


def _text(item, key, *, required=False, default=""):
    value = item.get(key, default)
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Invalid graph field: {key} must be {'nonempty text' if required else 'text'}.")
    return value


def _portable_key(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("Invalid portable graph identity.")
    return value


def validate_graph(payload: dict) -> tuple[dict, list[dict]]:
    if not isinstance(payload, dict) or payload.get("format") not in ("bosuil-knowledge-graph-v1", FORMAT):
        raise ValueError("Choose a Bosuil knowledge graph JSON export (v1 or v2).")
    if not isinstance(payload.get("nodes"), list) or not isinstance(payload.get("claims"), list):
        raise ValueError("Graph export must contain nodes and claims arrays.")
    identity = payload.get("graph_id") or "legacy-" + digest([
        payload.get("investigation_id"),
        sorted((str(n.get("id")) for n in payload["nodes"] if isinstance(n, dict))),
        sorted(tuple(str(c.get(k, "")) for k in ("claim_key", "chunk_id", "source_hash", "path"))
               for c in payload["claims"] if isinstance(c, dict)),
        sorted((str(n.get("name")), str(n.get("node_type"))) for n in payload["nodes"] if isinstance(n, dict)) if not payload["claims"] else [],
    ])
    _portable_key(identity)
    nodes, keys = {}, {}
    for item in payload["nodes"]:
        if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] <= 0 or item["id"] in nodes:
            raise ValueError("Graph nodes must have unique positive integer IDs.")
        name, node_type = _text(item, "name", required=True), _text(item, "node_type", required=True)
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type}.")
        origins = item.get("portable_ids", [f"{identity}/node/{item['id']}"])
        if not isinstance(origins, list) or not origins:
            raise ValueError("Node portable_ids must be a nonempty list.")
        for key in origins:
            _portable_key(key)
            if key in keys and keys[key] != item["id"]:
                raise ValueError("Multiple exported nodes claim the same portable identity.")
            keys[key] = item["id"]
        nodes[item["id"]] = {"name": name, "node_type": node_type, "origins": list(set(origins))}
    claims, ids, claim_keys = [], set(), set()
    for item in payload["claims"]:
        if not isinstance(item, dict) or type(item.get("id")) is not int or not item["id"] or item["id"] in ids:
            raise ValueError("Graph claims must have unique nonzero integer IDs.")
        ids.add(item["id"])
        record = dict(item)
        for role in ("subject", "object"):
            node_id = item.get(role + "_id")
            if type(node_id) is not int or node_id not in nodes:
                raise ValueError("A graph claim refers to an unknown node.")
            if item.get(role, nodes[node_id]["name"]) != nodes[node_id]["name"] or item.get(role + "_type", nodes[node_id]["node_type"]) != nodes[node_id]["node_type"]:
                raise ValueError("Claim endpoint names/types disagree with their exported nodes.")
        for key in ("predicate", "polarity", "quote", "chunk_id", "source_hash", "path", "verification", "review_status"):
            record[key] = _text(item, key, required=True)
        if record["predicate"] not in RELATIONS or record["polarity"] not in ("affirmed", "negated", "uncertain"):
            raise ValueError("Unknown relationship or polarity in graph export.")
        if record["review_status"] not in ("suggested", "accepted", "rejected"):
            raise ValueError("Unknown graph review status.")
        for key in ("attribution", "time_text", "attribute", "review_note", "model"):
            record[key] = _text(item, key)
        for key, default in (("chunk_index", 0), ("page_number", None), ("run_id", None)):
            value = item.get(key, default)
            if value is not None and (type(value) is not int or (key != "run_id" and value < 0)):
                raise ValueError(f"Invalid graph field: {key}.")
            if key == "chunk_index" and value is None:
                raise ValueError("Missing source chunk index.")
            record[key] = value
        if item.get("attachment_name") is not None and not isinstance(item["attachment_name"], str):
            raise ValueError("Invalid attachment name.")
        record["attachment_name"] = item.get("attachment_name")
        current = item.get("exported_source_current", item.get("source_current", 0))
        if type(current) not in (int, bool) or current not in (0, 1):
            raise ValueError("Invalid evidence currency flag.")
        record["exported_source_current"] = int(current)
        record["portable_key"] = _portable_key(item.get("portable_id", f"{identity}/claim/{item['id']}"))
        record["claim_key"] = _text(item, "claim_key") or digest([identity, item["id"]])
        record["created_at"] = _text(item, "created_at") or utc_now()
        if record["portable_key"] in claim_keys:
            raise ValueError("Duplicate portable claim identity.")
        claim_keys.add(record["portable_key"])
        claims.append(record)
    return nodes, claims


def import_graph(store, investigation_id: int, payload: dict, *, cancel: Event | None = None, progress=lambda message: None) -> dict:
    cancel = cancel or Event()
    def check_cancel():
        if cancel.is_set():
            raise KGCancelled("Knowledge graph import cancelled; no changes saved.")
    check_cancel()
    nodes, claims = validate_graph(payload)
    result = {"operation": "import", "nodes": 0, "claims": 0, "existing": 0, "linked": 0, "unlinked": 0}
    with store.db.transaction() as conn:
        if not conn.execute("SELECT 1 FROM investigations WHERE id=?", (investigation_id,)).fetchone():
            raise ValueError("Load an investigation before importing a graph.")
        local_nodes = {r["portable_key"]: r["node_id"] for r in conn.execute(
            "SELECT portable_key,node_id FROM kg_import_nodes WHERE investigation_id=?", (investigation_id,))}
        existing = {r["portable_key"]: dict(r) for r in conn.execute("SELECT * FROM kg_imported_claims WHERE investigation_id=?", (investigation_id,))}
        native = {r["claim_key"]: dict(r) for r in conn.execute("SELECT * FROM kg_claims WHERE investigation_id=?", (investigation_id,))}
        identity_row = conn.execute("SELECT value FROM settings WHERE key='knowledge_graph_identity'").fetchone()
        native_portable = {f"{identity_row[0]}/claim/{r['id']}": r for r in native.values()} if identity_row else {}
        by_id, by_path = {}, {}
        source_query = "SELECT c.*,f.status AS file_status FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.investigation_id=? "
        def resolve(record):
            if record["chunk_id"] not in by_id:
                row = conn.execute(source_query + "AND c.id=?", (investigation_id, record["chunk_id"])).fetchone()
                by_id[record["chunk_id"]] = dict(row) if row else None
            source = by_id[record["chunk_id"]]
            if source and source["chunk_text_hash"] == record["source_hash"] and record["quote"] in source["chunk_text"]:
                return source
            path = ntpath.normcase(ntpath.normpath(record["path"]))
            key = (path, record["source_hash"], record["chunk_index"], record["attachment_name"])
            if key not in by_path:
                by_path[key] = [dict(r) for r in conn.execute(source_query + "AND c.chunk_text_hash=? AND c.chunk_index=? AND c.attachment_name IS ?",
                                                            (investigation_id, record["source_hash"], record["chunk_index"], record["attachment_name"]))
                                if ntpath.normcase(ntpath.normpath(r["path"])) == path]
            options = [c for c in by_path[key] if record["quote"] in c["chunk_text"]]
            return options[0] if len(options) == 1 else None
        preferred, linked_sources, native_matches = {}, {}, {}
        for index, record in enumerate(claims):
            check_cancel()
            source = resolve(record)
            linked_sources[record["portable_key"]] = source
            native_claim = native_portable.get(record["portable_key"]) or native.get(record["claim_key"])
            same_source = bool(native_claim and native_claim["chunk_id"] == record["chunk_id"] and
                               native_claim["source_hash"] == record["source_hash"] and native_claim["quote"] == record["quote"])
            if record["portable_key"] in native_portable and not same_source:
                raise ValueError("An imported claim identity conflicts with existing evidence. No changes saved.")
            native_matches[record["portable_key"]] = native_claim if same_source else None
            known = existing.get(record["portable_key"]) or native_matches[record["portable_key"]]
            if known and (record["portable_key"] in existing or
                          (known["chunk_id"] == record["chunk_id"] and known["source_hash"] == record["source_hash"] and known["quote"] == record["quote"])):
                for role in ("subject", "object"):
                    preferred.setdefault(record[role + "_id"], set()).add(known[role + "_id"])
            if index % 200 == 0:
                progress(f"Checking imported evidence {index + 1}/{len(claims)}")
        mapping = {}
        for external_id, node in nodes.items():
            check_cancel()
            choices = {local_nodes[key] for key in node["origins"] if key in local_nodes} | preferred.get(external_id, set())
            if len(choices) > 1:
                raise ValueError("Imported identities conflict with existing nodes. No nodes were merged or overwritten.")
            if choices:
                node_id = choices.pop()
            else:
                node_id = conn.execute("INSERT INTO kg_nodes(investigation_id,name,node_type,created_at) VALUES(?,?,?,?)",
                                       (investigation_id, node["name"], node["node_type"], utc_now())).lastrowid
                result["nodes"] += 1
            mapping[external_id] = node_id
            for key in node["origins"]:
                conn.execute("INSERT OR IGNORE INTO kg_import_nodes VALUES(?,?,?)", (investigation_id, key, node_id))
        columns = ("investigation_id", "portable_key", "claim_key", "subject_id", "object_id", "predicate", "polarity",
                   "attribution", "time_text", "attribute", "chunk_id", "linked_chunk_id", "native_claim_id", "path", "chunk_index",
                   "page_number", "attachment_name", "quote", "source_hash", "exported_source_current", "model", "verification",
                   "review_status", "review_note", "original_run_id", "created_at")
        for index, record in enumerate(claims):
            check_cancel()
            source = linked_sources[record["portable_key"]]
            current = bool(source and source["is_active"] and source["file_status"] == "active")
            result["linked" if current else "unlinked"] += 1
            if record["portable_key"] in existing:
                # Reimport may attach evidence, but never overwrite local reviews/corrections.
                saved = existing[record["portable_key"]]
                if source and saved["source_hash"] == record["source_hash"] and saved["quote"] == record["quote"]:
                    conn.execute("UPDATE kg_imported_claims SET linked_chunk_id=? WHERE id=?", (source["id"], saved["id"]))
                result["existing"] += 1
                continue
            known = native_matches[record["portable_key"]]
            native_id = known["id"] if known else None
            values = {**record, "investigation_id": investigation_id,
                      "subject_id": mapping[record["subject_id"]], "object_id": mapping[record["object_id"]],
                      "linked_chunk_id": source["id"] if source else None, "native_claim_id": native_id,
                      "original_run_id": record["run_id"]}
            if native_id:
                # A portable snapshot of an existing claim follows the receiving case's latest review.
                for key in ("predicate", "polarity", "attribution", "time_text", "review_status", "review_note", "model", "verification"):
                    values[key] = known[key]
                attribute = conn.execute("SELECT attribute_name FROM kg_claim_attributes WHERE claim_id=?", (native_id,)).fetchone()
                values["attribute"] = attribute[0] if attribute else ""
            conn.execute("INSERT INTO kg_imported_claims(" + ",".join(columns) + ") VALUES(" + ",".join("?" for _ in columns) + ")",
                         [values[c] for c in columns])
            result["existing" if native_id else "claims"] += 1
            if index % 200 == 0:
                progress(f"Importing relationships {index + 1}/{len(claims)}")
        check_cancel()
    return result
