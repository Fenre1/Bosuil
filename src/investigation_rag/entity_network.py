"""Complete entity membership index and evidence queries for graph exploration.

The index stores memberships, not a quadratic matrix of every entity pair.
Neighbours and evidence are calculated on demand, without sampling or limits.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable

from investigation_rag.db.sqlite_db import SQLiteDB, utc_now


@dataclass
class NetworkEntity:
    id: int
    name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    chunks: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class NetworkChunk:
    id: str
    document: str
    file_id: int
    path: str
    index: int
    page: int | None
    text_hash: str
    embedding_model: str


@dataclass(frozen=True)
class NetworkDocument:
    id: str
    file_id: int
    path: str
    identity: str
    content_key: str


@dataclass(frozen=True)
class Connection:
    first: int
    second: int
    chunks: frozenset[str]
    documents: frozenset[str]
    unique_passages: int
    unique_documents: int
    specificity: float


class EntityNetwork:
    def __init__(self, entities, chunks, documents, indexed_files=0):
        self.entities: dict[int, NetworkEntity] = entities
        self.chunks: dict[str, NetworkChunk] = chunks
        self.documents: dict[str, NetworkDocument] = documents
        self.indexed_files = indexed_files
        self.by_chunk: dict[str, set[int]] = defaultdict(set)
        self.by_document: dict[str, set[int]] = defaultdict(set)
        self.entity_documents: dict[int, set[str]] = defaultdict(set)
        for entity in entities.values():
            for chunk_id in entity.chunks:
                document = chunks[chunk_id].document
                self.by_chunk[chunk_id].add(entity.id)
                self.by_document[document].add(entity.id)
                self.entity_documents[entity.id].add(document)

    def scoped_chunks(self, entity_id: int, scope: frozenset[str] | None):
        chunks = self.entities[entity_id].chunks
        return chunks if scope is None else {c for c in chunks if self.chunks[c].document in scope}

    def eligible(self, scope=None, types=None):
        return {i for i, e in self.entities.items()
                if (types is None or e.entity_type in types)
                and (scope is None or self.entity_documents[i] & scope)}

    def connection(self, first: int, second: int, mode="passage", scope=None):
        a, b = self.scoped_chunks(first, scope), self.scoped_chunks(second, scope)
        shared = a & b
        if mode == "passage":
            documents = {self.chunks[c].document for c in shared}
            evidence = shared
        else:
            documents = {self.chunks[c].document for c in a} & {self.chunks[c].document for c in b}
            evidence = {c for c in a | b if self.chunks[c].document in documents}
        if not documents:
            return None
        # Exact file copies and repeated passage text do not increase these counts.
        doc_keys = {self.documents[d].content_key for d in documents}
        passage_keys = {self.chunks[c].text_hash or c for c in shared}
        a_keys = {self.documents[self.chunks[c].document].content_key for c in a}
        b_keys = {self.documents[self.chunks[c].document].content_key for c in b}
        specificity = len(doc_keys) / len(a_keys | b_keys)
        return Connection(min(first, second), max(first, second), frozenset(evidence),
                          frozenset(documents), len(passage_keys), len(doc_keys), specificity)

    def neighbors(self, entity_id: int, mode="passage", scope=None, types=None):
        candidates: set[int] = set()
        chunks = self.scoped_chunks(entity_id, scope)
        if mode == "passage":
            for chunk in chunks:
                candidates.update(self.by_chunk[chunk])
        else:
            for document in {self.chunks[c].document for c in chunks}:
                candidates.update(self.by_document[document])
        candidates.discard(entity_id)
        if types is not None:
            candidates = {i for i in candidates if self.entities[i].entity_type in types}
        return [self.connection(entity_id, other, mode, scope) for other in sorted(candidates)]

    def edges(self, visible: Iterable[int], mode="passage", scope=None, cancelled=lambda: False):
        visible = set(visible)
        pairs = set()
        buckets = self.by_chunk if mode == "passage" else self.by_document
        for key, members in buckets.items():
            if cancelled():
                raise InterruptedError()
            document = self.chunks[key].document if mode == "passage" else key
            if scope is not None and document not in scope:
                continue
            for index, pair in enumerate(combinations(sorted(members & visible), 2)):
                if index % 4096 == 0 and cancelled():
                    raise InterruptedError()
                pairs.add(pair)
        result = []
        for first, second in sorted(pairs):
            if cancelled():
                raise InterruptedError()
            result.append(self.connection(first, second, mode, scope))
        return result

    def groups(self, mode="passage", scope=None, types=None):
        """Connected components, computed directly from memberships in linear space."""
        eligible = self.eligible(scope, types)
        parent = {i: i for i in eligible}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        buckets = self.by_chunk if mode == "passage" else self.by_document
        for key, members in buckets.items():
            document = self.chunks[key].document if mode == "passage" else key
            if scope is not None and document not in scope:
                continue
            included = sorted(members & eligible)
            if included:
                for i in included[1:]:
                    parent[root(i)] = root(included[0])
        groups = defaultdict(set)
        for i in eligible:
            groups[root(i)].add(i)
        return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def rank_connections(connections, ranking="documents"):
    def key(c):
        score = c.specificity if ranking == "specificity" else (
            c.unique_passages if ranking == "passages" else c.unique_documents)
        return (-score, -c.unique_documents, -c.unique_passages, c.first, c.second)
    return sorted(connections, key=key)


def connection_layout(visible, edges, previous=None, cancelled=lambda: False):
    """Place new nodes beside connected nodes; retain all existing coordinates.

    A spatial grid avoids pairwise collision calculations. Components and breadth
    first traversal provide a deterministic connection-driven initial placement.
    """
    visible = set(visible)
    positions = {i: p for i, p in (previous or {}).items() if i in visible}
    occupied = {(round(x / 155), round(y / 85)) for x, y in positions.values()}
    adjacent = defaultdict(set)
    for edge in edges:
        adjacent[edge.first].add(edge.second)
        adjacent[edge.second].add(edge.first)
    order = sorted(visible, key=lambda i: (-len(adjacent[i]), i))
    queue = list(positions)
    queued = set(queue)
    for seed in order:
        if seed in queued:
            continue
        queue.append(seed)
        queued.add(seed)
        index = len(queue) - 1
        while index < len(queue):
            current = queue[index]
            for other in sorted(adjacent[current]):
                if other not in queued:
                    queued.add(other)
                    queue.append(other)
            index += 1
    empty_cursor = 0
    radius_by_center = {}
    columns = max(1, math.ceil(math.sqrt(len(visible))))
    for i in queue:
        if cancelled():
            raise InterruptedError()
        if i in positions:
            continue
        neighbors = [positions[j] for j in adjacent[i] if j in positions]
        if not neighbors:
            while (empty_cursor % columns, empty_cursor // columns) in occupied:
                empty_cursor += 1
            cell = (empty_cursor % columns, empty_cursor // columns)
            occupied.add(cell)
            positions[i] = (cell[0] * 155.0, cell[1] * 85.0)
            empty_cursor += 1
            continue
        cx = round(sum(p[0] for p in neighbors) / len(neighbors) / 155) if neighbors else 0
        cy = round(sum(p[1] for p in neighbors) / len(neighbors) / 85) if neighbors else 0
        # Square spiral, with an entity-specific angle for varied neighbourhoods.
        radius = radius_by_center.get((cx, cy), 0)
        while True:
            cells = [(cx, cy)] if radius == 0 else (
                [(cx + x, cy - radius) for x in range(-radius, radius + 1)]
                + [(cx + radius, cy + y) for y in range(-radius + 1, radius + 1)]
                + [(cx + x, cy + radius) for x in range(radius - 1, -radius - 1, -1)]
                + [(cx - radius, cy + y) for y in range(radius - 1, -radius, -1)])
            free = next((cell for cell in cells if cell not in occupied), None)
            if free is not None:
                occupied.add(free)
                positions[i] = (free[0] * 155.0, free[1] * 85.0)
                radius_by_center[(cx, cy)] = radius
                break
            radius += 1
    return positions


def separate_topic_positions(positions, fixed=None, cancelled=lambda: False):
    """Keep coincident semantic contexts selectable without moving retained nodes."""
    output = {i: p for i, p in (fixed or {}).items() if i in positions}
    occupied = {(round(x / 155), round(y / 85)) for x, y in output.values()}
    radii = {}
    for i, (x, y) in positions.items():
        if cancelled():
            raise InterruptedError()
        if i in output:
            continue
        cx, cy = round(x / 155), round(y / 85)
        radius = radii.get((cx, cy), 0)
        while True:
            cells = [(cx, cy)] if radius == 0 else (
                [(cx + dx, cy + dy) for dx in range(-radius, radius + 1) for dy in (-radius, radius)]
                + [(cx + dx, cy + dy) for dy in range(-radius + 1, radius) for dx in (-radius, radius)])
            cell = next((p for p in cells if p not in occupied), None)
            if cell is not None:
                occupied.add(cell)
                output[i] = (cell[0] * 155.0, cell[1] * 85.0)
                radii[(cx, cy)] = radius
                break
            radius += 1
    return output


def load_network(db: SQLiteDB, investigation_id: int, cancelled: Callable = lambda: False):
    # One read transaction prevents a scan from mixing old chunks and new mentions.
    with db.connection() as conn:
        conn.set_progress_handler(lambda: int(cancelled()), 10000)
        conn.execute("BEGIN")
        entity_rows = conn.execute("SELECT * FROM entities WHERE investigation_id = ?", (investigation_id,)).fetchall()
        entities = {int(r["id"]): NetworkEntity(int(r["id"]), r["name"], r["entity_type"])
                    for r in entity_rows if r["merged_into_id"] is None}
        aliases = defaultdict(list)
        targets = {int(r["id"]): r["merged_into_id"] for r in entity_rows}
        for row in entity_rows:
            target = row["merged_into_id"]
            seen = set()
            while target is not None and target not in seen and targets.get(target) is not None:
                seen.add(target)
                target = targets[target]
            if target in entities:
                aliases[target].append(row["name"])
        for i, values in aliases.items():
            entities[i].aliases = tuple(sorted(set(values)))
        chunk_rows = conn.execute("""
            SELECT c.id, c.file_id, c.path, c.chunk_index, c.page_number, c.metadata_json,
                   c.attachment_name, c.chunk_text_hash, c.embedding_model, f.content_hash
            FROM chunks c JOIN files f ON f.id = c.file_id AND f.investigation_id = c.investigation_id
            WHERE c.investigation_id = ? AND c.is_active = 1 AND f.status = 'active'
        """, (investigation_id,))
        chunks, documents, files = {}, {}, set()
        for row in chunk_rows:
            if cancelled():
                raise InterruptedError()
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (ValueError, TypeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            identity = str(metadata.get("source_document_id") or row["attachment_name"] or "root")
            document = f"{row['file_id']}:{identity}"
            files.add(row["file_id"])
            documents[document] = NetworkDocument(document, row["file_id"], row["path"], identity,
                f"{row['content_hash'] or 'file:' + str(row['file_id'])}:{identity}")
            chunks[row["id"]] = NetworkChunk(row["id"], document, row["file_id"], row["path"],
                row["chunk_index"], row["page_number"], row["chunk_text_hash"], row["embedding_model"])
        for row in conn.execute("""
            SELECT DISTINCT em.entity_id, em.chunk_id, em.file_id FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE e.investigation_id = ? AND e.merged_into_id IS NULL
        """, (investigation_id,)):
            if cancelled():
                raise InterruptedError()
            if row["chunk_id"] in chunks and chunks[row["chunk_id"]].file_id == row["file_id"]:
                entities[row["entity_id"]].chunks.add(row["chunk_id"])
        entities = {i: e for i, e in entities.items() if e.chunks}
        return EntityNetwork(entities, chunks, documents, len(files))


class ExplorationRepository:
    def __init__(self, db):
        self.db = db

    def save(self, investigation_id, name, state):
        with self.db.transaction() as conn:
            conn.execute("""INSERT INTO entity_graph_views(investigation_id, name, state_json, updated_at)
                VALUES(?, ?, ?, ?) ON CONFLICT(investigation_id, name) DO UPDATE SET
                state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (investigation_id, name, json.dumps(state), utc_now()))

    def saved(self, investigation_id):
        with self.db.connection() as conn:
            return {r["name"]: json.loads(r["state_json"]) for r in conn.execute(
                "SELECT name, state_json FROM entity_graph_views WHERE investigation_id = ? ORDER BY name COLLATE NOCASE",
                (investigation_id,))}

    @staticmethod
    def _snapshot(conn, investigation_id):
        return {"entities": [dict(r) for r in conn.execute(
                    "SELECT * FROM entities WHERE investigation_id = ? ORDER BY id", (investigation_id,))],
                "entity_mentions": [dict(r) for r in conn.execute("""SELECT em.* FROM entity_mentions em
                    JOIN entities e ON e.id = em.entity_id WHERE e.investigation_id = ? ORDER BY em.id""", (investigation_id,))]}

    def edit(self, investigation_id, ids, name, entity_type):
        # Reuse the DB session's nested transactions to keep edit and undo atomic.
        with self.db.scan_session(investigation_id):
            with self.db.transaction() as conn:
                before = self._snapshot(conn, investigation_id)
                if len(ids) > 1:
                    target = self.db.merge_entities_as(investigation_id, ids, name, entity_type)
                else:
                    target = ids[0]
                    row = conn.execute("SELECT * FROM entities WHERE id=? AND investigation_id=? AND merged_into_id IS NULL",
                                       (target, investigation_id)).fetchone()
                    if row is None or not entity_type.strip():
                        raise ValueError("Select an active entity and enter its type.")
                    new_type = entity_type.strip().casefold()
                    if new_type != row["entity_type"]:
                        conflict = conn.execute("""SELECT id, merged_into_id FROM entities
                            WHERE investigation_id=? AND normalized_name=? AND entity_type=?""",
                            (investigation_id, row["normalized_name"], new_type)).fetchone()
                        if conflict is not None:
                            if conflict["merged_into_id"] != target:
                                raise ValueError("That name and type already belongs to another entity. Merge the entities instead.")
                            conn.execute("DELETE FROM entities WHERE id=?", (conflict["id"],))
                        conn.execute("UPDATE entities SET entity_type=?, updated_at=? WHERE id=?", (new_type, utc_now(), target))
                        conn.execute("""INSERT INTO entities(investigation_id,name,normalized_name,entity_type,
                            merged_into_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?)""",
                            (investigation_id,row["name"],row["normalized_name"],row["entity_type"],target,utc_now(),utc_now()))
                after = self._snapshot(conn, investigation_id)
                conn.execute("""INSERT INTO entity_graph_edits(investigation_id,before_json,after_json,created_at)
                    VALUES(?,?,?,?)""", (investigation_id,json.dumps(before),json.dumps(after),utc_now()))
                return target

    def undo_edit(self, investigation_id):
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM entity_graph_edits WHERE investigation_id=? ORDER BY id DESC LIMIT 1",
                               (investigation_id,)).fetchone()
            if row is None:
                raise ValueError("There is no graph entity correction to undo.")
            if self._snapshot(conn, investigation_id) != json.loads(row["after_json"]):
                raise ValueError("Entity data changed after this correction. Undo would overwrite newer work; review the entities instead.")
            before = json.loads(row["before_json"])
            conn.execute("DELETE FROM entity_mentions WHERE entity_id IN (SELECT id FROM entities WHERE investigation_id=?)", (investigation_id,))
            conn.execute("UPDATE entities SET merged_into_id=NULL WHERE investigation_id=?", (investigation_id,))
            conn.execute("DELETE FROM entities WHERE investigation_id=?", (investigation_id,))
            # Restore canonical rows before their aliases (foreign key targets).
            pending = list(before["entities"])
            restored = set()
            while pending:
                ready = [r for r in pending if r["merged_into_id"] is None or r["merged_into_id"] in restored]
                if not ready:
                    raise ValueError("Cannot restore cyclic aliases.")
                for r in ready:
                    conn.execute(f"INSERT INTO entities({','.join(r)}) VALUES({','.join('?' for _ in r)})", tuple(r.values()))
                    restored.add(r["id"])
                    pending.remove(r)
            for r in before["entity_mentions"]:
                conn.execute(f"INSERT INTO entity_mentions({','.join(r)}) VALUES({','.join('?' for _ in r)})", tuple(r.values()))
            conn.execute("DELETE FROM entity_graph_edits WHERE id=?", (row["id"],))
