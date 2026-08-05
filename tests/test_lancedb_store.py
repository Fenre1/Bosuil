from pathlib import Path
from types import SimpleNamespace

from investigation_rag.db.lancedb_store import LanceDBStore


def test_drop_deprecated_text_column_when_present():
    table = FakeTable(["chunk_id", "text", "vector"])

    LanceDBStore._drop_deprecated_text_column(table)

    assert table.dropped_columns == [["text"]]


def test_drop_deprecated_text_column_noops_when_absent():
    table = FakeTable(["chunk_id", "vector"])

    LanceDBStore._drop_deprecated_text_column(table)

    assert table.dropped_columns == []


def test_upsert_deletes_records_in_one_batch():
    table = FakeUpsertTable()
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    store.upsert_chunks(
        [
            {"chunk_id": "one", "vector": [1.0]},
            {"chunk_id": "two", "vector": [2.0]},
        ]
    )

    assert table.deletes == ["chunk_id IN ('one', 'two')"]
    assert len(table.added) == 2


def test_table_handle_is_opened_once_and_reused():
    table = FakeUpsertTable()
    db = FakeDB(table)
    store = LanceDBStore(Path("unused"))
    store._db = db

    assert store.has_table() is True
    store.upsert_chunks([{"chunk_id": "one", "vector": [1.0]}])
    assert store.has_table() is True

    assert db.open_calls == 1
    assert db.table_name_calls == 1


def test_delete_vectors_for_files_is_batched_and_scoped():
    table = FakeMaintenanceTable()
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    store.delete_vectors_for_files([3, 2, 3], investigation_id=7)

    assert table.deletes == ["file_id IN (2, 3) AND investigation_id = 7"]


def test_delete_file_wraps_the_batch_api():
    table = FakeMaintenanceTable()
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    store.delete_file(3)

    assert table.deletes == ["file_id IN (3)"]


def test_set_files_active_updates_vector_metadata_without_deleting():
    table = FakeMaintenanceTable()
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    updated = store.set_files_active([3, 2, 3], False, investigation_id=7)

    assert updated == 2
    assert table.updates == [
        ("file_id IN (2, 3) AND investigation_id = 7", {"is_active": False})
    ]
    assert table.deletes == []


def test_reconcile_active_chunks_removes_only_stale_ids():
    table = FakeMaintenanceTable(
        rows=[
            {"chunk_id": "keep", "is_active": True},
            {"chunk_id": "stale", "is_active": True},
            {"chunk_id": "pending-review", "is_active": False},
        ]
    )
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    removed = store.reconcile_active_chunks(7, ["keep"])

    assert removed == 1
    assert table.queries[0].where_clause == "investigation_id = 7"
    assert table.queries[0].selected == ["chunk_id", "is_active"]
    assert table.deletes == ["chunk_id IN ('stale') AND investigation_id = 7"]


def test_search_filters_inactive_vector_rows_when_column_is_available():
    table = FakeMaintenanceTable(rows=[{"chunk_id": "keep", "_distance": 0.1}])
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    rows = store.search([0.1], investigation_id=7, embedding_model="model", limit=1)

    assert rows == [{"chunk_id": "keep", "_distance": 0.1}]
    assert table.queries[0].where_clause == (
        "investigation_id = 7 AND is_active = true AND embedding_model = 'model'"
    )


def test_update_file_path_does_not_require_new_vectors():
    table = FakeMaintenanceTable()
    store = LanceDBStore(Path("unused"))
    store._db = FakeDB(table)

    updated = store.update_file_path(3, Path("C:/case/moved.txt"), investigation_id=7)

    assert updated == 2
    assert table.updates == [
        ("file_id = 3 AND investigation_id = 7", {"path": "C:\\case\\moved.txt"})
    ]


class FakeTable:
    def __init__(self, columns):
        self.schema = SimpleNamespace(names=columns)
        self.dropped_columns = []

    def drop_columns(self, columns):
        self.dropped_columns.append(columns)


class FakeUpsertTable(FakeTable):
    def __init__(self):
        super().__init__(["chunk_id", "vector"])
        self.deletes = []
        self.added = []

    def delete(self, predicate):
        self.deletes.append(predicate)

    def add(self, records):
        self.added.extend(records)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.where_clause = ""
        self.selected = None
        self.query_limit = None

    def where(self, where_clause):
        self.where_clause = where_clause
        return self

    def select(self, columns):
        self.selected = columns
        return self

    def limit(self, limit):
        self.query_limit = limit
        return self

    def to_list(self):
        return self.rows


class FakeMaintenanceTable(FakeTable):
    def __init__(self, rows=None):
        super().__init__([
            "chunk_id",
            "investigation_id",
            "file_id",
            "path",
            "source_type",
            "is_active",
            "embedding_model",
            "vector",
        ])
        self.rows = rows or []
        self.deletes = []
        self.queries = []
        self.updates = []

    def search(self, *args):
        query = FakeQuery(self.rows)
        self.queries.append(query)
        return query

    def delete(self, predicate):
        self.deletes.append(predicate)

    def update(self, where, values):
        self.updates.append((where, values))
        return SimpleNamespace(rows_updated=2)


class FakeDB:
    def __init__(self, table):
        self.table = table
        self.open_calls = 0
        self.table_name_calls = 0

    def table_names(self):
        self.table_name_calls += 1
        return ["chunks"]

    def open_table(self, name):
        assert name == "chunks"
        self.open_calls += 1
        return self.table
