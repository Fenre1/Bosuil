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
    store = LanceDBStore.__new__(LanceDBStore)
    store.db_dir = None
    store.table_name = "chunks"
    store._db = FakeDB(table)

    store.upsert_chunks(
        [
            {"chunk_id": "one", "vector": [1.0]},
            {"chunk_id": "two", "vector": [2.0]},
        ]
    )

    assert table.deletes == ["chunk_id IN ('one', 'two')"]
    assert len(table.added) == 2


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


class FakeDB:
    def __init__(self, table):
        self.table = table

    def table_names(self):
        return ["chunks"]

    def open_table(self, name):
        assert name == "chunks"
        return self.table
