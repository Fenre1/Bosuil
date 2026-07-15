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


class FakeTable:
    def __init__(self, columns):
        self.schema = SimpleNamespace(names=columns)
        self.dropped_columns = []

    def drop_columns(self, columns):
        self.dropped_columns.append(columns)
