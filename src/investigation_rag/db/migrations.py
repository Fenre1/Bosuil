from __future__ import annotations

from investigation_rag.db.sqlite_db import SQLiteDB


def initialize_database(db: SQLiteDB) -> None:
    db.initialize()
