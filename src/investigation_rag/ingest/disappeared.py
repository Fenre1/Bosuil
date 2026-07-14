from __future__ import annotations

from investigation_rag.db.sqlite_db import SQLiteDB


def queue_disappeared_files(db: SQLiteDB, investigation_id: int, seen_file_ids: set[int]) -> int:
    count = 0
    for row in db.active_files_for_investigation(investigation_id):
        file_id = int(row["id"])
        if file_id in seen_file_ids:
            continue
        db.add_disappeared_file(
            investigation_id=investigation_id,
            file_id=file_id,
            path=str(row["path"]),
            last_seen_at=str(row["last_seen_at"]),
        )
        count += 1
    return count
