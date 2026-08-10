from investigation_rag.db.sqlite_db import SQLiteDB, utc_now
from investigation_rag.entities import extract_entities, normalize_entity_name


class FakeGLiNER:
    def predict_entities(self, text, labels, threshold):
        results = []
        for name, label in (("Alice", "person"), ("ACME", "organization")):
            start = text.find(name)
            if start >= 0 and label in labels:
                results.append(
                    {"text": name, "label": label, "start": start, "end": start + len(name), "score": 0.9}
                )
        return results


def make_database(tmp_path):
    db = SQLiteDB(tmp_path / "entities.sqlite3")
    db.initialize()
    investigation_id = db.create_investigation("Case", [tmp_path])
    now = utc_now()
    with db.transaction() as conn:
        file_id = conn.execute(
            """
            INSERT INTO files(
              investigation_id, path, normalized_path, status, source_type,
              first_seen_at, last_seen_at, updated_at
            ) VALUES(?, ?, ?, 'active', 'txt', ?, ?, ?)
            """,
            (investigation_id, "doc.txt", "doc.txt", now, now, now),
        ).lastrowid
        for index, text in enumerate(("Alice works for ACME.", "Alice returned.")):
            conn.execute(
                """
                INSERT INTO chunks(
                  id, investigation_id, file_id, content_hash, chunk_index, chunk_text,
                  chunk_text_hash, source_type, path, extractor_version, chunker_version,
                  created_at, updated_at
                ) VALUES(?, ?, ?, 'hash', ?, ?, ?, 'txt', 'doc.txt', '1', '1', ?, ?)
                """,
                (f"chunk-{index}", investigation_id, file_id, index, text, f"hash-{index}", now, now),
            )
    return db, investigation_id


def add_entity_mention(db, investigation_id, name, entity_type):
    now = utc_now()
    with db.transaction() as conn:
        file_id = conn.execute(
            "SELECT id FROM files WHERE investigation_id = ? LIMIT 1", (investigation_id,)
        ).fetchone()["id"]
        entity_id = conn.execute(
            """
            INSERT INTO entities(
              investigation_id, name, normalized_name, entity_type, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (investigation_id, name, normalize_entity_name(name), entity_type, now, now),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO entity_mentions(
              entity_id, file_id, chunk_id, mention_text, start_offset,
              end_offset, confidence, created_at
            ) VALUES(?, ?, 'chunk-0', ?, 0, ?, 1.0, ?)
            """,
            (entity_id, file_id, name, len(name), now),
        )
    return entity_id


def test_extracts_unique_entities_and_retains_document_mentions(tmp_path):
    db, investigation_id = make_database(tmp_path)
    mention_count = extract_entities(
        db, investigation_id, ["person", "organization"], model=FakeGLiNER()
    )

    assert mention_count == 3
    entities = db.list_entities(investigation_id)
    assert [(row["name"], row["mention_count"], row["document_count"]) for row in entities] == [
        ("ACME", 1, 1),
        ("Alice", 2, 1),
    ]


def test_merge_survives_reextraction(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    entities = {row["name"]: row for row in db.list_entities(investigation_id)}
    db.merge_entities(investigation_id, entities["ACME"]["id"], entities["Alice"]["id"])

    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())

    remaining = db.list_entities(investigation_id)
    assert [(row["name"], row["mention_count"]) for row in remaining] == [("Alice", 3)]


def test_merges_multiple_entities_with_a_new_name_and_type(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    add_entity_mention(db, investigation_id, "A. Smith", "person")
    entities = db.list_entities(investigation_id)

    target_id = db.merge_entities_as(
        investigation_id,
        [row["id"] for row in entities],
        "Alice Smith",
        "Subject",
    )

    remaining = db.list_entities(investigation_id)
    assert [
        (row["id"], row["name"], row["entity_type"], row["mention_count"])
        for row in remaining
    ] == [(target_id, "Alice Smith", "subject", 4)]

    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    remaining = db.list_entities(investigation_id)
    assert [(row["name"], row["entity_type"], row["mention_count"]) for row in remaining] == [
        ("Alice Smith", "subject", 3)
    ]


def test_multi_merge_can_use_the_name_and_type_of_any_selected_entity(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    entities = {row["name"]: row for row in db.list_entities(investigation_id)}

    target_id = db.merge_entities_as(
        investigation_id,
        [entities["Alice"]["id"], entities["ACME"]["id"]],
        "ACME",
        "organization",
    )

    assert target_id == entities["ACME"]["id"]
    remaining = db.list_entities(investigation_id)
    assert [(row["name"], row["entity_type"], row["mention_count"]) for row in remaining] == [
        ("ACME", "organization", 3)
    ]


def test_entity_name_normalization_is_case_and_whitespace_insensitive():
    assert normalize_entity_name("  Alice\n Smith ") == "alice smith"


def test_list_entities_counts_only_mentions_from_active_chunks_and_files(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    with db.transaction() as conn:
        conn.execute("UPDATE chunks SET is_active = 0 WHERE id = 'chunk-1'")

    rows = db.list_entities(investigation_id)

    assert [(row["name"], row["mention_count"]) for row in rows] == [
        ("ACME", 1),
        ("Alice", 1),
    ]

    with db.transaction() as conn:
        conn.execute(
            "UPDATE files SET status = 'inactive' WHERE investigation_id = ?",
            (investigation_id,),
        )

    assert db.list_entities(investigation_id) == []


def test_entity_chunk_memberships_collapse_duplicate_mentions_in_one_chunk(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    alice = next(row for row in db.list_entities(investigation_id) if row["name"] == "Alice")
    now = utc_now()
    with db.transaction() as conn:
        file_id = conn.execute(
            "SELECT id FROM files WHERE investigation_id = ?", (investigation_id,)
        ).fetchone()["id"]
        conn.execute(
            """
            INSERT INTO entity_mentions(
              entity_id, file_id, chunk_id, mention_text, start_offset,
              end_offset, confidence, created_at
            ) VALUES(?, ?, 'chunk-0', 'Alice', 12, 17, 0.8, ?)
            """,
            (alice["id"], file_id, now),
        )

    rows = db.list_entity_chunk_memberships(investigation_id, [alice["id"]])

    assert [row["chunk_id"] for row in rows].count("chunk-0") == 1
    assert {row["entity_name"] for row in rows} == {"Alice"}


def test_entity_chunk_counts_collapse_mentions_and_ignore_inactive_sources(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    entities = {row["name"]: row for row in db.list_entities(investigation_id)}
    alice_id = entities["Alice"]["id"]
    acme_id = entities["ACME"]["id"]
    now = utc_now()
    with db.transaction() as conn:
        file_id = conn.execute(
            "SELECT id FROM files WHERE investigation_id = ?", (investigation_id,)
        ).fetchone()["id"]
        conn.execute(
            """
            INSERT INTO entity_mentions(
              entity_id, file_id, chunk_id, mention_text, start_offset,
              end_offset, confidence, created_at
            ) VALUES(?, ?, 'chunk-0', 'Alice', 12, 17, 0.8, ?)
            """,
            (alice_id, file_id, now),
        )

    assert db.count_entity_chunks(investigation_id, [alice_id, acme_id]) == {
        alice_id: 2,
        acme_id: 1,
    }

    with db.transaction() as conn:
        conn.execute("UPDATE chunks SET is_active = 0 WHERE id = 'chunk-1'")

    assert db.count_entity_chunks(investigation_id, [alice_id, acme_id]) == {
        alice_id: 1,
        acme_id: 1,
    }

    with db.transaction() as conn:
        conn.execute("UPDATE files SET status = 'inactive' WHERE id = ?", (file_id,))

    assert db.count_entity_chunks(investigation_id, [alice_id, acme_id, 999999]) == {
        alice_id: 0,
        acme_id: 0,
        999999: 0,
    }


def test_entity_chunk_memberships_keep_distinct_chunks_from_one_document(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    alice = next(row for row in db.list_entities(investigation_id) if row["name"] == "Alice")

    rows = db.list_entity_chunk_memberships(investigation_id, [alice["id"]])

    assert [row["chunk_id"] for row in rows] == ["chunk-0", "chunk-1"]
    assert len({row["file_id"] for row in rows}) == 1
    assert [(row["chunk_index"], row["chunk_text"]) for row in rows] == [
        (0, "Alice works for ACME."),
        (1, "Alice returned."),
    ]


def test_entity_chunk_memberships_use_the_merged_canonical_entity(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    entities = {row["name"]: row for row in db.list_entities(investigation_id)}
    source_id = entities["ACME"]["id"]
    target_id = entities["Alice"]["id"]
    db.merge_entities(investigation_id, source_id, target_id)

    rows = db.list_entity_chunk_memberships(
        investigation_id,
        [source_id, target_id],
    )

    assert [row["chunk_id"] for row in rows] == ["chunk-0", "chunk-1"]
    assert {row["entity_id"] for row in rows} == {target_id}
    assert {row["entity_name"] for row in rows} == {"Alice"}


def test_entity_chunk_memberships_with_no_entity_ids_returns_no_rows(tmp_path):
    db, investigation_id = make_database(tmp_path)

    assert db.list_entity_chunk_memberships(investigation_id, []) == []
    assert db.count_entity_chunks(investigation_id, []) == {}


def test_entity_chunk_membership_cap_round_robins_globally_across_id_batches(tmp_path):
    db, investigation_id = make_database(tmp_path)
    now = utc_now()
    entity_ids = []
    with db.transaction() as conn:
        file_id = conn.execute(
            "SELECT id FROM files WHERE investigation_id = ?", (investigation_id,)
        ).fetchone()["id"]
        for index in range(901):
            if index == 0:
                name = "Alpha first"
            elif index == 900:
                name = "Alpha second"
            else:
                name = f"Zulu {index:04d}"
            entity_id = conn.execute(
                """
                INSERT INTO entities(
                  investigation_id, name, normalized_name, entity_type, created_at, updated_at
                ) VALUES(?, ?, ?, 'person', ?, ?)
                """,
                (investigation_id, name, normalize_entity_name(name), now, now),
            ).lastrowid
            entity_ids.append(entity_id)
            conn.execute(
                """
                INSERT INTO entity_mentions(
                  entity_id, file_id, chunk_id, mention_text, start_offset,
                  end_offset, confidence, created_at
                ) VALUES(?, ?, 'chunk-0', ?, 0, 1, 1.0, ?)
                """,
                (entity_id, file_id, name, now),
            )
            if index == 0:
                conn.execute(
                    """
                    INSERT INTO entity_mentions(
                      entity_id, file_id, chunk_id, mention_text, start_offset,
                      end_offset, confidence, created_at
                    ) VALUES(?, ?, 'chunk-1', ?, 0, 1, 1.0, ?)
                    """,
                    (entity_id, file_id, name, now),
                )

    counts = db.count_entity_chunks(investigation_id, entity_ids)
    assert len(counts) == 901
    assert counts[entity_ids[0]] == 2
    assert all(counts[entity_id] == 1 for entity_id in entity_ids[1:])

    rows = db.list_entity_chunk_memberships(
        investigation_id,
        entity_ids,
        include_content=False,
        max_rows=2,
    )

    assert [(row["entity_name"], row["chunk_id"]) for row in rows] == [
        ("Alpha first", "chunk-0"),
        ("Alpha second", "chunk-0"),
    ]


def test_entity_chunk_membership_cap_accepts_zero_and_rejects_negative(tmp_path):
    db, investigation_id = make_database(tmp_path)

    assert db.list_entity_chunk_memberships(investigation_id, [1], max_rows=0) == []
    try:
        db.list_entity_chunk_memberships(investigation_id, [1], max_rows=-1)
    except ValueError as exc:
        assert str(exc) == "max_rows must be non-negative or None."
    else:
        raise AssertionError("Negative max_rows should be rejected.")


def test_entity_chunk_memberships_can_return_lightweight_graph_snippets(tmp_path):
    db, investigation_id = make_database(tmp_path)
    extract_entities(db, investigation_id, ["person", "organization"], model=FakeGLiNER())
    alice = next(row for row in db.list_entities(investigation_id) if row["name"] == "Alice")

    rows = db.list_entity_chunk_memberships(
        investigation_id,
        [alice["id"]],
        include_content=False,
    )

    assert rows[0]["chunk_snippet"] == "Alice works for ACME."
    assert "chunk_text" not in rows[0].keys()
    assert "display_markdown" not in rows[0].keys()
