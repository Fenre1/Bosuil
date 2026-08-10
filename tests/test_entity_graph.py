import math

import numpy as np

from investigation_rag.gui.entity_graph import (
    MIN_ENTITY_DIAMETER,
    _rectangle_overlap_count,
    _reduce_to_2d,
    _resolve_rectangle_overlaps,
    build_entity_graph_model,
    entity_diameter,
    entity_type_color,
    format_entity_label,
    limit_membership_rows,
    limit_shared_chunk_links,
    shared_chunk_links,
)


def graph_row(entity_id, name, entity_type, chunk_id, chunk_index=0, path="doc.txt"):
    return {
        "entity_id": entity_id,
        "entity_name": name,
        "entity_type": entity_type,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_snippet": f"Text for {chunk_id}",
        "file_id": 1,
        "file_path": path,
        "source_type": "txt",
        "metadata_json": "{}",
    }


def test_entity_diameter_uses_reference_square_root_scaling():
    assert entity_diameter(0) == MIN_ENTITY_DIAMETER
    assert entity_diameter(1) == MIN_ENTITY_DIAMETER
    assert entity_diameter(100) == 1.0
    assert entity_diameter(10_000) == 10.0


def test_shared_links_are_unique_per_entity_and_chunk():
    memberships = {
        1: ["chunk-a", "chunk-a", "chunk-b"],
        2: ["chunk-a", "chunk-a", "chunk-b"],
        3: ["chunk-b", "chunk-c"],
    }

    assert shared_chunk_links(memberships, 1) == [
        (2, "chunk-a"),
        (2, "chunk-b"),
        (3, "chunk-b"),
    ]


def test_link_limit_round_robins_across_connected_entities():
    links = [
        (2, "chunk-a"),
        (2, "chunk-b"),
        (2, "chunk-c"),
        (3, "chunk-d"),
        (3, "chunk-e"),
    ]

    assert limit_shared_chunk_links(links, 4) == [
        (2, "chunk-a"),
        (3, "chunk-d"),
        (2, "chunk-b"),
        (3, "chunk-e"),
    ]


def test_chunk_limit_keeps_entity_coverage_then_connection_rich_chunks():
    rows = [
        graph_row(1, "Alice", "person", "first-a"),
        graph_row(1, "Alice", "person", "shared"),
        graph_row(1, "Alice", "person", "extra"),
        graph_row(2, "Bob", "person", "first-b"),
        graph_row(2, "Bob", "person", "shared"),
        graph_row(3, "ACME", "organization", "first-c"),
    ]

    limited, total = limit_membership_rows(rows, 4)

    assert total == 5
    assert {row["entity_id"] for row in limited} == {1, 2, 3}
    assert {row["chunk_id"] for row in limited} == {
        "first-a",
        "first-b",
        "first-c",
        "shared",
    }


def test_graph_model_collapses_memberships_but_keeps_distinct_chunks():
    rows = [
        graph_row(1, "Alice", "person", "chunk-a", 0),
        graph_row(1, "Alice", "person", "chunk-a", 0),
        graph_row(1, "Alice", "person", "chunk-b", 1),
        graph_row(2, "ACME", "organization", "chunk-a", 0),
        graph_row(2, "ACME", "organization", "chunk-b", 1),
    ]
    vectors = {
        "chunk-a": [1.0, 0.0, 0.0],
        "chunk-b": [0.0, 1.0, 0.0],
    }

    model = build_entity_graph_model(rows, vectors)

    assert model.entities[1].chunk_ids == ("chunk-a", "chunk-b")
    assert model.entities[2].chunk_ids == ("chunk-a", "chunk-b")
    assert len(model.chunks) == 2
    assert shared_chunk_links(
        {entity_id: entity.chunk_ids for entity_id, entity in model.entities.items()}, 1
    ) == [(2, "chunk-a"), (2, "chunk-b")]
    assert set(model.chunk_offsets) == {
        (1, "chunk-a"),
        (1, "chunk-b"),
        (2, "chunk-a"),
        (2, "chunk-b"),
    }


def test_graph_model_sizes_entities_by_true_count_when_chunks_are_sampled():
    rows = [graph_row(1, "Alice", "person", "sampled-chunk")]

    model = build_entity_graph_model(rows, {}, distinct_chunk_counts={1: 10_000})

    assert model.entities[1].chunk_ids == ("sampled-chunk",)
    assert model.entities[1].distinct_chunk_count == 10_000
    assert model.diameters[1] == 10.0


def test_graph_model_has_deterministic_structural_fallback_without_vectors():
    rows = [
        graph_row(1, "Alice", "person", "chunk-a"),
        graph_row(2, "ACME", "organization", "chunk-a"),
        graph_row(3, "Utrecht", "location", "chunk-b"),
    ]

    first = build_entity_graph_model(rows, {})
    second = build_entity_graph_model(rows, {})

    assert first.centers == second.centers
    assert first.chunk_offsets == second.chunk_offsets
    assert first.layout_method == "structural"
    assert "embeddings were unavailable" in first.warning
    for x, y in first.centers.values():
        assert math.isfinite(x)
        assert math.isfinite(y)


def test_entity_type_colors_are_stable_and_type_specific():
    assert entity_type_color("person") == entity_type_color("Person")
    assert entity_type_color("person") != entity_type_color("organization")


def test_organization_labels_wrap_to_two_balanced_lines():
    assert format_entity_label("Open Artificial Intelligence", "organization") == (
        "Open Artificial\nIntelligence"
    )
    assert format_entity_label("ABC DEF", "organisation") == "ABC\nDEF"
    assert format_entity_label("Ada Lovelace", "person") == "Ada Lovelace"


def test_long_single_words_break_at_fifteen_characters_without_losing_text():
    name = "North ABCDEFGHIJKLMNOPQRSTUVWXYZ South"

    label = format_entity_label(name, "organization")

    assert all(len(line) <= 15 for line in label.splitlines())
    assert "".join(label.split()) == "".join(name.split())


def test_embedding_layout_uses_local_neighborhood_manifold_projection():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    coords, method, warning = _reduce_to_2d(matrix, allow_manifold=True)

    assert coords.shape == (4, 2)
    assert np.isfinite(coords).all()
    assert method == "manifold"
    assert warning == ""


def test_rectangle_layout_separates_overlapping_entity_labels():
    positions = np.zeros((3, 2), dtype=np.float64)
    sizes = np.array([[40.0, 16.0], [60.0, 18.0], [30.0, 14.0]])

    resolved = _resolve_rectangle_overlaps(positions, sizes)

    assert _rectangle_overlap_count(resolved, sizes) == 0
    for first in range(len(resolved) - 1):
        for second in range(first + 1, len(resolved)):
            assert (
                abs(resolved[first, 0] - resolved[second, 0])
                >= (sizes[first, 0] + sizes[second, 0]) / 2.0
                or abs(resolved[first, 1] - resolved[second, 1])
                >= (sizes[first, 1] + sizes[second, 1]) / 2.0
            )
