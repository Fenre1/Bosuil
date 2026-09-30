from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping

import numpy as np


MIN_ENTITY_DIAMETER = 0.5
ENTITY_SIZE_SCALER = 0.1
LAYOUT_SCALE = 10.0
MAX_LAYOUT_CHUNKS = 5_000
MAX_MANIFOLD_LANDMARKS = 2_000

TYPE_COLORS = (
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#dc2626",
    "#4f46e5",
    "#65a30d",
    "#c026d3",
    "#0f766e",
    "#b45309",
    "#475569",
)


@dataclass(frozen=True)
class GraphChunk:
    chunk_id: str
    snippet: str
    file_id: int
    path: str
    chunk_index: int
    source_type: str
    metadata_json: str


@dataclass(frozen=True)
class GraphEntity:
    entity_id: int
    name: str
    entity_type: str
    chunk_ids: tuple[str, ...]
    distinct_chunk_count: int


@dataclass(frozen=True)
class EntityGraphModel:
    entities: dict[int, GraphEntity]
    chunks: dict[str, GraphChunk]
    centers: dict[int, tuple[float, float]]
    diameters: dict[int, float]
    chunk_offsets: dict[tuple[int, str], tuple[float, float]]
    layout_method: str
    warning: str = ""


def entity_diameter(distinct_chunk_count: int) -> float:
    """Use the reference view's square-root scaling so large entities stay readable."""
    return max(math.sqrt(max(0, distinct_chunk_count)) * ENTITY_SIZE_SCALER, MIN_ENTITY_DIAMETER)


def entity_type_color(entity_type: str) -> str:
    digest = sha256(entity_type.strip().casefold().encode("utf-8")).digest()
    return TYPE_COLORS[int.from_bytes(digest[:2], "big") % len(TYPE_COLORS)]


def format_entity_label(name: str, entity_type: str) -> str:
    """Return an explicit, compact multiline label without losing any text."""
    compact_name = " ".join(name.split())
    if not compact_name:
        return ""
    words = compact_name.split(" ")
    if any(len(word) > 15 for word in words):
        tokens = [
            word[start : start + 15]
            for word in words
            for start in range(0, len(word), 15)
        ]
        lines: list[str] = []
        current = ""
        for token in tokens:
            candidate = f"{current} {token}".strip()
            if current and len(candidate) > 15:
                lines.append(current)
                current = token
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    normalized_type = entity_type.strip().casefold()
    is_organization = normalized_type in {"organization", "organisation", "org"}
    if is_organization and len(compact_name) > 5 and len(words) > 1:
        split_at = min(
            range(1, len(words)),
            key=lambda index: abs(
                len(" ".join(words[:index])) - len(" ".join(words[index:]))
            ),
        )
        return f"{' '.join(words[:split_at])}\n{' '.join(words[split_at:])}"
    return compact_name


def shared_chunk_links(
    memberships: Mapping[int, Iterable[str]],
    selected_entity_id: int,
) -> list[tuple[int, str]]:
    """Return one link descriptor per other entity and shared chunk."""
    selected = set(memberships.get(selected_entity_id, ()))
    links: list[tuple[int, str]] = []
    for entity_id in sorted(memberships):
        if entity_id == selected_entity_id:
            continue
        for chunk_id in sorted(selected & set(memberships[entity_id])):
            links.append((entity_id, chunk_id))
    return links


def limit_shared_chunk_links(
    links: Iterable[tuple[int, str]],
    maximum: int,
) -> list[tuple[int, str]]:
    """Limit detail items while retaining coverage across connected entities."""
    if maximum <= 0:
        return []
    grouped: dict[int, list[str]] = {}
    for entity_id, chunk_id in links:
        grouped.setdefault(entity_id, []).append(chunk_id)
    positions = {entity_id: 0 for entity_id in grouped}
    limited: list[tuple[int, str]] = []
    active = sorted(grouped)
    while active and len(limited) < maximum:
        next_active: list[int] = []
        for entity_id in active:
            position = positions[entity_id]
            chunk_ids = grouped[entity_id]
            if position >= len(chunk_ids):
                continue
            limited.append((entity_id, chunk_ids[position]))
            positions[entity_id] = position + 1
            if positions[entity_id] < len(chunk_ids):
                next_active.append(entity_id)
            if len(limited) >= maximum:
                break
        active = next_active
    return limited


def limit_membership_rows(
    rows: Iterable[Mapping[str, Any]],
    maximum_chunks: int,
) -> tuple[list[Mapping[str, Any]], int]:
    """Choose a deterministic, connection-rich subset of chunks for a large graph.

    One first chunk per entity is retained before remaining chunks are ranked by
    how many of the requested entities they connect. This avoids letting one
    prolific entity crowd every other visible entity out of a bounded graph.
    """
    source_rows = list(rows)
    chunk_entities: dict[str, set[int]] = {}
    chunk_first_seen: dict[str, int] = {}
    first_chunk_by_entity: dict[int, str] = {}
    for index, row in enumerate(source_rows):
        chunk_id = str(row["chunk_id"])
        entity_id = int(row["entity_id"])
        chunk_entities.setdefault(chunk_id, set()).add(entity_id)
        chunk_first_seen.setdefault(chunk_id, index)
        first_chunk_by_entity.setdefault(entity_id, chunk_id)

    total_chunks = len(chunk_entities)
    if maximum_chunks <= 0:
        return [], total_chunks
    if total_chunks <= maximum_chunks:
        return source_rows, total_chunks

    selected: list[str] = []
    selected_set: set[str] = set()
    for chunk_id in first_chunk_by_entity.values():
        if chunk_id not in selected_set:
            selected.append(chunk_id)
            selected_set.add(chunk_id)
        if len(selected) >= maximum_chunks:
            break
    if len(selected) < maximum_chunks:
        remaining = sorted(
            (chunk_id for chunk_id in chunk_entities if chunk_id not in selected_set),
            key=lambda chunk_id: (
                -len(chunk_entities[chunk_id]),
                chunk_first_seen[chunk_id],
                chunk_id,
            ),
        )
        selected.extend(remaining[: maximum_chunks - len(selected)])
        selected_set.update(selected)
    return [row for row in source_rows if str(row["chunk_id"]) in selected_set], total_chunks


def build_entity_graph_model(
    rows: Iterable[Mapping[str, Any]],
    vectors_by_chunk: Mapping[str, Iterable[float]],
    *,
    source_warning: str = "",
    distinct_chunk_counts: Mapping[int, int] | None = None,
) -> EntityGraphModel:
    entity_meta: dict[int, tuple[str, str]] = {}
    memberships: dict[int, set[str]] = {}
    chunks: dict[str, GraphChunk] = {}
    for row in rows:
        entity_id = int(row["entity_id"])
        chunk_id = str(row["chunk_id"])
        entity_meta[entity_id] = (str(row["entity_name"]), str(row["entity_type"]))
        memberships.setdefault(entity_id, set()).add(chunk_id)
        if chunk_id not in chunks:
            chunks[chunk_id] = GraphChunk(
                chunk_id=chunk_id,
                snippet=_compact_text(str(_row_value(row, "chunk_snippet", ""))),
                file_id=int(row["file_id"]),
                path=str(_row_value(row, "file_path", _row_value(row, "path", ""))),
                chunk_index=int(row["chunk_index"]),
                source_type=str(_row_value(row, "source_type", "")),
                metadata_json=str(_row_value(row, "metadata_json", "{}") or "{}"),
            )

    entities = {
        entity_id: GraphEntity(
            entity_id=entity_id,
            name=entity_meta[entity_id][0],
            entity_type=entity_meta[entity_id][1],
            chunk_ids=tuple(
                sorted(
                    chunk_ids,
                    key=lambda chunk_id: (
                        chunks[chunk_id].path.casefold(),
                        chunks[chunk_id].chunk_index,
                        chunk_id,
                    ),
                )
            ),
            distinct_chunk_count=max(
                len(chunk_ids),
                int((distinct_chunk_counts or {}).get(entity_id, len(chunk_ids))),
            ),
        )
        for entity_id, chunk_ids in memberships.items()
    }
    if not entities:
        return EntityGraphModel({}, {}, {}, {}, {}, "none", source_warning)

    clean_vectors = _clean_vectors(vectors_by_chunk, chunks)
    warnings = [source_warning] if source_warning else []
    chunk_ids = sorted(chunks)
    chunk_xy: dict[str, np.ndarray] = {}
    layout_methods: list[str] = []

    valid_chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id in clean_vectors]
    if valid_chunk_ids:
        matrix = np.stack([clean_vectors[chunk_id] for chunk_id in valid_chunk_ids])
        allow_manifold = len(valid_chunk_ids) <= MAX_LAYOUT_CHUNKS
        coords, method, warning = _reduce_to_2d(matrix, allow_manifold=allow_manifold)
        layout_methods.append(method)
        if warning:
            warnings.append(warning)
        if not allow_manifold:
            warnings.append(
                f"More than {MAX_LAYOUT_CHUNKS:,} chunks matched; internal chunk placement used "
                "a bounded projection."
            )
        for chunk_id, coord in zip(valid_chunk_ids, coords):
            chunk_xy[chunk_id] = coord

    missing_chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_xy]
    if missing_chunk_ids:
        fallback = _radial_points(len(missing_chunk_ids), radius=1.0)
        for chunk_id, coord in zip(missing_chunk_ids, fallback):
            chunk_xy[chunk_id] = coord
        if not valid_chunk_ids:
            warnings.append("Stored chunk embeddings were unavailable; placement is structural.")
            layout_methods.append("structural")
        else:
            warnings.append(
                f"{len(missing_chunk_ids):,} chunks had no compatible stored vector and use fallback positions."
            )

    entity_ids = list(entities)
    if clean_vectors:
        dimension = len(next(iter(clean_vectors.values())))
        global_mean = np.mean(np.stack(list(clean_vectors.values())), axis=0)
        centroids = []
        for index, entity_id in enumerate(entity_ids):
            member_vectors = [
                clean_vectors[chunk_id]
                for chunk_id in entities[entity_id].chunk_ids
                if chunk_id in clean_vectors
            ]
            if member_vectors:
                centroids.append(np.mean(np.stack(member_vectors), axis=0))
            else:
                jitter = np.zeros(dimension, dtype=np.float32)
                jitter[index % dimension] = 1e-4
                centroids.append(global_mean + jitter)
        center_coords, method, warning = _reduce_to_2d(
            np.stack(centroids), allow_manifold=True
        )
        layout_methods.append(method)
        if warning:
            warnings.append(warning)
    else:
        center_coords = _radial_points(len(entity_ids), radius=LAYOUT_SCALE)

    diameters = {
        entity_id: entity_diameter(entities[entity_id].distinct_chunk_count)
        for entity_id in entity_ids
    }
    radii = np.array(
        [diameters[entity_id] / 2.0 for entity_id in entity_ids], dtype=np.float64
    )
    adaptive_scale = max(
        LAYOUT_SCALE,
        1.5 * math.sqrt(float(np.sum(np.square(radii + 0.04)))),
    )
    center_coords = _normalize_coordinates(center_coords, adaptive_scale)
    center_coords = _resolve_overlaps(
        center_coords,
        radii,
    )
    centers = {
        entity_id: (float(center_coords[index, 0]), float(center_coords[index, 1]))
        for index, entity_id in enumerate(entity_ids)
    }

    chunk_offsets: dict[tuple[int, str], tuple[float, float]] = {}
    for entity_id, entity in entities.items():
        member_coords = np.stack([chunk_xy[chunk_id] for chunk_id in entity.chunk_ids])
        member_coords = member_coords - member_coords.mean(axis=0)
        norms = np.linalg.norm(member_coords, axis=1)
        max_norm = float(norms.max()) if len(norms) else 0.0
        if max_norm <= 1e-9:
            member_coords = _radial_points(len(entity.chunk_ids), radius=1.0)
            max_norm = 1.0
        radius = diameters[entity_id] * 0.39
        member_coords = member_coords / max_norm * radius
        for chunk_id, coord in zip(entity.chunk_ids, member_coords):
            chunk_offsets[(entity_id, chunk_id)] = (float(coord[0]), float(coord[1]))

    method = "+".join(dict.fromkeys(layout_methods)) or "structural"
    return EntityGraphModel(
        entities=entities,
        chunks=chunks,
        centers=centers,
        diameters=diameters,
        chunk_offsets=chunk_offsets,
        layout_method=method,
        warning=" ".join(dict.fromkeys(value for value in warnings if value)),
    )


def _clean_vectors(
    vectors_by_chunk: Mapping[str, Iterable[float]],
    chunks: Mapping[str, GraphChunk],
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    dimensions: Counter[int] = Counter()
    for chunk_id, values in vectors_by_chunk.items():
        if chunk_id not in chunks:
            continue
        try:
            vector = np.asarray(list(values), dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
            continue
        candidates[str(chunk_id)] = vector
        dimensions[len(vector)] += 1
    if not dimensions:
        return {}
    dimension = dimensions.most_common(1)[0][0]
    return {chunk_id: vector for chunk_id, vector in candidates.items() if len(vector) == dimension}


def _reduce_to_2d(
    matrix: np.ndarray,
    *,
    allow_manifold: bool,
) -> tuple[np.ndarray, str, str]:
    matrix = np.asarray(matrix, dtype=np.float32)
    count = len(matrix)
    if count <= 1:
        return np.zeros((count, 2), dtype=np.float64), "single", ""
    if count == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64), "pair", ""
    if count > MAX_LAYOUT_CHUNKS:
        return (
            _variance_projection_2d(matrix),
            "projection",
            f"Layout input exceeded {MAX_LAYOUT_CHUNKS:,} vectors; used a bounded projection.",
        )
    if allow_manifold and count >= 4:
        try:
            coords = _manifold_2d(matrix)
            if coords.shape == (count, 2) and np.isfinite(coords).all():
                return coords, "manifold", ""
        except Exception as exc:
            warning = f"Neighborhood placement was unavailable ({exc}); used PCA fallback."
        else:
            warning = "Neighborhood placement returned an invalid layout; used PCA fallback."
    else:
        warning = ""
    return _pca_2d(matrix), "pca", warning


def _manifold_2d(matrix: np.ndarray) -> np.ndarray:
    """Local-neighborhood embedding with scalable landmark interpolation.

    This gives the UMAP-style semantic grouping the view needs without a
    runtime JIT. Large layouts embed deterministic landmarks and place the
    remaining chunks from their nearest landmark neighborhoods.
    """
    import os
    import warnings

    # New Windows installations may not include WMIC, which joblib otherwise
    # probes just to discover the physical CPU count. We use one worker below;
    # supplying that count avoids a noisy, failed subprocess probe.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    from sklearn.decomposition import PCA
    from sklearn.manifold import SpectralEmbedding

    matrix = np.asarray(matrix, dtype=np.float32)
    count, dimension = matrix.shape
    if dimension > 32:
        components = min(32, count - 1, dimension)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            matrix = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=42,
            ).fit_transform(matrix)
    matrix = _normalize_rows(matrix)

    landmark_count = min(count, MAX_MANIFOLD_LANDMARKS)
    if landmark_count == count:
        landmark_indices = np.arange(count)
    else:
        rng = np.random.default_rng(42)
        landmark_indices = np.sort(rng.choice(count, landmark_count, replace=False))
    landmarks = matrix[landmark_indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        landmark_coords = np.asarray(
            SpectralEmbedding(
                n_components=2,
                affinity="nearest_neighbors",
                n_neighbors=min(15, landmark_count - 1),
                random_state=42,
                n_jobs=1,
            ).fit_transform(landmarks),
            dtype=np.float64,
        )
    if landmark_count == count:
        return landmark_coords

    coords = np.empty((count, 2), dtype=np.float64)
    coords[landmark_indices] = landmark_coords
    remaining_mask = np.ones(count, dtype=bool)
    remaining_mask[landmark_indices] = False
    remaining_indices = np.flatnonzero(remaining_mask)
    neighbor_count = min(5, landmark_count)
    for start in range(0, len(remaining_indices), 256):
        batch_indices = remaining_indices[start : start + 256]
        similarities = matrix[batch_indices] @ landmarks.T
        nearest = np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
        nearest_similarity = np.take_along_axis(similarities, nearest, axis=1)
        weights = np.exp((nearest_similarity - nearest_similarity.max(axis=1, keepdims=True)) * 8)
        weights /= weights.sum(axis=1, keepdims=True)
        coords[batch_indices] = np.sum(
            landmark_coords[nearest] * weights[:, :, None], axis=1
        )
    return coords


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _variance_projection_2d(matrix: np.ndarray) -> np.ndarray:
    """Cheap deterministic fallback that never performs an unbounded SVD."""
    centered = np.asarray(matrix, dtype=np.float64) - np.mean(matrix, axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return np.column_stack((centered[:, 0], np.zeros(len(centered))))
    variances = np.var(centered, axis=0)
    dimensions = np.argsort(variances, kind="stable")[-2:]
    coords = centered[:, dimensions]
    if not np.isfinite(coords).all() or np.max(np.linalg.norm(coords, axis=1)) <= 1e-12:
        return _radial_points(len(matrix), radius=1.0)
    return coords


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = np.asarray(matrix, dtype=np.float64) - np.mean(matrix, axis=0, keepdims=True)
    if not np.any(centered):
        return _radial_points(len(matrix), radius=1.0)
    if centered.size > 1_000_000:
        try:
            import os

            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
            from sklearn.decomposition import PCA

            coords = PCA(
                n_components=2,
                svd_solver="randomized",
                random_state=42,
            ).fit_transform(centered)
            if np.isfinite(coords).all():
                return np.asarray(coords, dtype=np.float64)
        except Exception:
            return _variance_projection_2d(centered)
    try:
        u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * singular_values[:2]
    except np.linalg.LinAlgError:
        return _radial_points(len(matrix), radius=1.0)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    if not np.isfinite(coords).all() or np.max(np.linalg.norm(coords, axis=1)) <= 1e-12:
        return _radial_points(len(matrix), radius=1.0)
    return coords


def _normalize_coordinates(coords: np.ndarray, scale: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if not len(coords):
        return coords.reshape((0, 2))
    coords = coords - coords.mean(axis=0)
    raw_scale = float(np.max(np.abs(coords)))
    return coords * (scale / raw_scale) if raw_scale > 1e-12 else coords


def _radial_points(count: int, radius: float) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    if count == 1:
        return np.zeros((1, 2), dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    return np.array(
        [
            (
                math.cos(index * golden_angle) * radius * math.sqrt((index + 1) / count),
                math.sin(index * golden_angle) * radius * math.sqrt((index + 1) / count),
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )


def _resolve_overlaps(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    iterations: int = 80,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64).copy()
    count = len(positions)
    if count < 2:
        return positions - positions.mean(axis=0) if count else positions
    required = radii[:, None] + radii[None, :] + 0.08
    upper_triangle = np.triu(np.ones((count, count), dtype=bool), 1)
    for _ in range(iterations):
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        overlapping = upper_triangle & (distances < required - 1e-7)
        if not np.any(overlapping):
            break
        for first, second in np.argwhere(overlapping):
            delta = positions[first] - positions[second]
            distance = float(np.linalg.norm(delta))
            minimum = float(required[first, second])
            if distance >= minimum:
                continue
            if distance <= 1e-12:
                angle = (first * 37 + second * 101) * math.pi / 180.0
                direction = np.array([math.cos(angle), math.sin(angle)])
            else:
                direction = delta / distance
            push = direction * ((minimum - distance) * 0.51)
            positions[first] += push
            positions[second] -= push
    return positions - positions.mean(axis=0) if len(positions) else positions


def _resolve_rectangle_overlaps(
    positions: np.ndarray,
    sizes: np.ndarray,
    *,
    gap: float = 2.0,
    iterations: int = 100,
) -> np.ndarray:
    """Relax axis-aligned label rectangles while preserving semantic proximity."""
    positions = np.asarray(positions, dtype=np.float64).copy()
    sizes = np.asarray(sizes, dtype=np.float64)
    count = len(positions)
    if count < 2:
        return positions - positions.mean(axis=0) if count else positions
    half_widths = sizes[:, 0] / 2.0
    half_heights = sizes[:, 1] / 2.0
    required_x = half_widths[:, None] + half_widths[None, :] + gap
    required_y = half_heights[:, None] + half_heights[None, :] + gap
    upper_triangle = np.triu(np.ones((count, count), dtype=bool), 1)
    for _ in range(iterations):
        deltas = positions[:, None, :] - positions[None, :, :]
        overlap_x = required_x - np.abs(deltas[:, :, 0])
        overlap_y = required_y - np.abs(deltas[:, :, 1])
        overlapping = upper_triangle & (overlap_x > 1e-6) & (overlap_y > 1e-6)
        if not np.any(overlapping):
            break
        for first, second in np.argwhere(overlapping):
            delta = positions[first] - positions[second]
            x_depth = float(required_x[first, second] - abs(delta[0]))
            y_depth = float(required_y[first, second] - abs(delta[1]))
            if x_depth <= 0 or y_depth <= 0:
                continue
            if x_depth < y_depth:
                direction = 1.0 if delta[0] >= 0 else -1.0
                if abs(delta[0]) <= 1e-12:
                    direction = 1.0 if (first + second) % 2 else -1.0
                push = np.array([direction * x_depth * 0.51, 0.0])
            else:
                direction = 1.0 if delta[1] >= 0 else -1.0
                if abs(delta[1]) <= 1e-12:
                    direction = 1.0 if (first * 3 + second) % 2 else -1.0
                push = np.array([0.0, direction * y_depth * 0.51])
            positions[first] += push
            positions[second] -= push
    return positions - positions.mean(axis=0)


def _rectangle_overlap_count(
    positions: np.ndarray,
    sizes: np.ndarray,
    *,
    gap: float = 2.0,
) -> int:
    if len(positions) < 2:
        return 0
    positions = np.asarray(positions, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    deltas = np.abs(positions[:, None, :] - positions[None, :, :])
    required = (sizes[:, None, :] + sizes[None, :, :]) / 2.0 + gap
    overlaps = (deltas[:, :, 0] < required[:, :, 0] - 1e-6) & (
        deltas[:, :, 1] < required[:, :, 1] - 1e-6
    )
    return int(np.triu(overlaps, 1).sum())


def _row_value(row: Mapping[str, Any], key: str, default: Any) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


# Public UI entry points; the legacy projection helpers remain available for topic layouts.
from investigation_rag.gui.entity_explorer import EntityGraphPanel, NetworkView as EntityGraphView
