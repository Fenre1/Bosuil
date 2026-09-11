from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from threading import Event, Thread
from typing import Any, Iterable, Mapping

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.gui.chunk_preview import ChunkPreview


MIN_ENTITY_DIAMETER = 0.5
ENTITY_SIZE_SCALER = 0.1
LAYOUT_SCALE = 10.0
MAX_LAYOUT_CHUNKS = 5_000
DEFAULT_GRAPH_CHUNKS = 2_000
MAX_GRAPH_MEMBERSHIPS = 50_000
MAX_SELECTED_LINKS = 2_500
MAX_UNSHARED_SELECTED_CHUNKS = 2_000
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


_BUILD_CANCELLED = "__entity_graph_build_cancelled__"


class EntityGraphBuildEmitter(QObject):
    finished = pyqtSignal(int, object, str)


def _build_entity_graph_request(
    emitter: EntityGraphBuildEmitter,
    cancelled: Event,
    generation: int,
    db: SQLiteDB,
    store: LanceDBStore | None,
    investigation_id: int,
    entity_ids: list[int],
    maximum_chunks: int,
) -> None:
    """Build one immutable request on a daemon thread and report through Qt."""
    try:
        count_chunks = getattr(db, "count_entity_chunks", None)
        distinct_chunk_counts = (
            count_chunks(investigation_id, entity_ids) if callable(count_chunks) else {}
        )
        if cancelled.is_set():
            _emit_graph_result(emitter, generation, None, _BUILD_CANCELLED)
            return
        rows = db.list_entity_chunk_memberships(
            investigation_id,
            entity_ids,
            include_content=False,
            max_rows=MAX_GRAPH_MEMBERSHIPS,
        )
        if cancelled.is_set():
            _emit_graph_result(emitter, generation, None, _BUILD_CANCELLED)
            return

        warnings: list[str] = []
        true_memberships = sum(distinct_chunk_counts.values())
        membership_limited = true_memberships > len(rows)
        if membership_limited:
            warnings.append(
                f"Membership safety limit applied: using {len(rows):,} of "
                f"{true_memberships:,} entity-chunk memberships. Narrow the entity filters "
                "for a complete graph."
            )
        rows, total_chunks = limit_membership_rows(rows, maximum_chunks)
        included_chunk_ids = list(dict.fromkeys(str(row["chunk_id"]) for row in rows))
        if total_chunks > len(included_chunk_ids):
            qualifier = "at least " if membership_limited else ""
            warnings.append(
                f"Chunk limit applied: using {len(included_chunk_ids):,} of {qualifier}"
                f"{total_chunks:,} matched chunks."
            )

        vectors: dict[str, list[float]] = {}
        model_by_chunk = {
            str(row["chunk_id"]): str(_row_value(row, "embedding_model", "") or "")
            for row in rows
        }
        model_counts = Counter(model for model in model_by_chunk.values() if model)
        embedding_model = ""
        vector_chunk_ids: list[str] = []
        if model_counts:
            embedding_model = sorted(
                model_counts,
                key=lambda model: (-model_counts[model], model.casefold(), model),
            )[0]
            vector_chunk_ids = [
                chunk_id
                for chunk_id in included_chunk_ids
                if model_by_chunk.get(chunk_id) == embedding_model
            ]
            if len(vector_chunk_ids) < len(included_chunk_ids):
                warnings.append(
                    f"Semantic placement uses embedding model {embedding_model!r} for "
                    f"{len(vector_chunk_ids):,} of {len(included_chunk_ids):,} chunks; "
                    "chunks from other or missing models use structural positions."
                )
        elif included_chunk_ids:
            warnings.append("Matched chunks do not have a recorded embedding model.")

        if cancelled.is_set():
            _emit_graph_result(emitter, generation, None, _BUILD_CANCELLED)
            return
        if store is not None and vector_chunk_ids:
            try:
                vectors = store.get_vectors_for_chunks(
                    vector_chunk_ids,
                    investigation_id=investigation_id,
                    embedding_model=embedding_model,
                )
            except Exception as exc:
                warnings.append(f"Stored vectors could not be loaded ({exc}).")
        elif store is None:
            warnings.append("No vector store is available.")

        if cancelled.is_set():
            _emit_graph_result(emitter, generation, None, _BUILD_CANCELLED)
            return
        model = build_entity_graph_model(
            rows,
            vectors,
            source_warning=" ".join(warnings),
            distinct_chunk_counts=distinct_chunk_counts,
        )
        _emit_graph_result(
            emitter,
            generation,
            model,
            _BUILD_CANCELLED if cancelled.is_set() else "",
        )
    except Exception as exc:
        _emit_graph_result(emitter, generation, None, str(exc))


def _emit_graph_result(
    emitter: EntityGraphBuildEmitter,
    generation: int,
    model: EntityGraphModel | None,
    error: str,
) -> None:
    try:
        emitter.finished.emit(generation, model, error)
    except RuntimeError:
        # The panel can be destroyed while a daemon build finishes during app exit.
        return


class EntityRectangleItem(QGraphicsRectItem):
    def __init__(self, entity_id: int, rect: QRectF):
        super().__init__(rect)
        self.entity_id = entity_id


class ChunkNodeItem(QGraphicsEllipseItem):
    def __init__(self, entity_id: int, chunk_id: str, rect: QRectF):
        super().__init__(rect)
        self.entity_id = entity_id
        self.chunk_id = chunk_id


class EntityGraphView(QGraphicsView):
    chunk_selected = pyqtSignal(str)
    entity_selected = pyqtSignal(int)
    detail_status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor("#f8fafc"))
        self.setMouseTracking(True)
        self.model: EntityGraphModel | None = None
        self.minimum_font_size = 9
        self.entity_items: dict[int, EntityRectangleItem] = {}
        self.entity_labels: dict[int, QGraphicsSimpleTextItem] = {}
        self.entity_centers: dict[int, tuple[float, float]] = {}
        self.entity_sizes: dict[int, tuple[float, float]] = {}
        self.node_items: list[ChunkNodeItem] = []
        self.link_items: list[QGraphicsLineItem] = []
        self.selected_entity_id: int | None = None
        self.selected_chunk_id: str | None = None
        self._hovered_entity_id: int | None = None
        self._tooltip_key: tuple[str, object] | None = None

    def set_model(self, model: EntityGraphModel | None) -> None:
        self.scene().clear()
        self.model = model
        self.entity_items.clear()
        self.entity_labels.clear()
        self.entity_centers.clear()
        self.entity_sizes.clear()
        self.node_items.clear()
        self.link_items.clear()
        self.selected_entity_id = None
        self.selected_chunk_id = None
        self._hovered_entity_id = None
        if model is None:
            return
        font = QFont(self.font())
        font.setPointSize(self.minimum_font_size)
        entity_ids = list(model.entities)
        sizes: list[tuple[float, float]] = []
        labels: list[QGraphicsSimpleTextItem] = []
        for entity_id in entity_ids:
            entity = model.entities[entity_id]
            diameter = model.diameters[entity_id]
            padding = max(self.minimum_font_size * 0.35, self.minimum_font_size * diameter * 0.28)
            label = QGraphicsSimpleTextItem(format_entity_label(entity.name, entity.entity_type))
            label.setFont(font)
            label.setBrush(QBrush(QColor("#0f172a")))
            label.setAcceptedMouseButtons(Qt.NoButton)
            label_bounds = label.boundingRect()
            text_width = label_bounds.width()
            text_height = label_bounds.height()
            count_extent = self.minimum_font_size * diameter * 0.75
            width = max(text_width + padding * 2.0, count_extent, text_height + padding * 2.0)
            height = max(text_height + padding * 2.0, count_extent)
            sizes.append((width, height))
            labels.append(label)
        size_array = np.asarray(sizes, dtype=np.float64)
        raw_centers = np.asarray([model.centers[entity_id] for entity_id in entity_ids])
        base_scale = max(
            40.0,
            math.sqrt(float(np.sum(size_array[:, 0] * size_array[:, 1]))) * 2.0,
        )
        gap = max(2.0, self.minimum_font_size * 0.25)
        for expansion in (1.0, 1.5, 2.25, 3.5):
            positions = _normalize_coordinates(raw_centers, base_scale * expansion)
            positions = _resolve_rectangle_overlaps(positions, size_array, gap=gap)
            if _rectangle_overlap_count(positions, size_array, gap=gap) == 0:
                break

        for index, entity_id in enumerate(entity_ids):
            entity = model.entities[entity_id]
            width, height = sizes[index]
            center = (float(positions[index, 0]), float(positions[index, 1]))
            item = EntityRectangleItem(
                entity_id,
                QRectF(-width / 2.0, -height / 2.0, width, height),
            )
            color = QColor(entity_type_color(entity.entity_type))
            fill = QColor(color)
            fill.setAlpha(105)
            item.setBrush(QBrush(fill))
            item.setPen(QPen(color.darker(125), 0.5))
            item.setPos(QPointF(*center))
            item.setZValue(-10)
            self.scene().addItem(item)
            self.entity_items[entity_id] = item
            self.entity_centers[entity_id] = center
            self.entity_sizes[entity_id] = (width, height)

            label = labels[index]
            label_bounds = label.boundingRect()
            label.setPos(
                center[0] - label_bounds.width() / 2.0,
                center[1] - label_bounds.height() / 2.0,
            )
            label.setZValue(30)
            self.scene().addItem(label)
            self.entity_labels[entity_id] = label
        self._update_scene_rect()
        QTimer.singleShot(0, self.reset_view)

    def set_minimum_font_size(self, point_size: int) -> None:
        point_size = max(1, int(point_size))
        if point_size == self.minimum_font_size:
            return
        self.minimum_font_size = point_size
        if self.model is None:
            return
        model = self.model
        selected_entity_id = self.selected_entity_id
        selected_chunk_id = self.selected_chunk_id
        self.set_model(model)
        if selected_entity_id in model.entities:
            self.blockSignals(True)
            try:
                self.select_entity(selected_entity_id)
                if selected_chunk_id is not None:
                    self.highlight_chunk(selected_chunk_id)
            finally:
                self.blockSignals(False)

    def reset_view(self) -> None:
        bounds = self.scene().itemsBoundingRect()
        if bounds.isValid() and not bounds.isEmpty():
            self.fitInView(bounds.adjusted(-0.7, -0.7, 0.7, 0.7), Qt.KeepAspectRatio)

    def select_entity(self, entity_id: int | None) -> None:
        if self.model is None or entity_id not in self.model.entities:
            self.selected_entity_id = None
            self._clear_detail_items()
            self._style_entity_items(set())
            self.entity_selected.emit(-1)
            self.detail_status_changed.emit("")
            return

        self.selected_entity_id = entity_id
        self.selected_chunk_id = None
        self._clear_detail_items()
        memberships = {
            current_id: set(entity.chunk_ids) for current_id, entity in self.model.entities.items()
        }
        all_links = shared_chunk_links(memberships, entity_id)
        links = limit_shared_chunk_links(all_links, MAX_SELECTED_LINKS)
        connected_ids = {other_id for other_id, _ in all_links}
        connected_ids.add(entity_id)
        self._style_entity_items(connected_ids)

        selected_members = set(memberships[entity_id])
        all_shared_members = {chunk_id for _, chunk_id in all_links}
        shown_shared_members = {chunk_id for _, chunk_id in links}
        unshared = sorted(selected_members - all_shared_members)
        shown_selected = shown_shared_members | set(unshared[:MAX_UNSHARED_SELECTED_CHUNKS])

        positions: dict[tuple[int, str], QPointF] = {}
        for chunk_id in sorted(shown_selected):
            positions[(entity_id, chunk_id)] = self._add_chunk_node(entity_id, chunk_id)
        for other_id, chunk_id in links:
            key = (other_id, chunk_id)
            if key not in positions:
                positions[key] = self._add_chunk_node(other_id, chunk_id)
            start = positions.get((entity_id, chunk_id))
            if start is None:
                start = self._add_chunk_node(entity_id, chunk_id)
                positions[(entity_id, chunk_id)] = start
            line = QGraphicsLineItem(start.x(), start.y(), positions[key].x(), positions[key].y())
            line.setPen(QPen(QColor(71, 85, 105, 125), 0.35))
            line.setZValue(0)
            self.scene().addItem(line)
            self.link_items.append(line)

        entity = self.model.entities[entity_id]
        omitted = len(selected_members) - len(shown_selected)
        message = f"{entity.name}: {entity.distinct_chunk_count:,} distinct chunks"
        if entity.distinct_chunk_count != len(selected_members):
            message += f" ({len(selected_members):,} retained in this bounded graph)"
        message += (
            f", {len(all_links):,} retained chunk-level links to "
            f"{len(connected_ids) - 1:,} entities"
        )
        omitted_links = len(all_links) - len(links)
        omissions = []
        if omitted_links:
            omissions.append(f"{omitted_links:,} links")
        if omitted:
            omissions.append(f"{omitted:,} unlinked chunk markers")
        if omissions:
            message += " · display safety limit omitted " + " and ".join(omissions)
        self.entity_selected.emit(entity_id)
        self.detail_status_changed.emit(message)

    def highlight_chunk(self, chunk_id: str) -> None:
        self.selected_chunk_id = chunk_id
        for item in self.node_items:
            if item.chunk_id == chunk_id:
                item.setBrush(QColor("#dc2626"))
                item.setPen(QPen(QColor("#7f1d1d"), 0.5))
            else:
                item.setBrush(QColor("#ffffff"))
                item.setPen(QPen(QColor("#334155"), 0.3))

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            items = self.items(event.pos())
            node = next((item for item in items if isinstance(item, ChunkNodeItem)), None)
            if node is not None:
                self.highlight_chunk(node.chunk_id)
                self.chunk_selected.emit(node.chunk_id)
                event.accept()
                return
            entity = next((item for item in items if isinstance(item, EntityRectangleItem)), None)
            if entity is not None:
                self.select_entity(entity.entity_id)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.itemAt(event.pos()) is None:
            self.select_entity(None)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:
        items = self.items(event.pos())
        item = next((value for value in items if isinstance(value, ChunkNodeItem)), None)
        if item is None:
            item = next((value for value in items if isinstance(value, EntityRectangleItem)), None)
        hovered_entity_id = item.entity_id if isinstance(
            item, (ChunkNodeItem, EntityRectangleItem)
        ) else None
        self._set_hovered_entity(hovered_entity_id)
        tooltip_key: tuple[str, object] | None = None
        tooltip = ""
        if self.model is not None and isinstance(item, ChunkNodeItem):
            chunk = self.model.chunks[item.chunk_id]
            tooltip_key = ("chunk", item.chunk_id)
            tooltip = (
                f"<b>{escape(Path(chunk.path).name)}</b>"
                f" · chunk {chunk.chunk_index + 1}<br>{escape(chunk.snippet[:280])}"
            )
        elif self.model is not None and isinstance(item, EntityRectangleItem):
            entity = self.model.entities[item.entity_id]
            tooltip_key = ("entity", item.entity_id)
            tooltip = (
                f"<b>{escape(entity.name)}</b><br>{escape(entity.entity_type)} · "
                f"{entity.distinct_chunk_count:,} distinct chunks"
            )
        if tooltip_key != self._tooltip_key:
            self._tooltip_key = tooltip_key
            if tooltip:
                QToolTip.showText(event.globalPos(), tooltip, self)
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_hovered_entity(None)
        self._tooltip_key = None
        QToolTip.hideText()
        super().leaveEvent(event)

    def _add_chunk_node(self, entity_id: int, chunk_id: str) -> QPointF:
        assert self.model is not None
        center = self.entity_centers[entity_id]
        width, height = self.entity_sizes[entity_id]
        offset = self.model.chunk_offsets[(entity_id, chunk_id)]
        base_radius = max(self.model.diameters[entity_id] * 0.39, 1e-9)
        normalized_x = max(-1.0, min(1.0, offset[0] / base_radius))
        normalized_y = max(-1.0, min(1.0, offset[1] / base_radius))
        position = QPointF(
            center[0] + normalized_x * width * 0.42,
            center[1] + normalized_y * height * 0.38,
        )
        radius = max(0.8, min(2.2, min(width, height) * 0.055))
        item = ChunkNodeItem(entity_id, chunk_id, QRectF(-radius, -radius, radius * 2, radius * 2))
        item.setPos(position)
        item.setBrush(QColor("#ffffff"))
        item.setPen(QPen(QColor("#334155"), 0.3))
        item.setZValue(10)
        self.scene().addItem(item)
        self.node_items.append(item)
        return position

    def _clear_detail_items(self) -> None:
        for item in [*self.node_items, *self.link_items]:
            self.scene().removeItem(item)
        self.node_items.clear()
        self.link_items.clear()

    def _style_entity_items(self, connected_ids: set[int]) -> None:
        if self.model is None:
            return
        for entity_id, item in self.entity_items.items():
            entity = self.model.entities[entity_id]
            color = QColor(entity_type_color(entity.entity_type))
            fill = QColor(color)
            fill.setAlpha(110 if not connected_ids or entity_id in connected_ids else 35)
            item.setBrush(fill)
            label = self.entity_labels.get(entity_id)
            if label is not None:
                label.setOpacity(1.0 if not connected_ids or entity_id in connected_ids else 0.35)
            if entity_id == self.selected_entity_id:
                item.setPen(QPen(QColor("#0f172a"), 1.25))
            else:
                item.setPen(QPen(color.darker(125), 0.5))

    def _set_hovered_entity(self, entity_id: int | None) -> None:
        if entity_id == self._hovered_entity_id:
            return
        self._hovered_entity_id = entity_id
        for current_id, label in self.entity_labels.items():
            label.setVisible(current_id != entity_id)

    def _update_scene_rect(self) -> None:
        bounds = self.scene().itemsBoundingRect()
        margin = max(4.0, self.minimum_font_size * 0.5)
        self.scene().setSceneRect(bounds.adjusted(-margin, -margin, margin, margin))


class EntityGraphPanel(QWidget):
    def __init__(self, db: SQLiteDB, store: LanceDBStore | None = None):
        super().__init__()
        self.db = db
        self.store = store
        self.investigation_id: int | None = None
        self.visible_entity_ids: list[int] = []
        self._active = False
        self._dirty = True
        self._building = False
        self._closed = False
        self._generation = 0
        self._build_cancel: Event | None = None
        self._build_thread: Thread | None = None
        self._build_emitter: EntityGraphBuildEmitter | None = None
        self.model: EntityGraphModel | None = None
        self._preview_chunk_ids: list[str] = []
        self._current_preview_index = -1

        self.max_entities = QSpinBox()
        self.max_entities.setRange(10, 500)
        self.max_entities.setValue(100)
        self.max_entities.setSuffix(" entities")
        self.max_entities.setToolTip(
            "Maximum number of visible entities included, in the current table sort order"
        )
        self.max_entities.setFixedHeight(32)
        self.max_entities.valueChanged.connect(self._limit_changed)
        self.max_chunks = QSpinBox()
        self.max_chunks.setRange(100, MAX_LAYOUT_CHUNKS)
        self.max_chunks.setSingleStep(100)
        self.max_chunks.setValue(DEFAULT_GRAPH_CHUNKS)
        self.max_chunks.setSuffix(" chunks")
        self.max_chunks.setToolTip(
            "Maximum distinct chunks included in the graph; lower this for faster layouts"
        )
        self.max_chunks.setFixedHeight(32)
        self.max_chunks.valueChanged.connect(self._limit_changed)
        self.minimum_font_size = QSpinBox()
        self.minimum_font_size.setRange(6, 24)
        self.minimum_font_size.setValue(9)
        self.minimum_font_size.setSuffix(" pt")
        self.minimum_font_size.setFixedHeight(32)
        self.minimum_font_size.setToolTip(
            "Minimum entity-label font size; updates the drawing without rebuilding graph data"
        )
        self.minimum_font_size.valueChanged.connect(self._font_size_changed)
        self.rebuild_button = QPushButton("Rebuild graph")
        self.rebuild_button.clicked.connect(self.refresh)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self._reset_view)
        self.rebuild_button.setFixedHeight(32)
        self.reset_view_button.setFixedHeight(32)
        self.status_label = QLabel("Open this tab to build the graph")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(QLabel("Graph limit:"))
        controls.addWidget(self.max_entities)
        controls.addWidget(self.max_chunks)
        controls.addWidget(QLabel("Min. font:"))
        controls.addWidget(self.minimum_font_size)
        controls.addWidget(self.rebuild_button)
        controls.addWidget(self.reset_view_button)
        controls.addStretch(1)

        self.legend_label = QLabel()
        self.legend_label.setWordWrap(True)
        self.selection_label = QLabel()
        self.selection_label.setObjectName("StatusLabel")
        self.selection_label.setWordWrap(True)
        self.selection_label.setVisible(False)

        self.view = EntityGraphView()
        self.view.setMinimumWidth(420)
        self.preview = ChunkPreview("Click a chunk marker to keep its full text open here")
        self.preview.setMinimumWidth(280)
        self.preview.setVisible(False)
        self.preview.previous_requested.connect(lambda: self.select_adjacent_chunk(-1))
        self.preview.next_requested.connect(lambda: self.select_adjacent_chunk(1))
        self.view.chunk_selected.connect(self.show_chunk)
        self.view.entity_selected.connect(self._entity_selected)
        self.view.detail_status_changed.connect(self._set_selection_status)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1100, 420])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addWidget(self.legend_label)
        layout.addWidget(self.selection_label)
        layout.addWidget(self.splitter, 1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._start_build)

    def set_context(self, investigation_id: int | None, visible_entity_ids: Iterable[int]) -> None:
        ids = list(dict.fromkeys(int(entity_id) for entity_id in visible_entity_ids))
        if investigation_id == self.investigation_id and ids == self.visible_entity_ids:
            return
        self.investigation_id = investigation_id
        self.visible_entity_ids = ids
        self._invalidate(clear_model=True)
        if self._active:
            self._debounce.start()

    def set_data_sources(self, db: SQLiteDB, store: LanceDBStore | None) -> None:
        self.db = db
        self.store = store
        self._invalidate(clear_model=True)

    def set_active(self, active: bool) -> None:
        self._active = active
        if active and self._dirty:
            self._debounce.start(0)

    def refresh(self) -> None:
        self._invalidate(clear_model=True)
        if self._active:
            self._debounce.start(0)

    def show_chunk(self, chunk_id: str) -> None:
        row = self.db.get_chunk(chunk_id)
        if row is None:
            return
        if self.preview.isHidden():
            self.preview.setVisible(True)
            QTimer.singleShot(0, self._balance_splitter)
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        self.preview.set_chunk(
            str(row["chunk_text"]),
            str(row["display_markdown"]) if row["display_markdown"] else None,
            path=str(row["path"]),
            chunk_index=int(row["chunk_index"]),
            metadata=metadata,
            has_previous=self._preview_chunk_ids.index(chunk_id) > 0
            if chunk_id in self._preview_chunk_ids
            else False,
            has_next=self._preview_chunk_ids.index(chunk_id) + 1 < len(self._preview_chunk_ids)
            if chunk_id in self._preview_chunk_ids
            else False,
        )
        if chunk_id in self._preview_chunk_ids:
            self._current_preview_index = self._preview_chunk_ids.index(chunk_id)
        self.view.highlight_chunk(chunk_id)

    def select_adjacent_chunk(self, delta: int) -> None:
        target = self._current_preview_index + delta
        if 0 <= target < len(self._preview_chunk_ids):
            self.show_chunk(self._preview_chunk_ids[target])

    def shutdown(self) -> None:
        self._closed = True
        self._debounce.stop()
        if self._build_cancel is not None:
            self._build_cancel.set()

    def _limit_changed(self) -> None:
        self.refresh()

    def _font_size_changed(self, point_size: int) -> None:
        self.view.set_minimum_font_size(point_size)

    def _reset_view(self) -> None:
        self.view.reset_view()

    def _balance_splitter(self) -> None:
        width = max(self.splitter.width(), 900)
        preview_width = max(300, int(width * 0.32))
        self.splitter.setSizes([width - preview_width, preview_width])
        QTimer.singleShot(0, self.view.reset_view)

    def _set_selection_status(self, message: str) -> None:
        self.selection_label.setText(message)
        self.selection_label.setVisible(bool(message))

    def _start_build(self) -> None:
        if not self._active or self._closed:
            return
        if self._building:
            self._dirty = True
            return
        if self.investigation_id is None:
            self.status_label.setText("No investigation loaded")
            self._clear_model_state()
            return
        if not self.visible_entity_ids:
            self.status_label.setText("No entities match the current table filters")
            self._clear_model_state()
            return
        entity_ids = self.visible_entity_ids[: self.max_entities.value()]
        self._dirty = False
        self._building = True
        self.status_label.setText(
            f"Calculating semantic graph for {len(entity_ids):,} entities and their chunks..."
        )
        self.rebuild_button.setEnabled(False)
        cancel_event = Event()
        emitter = EntityGraphBuildEmitter(self)
        emitter.finished.connect(self._build_finished, Qt.QueuedConnection)
        self._build_cancel = cancel_event
        self._build_emitter = emitter
        self._build_thread = Thread(
            target=_build_entity_graph_request,
            args=(
                emitter,
                cancel_event,
                self._generation,
                self.db,
                self.store,
                self.investigation_id,
                entity_ids,
                self.max_chunks.value(),
            ),
            name=f"entity-graph-{self._generation}",
            daemon=True,
        )
        self._build_thread.start()

    @pyqtSlot(int, object, str)
    def _build_finished(self, generation: int, model: EntityGraphModel | None, error: str) -> None:
        self._building = False
        self._build_cancel = None
        self._build_thread = None
        if self._build_emitter is not None:
            self._build_emitter.deleteLater()
            self._build_emitter = None
        if self._closed:
            return
        self.rebuild_button.setEnabled(True)
        if error == _BUILD_CANCELLED or generation != self._generation:
            if self._active:
                self._debounce.start(0)
            return
        if error:
            self.status_label.setText(f"Could not build entity graph: {error}")
            self._clear_model_state()
            return
        self.model = model
        self.view.set_model(model)
        if model is None:
            self._clear_model_state()
            return
        shown = len(model.entities)
        total = len(self.visible_entity_ids)
        requested = min(total, self.max_entities.value())
        chunk_count = len(model.chunks)
        status = f"Showing {shown:,} entities and {chunk_count:,} distinct chunks"
        if total > requested:
            status += (
                f" · entity limit applied to {requested:,} of {total:,} visible rows in current "
                "table order; "
                "narrow the filters or raise the graph limit"
            )
        if shown < requested:
            status += (
                f" · {requested - shown:,} requested entities had no chunk retained after "
                "active-source and graph safety limits"
            )
        status += f" · layout: {model.layout_method}"
        if model.warning:
            status += f" · {model.warning}"
        self.status_label.setText(status)
        self._update_legend(model)

    def _invalidate(self, *, clear_model: bool) -> None:
        self._generation += 1
        self._dirty = True
        if self._build_cancel is not None:
            self._build_cancel.set()
        if clear_model:
            self._clear_model_state()

    def _clear_model_state(self) -> None:
        self.model = None
        self._preview_chunk_ids = []
        self._current_preview_index = -1
        self.preview.clear()
        self.preview.setVisible(False)
        self.selection_label.clear()
        self.selection_label.setVisible(False)
        self.legend_label.clear()
        self.view.set_model(None)

    def _entity_selected(self, entity_id: int) -> None:
        self.preview.clear()
        self.preview.setVisible(False)
        QTimer.singleShot(0, self.view.reset_view)
        self._current_preview_index = -1
        if self.model is None or entity_id not in self.model.entities:
            self._preview_chunk_ids = []
            return
        self._preview_chunk_ids = list(self.model.entities[entity_id].chunk_ids)

    def _update_legend(self, model: EntityGraphModel) -> None:
        entity_types = sorted(
            {entity.entity_type for entity in model.entities.values()}, key=str.casefold
        )
        parts = [
            f"<span style='color:{entity_type_color(entity_type)}'>■</span> {escape(entity_type)}"
            for entity_type in entity_types
        ]
        self.legend_label.setText(" &nbsp; ".join(parts))
