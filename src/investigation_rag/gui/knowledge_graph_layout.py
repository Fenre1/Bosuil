"""Deterministic compact layouts, with linear storage for large graphs."""
from collections import deque
import math

import numpy as np


def _spring_positions(group, adjacency, local, spacing):
    """Bounded refinement for small components; large components keep the compact traversal grid."""
    if not 3 <= len(group) <= 160:
        return local
    dx, dy = spacing
    positions = np.array([(local[node][0] / dx, local[node][1] / dy) for node in group], dtype=float)
    order = np.arange(len(group), dtype=float)
    positions += np.column_stack((np.sin(order * 2.4), np.cos(order * 2.4))) * .08
    slots = {node: i for i, node in enumerate(group)}
    links = np.zeros((len(group), len(group)), dtype=float)
    for node in group:
        for other in adjacency[node]:
            links[slots[node], slots[other]] = 1.0
    for step in range(55):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), .01)
        np.fill_diagonal(distance, np.inf)
        repulsion = (delta / distance[:, :, None] ** 2).sum(axis=1)
        attraction = (delta * (links * np.where(np.isfinite(distance), distance, 0))[:, :, None]).sum(axis=1)
        movement = repulsion - attraction
        length = np.maximum(np.linalg.norm(movement, axis=1), .01)
        temperature = .25 * (1 - step / 55) + .02
        positions += movement * (np.minimum(length, temperature) / length)[:, None]
    # Keep enough room for circles and their labels after the spring relaxation.
    for _ in range(20):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), .01)
        np.fill_diagonal(distance, np.inf)
        separation = np.maximum(1.15 - distance, 0) / distance
        positions += (delta * separation[:, :, None]).sum(axis=1) * .5
    # Put each component's longest axis along the wider dimension of the graph pane.
    centered = positions - positions.mean(axis=0)
    axis = np.linalg.eigh(centered.T @ centered)[1][:, -1]
    if axis[0] < 0:
        axis = -axis
    positions = centered @ np.column_stack((axis, np.array([-axis[1], axis[0]])))
    positions -= positions.min(axis=0)
    return {node: (float(positions[i, 0] * dx), float(positions[i, 1] * dy)) for i, node in enumerate(group)}


def graph_positions(node_ids, edges, *, mode="connected", spacing=(220.0, 135.0), aspect=1.5):
    adjacency = {node: set() for node in node_ids}
    for a, b in edges:
        if a in adjacency and b in adjacency and a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    ranking = sorted(adjacency, key=lambda node: (-len(adjacency[node]), node))
    unseen = set(adjacency)
    groups = []
    for root in ranking:
        if root not in unseen:
            continue
        unseen.remove(root)
        queue, group = deque([root]), []
        while queue:
            node = queue.popleft()
            group.append(node)
            neighbors = sorted(adjacency[node] & unseen, key=lambda other: (-len(adjacency[other]), other))
            unseen.difference_update(neighbors)
            queue.extend(neighbors)
        groups.append(group)
    groups.sort(key=lambda group: (-len(group), group[0]))
    if mode == "grid":
        groups = [[node for group in groups for node in group]] if groups else []
    elif mode != "connected":
        raise ValueError("Unknown graph layout.")

    dx, dy = spacing
    aspect = max(.7, min(2.4, aspect))
    tiles = []
    for group in groups:
        columns = min(len(group), max(1, math.ceil(math.sqrt(len(group) * aspect * dy / dx))))
        rows = math.ceil(len(group) / columns)
        local = {}
        for index, node in enumerate(group):
            row, column = divmod(index, columns)
            # Neighboring traversal entries stay close at row boundaries too.
            if row % 2:
                column = columns - column - 1
            local[node] = (column * dx, row * dy)
        if mode == "connected":
            local = _spring_positions(group, adjacency, local, spacing)
        width = max(x for x, y in local.values()) + dx
        height = max(y for x, y in local.values()) + dy
        tiles.append((width, height, local))
    if not tiles:
        return {}
    gap = 35.0
    target_width = max(max(width for width, _, _ in tiles),
                       math.sqrt(sum((width + gap) * (height + gap) for width, height, _ in tiles) * aspect))
    bin_width = dx + gap
    column_count = max(1, math.ceil(target_width / bin_width), max(math.ceil((width + gap) / bin_width) for width, _, _ in tiles))
    skyline = [0.0] * column_count
    positions = {}
    for width, height, local in tiles:
        span = math.ceil((width + gap) / bin_width)
        column = min(range(column_count - span + 1), key=lambda start: (max(skyline[start:start + span]), start))
        x, y = column * bin_width, max(skyline[column:column + span])
        positions.update({node: (x + px, y + py) for node, (px, py) in local.items()})
        skyline[column:column + span] = [y + height + gap] * span
    return positions
