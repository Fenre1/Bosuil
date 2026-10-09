import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from investigation_rag.gui.knowledge_graph_layout import graph_positions
from investigation_rag.gui.knowledge_graph_view import KnowledgeGraphView, KGNodeItem


def claims(count):
    return [{"id": i + 1, "subject_id": i + 1, "object_id": i + 2, "subject": f"Node {i + 1}",
             "object": f"Node {i + 2}", "subject_type": "object", "object_type": "object",
             "predicate": "part_of", "polarity": "affirmed", "attribution": "", "time_text": "",
             "quote": "Synthetic graph display test.", "review_status": "suggested"} for i in range(count)]


@pytest.mark.parametrize("mode", ["connected", "grid"])
def test_compact_layout_bounds_large_graph_without_missing_nodes(mode):
    ids = list(range(1000))
    edges = [(i, i + 1) for i in range(999)]
    positions = graph_positions(ids, edges, mode=mode)
    assert len(positions) == len(set(positions.values())) == 1000
    assert max(x for x, y in positions.values()) < 10000
    assert max(y for x, y in positions.values()) < 10000
    assert positions == graph_positions(list(reversed(ids)), list(reversed(edges)), mode=mode)
    assert graph_positions([], [], mode=mode) == {}


def test_connected_components_are_separate_and_nodes_do_not_overlap():
    positions = graph_positions(range(12), [(i, i + 1) for i in range(5)] + [(i, i + 1) for i in range(6, 11)])
    first, second = [positions[i] for i in range(6)], [positions[i] for i in range(6, 12)]
    assert (max(x for x, y in first) < min(x for x, y in second) or
            max(y for x, y in first) < min(y for x, y in second))
    assert all(abs(a[0] - b[0]) >= 100 or abs(a[1] - b[1]) >= 100
               for i, a in enumerate(positions.values()) for b in list(positions.values())[i + 1:])


def test_graph_can_display_more_than_sixty_and_reset_manual_positions():
    app = QApplication.instance() or QApplication([])
    view = KnowledgeGraphView()
    assert view.draw_claims(claims(120)) == (121, 120)
    assert view.draw_claims(claims(120), limit=17) == (17, 16)
    assert view.draw_claims(claims(120), limit=0) == (121, 120)
    node = next(i for i in view.scene().items() if isinstance(i, KGNodeItem))
    node_id = node.node_id
    node.setPos(90000, 90000)
    view.draw_claims(claims(120))
    moved = next(i for i in view.scene().items() if isinstance(i, KGNodeItem) and i.node_id == node_id)
    assert moved.pos().x() == 90000
    view.positions.clear()
    view.draw_claims(claims(120), layout="grid")
    restored = next(i for i in view.scene().items() if isinstance(i, KGNodeItem) and i.node_id == node_id)
    assert restored.pos().x() < 10000
    assert view.draw_claims([], extra_nodes=[{"id": 90, "name": "Isolated node", "node_type": "concept"}]) == (1, 0)
    view.close()
