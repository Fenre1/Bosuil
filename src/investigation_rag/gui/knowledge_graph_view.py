"""Small Qt-native directed graph viewer for independently extracted KG claims."""
from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView

from investigation_rag.gui.knowledge_graph_layout import graph_positions


class KGNodeItem(QGraphicsEllipseItem):
    def __init__(self, node_id: int, name: str, node_type: str, view: "KnowledgeGraphView"):
        super().__init__(-31, -31, 62, 62)
        self.node_id, self.view, self.edges = node_id, view, []
        colors = {"person": "#64b5f6", "organization": "#81c784", "event": "#ffb74d", "location": "#ba68c8",
                  "date": "#4db6ac", "time": "#4db6ac", "time_period": "#4db6ac", "duration": "#4db6ac", "quantity": "#fff176", "value": "#fff176"}
        self.setBrush(QBrush(QColor(colors.get(node_type, "#90a4ae"))))
        self.setPen(QPen(QColor("#263238"), 2))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setToolTip(f"{name}\n{node_type} · node {node_id}\nClick to focus; drag to move")
        label = view.scene().addSimpleText(name if len(name) <= 28 else name[:25] + "…")
        label.setParentItem(self)
        label.setPos(-label.boundingRect().width() / 2, 35)
        self.setZValue(2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            if not self.view._laying_out:
                self.view.positions[self.node_id] = QPointF(value)
            for edge in self.edges:
                edge.update_path()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self._press_position = event.scenePos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and (event.scenePos() - self._press_position).manhattanLength() < 4:
            # Defer scene rebuilding until native mouse dispatch has finished.
            view, node_id = self.view, self.node_id
            QTimer.singleShot(0, lambda: view.node_selected.emit(node_id))


class KGEdgeItem(QGraphicsPathItem):
    def __init__(self, source: KGNodeItem, target: KGNodeItem, claim: dict, view: "KnowledgeGraphView", offset: float):
        super().__init__()
        self.source, self.target, self.claim, self.view, self.offset = source, target, claim, view, offset
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        colour = "#546e7a" if claim["review_status"] == "accepted" else "#9e9e9e"
        pen = QPen(QColor(colour), 2)
        if claim["polarity"] != "affirmed":
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        qualifier = "" if claim["polarity"] == "affirmed" else f" [{claim['polarity']}]"
        attribute = f" ({claim['attribute']})" if claim.get("attribute") else ""
        self.label = view.scene().addSimpleText(claim["predicate"] + attribute + qualifier)
        self.label.setBrush(QBrush(QColor("#37474f")))
        self.label.setZValue(1)
        self.label.setAcceptedMouseButtons(Qt.NoButton)
        self.setToolTip(f"{claim['subject']} → {claim['predicate']} → {claim['object']}\n"
                        f"{claim['polarity']} · {claim['review_status']}\n{claim['quote']}")
        source.edges.append(self)
        target.edges.append(self)
        self.update_path()

    def update_path(self):
        a, b = self.source.pos(), self.target.pos()
        if self.source is self.target:
            path = QPainterPath(a + QPointF(20, -25))
            path.cubicTo(a + QPointF(110, -110), a + QPointF(-110, -110), a + QPointF(-20, -25))
            tip = a + QPointF(-20, -25)
            path.addPolygon(QPolygonF([tip, tip + QPointF(-12, -4), tip + QPointF(-3, -12), tip]))
            self.setPath(path)
            self.label.setPos(a + QPointF(-self.label.boundingRect().width() / 2, -105))
            return
        delta = b - a
        length = max(math.hypot(delta.x(), delta.y()), 1)
        unit = QPointF(delta.x() / length, delta.y() / length)
        normal = QPointF(-unit.y(), unit.x())
        start, end = a + unit * 33, b - unit * 33
        mid = (a + b) / 2 + normal * self.offset
        path = QPainterPath(start)
        path.quadTo(mid, end)
        tip_delta = end - mid
        tip_length = max(math.hypot(tip_delta.x(), tip_delta.y()), 1)
        direction = QPointF(tip_delta.x() / tip_length, tip_delta.y() / tip_length)
        perpendicular = QPointF(-direction.y(), direction.x())
        path.addPolygon(QPolygonF([end, end - direction * 12 + perpendicular * 5,
                                   end - direction * 12 - perpendicular * 5, end]))
        self.setPath(path)
        label_bounds = self.label.boundingRect()
        label_y = mid.y() - 14
        if abs(delta.y()) < length * .35 and label_bounds.width() > length - 70:
            # Long property labels sit above the nodes instead of disappearing behind them.
            label_y = min(a.y(), b.y()) - label_bounds.height() - 42 - abs(self.offset)
        self.label.setPos(mid.x() - label_bounds.width() / 2, label_y)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.view.claim_selected.emit(self.claim["id"])


class KnowledgeGraphView(QGraphicsView):
    node_selected = pyqtSignal(int)
    claim_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(300, 220)
        self.setBackgroundBrush(QColor("#fafcfd"))
        self.setAccessibleName("Directed knowledge graph; select a relationship in the table to read evidence")
        self.positions: dict[int, QPointF] = {}
        self._laying_out = False

    def draw_claims(self, claims: list[dict], limit: int = 0, *, layout: str = "connected",
                    extra_nodes: list[dict] | None = None) -> tuple[int, int]:
        """Zero means all matching nodes. Only manual positions survive a redraw."""
        self._laying_out = True
        self.scene().clear()
        nodes = {}
        for claim in claims:
            for role in ("subject", "object"):
                node_id = claim[role + "_id"]
                if node_id not in nodes and (not limit or len(nodes) < limit):
                    nodes[node_id] = (claim[role], claim[role + "_type"])
        for node in extra_nodes or []:
            if node["id"] not in nodes and (not limit or len(nodes) < limit):
                nodes[node["id"]] = (node["name"], node["node_type"])
        items = {}
        for node_id, (name, node_type) in nodes.items():
            item = KGNodeItem(node_id, name, node_type, self)
            self.scene().addItem(item)
            items[node_id] = item
        width = max((item.childrenBoundingRect().width() for item in items.values()), default=170)
        locations = graph_positions(nodes, [(c["subject_id"], c["object_id"]) for c in claims], mode=layout,
                                    spacing=(max(200, width + 45), 170),
                                    aspect=self.viewport().width() / max(1, self.viewport().height()))
        for node_id, item in items.items():
            item.setPos(self.positions.get(node_id, QPointF(*locations[node_id])))
        drawn, pairs = 0, {}
        # Group identical labelled edges; each source claim remains in the complete table.
        seen = set()
        for claim in claims:
            a, b = claim["subject_id"], claim["object_id"]
            key = (a, b, claim["predicate"], claim.get("attribute", ""), claim["polarity"], claim["attribution"], claim["time_text"])
            if a not in items or b not in items or key in seen:
                continue
            seen.add(key)
            pair = tuple(sorted((a, b)))
            offset = pairs.get(pair, 0) * 35
            pairs[pair] = pairs.get(pair, 0) + 1
            edge = KGEdgeItem(items[a], items[b], claim, self, offset)
            self.scene().addItem(edge)
            drawn += 1
        self.setSceneRect(self.scene().itemsBoundingRect().adjusted(-80, -80, 80, 80))
        self._laying_out = False
        return len(items), drawn

    def fit_graph(self):
        if self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect().adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        scale = self.transform().m11() * factor
        if 0.04 < scale < 6:
            self.scale(factor, factor)
        event.accept()
