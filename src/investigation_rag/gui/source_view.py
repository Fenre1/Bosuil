from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from PyQt5.QtCore import QRectF, Qt, QTimer, QVariant
from PyQt5.QtGui import QBrush, QColor, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from investigation_rag.gui.source_document import load_source


_executor: ProcessPoolExecutor | None = None


def shutdown_source_renderer() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


atexit.register(shutdown_source_renderer)


class LocalTextBrowser(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)

    def loadResource(self, resource_type, name):
        # Excerpts and source HTML consist solely of escaped text and local
        # formatting. Never load an email tracking image or a file URL.
        return QVariant()


class SourceView(QWidget):
    def __init__(self):
        super().__init__()
        self.path = ""
        self.metadata = {}
        self.future = None
        self.generation = 0
        self.request_generation = 0
        self.pending = False
        self.requested_page = 1
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.page = QSpinBox()
        self.page.setMinimumWidth(95)
        self.page.setPrefix("Page ")
        self.page.setRange(1, 1)
        self.page.valueChanged.connect(self.request_page)
        self.page_total = QLabel()
        self.page_total.setMinimumWidth(45)
        self.previous = QPushButton("‹")
        self.previous.setToolTip("Previous source page")
        self.previous.clicked.connect(lambda: self.page.setValue(self.page.value() - 1))
        self.next = QPushButton("›")
        self.next.setToolTip("Next source page")
        self.next.clicked.connect(lambda: self.page.setValue(self.page.value() + 1))
        self.fit = QPushButton("Fit width")
        self.fit.clicked.connect(self.fit_width)
        self.zoom_in = QPushButton("+")
        self.zoom_out = QPushButton("−")
        for button in (self.previous, self.next, self.zoom_in, self.zoom_out):
            button.setFixedWidth(30)
        self.zoom_in.clicked.connect(lambda: self.zoom(1.25))
        self.zoom_out.clicked.connect(lambda: self.zoom(0.8))
        self.focus = QPushButton("Show passage")
        self.focus.clicked.connect(self.focus_selection)
        self.controls = QWidget()
        controls = QHBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        for widget in (self.previous, self.page, self.page_total, self.next, self.fit, self.zoom_out, self.zoom_in, self.focus):
            controls.addWidget(widget)
        controls.addStretch()
        self.scene = QGraphicsScene(self)
        self.graphics = QGraphicsView(self.scene)
        self.graphics.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics.setBackgroundBrush(QColor("#e2e8f0"))
        self.text = LocalTextBrowser()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.graphics)
        self.stack.addWidget(self.text)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.status)
        layout.addWidget(self.controls)
        layout.addWidget(self.stack, 1)
        self.selection_rect = QRectF()
        self.image_rect = QRectF()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)
        self.clear()

    def clear(self):
        self.generation += 1
        self.pending = False
        self.path = ""
        self.scene.clear()
        self.text.clear()
        self.status.clear()
        self.controls.hide()

    def set_source(self, path: str, metadata: dict, fallback: str):
        self.clear()
        self.path, self.metadata = path, metadata
        self.stack.setCurrentWidget(self.text)
        self.text.setPlainText(fallback)
        if not metadata.get("source_reference_version") or not metadata.get("source_content_hash"):
            self.status.setText("Indexed text · rescan this document to enable Source view.")
            return
        self.request_page(int(metadata.get("page_start") or 1))

    def request_page(self, page: int):
        self.generation += 1
        self.requested_page = page
        self.pending = True
        self.status.setText("Loading source…")
        # Never leave a previous document's highlight visible during a load.
        self.scene.clear()
        self.controls.setEnabled(False)
        self.timer.start()
        self.poll()

    def poll(self):
        global _executor
        if self.future is not None:
            if not self.future.done():
                return
            try:
                result = self.future.result()
            except Exception as exc:
                result = {"kind": "error", "status": f"Source preview unavailable: {exc}"}
            self.future = None
            if self.request_generation == self.generation:
                self.show_result(result)
        if self.pending:
            self.pending = False
            self.request_generation = self.generation
            try:
                if _executor is None:
                    _executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                self.future = _executor.submit(load_source, self.path, self.metadata, self.requested_page)
            except Exception as exc:
                shutdown_source_renderer()
                self.show_result({"kind": "error", "status": f"Source preview unavailable: {exc}"})
        elif self.future is None:
            self.timer.stop()

    def show_result(self, result: dict):
        self.status.setText(result["status"])
        self.controls.setEnabled(True)
        self.controls.setVisible(result["kind"] == "pdf")
        if result["kind"] == "pdf":
            self.stack.setCurrentWidget(self.graphics)
            self.scene.clear()
            pixmap = QPixmap()
            pixmap.loadFromData(result["png"])
            self.scene.addPixmap(pixmap)
            self.image_rect = QRectF(0, 0, result["width"], result["height"])
            self.scene.setSceneRect(self.image_rect)
            self.selection_rect = QRectF()
            for coordinates in result["rectangles"]:
                x0, y0, x1, y1 = coordinates
                rect = QRectF(x0, y0, x1 - x0, y1 - y0)
                self.scene.addRect(rect, QPen(Qt.NoPen), QBrush(QColor(250, 204, 21, 85)))
                if self.selection_rect.isNull():
                    self.selection_rect = rect
            self.page.blockSignals(True)
            self.page.setRange(1, result["page_count"])
            self.page.setValue(result["page"])
            self.page.blockSignals(False)
            self.page_total.setText(f"of {result['page_count']}")
            self.previous.setEnabled(result["page"] > 1)
            self.next.setEnabled(result["page"] < result["page_count"])
            self.fit_width()
            if not self.selection_rect.isNull():
                self.focus_selection()
            else:
                self.graphics.verticalScrollBar().setValue(0)
        elif result["kind"] == "text":
            self.stack.setCurrentWidget(self.text)
            self.text.setHtml(result["html"])
            self.text.scrollToAnchor("selection")
        else:
            self.stack.setCurrentWidget(self.text)

    def fit_width(self):
        if self.image_rect.width():
            self.graphics.resetTransform()
            factor = max(0.1, (self.graphics.viewport().width() - 24) / self.image_rect.width())
            self.graphics.scale(factor, factor)

    def zoom(self, factor: float):
        if 0.1 <= self.graphics.transform().m11() * factor <= 5:
            self.graphics.scale(factor, factor)

    def focus_selection(self):
        if self.selection_rect.isNull():
            first_page = int(self.metadata.get("page_start") or 1)
            if self.page.value() != first_page:
                self.page.setValue(first_page)
            else:
                self.graphics.verticalScrollBar().setValue(0)
        else:
            self.graphics.centerOn(self.selection_rect.center())
