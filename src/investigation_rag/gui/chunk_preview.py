from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from investigation_rag.gui.source_view import LocalTextBrowser, SourceView
from investigation_rag.ingest.searchable_text import stable_embedding_text


class ChunkPreview(QWidget):
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    rendered = pyqtSignal()

    def __init__(self, placeholder: str = "Select a chunk to preview its text", *, db=None, navigation_label: str = "chunk"):
        super().__init__()
        self.db = db
        self._mode = "source"
        self._plain_text, self._markdown, self._metadata = "", None, {}
        self._path, self._chunk_index = "", None
        self._source_loaded, self._neighbors = False, {}
        self.context_label = QLabel()
        self.context_label.setWordWrap(True)
        self.context_label.setObjectName("ChunkPreviewContext")
        self.context_label.setTextFormat(Qt.RichText)
        self.source_button = QPushButton("Source")
        self.formatted_button = QPushButton("Readable text")
        self.plain_button = QPushButton("Embedding input")
        group = QButtonGroup(self)
        group.setExclusive(True)
        modes = QHBoxLayout()
        for button, mode in ((self.source_button, "source"), (self.formatted_button, "readable"), (self.plain_button, "embedding")):
            button.setCheckable(True)
            group.addButton(button)
            button.clicked.connect(lambda checked, value=mode: self.set_mode(value))
            modes.addWidget(button)
        modes.addStretch()
        self.source_button.setChecked(True)
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.previous_button.setToolTip(f"Previous {navigation_label}")
        self.next_button.setToolTip(f"Next {navigation_label}")
        self.previous_button.setAccessibleName(f"Previous {navigation_label}")
        self.next_button.setAccessibleName(f"Next {navigation_label}")
        self.previous_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)
        self.earlier_button = QPushButton("Earlier")
        self.later_button = QPushButton("Later")
        self.earlier_button.setToolTip("Earlier passage in this document")
        self.later_button.setToolTip("Later passage in this document")
        self.earlier_button.clicked.connect(lambda: self.show_neighbor(-1))
        self.later_button.clicked.connect(lambda: self.show_neighbor(1))
        navigation = QHBoxLayout()
        for button in (self.earlier_button, self.later_button):
            navigation.addWidget(button)
            button.setVisible(db is not None)
        navigation.addStretch()
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        if navigation_label == "chunk" and db is not None:
            self.previous_button.hide()
            self.next_button.hide()
        self.viewer = LocalTextBrowser()
        self.viewer.setPlaceholderText(placeholder)
        self.viewer.document().setDefaultStyleSheet(
            "body { color:#1f2937; } p { margin:0 0 10px 0; } "
            "h1,h2,h3,h4 { color:#0f172a; } td { padding:5px; } pre { white-space:pre-wrap; }"
        )
        self.source_view = SourceView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.source_view)
        self.stack.addWidget(self.viewer)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.context_label)
        layout.addLayout(modes)
        layout.addLayout(navigation)
        layout.addWidget(self.stack, 1)
        self.clear()

    def set_chunk(self, plain_text: str, display_markdown: str | None = None, *,
                  path: str = "", chunk_index: int | None = None, metadata: dict[str, Any] | None = None,
                  has_previous: bool = False, has_next: bool = False) -> None:
        self._plain_text, self._markdown = plain_text, display_markdown
        self._metadata = metadata or {}
        self._path, self._chunk_index = path, chunk_index
        self.context_label.setText(_context_html(path, chunk_index, self._metadata))
        self.context_label.show()
        self.previous_button.setEnabled(has_previous)
        self.next_button.setEnabled(has_next)
        for button in (self.source_button, self.formatted_button, self.plain_button):
            button.setEnabled(True)
        self.source_view.clear()
        self._source_loaded = False
        self.load_neighbors()
        self._render()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._render()

    def set_formatted(self, formatted: bool) -> None:
        self.set_mode("readable" if formatted else "embedding")

    def clear(self) -> None:
        self._plain_text, self._markdown, self._metadata = "", None, {}
        self._source_loaded, self._neighbors = False, {}
        self.source_view.clear()
        self.context_label.clear()
        self.context_label.hide()
        for button in (self.previous_button, self.next_button, self.source_button, self.formatted_button,
                       self.plain_button, self.earlier_button, self.later_button):
            button.setEnabled(False)
        self.viewer.clear()
        self.stack.setCurrentWidget(self.viewer)

    def setPlainText(self, text: str) -> None:
        self.clear()
        self._plain_text = text
        self.viewer.setPlainText(text)
        self.stack.setCurrentWidget(self.viewer)

    def setMarkdown(self, markdown: str) -> None:
        self.clear()
        self._markdown = markdown
        self.viewer.setMarkdown(markdown)
        self.stack.setCurrentWidget(self.viewer)

    def toPlainText(self) -> str:
        return self.viewer.toPlainText()

    def textCursor(self):
        return self.viewer.textCursor()

    def setTextCursor(self, cursor) -> None:
        self.viewer.setTextCursor(cursor)

    def ensureCursorVisible(self) -> None:
        self.viewer.ensureCursorVisible()

    def _render(self) -> None:
        self.source_button.setChecked(self._mode == "source")
        self.formatted_button.setChecked(self._mode == "readable")
        self.plain_button.setChecked(self._mode == "embedding")
        if self._mode == "embedding":
            self.viewer.setPlainText(stable_embedding_text(self._plain_text, self._metadata))
        elif self._metadata.get("display_html"):
            self.viewer.setHtml(str(self._metadata["display_html"]))
        elif self._markdown:
            self.viewer.setMarkdown(self._markdown)
        else:
            self.viewer.setPlainText(self._plain_text)
        self.stack.setCurrentWidget(self.source_view if self._mode == "source" else self.viewer)
        if self._mode == "source" and not self._source_loaded:
            self._source_loaded = True
            self.source_view.set_source(self._path, self._metadata, self._plain_text)
        self.rendered.emit()

    def load_neighbors(self) -> None:
        from investigation_rag.retrieval.ranking import row_to_result
        self._neighbors = {}
        if self.db is not None and self._metadata.get("file_id") is not None and self._chunk_index is not None:
            rows = self.db.get_neighbor_chunks(int(self._metadata["file_id"]), self._chunk_index)
            identity = self._metadata.get("source_document_id", self._metadata.get("attachment_name") or "root")
            for row in rows:
                neighbor = row_to_result(row)
                nm = neighbor.metadata or {}
                if nm.get("source_document_id", neighbor.attachment_name or "root") != identity:
                    continue
                offset = neighbor.chunk_index - self._chunk_index
                if offset in (-1, 1):
                    self._neighbors[offset] = neighbor
        self.earlier_button.setEnabled(-1 in self._neighbors)
        self.later_button.setEnabled(1 in self._neighbors)

    def show_neighbor(self, offset: int) -> None:
        result = self._neighbors.get(offset)
        if result is not None:
            self.set_chunk(result.chunk_text, result.display_markdown, path=result.path,
                           chunk_index=result.chunk_index, metadata=result.metadata,
                           has_previous=self.previous_button.isEnabled(), has_next=self.next_button.isEnabled())


def _context_html(path: str, chunk_index: int | None, metadata: dict[str, Any]) -> str:
    primary = [f"<b>{escape(Path(path).name)}</b>"] if path else []
    if chunk_index is not None:
        primary.append(f"Chunk {chunk_index + 1}")
    pages = metadata.get("pages")
    if isinstance(pages, list) and pages:
        primary.append("Pages " + ", ".join(str(page) for page in pages) if len(pages) > 1 else f"Page {pages[0]}")
    elif metadata.get("page_start"):
        primary.append(f"Page {metadata['page_start']}")
    if metadata.get("line_start"):
        primary.append(f"Lines {metadata['line_start']}–{metadata.get('line_end', metadata['line_start'])}")
    secondary = []
    if metadata.get("attachment_chain"):
        secondary.append("Attachment: " + " → ".join(escape(str(part["name"])) for part in metadata["attachment_chain"]))
    elif metadata.get("attachment_name"):
        secondary.append("Attachment: " + escape(str(metadata["attachment_name"])))
    if metadata.get("heading_path_text"):
        secondary.append(escape(str(metadata["heading_path_text"])))
    for key, label in (("contains_table", "Table"), ("contains_figure", "Figure"),
                       ("contains_list", "List"), ("is_continuation", "Continued passage")):
        if metadata.get(key):
            secondary.append(label)
    return "<br>".join(" · ".join(row) for row in (primary, secondary) if row)
