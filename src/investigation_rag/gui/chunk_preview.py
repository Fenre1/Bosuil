from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ChunkPreview(QWidget):
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()

    def __init__(self, placeholder: str = "Select a chunk to preview its text"):
        super().__init__()
        self._markdown: str | None = None
        self._plain_text = ""
        self._formatted = True

        self.context_label = QLabel()
        self.context_label.setWordWrap(True)
        self.context_label.setObjectName("ChunkPreviewContext")

        self.formatted_button = QPushButton("Formatted")
        self.formatted_button.setCheckable(True)
        self.formatted_button.setChecked(True)
        self.plain_button = QPushButton("Plain text")
        self.plain_button.setCheckable(True)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.formatted_button)
        mode_group.addButton(self.plain_button)
        self.formatted_button.clicked.connect(lambda: self.set_formatted(True))
        self.plain_button.clicked.connect(lambda: self.set_formatted(False))

        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.previous_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.formatted_button)
        controls.addWidget(self.plain_button)
        controls.addStretch(1)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)

        self.viewer = QTextBrowser()
        self.viewer.setReadOnly(True)
        self.viewer.setPlaceholderText(placeholder)
        self.viewer.document().setDefaultStyleSheet(
            "body { color: #1f2937; } "
            "p { margin: 0 0 10px 0; line-height: 135%; } "
            "h1, h2, h3, h4 { color: #0f172a; margin-top: 12px; margin-bottom: 6px; } "
            "pre { background: #f1f5f9; padding: 8px; white-space: pre-wrap; } "
            "blockquote { color: #64748b; border-left: 3px solid #cbd5e1; margin-left: 4px; padding-left: 10px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.context_label)
        layout.addLayout(controls)
        layout.addWidget(self.viewer, 1)
        self.clear()

    def set_chunk(
        self,
        plain_text: str,
        display_markdown: str | None = None,
        *,
        path: str = "",
        chunk_index: int | None = None,
        metadata: dict[str, Any] | None = None,
        has_previous: bool = False,
        has_next: bool = False,
    ) -> None:
        self._plain_text = plain_text
        self._markdown = display_markdown
        self.context_label.setTextFormat(Qt.RichText)
        self.context_label.setText(_context_html(path, chunk_index, metadata or {}))
        self.context_label.setVisible(True)
        self.previous_button.setEnabled(has_previous)
        self.next_button.setEnabled(has_next)
        self.formatted_button.setEnabled(bool(display_markdown))
        if not display_markdown:
            self._formatted = False
            self.plain_button.setChecked(True)
        self._render()

    def set_formatted(self, formatted: bool) -> None:
        self._formatted = formatted and bool(self._markdown)
        self.formatted_button.setChecked(self._formatted)
        self.plain_button.setChecked(not self._formatted)
        self._render()

    def clear(self) -> None:
        self._plain_text = ""
        self._markdown = None
        self.context_label.clear()
        self.context_label.setVisible(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.formatted_button.setEnabled(False)
        self.viewer.clear()

    def setPlainText(self, text: str) -> None:
        self._plain_text = text
        self._markdown = None
        self.viewer.setPlainText(text)

    def setMarkdown(self, markdown: str) -> None:
        self._markdown = markdown
        self._formatted = True
        self.viewer.setMarkdown(markdown)

    def toPlainText(self) -> str:
        return self.viewer.toPlainText()

    def textCursor(self):
        return self.viewer.textCursor()

    def setTextCursor(self, cursor) -> None:
        self.viewer.setTextCursor(cursor)

    def ensureCursorVisible(self) -> None:
        self.viewer.ensureCursorVisible()

    def _render(self) -> None:
        if self._formatted and self._markdown:
            self.viewer.setMarkdown(self._markdown)
        else:
            self.viewer.setPlainText(self._plain_text)


def _context_html(path: str, chunk_index: int | None, metadata: dict[str, Any]) -> str:
    primary: list[str] = []
    if path:
        primary.append(f"<b>{escape(Path(path).name)}</b>")
    if chunk_index is not None:
        primary.append(f"Chunk {chunk_index + 1}")
    pages = metadata.get("pages")
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if isinstance(pages, list) and pages:
        primary.append(f"Pages {min(pages)}–{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}")
    elif page_start:
        primary.append(f"Pages {page_start}–{page_end}" if page_end and page_end != page_start else f"Page {page_start}")

    secondary: list[str] = []
    heading = str(metadata.get("heading_path_text", "")).strip()
    if heading:
        secondary.append(escape(heading))
    for key, label in (
        ("contains_table", "TABLE"),
        ("contains_figure", "FIGURE"),
        ("contains_list", "LIST"),
        ("is_continuation", "CONTINUATION"),
    ):
        if metadata.get(key):
            secondary.append(f"<span style='color:#475569'><b>{label}</b></span>")
    rows = [" · ".join(primary)]
    if secondary:
        rows.append(" · ".join(secondary))
    return "<br>".join(row for row in rows if row)
