from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.cli import build_rag_searcher
from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.gui.folder_scope import FolderScope
from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.rag.prompt_builder import ConversationTurn
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.rag.strict_answering import StrictAnswerer
from investigation_rag.utils.windows_open import open_containing_folder, open_file


CITATION_REF_RE = re.compile(r"\[(\d+)\]")


@dataclass
class ChatExchange:
    question: str
    answer: str
    results: list[SearchResult]
    retrieval_query: str
    evidence_ledger: list[dict[str, object]] | None = None
    process_feedback: list[str] | None = None


class RagPanel(QWidget):
    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore, folder_scope: FolderScope | None = None):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.folder_scope = folder_scope
        self.investigation_id: int | None = None
        self.results: list[SearchResult] = []
        self.turns: list[ChatExchange] = []
        self.active_turn_index: int | None = None
        self._ledger_progress_lines: list[str] = []

        self.question_edit = QTextEdit()
        self.question_edit.setFixedHeight(90)
        ask_button = QPushButton("Ask")
        ask_button.clicked.connect(self.ask)
        new_button = QPushButton("New conversation")
        new_button.clicked.connect(self.new_conversation)
        save_button = QPushButton("Save to PDF")
        save_button.clicked.connect(self.save_conversation_pdf)
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Standard retrieval", "standard")
        self.profile_combo.addItem("Deep retrieval", "deep")
        self.profile_combo.addItem("Evidence ledger RAG", "ledger")
        self.profile_combo.addItem("Legacy retrieval", "legacy")

        open_button = QPushButton("Open file")
        open_button.clicked.connect(self.open_selected_file)
        folder_button = QPushButton("Open folder")
        folder_button.clicked.connect(self.open_selected_folder)

        self.answer_view = QTextBrowser()
        self.answer_view.setReadOnly(True)
        self.answer_view.setOpenExternalLinks(False)
        self.answer_view.setOpenLinks(False)
        self.answer_view.anchorClicked.connect(self.handle_answer_link)

        self.citation_table = QTableWidget(0, 4)
        self.citation_table.setHorizontalHeaderLabels(["#", "Source", "Location", "Path"])
        self.citation_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.citation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.citation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.citation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.citation_table.itemSelectionChanged.connect(self.update_chunk_preview)

        self.chunk_preview = QTextEdit()
        self.chunk_preview.setReadOnly(True)

        citation_buttons = QHBoxLayout()
        citation_buttons.addWidget(open_button)
        citation_buttons.addWidget(folder_button)
        citation_buttons.addStretch(1)

        citation_splitter = QSplitter()
        citation_splitter.addWidget(self.citation_table)
        citation_splitter.addWidget(self.chunk_preview)
        citation_splitter.setStretchFactor(0, 2)
        citation_splitter.setStretchFactor(1, 3)

        citation_container = QWidget()
        citation_layout = QVBoxLayout(citation_container)
        citation_layout.setContentsMargins(0, 0, 0, 0)
        citation_layout.addLayout(citation_buttons)
        citation_layout.addWidget(citation_splitter, 1)

        result_splitter = QSplitter(Qt.Vertical)
        result_splitter.addWidget(self.answer_view)
        result_splitter.addWidget(citation_container)
        result_splitter.setStretchFactor(0, 3)
        result_splitter.setStretchFactor(1, 2)

        controls = QHBoxLayout()
        controls.addWidget(new_button)
        controls.addWidget(save_button)
        controls.addWidget(self.profile_combo)
        controls.addStretch(1)
        controls.addWidget(ask_button)

        layout = QVBoxLayout(self)
        layout.addWidget(result_splitter, 1)
        layout.addWidget(self.question_edit)
        layout.addLayout(controls)

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.new_conversation()

    def new_conversation(self) -> None:
        self.turns = []
        self.results = []
        self.active_turn_index = None
        self.answer_view.clear()
        self.citation_table.setRowCount(0)
        self.chunk_preview.clear()
        self.question_edit.clear()

    def ask(self) -> None:
        if self.investigation_id is None:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            return
        history = [ConversationTurn(turn.question, turn.answer) for turn in self.turns]
        try:
            llm = OllamaLLM(
                OllamaClient(self.config.ollama_base_url, timeout=self.config.ollama_timeout_seconds),
                self.config.ollama_llm_model,
            )
            profile_name = str(self.profile_combo.currentData())
            searcher_profile = "standard" if profile_name == "ledger" else profile_name
            searcher = build_rag_searcher(self.config, self.db, searcher_profile, llm)
            answerer = StrictAnswerer(searcher, llm)
            folder_paths = self.folder_scope.folder_paths() if self.folder_scope is not None else None
            if profile_name == "ledger":
                self._ledger_progress_lines = []
                response = answerer.answer_with_evidence_ledger(
                    self.investigation_id,
                    question,
                    history=history,
                    folder_paths=folder_paths,
                    progress_callback=self.show_ledger_progress,
                )
            else:
                response = answerer.answer(self.investigation_id, question, history=history, folder_paths=folder_paths)
        except Exception as exc:
            QMessageBox.critical(self, "RAG", str(exc))
            return
        self.turns.append(
            ChatExchange(
                question=question,
                answer=response.answer,
                results=response.results,
                retrieval_query=response.retrieval_query,
                evidence_ledger=response.evidence_ledger,
                process_feedback=response.process_feedback,
            )
        )
        self.active_turn_index = len(self.turns) - 1
        self.results = response.results
        self.answer_view.setHtml(self.render_conversation_html())
        self.answer_view.verticalScrollBar().setValue(self.answer_view.verticalScrollBar().maximum())
        self.populate_citations()
        self.question_edit.clear()

    def show_ledger_progress(self, message: str) -> None:
        self._ledger_progress_lines.append(message)
        body = "<br>".join(html.escape(line) for line in self._ledger_progress_lines)
        self.answer_view.setHtml(f"<html><body><p>{body}</p></body></html>")
        self.answer_view.verticalScrollBar().setValue(self.answer_view.verticalScrollBar().maximum())
        QApplication.processEvents()

    def save_conversation_pdf(self) -> None:
        if not self.turns:
            return
        default_name = f"rag-conversation-{datetime.now():%Y%m%d-%H%M}.pdf"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save conversation to PDF",
            default_name,
            "PDF files (*.pdf)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".pdf"):
            file_path = f"{file_path}.pdf"

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_path)

        document = QTextDocument()
        document.setHtml(self.render_conversation_html(for_export=True))
        document.print_(printer)
        QMessageBox.information(self, "Save to PDF", f"Conversation saved to:\n{file_path}")

    def render_conversation_html(self, for_export: bool = False) -> str:
        body = "\n".join(self.render_turn_html(turn, index) for index, turn in enumerate(self.turns))
        return f"""
        <html>
          <head>
            <style>
              body {{
                color: #202124;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 10pt;
                margin: {24 if for_export else 8}px;
              }}
              .turn {{
                margin-bottom: 16px;
              }}
              .role {{
                color: #525866;
                font-weight: 600;
                margin-bottom: 4px;
              }}
              .message {{
                border: 1px solid #d7dce2;
                border-radius: 8px;
                padding: 10px 12px;
              }}
              .user .message {{
                background: #f2f6fb;
              }}
              .assistant .message {{
                background: #ffffff;
              }}
              .citations {{
                color: #333842;
                font-size: 9pt;
                margin-top: 8px;
              }}
              .citation-title {{
                font-weight: 600;
                margin-bottom: 4px;
              }}
              p {{
                margin: 0 0 8px 0;
              }}
            </style>
          </head>
          <body>
            {body}
          </body>
        </html>
        """

    def render_turn_html(self, turn: ChatExchange, turn_index: int) -> str:
        question_html = self.render_plain_text(turn.question)
        answer_html = self.render_markdown_answer(turn.answer, turn.results, turn_index)
        process_html = self.render_process_feedback_html(turn.process_feedback)
        ledger_html = self.render_evidence_ledger_html(turn.evidence_ledger)
        citation_html = self.render_citations_html(turn.results, turn_index)
        if citation_html:
            citation_html = f'<div class="citations"><div class="citation-title">Citations</div>{citation_html}</div>'
        return f"""
        <div class="turn user">
          <div class="role">You</div>
          <div class="message">{question_html}</div>
        </div>
        <div class="turn assistant">
          <div class="role">Assistant</div>
          <div class="message">{answer_html}{process_html}{ledger_html}</div>
          {citation_html}
        </div>
        """

    def render_answer_html(self, answer: str) -> str:
        linked_answer = self.render_markdown_answer(answer, self.results, self.active_turn_index)
        citation_html = self.render_citations_html(self.results, self.active_turn_index)
        if citation_html:
            citation_html = f"<h3>Citations</h3>{citation_html}"
        return f"""
        <html>
          <body>
            <div>{linked_answer}</div>
            {citation_html}
          </body>
        </html>
        """

    @staticmethod
    def render_plain_text(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    def render_markdown_answer(
        self,
        answer: str,
        results: list[SearchResult] | None = None,
        turn_index: int | None = None,
    ) -> str:
        markdown = self.link_answer_citations(answer, results or [], turn_index)
        document = QTextDocument()
        document.setMarkdown(markdown)
        return self.html_body(document.toHtml())

    def link_answer_citations(
        self,
        answer: str,
        results: list[SearchResult],
        turn_index: int | None,
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            citation_number = int(match.group(1))
            row = citation_number - 1
            if 0 <= row < len(results):
                return f"[[{citation_number}]]({self.link_href('source', turn_index, row)})"
            return match.group(0)

        return CITATION_REF_RE.sub(replace, answer)

    @staticmethod
    def html_body(document_html: str) -> str:
        match = re.search(r"<body[^>]*>(.*)</body>", document_html, flags=re.DOTALL)
        if match is None:
            return document_html
        return match.group(1)

    def render_citations_html(
        self,
        results: list[SearchResult] | None = None,
        turn_index: int | None = None,
    ) -> str:
        citation_results = results if results is not None else self.results
        lines = []
        for row, result in enumerate(citation_results):
            citation_number = row + 1
            location = html.escape(self.location_label(result))
            path = html.escape(result.path)
            source = html.escape(result.source_type)
            lines.append(
                "<p>"
                f'<a href="{self.link_href("source", turn_index, row)}">[{citation_number}]</a> '
                f'<a href="{self.link_href("source", turn_index, row)}">{path}</a> '
                f"({source}, {location}) "
                f'<a href="{self.link_href("chunk", turn_index, row)}">view chunk</a>'
                "</p>"
            )
        return "\n".join(lines)

    def render_process_feedback_html(self, feedback: list[str] | None) -> str:
        if not feedback:
            return ""
        items = "".join(f"<li>{html.escape(item)}</li>" for item in feedback)
        return f'<div class="citations"><div class="citation-title">Process</div><ul>{items}</ul></div>'

    def render_evidence_ledger_html(self, ledger: list[dict[str, object]] | None) -> str:
        if ledger is None:
            return ""
        if not ledger:
            return '<div class="citations"><div class="citation-title">Evidence ledger</div><p>No evidence entries were added.</p></div>'
        rows = []
        for index, entry in enumerate(ledger, start=1):
            claim = html.escape(str(entry.get("claim", "")))
            support_type = html.escape(str(entry.get("support_type", "")))
            confidence = html.escape(str(entry.get("confidence", "")))
            chunk_id = html.escape(str(entry.get("source_chunk_id", "")))
            source_path = html.escape(str(entry.get("source_path", "")))
            source_location = html.escape(str(entry.get("source_location", "")))
            why = html.escape(str(entry.get("why_relevant", "")))
            quote = html.escape(str(entry.get("quoted_support", "")))
            rows.append(
                "<tr>"
                f"<td>E{index}</td>"
                f"<td>{claim}</td>"
                f"<td>{support_type}</td>"
                f"<td>{confidence}</td>"
                f"<td>{source_path}<br>{source_location}<br><code>{chunk_id}</code></td>"
                f"<td>{why}<br><em>{quote}</em></td>"
                "</tr>"
            )
        return (
            '<div class="citations"><div class="citation-title">Evidence ledger</div>'
            '<table border="1" cellspacing="0" cellpadding="4">'
            "<tr><th>ID</th><th>Claim</th><th>Type</th><th>Confidence</th><th>Source</th><th>Relevance</th></tr>"
            f"{''.join(rows)}</table></div>"
        )

    @staticmethod
    def link_href(action: str, turn_index: int | None, row: int) -> str:
        if turn_index is None:
            return f"{action}:{row}"
        return f"{action}:{turn_index}:{row}"

    def populate_citations(self, select_row: int = 0) -> None:
        self.citation_table.setRowCount(len(self.results))
        for row, result in enumerate(self.results):
            self.citation_table.setItem(row, 0, QTableWidgetItem(f"[{row + 1}]"))
            self.citation_table.setItem(row, 1, QTableWidgetItem(result.source_type))
            self.citation_table.setItem(row, 2, QTableWidgetItem(self.location_label(result)))
            self.citation_table.setItem(row, 3, QTableWidgetItem(result.path))
        if self.results:
            select_row = min(max(select_row, 0), len(self.results) - 1)
            self.citation_table.selectRow(select_row)
        else:
            self.chunk_preview.clear()

    def selected_result(self) -> SearchResult | None:
        row = self.citation_table.currentRow()
        if row < 0 or row >= len(self.results):
            return None
        return self.results[row]

    def select_citation(self, turn_index: int, row: int) -> None:
        if turn_index < 0 or turn_index >= len(self.turns):
            return
        results = self.turns[turn_index].results
        if row < 0 or row >= len(results):
            return
        self.active_turn_index = turn_index
        self.results = results
        self.populate_citations(row)
        item = self.citation_table.item(row, 0)
        if item is not None:
            self.citation_table.scrollToItem(item)
        self.update_chunk_preview()

    def update_chunk_preview(self) -> None:
        result = self.selected_result()
        if result is None:
            self.chunk_preview.clear()
            return
        if result.display_markdown:
            self.chunk_preview.setMarkdown(result.display_markdown)
        else:
            self.chunk_preview.setPlainText(result.chunk_text)

    def handle_answer_link(self, url: QUrl) -> None:
        action, _, value = url.toString().partition(":")
        target = self.parse_link_target(value)
        if target is None:
            return
        turn_index, row = target
        result = self.result_for_link(turn_index, row)
        if result is None:
            return
        if action == "source":
            self.open_result_file(result)
        elif action == "chunk":
            self.select_citation(turn_index, row)

    def parse_link_target(self, value: str) -> tuple[int, int] | None:
        parts = value.split(":")
        try:
            if len(parts) == 1:
                if self.active_turn_index is None:
                    return None
                return self.active_turn_index, int(parts[0])
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except ValueError:
            return None
        return None

    def result_for_link(self, turn_index: int, row: int) -> SearchResult | None:
        if turn_index < 0 or turn_index >= len(self.turns):
            return None
        results = self.turns[turn_index].results
        if row < 0 or row >= len(results):
            return None
        return results[row]

    def open_selected_file(self) -> None:
        result = self.selected_result()
        if result is not None:
            self.open_result_file(result)

    def open_selected_folder(self) -> None:
        result = self.selected_result()
        if result is None:
            return
        try:
            open_containing_folder(result.path)
        except Exception as exc:
            QMessageBox.warning(self, "Open folder", str(exc))

    def open_result_file(self, result: SearchResult) -> None:
        try:
            open_file(result.path)
        except Exception as exc:
            QMessageBox.warning(self, "Open file", str(exc))

    @staticmethod
    def location_label(result: SearchResult) -> str:
        if result.page_number:
            return f"page {result.page_number}"
        return f"chunk {result.chunk_index + 1}"
