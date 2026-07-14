from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
from investigation_rag.rag.strict_answering import (
    ANSWER_STYLE_BOTH,
    ANSWER_STYLE_LENIENT,
    ANSWER_STYLE_STRICT,
    StrictAnswerer,
)
from investigation_rag.utils.windows_open import open_containing_folder, open_file


CITATION_REF_RE = re.compile(r"\[(\d+)\]")
DEFAULT_ANSWER_MODEL = "qwen3.6:latest"


@dataclass
class ChatExchange:
    question: str
    answer: str
    results: list[SearchResult]
    retrieval_query: str
    answer_style: str = ANSWER_STYLE_STRICT
    evidence_ledger: list[dict[str, object]] | None = None
    process_feedback: list[str] | None = None


class SubmitTextEdit(QTextEdit):
    submitted = pyqtSignal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class OllamaModelsWorker(QObject):
    finished = pyqtSignal(list, str)

    def __init__(self, base_url: str, timeout: int, allow_remote: bool):
        super().__init__()
        self.base_url = base_url
        self.timeout = timeout
        self.allow_remote = allow_remote

    def run(self) -> None:
        try:
            client = OllamaClient(self.base_url, timeout=self.timeout, allow_remote=self.allow_remote)
            models = client.list_models()
        except Exception as exc:
            self.finished.emit([], str(exc))
            return

        answer_models = []
        capability_errors = 0
        for model in models:
            model_name = str(model.get("name") or model.get("model") or "").strip()
            if not model_name:
                continue
            model_details = dict(model)
            try:
                show_payload = client.show_model(model_name)
            except Exception:
                capability_errors += 1
                show_payload = {}

            capabilities = show_payload.get("capabilities")
            if isinstance(capabilities, list):
                model_details["capabilities"] = capabilities
                if not answer_model_is_usable(model_details):
                    continue
            answer_models.append(model_details)

        if capability_errors and answer_models:
            status = f"Loaded {len(answer_models)} answer model(s); some capabilities could not be verified."
        elif capability_errors:
            status = "Loaded models, but capabilities could not be verified."
        elif answer_models:
            status = f"Loaded {len(answer_models)} answer model(s)."
        else:
            status = "No answer-capable Ollama models found."
        self.finished.emit(answer_models, status)


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
        self._is_busy = False
        self.model_thread: QThread | None = None
        self.model_worker: OllamaModelsWorker | None = None

        self.question_edit = SubmitTextEdit()
        self.question_edit.setObjectName("Composer")
        self.question_edit.setFixedHeight(96)
        self.question_edit.setPlaceholderText("Ask a question")
        self.question_edit.submitted.connect(self.ask)
        self.question_edit.textChanged.connect(self.update_ask_button_state)
        self.ask_button = QPushButton("Ask")
        self.ask_button.setObjectName("PrimaryButton")
        self.ask_button.setMinimumWidth(96)
        self.ask_button.setMinimumHeight(38)
        self.ask_button.clicked.connect(self.ask)
        new_button = QPushButton("New conversation")
        new_button.setObjectName("GhostButton")
        new_button.clicked.connect(self.new_conversation)
        save_button = QPushButton("Save to PDF")
        save_button.setObjectName("GhostButton")
        save_button.clicked.connect(self.save_conversation_pdf)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(190)
        self.profile_combo.addItem("Standard retrieval", "standard")
        self.profile_combo.addItem("Deep retrieval", "deep")
        self.profile_combo.addItem("Evidence ledger RAG", "ledger")
        self.profile_combo.addItem("Legacy retrieval", "legacy")
        self.answer_style_combo = QComboBox()
        self.answer_style_combo.setMinimumWidth(150)
        self.answer_style_combo.addItem("Strict answer", ANSWER_STYLE_STRICT)
        self.answer_style_combo.addItem("Lenient answer", ANSWER_STYLE_LENIENT)
        self.answer_style_combo.addItem("Both answers", ANSWER_STYLE_BOTH)
        self.answer_model_combo = QComboBox()
        self.answer_model_combo.setMinimumWidth(260)
        self.answer_model_combo.addItem(DEFAULT_ANSWER_MODEL, DEFAULT_ANSWER_MODEL)
        if self.config.ollama_llm_model and self.config.ollama_llm_model != DEFAULT_ANSWER_MODEL:
            self.answer_model_combo.addItem(f"{self.config.ollama_llm_model} (.env)", self.config.ollama_llm_model)
        self.model_status_label = QLabel("Loading Ollama models...")
        self.model_status_label.setObjectName("StatusLabel")
        self.refresh_models_button = QPushButton("Refresh models")
        self.refresh_models_button.setObjectName("GhostButton")
        self.refresh_models_button.clicked.connect(self.refresh_answer_models)

        self.open_button = QPushButton("Open file")
        self.open_button.setObjectName("GhostButton")
        self.open_button.clicked.connect(self.open_selected_file)
        self.folder_button = QPushButton("Open folder")
        self.folder_button.setObjectName("GhostButton")
        self.folder_button.clicked.connect(self.open_selected_folder)

        self.answer_view = QTextBrowser()
        self.answer_view.setReadOnly(True)
        self.answer_view.setPlaceholderText("Ask a question to start a conversation")
        self.answer_view.setOpenExternalLinks(False)
        self.answer_view.setOpenLinks(False)
        self.answer_view.anchorClicked.connect(self.handle_answer_link)

        self.citation_table = QTableWidget(0, 4)
        self.citation_table.setHorizontalHeaderLabels(["#", "Source", "Location", "Path"])
        self.citation_table.setAlternatingRowColors(True)
        self.citation_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.citation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.citation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.citation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.citation_table.itemSelectionChanged.connect(self.update_chunk_preview)

        self.chunk_preview = QTextEdit()
        self.chunk_preview.setReadOnly(True)
        self.chunk_preview.setPlaceholderText("Select a citation to preview the source text")

        citation_buttons = QHBoxLayout()
        citation_buttons.setSpacing(8)
        citation_buttons.addWidget(self.open_button)
        citation_buttons.addWidget(self.folder_button)
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
        controls.setSpacing(8)
        controls.addWidget(new_button)
        controls.addWidget(save_button)
        controls.addStretch(1)
        controls.addWidget(self.profile_combo)
        controls.addWidget(self.answer_style_combo)
        controls.addWidget(self.answer_model_combo)
        controls.addWidget(self.refresh_models_button)

        composer = QWidget()
        composer.setObjectName("ComposerBar")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(0, 12, 0, 0)
        composer_layout.setSpacing(10)
        composer_layout.addWidget(self.question_edit, 1)
        composer_layout.addWidget(self.ask_button, 0, Qt.AlignBottom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addLayout(controls)
        layout.addWidget(self.model_status_label)
        layout.addWidget(result_splitter, 1)
        layout.addWidget(composer)
        self.update_ask_button_state()
        self.update_citation_action_buttons()
        self.refresh_answer_models()

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
        self.update_ask_button_state()
        self.update_citation_action_buttons()

    def update_ask_button_state(self) -> None:
        can_configure = not self._is_busy
        can_ask = can_configure and self.investigation_id is not None
        self.question_edit.setEnabled(can_ask)
        self.profile_combo.setEnabled(can_configure)
        self.answer_style_combo.setEnabled(can_configure)
        self.answer_model_combo.setEnabled(can_configure)
        self.refresh_models_button.setEnabled(can_configure)
        self.ask_button.setEnabled(can_ask and bool(self.question_edit.toPlainText().strip()))

    def set_busy(self, is_busy: bool) -> None:
        self._is_busy = is_busy
        self.ask_button.setText("Working..." if is_busy else "Ask")
        self.update_ask_button_state()

    def update_citation_action_buttons(self) -> None:
        has_selection = self.selected_result() is not None
        self.open_button.setEnabled(has_selection)
        self.folder_button.setEnabled(has_selection)

    def ask(self) -> None:
        if self.investigation_id is None:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            self.question_edit.setFocus()
            return
        history = [ConversationTurn(turn.question, turn.answer) for turn in self.turns]
        self.set_busy(True)
        try:
            llm = OllamaLLM(
                OllamaClient(
                    self.config.ollama_base_url,
                    timeout=self.config.ollama_timeout_seconds,
                    allow_remote=self.config.ollama_allow_remote,
                ),
                self.selected_answer_model(),
            )
            profile_name = str(self.profile_combo.currentData())
            answer_style = self.selected_answer_style()
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
                    answer_style=answer_style,
                )
            else:
                response = answerer.answer(
                    self.investigation_id,
                    question,
                    history=history,
                    folder_paths=folder_paths,
                    answer_style=answer_style,
                )
        except Exception as exc:
            QMessageBox.critical(self, "RAG", str(exc))
            return
        finally:
            self.set_busy(False)
        self.turns.append(
            ChatExchange(
                question=question,
                answer=response.answer,
                results=response.results,
                retrieval_query=response.retrieval_query,
                answer_style=response.answer_style,
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

    def selected_answer_model(self) -> str:
        data = self.answer_model_combo.currentData()
        if data:
            return str(data)
        text = self.answer_model_combo.currentText().strip()
        if text:
            return text
        return self.config.ollama_llm_model or DEFAULT_ANSWER_MODEL

    def selected_answer_style(self) -> str:
        data = self.answer_style_combo.currentData()
        if data:
            return str(data)
        return ANSWER_STYLE_STRICT

    def refresh_answer_models(self) -> None:
        if self.model_thread is not None:
            return
        self.model_status_label.setText("Loading Ollama models...")
        self.refresh_models_button.setEnabled(False)
        thread = QThread(self)
        worker = OllamaModelsWorker(
            self.config.ollama_base_url,
            timeout=min(max(self.config.ollama_timeout_seconds, 1), 15),
            allow_remote=self.config.ollama_allow_remote,
        )
        self.model_thread = thread
        self.model_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.apply_answer_models)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.answer_model_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def answer_model_thread_finished(self) -> None:
        self.model_thread = None
        self.model_worker = None
        self.update_ask_button_state()

    def apply_answer_models(self, models: list[dict[str, object]], status: str) -> None:
        previous_model = self.selected_answer_model()
        self.answer_model_combo.blockSignals(True)
        self.answer_model_combo.clear()
        answer_models = [model for model in models if answer_model_is_usable(model)]
        for model in sorted(answer_models, key=lambda item: str(item.get("name") or item.get("model") or "").casefold()):
            model_name = str(model.get("name") or model.get("model") or "").strip()
            if not model_name:
                continue
            self.answer_model_combo.addItem(answer_model_label(model), model_name)

        if self.answer_model_combo.count() == 0:
            fallback_model = self.config.ollama_llm_model or DEFAULT_ANSWER_MODEL
            self.answer_model_combo.addItem(f"{fallback_model} (fallback)", fallback_model)

        self.select_answer_model(DEFAULT_ANSWER_MODEL)
        if self.answer_model_combo.currentData() != DEFAULT_ANSWER_MODEL:
            self.select_answer_model(self.config.ollama_llm_model)
        if self.answer_model_combo.currentData() not in {DEFAULT_ANSWER_MODEL, self.config.ollama_llm_model}:
            self.select_answer_model(previous_model)
        self.answer_model_combo.blockSignals(False)
        self.model_status_label.setText(status)
        self.update_ask_button_state()

    def select_answer_model(self, model_name: str) -> None:
        if not model_name:
            return
        for index in range(self.answer_model_combo.count()):
            if self.answer_model_combo.itemData(index) == model_name:
                self.answer_model_combo.setCurrentIndex(index)
                return

    def shutdown(self, wait_ms: int = 3000) -> None:
        if self.model_thread is None or not self.model_thread.isRunning():
            return
        self.model_thread.quit()
        self.model_thread.wait(wait_ms)

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
        self.update_citation_action_buttons()

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
        self.update_citation_action_buttons()
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


def answer_model_label(model: dict[str, object]) -> str:
    name = str(model.get("name") or model.get("model") or "").strip()
    details = model.get("details")
    parts = []
    if isinstance(details, dict):
        parameter_size = details.get("parameter_size")
        quantization = details.get("quantization_level")
        if parameter_size:
            parts.append(str(parameter_size))
        if quantization:
            parts.append(str(quantization))
    return f"{name} ({', '.join(parts)})" if parts else name


def answer_model_is_usable(model: dict[str, object]) -> bool:
    capabilities = model.get("capabilities")
    if not isinstance(capabilities, list):
        return True
    capability_names = {str(capability).casefold() for capability in capabilities}
    return "completion" in capability_names
