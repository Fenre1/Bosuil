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
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.gui.file_type_scope import FileTypeScope
from investigation_rag.gui.folder_scope import FolderScope
from investigation_rag.gui.chunk_preview import ChunkPreview
from investigation_rag.rag.prompt_builder import ConversationTurn
from investigation_rag.rag.runner import LEDGER_PROFILE_NAMES, RagRunner
from investigation_rag.retrieval.ranking import SearchResult, result_location_label
from investigation_rag.rag.strict_answering import (
    ANSWER_STYLE_BOTH,
    ANSWER_STYLE_LENIENT,
    RAGAnswer,
    ANSWER_STYLE_STRICT,
)
from investigation_rag.utils.windows_open import open_containing_folder, open_file


CITATION_REF_RE = re.compile(r"\[(\d+)\]")
LEDGER_REF_RE = re.compile(r"\[E(\d+)\]", re.IGNORECASE)
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


@dataclass(frozen=True)
class RagAskResult:
    request_id: int
    investigation_id: int
    question: str
    response: RAGAnswer | None = None
    error: str = ""


class RagAskWorker(QObject):
    progress = pyqtSignal(int, str)
    ledger_updated = pyqtSignal(int, object)
    finished = pyqtSignal(object)

    def __init__(
        self,
        config: AppConfig,
        request_id: int,
        investigation_id: int,
        question: str,
        history: list[ConversationTurn],
        profile_name: str,
        answer_style: str,
        answer_model: str,
        folder_paths: list[str] | None,
        source_types: list[str] | None,
    ):
        super().__init__()
        self.config = config
        self.request_id = request_id
        self.investigation_id = investigation_id
        self.question = question
        self.history = history
        self.profile_name = profile_name
        self.answer_style = answer_style
        self.answer_model = answer_model
        self.folder_paths = folder_paths
        self.source_types = source_types

    def run(self) -> None:
        try:
            with RagRunner(
                self.config,
                self.investigation_id,
                self.profile_name,
                self.answer_style,
                self.answer_model,
                self.folder_paths,
                self.source_types,
            ) as runner:
                response = runner.answer(
                    self.question,
                    history=self.history,
                    progress_callback=lambda message: self.progress.emit(self.request_id, message),
                    ledger_callback=lambda ledger: self.ledger_updated.emit(self.request_id, ledger),
                )
            result = RagAskResult(
                request_id=self.request_id,
                investigation_id=self.investigation_id,
                question=self.question,
                response=response,
            )
        except Exception as exc:
            result = RagAskResult(
                request_id=self.request_id,
                investigation_id=self.investigation_id,
                question=self.question,
                error=str(exc),
            )
        self.finished.emit(result)


class RagPanel(QWidget):
    def __init__(
        self,
        config: AppConfig,
        db: SQLiteDB,
        store: LanceDBStore,
        folder_scope: FolderScope | None = None,
        file_type_scope: FileTypeScope | None = None,
    ):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.folder_scope = folder_scope
        self.file_type_scope = file_type_scope
        self.investigation_id: int | None = None
        self.results: list[SearchResult] = []
        self.turns: list[ChatExchange] = []
        self.active_turn_index: int | None = None
        self._ledger_progress_lines: list[str] = []
        self._pending_question = ""
        self._pending_progress_lines: list[str] = []
        self._pending_ledger: list[dict[str, object]] | None = None
        self._ask_request_id = 0
        self._updating_ledger_table = False
        self._is_busy = False
        self.model_thread: QThread | None = None
        self.model_worker: OllamaModelsWorker | None = None
        self.ask_thread: QThread | None = None
        self.ask_worker: RagAskWorker | None = None

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
        self.new_button = QPushButton("New conversation")
        self.new_button.setObjectName("GhostButton")
        self.new_button.clicked.connect(self.new_conversation)
        self.save_button = QPushButton("Save to PDF")
        self.save_button.setObjectName("GhostButton")
        self.save_button.clicked.connect(self.save_conversation_pdf)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(190)
        self.profile_combo.addItem("Standard retrieval", "standard")
        self.profile_combo.addItem("Deep retrieval", "deep")
        self.profile_combo.addItem("Evidence ledger RAG", "ledger")
        self.profile_combo.addItem("Hyde", "hyde")
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

        self.process_view = QTextEdit()
        self.process_view.setReadOnly(True)
        self.process_view.setPlaceholderText("Process details will appear here")

        self.ledger_table = QTableWidget(0, 4)
        self.ledger_table.setHorizontalHeaderLabels(["ID", "Claim", "Source", "Relevance"])
        self.ledger_table.setAlternatingRowColors(True)
        self.ledger_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.ledger_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ledger_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ledger_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ledger_table.itemSelectionChanged.connect(self.select_selected_ledger_source)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setMinimumWidth(320)
        self.detail_tabs.addTab(self.process_view, "Process")
        self.detail_tabs.addTab(self.ledger_table, "Evidence ledger")

        self.citation_table = QTableWidget(0, 4)
        self.citation_table.setHorizontalHeaderLabels(["#", "Source", "Location", "Path"])
        self.citation_table.setAlternatingRowColors(True)
        self.citation_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.citation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.citation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.citation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.citation_table.itemSelectionChanged.connect(self.update_chunk_preview)

        self.chunk_preview = ChunkPreview("Select a citation to preview the source text")
        self.chunk_preview.previous_requested.connect(lambda: self.select_adjacent_citation(-1))
        self.chunk_preview.next_requested.connect(lambda: self.select_adjacent_citation(1))

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
        answer_detail_splitter = QSplitter(Qt.Horizontal)
        answer_detail_splitter.addWidget(self.answer_view)
        answer_detail_splitter.addWidget(self.detail_tabs)
        answer_detail_splitter.setStretchFactor(0, 3)
        answer_detail_splitter.setStretchFactor(1, 1)

        result_splitter.addWidget(answer_detail_splitter)
        result_splitter.addWidget(citation_container)
        result_splitter.setStretchFactor(0, 3)
        result_splitter.setStretchFactor(1, 2)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self.new_button)
        controls.addWidget(self.save_button)
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
        if self._is_busy:
            self._ask_request_id += 1
            self.clear_conversation()
            return
        self.new_conversation()

    def new_conversation(self) -> None:
        if self._is_busy:
            return
        self.clear_conversation()

    def clear_conversation(self) -> None:
        self.turns = []
        self.results = []
        self.active_turn_index = None
        self._pending_question = ""
        self._pending_progress_lines = []
        self._pending_ledger = None
        self.answer_view.clear()
        self.update_detail_panel()
        self.citation_table.setRowCount(0)
        self.chunk_preview.clear()
        self.question_edit.clear()
        self.update_ask_button_state()
        self.update_citation_action_buttons()

    def update_ask_button_state(self) -> None:
        can_configure = not self._is_busy
        can_ask = can_configure and self.investigation_id is not None
        self.question_edit.setEnabled(can_ask)
        self.new_button.setEnabled(can_configure)
        self.save_button.setEnabled(can_configure and bool(self.turns))
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
        if self.investigation_id is None or self.ask_thread is not None:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            self.question_edit.setFocus()
            return
        history = [ConversationTurn(turn.question, turn.answer) for turn in self.turns]
        profile_name = str(self.profile_combo.currentData())
        answer_style = self.selected_answer_style()
        answer_model = self.selected_answer_model()
        folder_paths = self.folder_scope.folder_paths() if self.folder_scope is not None else None
        source_types = self.file_type_scope.source_types() if self.file_type_scope is not None else None
        self._ask_request_id += 1
        request_id = self._ask_request_id
        self._pending_question = question
        self._pending_progress_lines = ["Answer task started in the background."]
        self._pending_ledger = [] if profile_name in LEDGER_PROFILE_NAMES else None
        self._ledger_progress_lines = self._pending_progress_lines
        self.set_busy(True)
        self.refresh_answer_view()
        self.question_edit.clear()
        self.start_ask_thread(
            request_id,
            self.investigation_id,
            question,
            history,
            profile_name,
            answer_style,
            answer_model,
            folder_paths,
            source_types,
        )

    def start_ask_thread(
        self,
        request_id: int,
        investigation_id: int,
        question: str,
        history: list[ConversationTurn],
        profile_name: str,
        answer_style: str,
        answer_model: str,
        folder_paths: list[str] | None,
        source_types: list[str] | None,
    ) -> None:
        thread = QThread(self)
        worker = RagAskWorker(
            self.config,
            request_id,
            investigation_id,
            question,
            history,
            profile_name,
            answer_style,
            answer_model,
            folder_paths,
            source_types,
        )
        self.ask_thread = thread
        self.ask_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.apply_ask_progress)
        worker.ledger_updated.connect(self.apply_ask_ledger)
        worker.finished.connect(self.apply_ask_result)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.ask_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def apply_ask_progress(self, request_id: int, message: str) -> None:
        if request_id != self._ask_request_id:
            return
        self._pending_progress_lines.append(message)
        self._ledger_progress_lines = self._pending_progress_lines
        self.refresh_answer_view()

    def apply_ask_ledger(self, request_id: int, ledger: object) -> None:
        if request_id != self._ask_request_id:
            return
        if isinstance(ledger, list):
            self._pending_ledger = [dict(entry) for entry in ledger if isinstance(entry, dict)]
            self.refresh_answer_view()

    def apply_ask_result(self, result: RagAskResult) -> None:
        if result.request_id != self._ask_request_id or result.investigation_id != self.investigation_id:
            return
        if result.error or result.response is None:
            self._pending_question = ""
            self._pending_progress_lines = []
            self._pending_ledger = None
            self.refresh_answer_view()
            QMessageBox.critical(self, "RAG", result.error or "Answer task did not return a response.")
            return
        response = result.response
        self.turns.append(
            ChatExchange(
                question=result.question,
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
        self._pending_question = ""
        self._pending_progress_lines = []
        self._pending_ledger = None
        self.refresh_answer_view()
        self.populate_citations()

    def ask_thread_finished(self) -> None:
        self.ask_thread = None
        self.ask_worker = None
        self.set_busy(False)

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
            pass
        else:
            self.model_thread.quit()
            self.model_thread.wait(wait_ms)
        if self.ask_thread is not None and self.ask_thread.isRunning():
            self._ask_request_id += 1
            self.ask_thread.requestInterruption()
            self.ask_thread.quit()
            self.ask_thread.wait(wait_ms)

    def refresh_answer_view(self) -> None:
        self.answer_view.setHtml(self.render_conversation_html())
        self.answer_view.verticalScrollBar().setValue(self.answer_view.verticalScrollBar().maximum())
        self.update_detail_panel()

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
        rendered_turns = [self.render_turn_html(turn, index, for_export=for_export) for index, turn in enumerate(self.turns)]
        if not for_export and self._pending_question:
            rendered_turns.append(self.render_pending_turn_html())
        body = "\n".join(rendered_turns)
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

    def render_pending_turn_html(self) -> str:
        question_html = self.render_plain_text(self._pending_question)
        return f"""
        <div class="turn user">
          <div class="role">You</div>
          <div class="message">{question_html}</div>
        </div>
        <div class="turn assistant">
          <div class="role">Assistant</div>
          <div class="message"><p>Working in the background...</p></div>
        </div>
        """

    def render_turn_html(self, turn: ChatExchange, turn_index: int, for_export: bool = False) -> str:
        question_html = self.render_plain_text(turn.question)
        answer_html = self.render_markdown_answer(turn.answer, turn.results, turn_index, turn.evidence_ledger)
        process_html = self.render_process_feedback_html(turn.process_feedback) if for_export else ""
        ledger_html = self.render_evidence_ledger_html(turn.evidence_ledger) if for_export else ""
        citation_html = self.render_citations_html(turn.results, turn_index) if for_export else ""
        if for_export and citation_html:
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
        ledger = self.active_turn().evidence_ledger if self.active_turn() is not None else None
        linked_answer = self.render_markdown_answer(answer, self.results, self.active_turn_index, ledger)
        return f"""
        <html>
          <body>
            <div>{linked_answer}</div>
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
        ledger: list[dict[str, object]] | None = None,
    ) -> str:
        markdown = self.link_answer_citations(answer, results or [], turn_index, ledger)
        document = QTextDocument()
        document.setMarkdown(markdown)
        return self.html_body(document.toHtml())

    def link_answer_citations(
        self,
        answer: str,
        results: list[SearchResult],
        turn_index: int | None,
        ledger: list[dict[str, object]] | None,
    ) -> str:
        def replace_ledger(match: re.Match[str]) -> str:
            ledger_number = int(match.group(1))
            row = ledger_number - 1
            if ledger is not None and 0 <= row < len(ledger):
                return f"[[E{ledger_number}]]({self.link_href('ledger', turn_index, row)})"
            return match.group(0)

        def replace_numeric(match: re.Match[str]) -> str:
            citation_number = int(match.group(1))
            row = citation_number - 1
            if 0 <= row < len(results):
                return f"[[{citation_number}]]({self.link_href('chunk', turn_index, row)})"
            return match.group(0)

        with_ledger_links = LEDGER_REF_RE.sub(replace_ledger, answer)
        return CITATION_REF_RE.sub(replace_numeric, with_ledger_links)

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

    def render_evidence_ledger_html(self, ledger: list[dict[str, object]] | None, pending: bool = False) -> str:
        if ledger is None:
            return ""
        if not ledger:
            message = "No evidence entries yet." if pending else "No evidence entries were added."
            return f'<div class="citations"><div class="citation-title">Evidence ledger</div><p>{message}</p></div>'
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

    def update_detail_panel(self, selected_ledger_row: int | None = None) -> None:
        if self._pending_question:
            process_feedback = self._pending_progress_lines
            ledger = self._pending_ledger
        else:
            turn = self.active_turn()
            process_feedback = turn.process_feedback if turn is not None else None
            ledger = turn.evidence_ledger if turn is not None else None
        self.populate_process_feedback(process_feedback)
        self.populate_ledger_table(ledger, selected_ledger_row=selected_ledger_row)

    def populate_process_feedback(self, feedback: list[str] | None) -> None:
        if not feedback:
            self.process_view.clear()
            self.detail_tabs.setTabText(0, "Process")
            return
        self.process_view.setPlainText("\n".join(f"- {line}" for line in feedback))
        self.detail_tabs.setTabText(0, f"Process ({len(feedback)})")

    def populate_ledger_table(
        self,
        ledger: list[dict[str, object]] | None,
        selected_ledger_row: int | None = None,
    ) -> None:
        self._updating_ledger_table = True
        try:
            rows = ledger or []
            self.ledger_table.setRowCount(len(rows))
            self.ledger_table.clearSelection()
            for row, entry in enumerate(rows):
                claim = str(entry.get("claim", "")).strip()
                source = str(entry.get("source_path", "")).strip()
                location = str(entry.get("source_location", "")).strip()
                relevance = str(entry.get("why_relevant", "")).strip()
                quote = str(entry.get("quoted_support", "")).strip()
                source_text = "\n".join(part for part in (source, location) if part)
                relevance_text = "\n".join(part for part in (relevance, quote) if part)
                id_item = QTableWidgetItem(f"E{row + 1}")
                id_item.setData(Qt.UserRole, str(entry.get("source_chunk_id", "")).strip())
                self.ledger_table.setItem(row, 0, id_item)
                self.ledger_table.setItem(row, 1, QTableWidgetItem(claim))
                self.ledger_table.setItem(row, 2, QTableWidgetItem(source_text))
                self.ledger_table.setItem(row, 3, QTableWidgetItem(relevance_text))
            if rows and selected_ledger_row is not None:
                self.ledger_table.selectRow(min(max(selected_ledger_row, 0), len(rows) - 1))
            elif not rows:
                self.ledger_table.clearSelection()
            self.ledger_table.resizeRowsToContents()
        finally:
            self._updating_ledger_table = False
        self.detail_tabs.setTabText(1, f"Evidence ledger ({len(ledger or [])})")

    def active_turn(self) -> ChatExchange | None:
        if self.active_turn_index is None:
            return None
        if self.active_turn_index < 0 or self.active_turn_index >= len(self.turns):
            return None
        return self.turns[self.active_turn_index]

    def select_selected_ledger_source(self) -> None:
        if self._updating_ledger_table or self._pending_question:
            return
        row = self.ledger_table.currentRow()
        if row < 0 or self.active_turn_index is None:
            return
        self.select_ledger_entry(self.active_turn_index, row)

    def select_ledger_entry(self, turn_index: int, ledger_row: int) -> None:
        if turn_index < 0 or turn_index >= len(self.turns):
            return
        turn = self.turns[turn_index]
        ledger = turn.evidence_ledger or []
        if ledger_row < 0 or ledger_row >= len(ledger):
            return
        source_row = self.source_row_for_ledger_entry(turn, ledger_row)
        self.active_turn_index = turn_index
        self.results = turn.results
        if source_row is not None:
            self.populate_citations(source_row)
            item = self.citation_table.item(source_row, 0)
            if item is not None:
                self.citation_table.scrollToItem(item)
            self.update_chunk_preview()
        else:
            self.populate_citations(0)
        self.update_detail_panel(selected_ledger_row=ledger_row)

    @staticmethod
    def source_row_for_ledger_entry(turn: ChatExchange, ledger_row: int) -> int | None:
        ledger = turn.evidence_ledger or []
        if ledger_row < 0 or ledger_row >= len(ledger):
            return None
        chunk_id = str(ledger[ledger_row].get("source_chunk_id", "")).strip()
        if not chunk_id:
            return None
        for row, result in enumerate(turn.results):
            if result.chunk_id == chunk_id:
                return row
        return None

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
        self.update_detail_panel()
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
        row = self.citation_table.currentRow()
        self.chunk_preview.set_chunk(
            result.chunk_text,
            result.display_markdown,
            path=result.path,
            chunk_index=result.chunk_index,
            metadata=result.metadata or {},
            has_previous=row > 0,
            has_next=row + 1 < len(self.results),
        )

    def select_adjacent_citation(self, offset: int) -> None:
        row = self.citation_table.currentRow() + offset
        if 0 <= row < len(self.results):
            self.citation_table.selectRow(row)

    def handle_answer_link(self, url: QUrl) -> None:
        action, _, value = url.toString().partition(":")
        target = self.parse_link_target(value)
        if target is None:
            return
        turn_index, row = target
        if action == "ledger":
            self.select_ledger_entry(turn_index, row)
            return
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
        return result_location_label(result)


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
