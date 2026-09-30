from __future__ import annotations

from datetime import datetime
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.evaluation import (
    DEFAULT_JUDGE_MODEL,
    EvaluationOutcome,
    EvaluationReportParameters,
    EvaluationRow,
    EvaluationWorkbook,
    build_judge_prompt,
    build_judge_retry_prompt,
    export_evaluation_report,
    format_points,
    load_evaluation_workbook,
    parse_judge_score,
    summarize_evaluation,
)
from investigation_rag.gui.file_type_scope import FileTypeScope
from investigation_rag.gui.folder_scope import FolderScope
from investigation_rag.gui.rag_panel import (
    DEFAULT_ANSWER_MODEL,
    OllamaModelsWorker,
    answer_model_is_usable,
    answer_model_label,
)
from investigation_rag.rag.runner import RagRunner
from investigation_rag.rag.strict_answering import ANSWER_STYLE_BOTH, ANSWER_STYLE_LENIENT, ANSWER_STYLE_STRICT


class EvaluationWorker(QObject):
    row_started = pyqtSignal(int, int, int, int)
    row_finished = pyqtSignal(int, object)
    status = pyqtSignal(int, str)
    finished = pyqtSignal(int, bool, str)

    def __init__(
        self,
        config: AppConfig,
        run_id: int,
        investigation_id: int,
        rows: tuple[EvaluationRow, ...],
        profile_name: str,
        answer_style: str,
        answer_model: str,
        folder_paths: list[str] | None,
        source_types: list[str] | None,
    ):
        super().__init__()
        self.config = config
        self.run_id = run_id
        self.investigation_id = investigation_id
        self.rows = rows
        self.profile_name = profile_name
        self.answer_style = answer_style
        self.answer_model = answer_model
        self.folder_paths = folder_paths
        self.source_types = source_types
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        runner: RagRunner | None = None
        judge_client: OllamaClient | None = None
        cancelled = False
        error = ""
        try:
            runner = RagRunner(
                self.config,
                self.investigation_id,
                self.profile_name,
                self.answer_style,
                self.answer_model,
                self.folder_paths,
                self.source_types,
            )
            judge_client = OllamaClient(
                self.config.ollama_base_url,
                timeout=self.config.ollama_timeout_seconds,
                allow_remote=self.config.ollama_allow_remote,
                num_ctx=self.config.ollama_num_ctx,
            )
            total = len(self.rows)
            for index, row in enumerate(self.rows, start=1):
                if self._cancel_requested or QThread.currentThread().isInterruptionRequested():
                    cancelled = True
                    break
                self.row_started.emit(self.run_id, index, total, row.worksheet_row)
                self.status.emit(self.run_id, f"Question {index} of {total}: running the RAG pipeline.")
                generated_answer = ""
                try:
                    response = runner.answer(
                        row.question,
                        history=[],
                        progress_callback=lambda message, i=index, count=total: self.status.emit(
                            self.run_id, f"Question {i} of {count}: {message}"
                        ),
                    )
                    generated_answer = response.answer
                    if self._cancel_requested or QThread.currentThread().isInterruptionRequested():
                        cancelled = True
                        break
                    self.status.emit(self.run_id, f"Question {index} of {total}: judging the answer.")
                    score = self._judge(judge_client, row, generated_answer)
                    outcome = EvaluationOutcome(
                        row=row,
                        generated_answer=generated_answer,
                        awarded_points=score,
                    )
                except Exception as exc:
                    outcome = EvaluationOutcome(
                        row=row,
                        generated_answer=generated_answer,
                        error=str(exc),
                    )
                self.row_finished.emit(self.run_id, outcome)
        except Exception as exc:
            error = str(exc)
        finally:
            if runner is not None:
                runner.close()
            if judge_client is not None:
                judge_client.close()
        self.finished.emit(self.run_id, cancelled, error)

    @staticmethod
    def _judge(client: OllamaClient, row: EvaluationRow, generated_answer: str):
        first_response = client.generate_detailed(
            DEFAULT_JUDGE_MODEL,
            build_judge_prompt(row, generated_answer),
            options={"temperature": 0, "num_predict": 16},
            think=False,
        ).response
        try:
            return parse_judge_score(first_response, row.max_points)
        except ValueError:
            second_response = client.generate_detailed(
                DEFAULT_JUDGE_MODEL,
                build_judge_retry_prompt(row, generated_answer, first_response),
                options={"temperature": 0, "num_predict": 16},
                think=False,
            ).response
            return parse_judge_score(second_response, row.max_points)


class EvaluationPanel(QWidget):
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
        self.evaluation: EvaluationWorkbook | None = None
        self.outcomes: dict[int, EvaluationOutcome] = {}
        self.run_parameters: EvaluationReportParameters | None = None
        self._table_rows: dict[int, int] = {}
        self._run_id = 0
        self._is_busy = False
        self.run_thread: QThread | None = None
        self.run_worker: EvaluationWorker | None = None
        self.model_thread: QThread | None = None
        self.model_worker: OllamaModelsWorker | None = None

        self.open_button = QPushButton("Open Excel")
        self.open_button.setObjectName("GhostButton")
        self.open_button.clicked.connect(self.choose_workbook)
        self.file_label = QLabel("No evaluation workbook loaded")
        self.file_label.setObjectName("StatusLabel")

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
        self.refresh_models_button = QPushButton("Refresh models")
        self.refresh_models_button.setObjectName("GhostButton")
        self.refresh_models_button.clicked.connect(self.refresh_answer_models)
        self.model_status_label = QLabel("Loading Ollama models...")
        self.model_status_label.setObjectName("StatusLabel")
        self.judge_label = QLabel(f"Judge model: {DEFAULT_JUDGE_MODEL}")
        self.judge_label.setObjectName("StatusLabel")

        self.start_button = QPushButton("Start evaluation")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_evaluation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("GhostButton")
        self.stop_button.clicked.connect(self.stop_evaluation)
        self.export_button = QPushButton("Export report")
        self.export_button.setObjectName("GhostButton")
        self.export_button.clicked.connect(self.export_report)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Load an Excel workbook to begin.")
        self.status_label.setObjectName("StatusLabel")
        self.summary_label = QLabel("")

        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            ["qID", "Question", "Ground truth", "Generated answer", "Score", "Max", "Status"]
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.file_label, 1)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(8)
        settings_row.addWidget(self.profile_combo)
        settings_row.addWidget(self.answer_style_combo)
        settings_row.addWidget(self.answer_model_combo)
        settings_row.addWidget(self.refresh_models_button)
        settings_row.addStretch(1)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.export_button)
        action_row.addStretch(1)
        action_row.addWidget(self.summary_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addLayout(file_row)
        layout.addLayout(settings_row)
        layout.addWidget(self.judge_label)
        layout.addWidget(self.model_status_label)
        layout.addLayout(action_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_table, 1)

        self.update_controls()
        self.update_summary()
        self.refresh_answer_models()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        if investigation_id == self.investigation_id:
            self.update_controls()
            return
        if self._is_busy:
            self.stop_evaluation()
        self._run_id += 1
        self.investigation_id = investigation_id
        self.outcomes.clear()
        self.run_parameters = None
        self.populate_table()
        if investigation_id is None:
            self.status_label.setText("Load an investigation before starting the evaluation.")
        else:
            self.status_label.setText("Ready to evaluate the loaded investigation.")
        self.update_controls()

    def choose_workbook(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open evaluation workbook",
            "",
            "Excel workbooks (*.xlsx)",
        )
        if not file_path:
            return
        try:
            evaluation = load_evaluation_workbook(file_path)
        except Exception as exc:
            QMessageBox.warning(self, "Evaluation workbook", str(exc))
            return
        self.evaluation = evaluation
        self.outcomes.clear()
        self.run_parameters = None
        self.file_label.setText(
            f"{evaluation.path.name} — {evaluation.sheet_name} — {len(evaluation.rows)} question(s)"
        )
        self.progress_bar.setRange(0, len(evaluation.rows))
        self.progress_bar.setValue(0)
        self.status_label.setText("Workbook loaded. Select settings and start the evaluation.")
        self.populate_table()
        self.update_controls()

    def populate_table(self) -> None:
        rows = self.evaluation.rows if self.evaluation is not None else ()
        self._table_rows.clear()
        self.results_table.setRowCount(len(rows))
        for table_row, evaluation_row in enumerate(rows):
            self._table_rows[evaluation_row.worksheet_row] = table_row
            self.results_table.setItem(table_row, 0, QTableWidgetItem(evaluation_row.display_id))
            self.results_table.setItem(table_row, 1, QTableWidgetItem(evaluation_row.question))
            self.results_table.setItem(table_row, 2, QTableWidgetItem(evaluation_row.ground_truth_answer))
            self.results_table.setItem(table_row, 3, QTableWidgetItem(""))
            self.results_table.setItem(table_row, 4, QTableWidgetItem(""))
            self.results_table.setItem(table_row, 5, QTableWidgetItem(format_points(evaluation_row.max_points)))
            self.results_table.setItem(table_row, 6, QTableWidgetItem("Pending"))
            outcome = self.outcomes.get(evaluation_row.worksheet_row)
            if outcome is not None:
                self.apply_outcome_to_table(outcome)
        self.results_table.resizeRowsToContents()
        self.update_summary()

    def start_evaluation(self) -> None:
        if self._is_busy or self.run_thread is not None:
            return
        if self.investigation_id is None:
            QMessageBox.information(self, "Evaluation", "Load an investigation first.")
            return
        if self.evaluation is None:
            QMessageBox.information(self, "Evaluation", "Open an evaluation workbook first.")
            return

        self._run_id += 1
        run_id = self._run_id
        self.outcomes.clear()
        self.run_parameters = EvaluationReportParameters(
            answer_model=self.selected_answer_model(),
            answer_style=str(self.answer_style_combo.currentData()),
            retrieval_type=str(self.profile_combo.currentData()),
            judge_model=DEFAULT_JUDGE_MODEL,
        )
        self.populate_table()
        self.progress_bar.setRange(0, len(self.evaluation.rows))
        self.progress_bar.setValue(0)
        folder_paths = self.folder_scope.folder_paths() if self.folder_scope is not None else None
        source_types = self.file_type_scope.source_types() if self.file_type_scope is not None else None
        worker = EvaluationWorker(
            self.config,
            run_id,
            self.investigation_id,
            self.evaluation.rows,
            str(self.profile_combo.currentData()),
            str(self.answer_style_combo.currentData()),
            self.run_parameters.answer_model,
            folder_paths,
            source_types,
        )
        thread = QThread(self)
        self.run_worker = worker
        self.run_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.row_started.connect(self.apply_row_started)
        worker.row_finished.connect(self.apply_row_finished)
        worker.status.connect(self.apply_status)
        worker.finished.connect(self.apply_run_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.run_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.set_busy(True)
        self.status_label.setText("Evaluation started.")
        thread.start()

    def stop_evaluation(self) -> None:
        if self.run_worker is None:
            return
        self.run_worker.cancel()
        if self.run_thread is not None:
            self.run_thread.requestInterruption()
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping after the current model request finishes...")

    def apply_row_started(self, run_id: int, index: int, total: int, worksheet_row: int) -> None:
        if run_id != self._run_id:
            return
        table_row = self._table_rows.get(worksheet_row)
        if table_row is not None:
            self.results_table.item(table_row, 6).setText("Running")
            self.results_table.selectRow(table_row)
        self.status_label.setText(f"Question {index} of {total}.")

    def apply_row_finished(self, run_id: int, outcome: EvaluationOutcome) -> None:
        if run_id != self._run_id:
            return
        self.outcomes[outcome.row.worksheet_row] = outcome
        self.apply_outcome_to_table(outcome)
        self.progress_bar.setValue(len(self.outcomes))
        self.update_summary()

    def apply_outcome_to_table(self, outcome: EvaluationOutcome) -> None:
        table_row = self._table_rows.get(outcome.row.worksheet_row)
        if table_row is None:
            return
        self.results_table.item(table_row, 3).setText(outcome.generated_answer)
        score = "" if outcome.awarded_points is None else format_points(outcome.awarded_points)
        self.results_table.item(table_row, 4).setText(score)
        status_item = self.results_table.item(table_row, 6)
        status_item.setText("Completed" if outcome.completed else "Error")
        status_item.setToolTip(outcome.error)
        self.results_table.resizeRowToContents(table_row)

    def apply_status(self, run_id: int, message: str) -> None:
        if run_id == self._run_id:
            self.status_label.setText(message)

    def apply_run_finished(self, run_id: int, cancelled: bool, error: str) -> None:
        if run_id != self._run_id:
            return
        if error:
            self.status_label.setText("Evaluation could not start.")
            QMessageBox.critical(self, "Evaluation", error)
        elif cancelled:
            self.status_label.setText(f"Evaluation stopped after {len(self.outcomes)} question(s).")
            for table_row in range(self.results_table.rowCount()):
                status_item = self.results_table.item(table_row, 6)
                if status_item is not None and status_item.text() == "Running":
                    status_item.setText("Stopped")
        else:
            self.status_label.setText("Evaluation completed.")
        self.set_busy(False)
        self.update_summary()

    def run_thread_finished(self) -> None:
        self.run_thread = None
        self.run_worker = None
        self.set_busy(False)

    def update_summary(self) -> None:
        if self.evaluation is None:
            self.summary_label.setText("")
            return
        summary = summarize_evaluation(self.evaluation.rows, self.outcomes)
        if not self.outcomes:
            self.summary_label.setText(f"Available points: {format_points(summary.possible_points)}")
            return
        text = (
            f"Score: {format_points(summary.earned_points)} / {format_points(summary.possible_points)}"
            f" — Accuracy: {summary.accuracy_percent:.1f}%"
            f" — Completed: {summary.completed_rows}/{summary.total_rows}"
        )
        if summary.failed_rows:
            text += f" — Errors: {summary.failed_rows}"
        self.summary_label.setText(text)

    def export_report(self) -> None:
        if self.evaluation is None or not self.outcomes or self.run_parameters is None:
            return
        default_name = f"{self.evaluation.path.stem}-report-{datetime.now():%Y%m%d-%H%M}.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export evaluation report",
            str(self.evaluation.path.with_name(default_name)),
            "Excel workbooks (*.xlsx)",
        )
        if not file_path:
            return
        try:
            exported_path = export_evaluation_report(
                self.evaluation,
                self.outcomes,
                file_path,
                self.run_parameters,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export evaluation report", str(exc))
            return
        QMessageBox.information(self, "Export evaluation report", f"Report saved to:\n{exported_path}")

    def selected_answer_model(self) -> str:
        data = self.answer_model_combo.currentData()
        if data:
            return str(data)
        text = self.answer_model_combo.currentText().strip()
        return text or self.config.ollama_llm_model or DEFAULT_ANSWER_MODEL

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

    def apply_answer_models(self, models: list[dict[str, object]], status: str) -> None:
        previous_model = self.selected_answer_model()
        self.answer_model_combo.blockSignals(True)
        self.answer_model_combo.clear()
        answer_models = [model for model in models if answer_model_is_usable(model)]
        for model in sorted(answer_models, key=lambda item: str(item.get("name") or item.get("model") or "").casefold()):
            model_name = str(model.get("name") or model.get("model") or "").strip()
            if model_name:
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

        available_names = {
            str(model.get("name") or model.get("model") or "").strip().casefold() for model in answer_models
        }
        if DEFAULT_JUDGE_MODEL.casefold() not in available_names:
            status += f" Judge model {DEFAULT_JUDGE_MODEL} was not found."
        self.model_status_label.setText(status)
        self.update_controls()

    def answer_model_thread_finished(self) -> None:
        self.model_thread = None
        self.model_worker = None
        self.update_controls()

    def select_answer_model(self, model_name: str) -> None:
        if not model_name:
            return
        for index in range(self.answer_model_combo.count()):
            if self.answer_model_combo.itemData(index) == model_name:
                self.answer_model_combo.setCurrentIndex(index)
                return

    def set_busy(self, is_busy: bool) -> None:
        self._is_busy = is_busy
        self.start_button.setText("Evaluating..." if is_busy else "Start evaluation")
        self.update_controls()

    def update_controls(self) -> None:
        can_configure = not self._is_busy
        self.open_button.setEnabled(can_configure)
        self.profile_combo.setEnabled(can_configure)
        self.answer_style_combo.setEnabled(can_configure)
        self.answer_model_combo.setEnabled(can_configure)
        self.refresh_models_button.setEnabled(can_configure and self.model_thread is None)
        self.start_button.setEnabled(
            can_configure and self.investigation_id is not None and self.evaluation is not None
        )
        self.stop_button.setEnabled(self._is_busy)
        self.export_button.setEnabled(can_configure and bool(self.outcomes))

    def shutdown(self, wait_ms: int = 3000) -> None:
        if self.model_thread is not None and self.model_thread.isRunning():
            self.model_thread.quit()
            self.model_thread.wait(wait_ms)
        if self.run_thread is not None and self.run_thread.isRunning():
            if self.run_worker is not None:
                self.run_worker.cancel()
            self.run_thread.requestInterruption()
            self.run_thread.quit()
            self.run_thread.wait(wait_ms)
