from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.gui.disappeared_files_panel import DisappearedFilesPanel
from investigation_rag.ingest.pipeline import ScanPipeline


class ScanWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore, investigation_id: int):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.investigation_id = investigation_id
        self._stop_event = Event()

    def run(self) -> None:
        try:
            pipeline = ScanPipeline(
                self.config,
                self.db,
                self.store,
                log=self.log.emit,
                stop_requested=self._stop_event.is_set,
            )
            pipeline.scan_investigation(self.investigation_id)
            self.finished.emit("")
        except Exception as exc:
            self.finished.emit(f"Scan failed: {exc}")

    def request_stop(self) -> None:
        self._stop_event.set()


class ScanPanel(QWidget):
    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.investigation_id: int | None = None
        self.thread: QThread | None = None
        self.worker: ScanWorker | None = None
        self.investigation_selector: QWidget | None = None

        self.scan_button = QPushButton("Scan investigation")
        self.scan_button.setObjectName("PrimaryButton")
        self.scan_button.setEnabled(False)
        self.stop_button = QPushButton("Stop scan")
        self.stop_button.setObjectName("GhostButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_scan)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Scan activity will appear here")
        self.disappeared_panel = DisappearedFilesPanel(db)

        scan_log_label = QLabel(" Scan log")
        scan_log_label.setObjectName("FieldLabel")
        disappeared_label = QLabel("Disappeared files")
        disappeared_label.setObjectName("FieldLabel")

        scan_log_container = QWidget()
        scan_log_layout = QVBoxLayout(scan_log_container)
        scan_log_layout.setContentsMargins(0, 0, 0, 0)
        scan_log_layout.setSpacing(8)
        scan_log_layout.addWidget(scan_log_label)
        scan_log_layout.addWidget(self.log_view, 1)

        disappeared_container = QWidget()
        disappeared_layout = QVBoxLayout(disappeared_container)
        disappeared_layout.setContentsMargins(0, 0, 0, 0)
        disappeared_layout.setSpacing(8)
        disappeared_layout.addWidget(disappeared_label)
        disappeared_layout.addWidget(self.disappeared_panel, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(scan_log_container)
        splitter.addWidget(disappeared_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.addWidget(splitter, 1)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.content_container)
        self.main_splitter.setStretchFactor(0, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self.main_splitter, 1)

    def set_investigation_selector(self, widget: QWidget) -> None:
        self.investigation_selector = widget
        widget.setMinimumWidth(320)
        self.main_splitter.insertWidget(0, widget)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([360, 900])

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.scan_button.setEnabled(investigation_id is not None)
        self.disappeared_panel.set_investigation_id(investigation_id)

    def start_scan(self) -> None:
        if self.investigation_id is None or self.thread is not None:
            return
        self.log_view.append("Starting scan.")
        self.scan_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        thread = QThread(self)
        worker = ScanWorker(self.config, self.db, self.store, self.investigation_id)
        self.thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self.worker.log.connect(self.log_view.append)
        self.worker.finished.connect(self.scan_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.scan_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def stop_scan(self) -> None:
        if self.worker is None or self.thread is None:
            return
        self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self.log_view.append("Stop requested. Finishing the current file…")

    def scan_finished(self, message: str) -> None:
        if message:
            self.log_view.append(message)

    def scan_thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self.scan_button.setEnabled(self.investigation_id is not None)
        self.stop_button.setEnabled(False)
        self.disappeared_panel.refresh()

    def shutdown(self, wait_ms: int = 5000) -> None:
        if self.thread is None or not self.thread.isRunning():
            return
        self.log_view.append("Waiting for scan thread to finish.")
        self.thread.quit()
        self.thread.wait(wait_ms)
