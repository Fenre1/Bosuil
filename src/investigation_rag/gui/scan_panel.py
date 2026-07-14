from __future__ import annotations

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
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

    def run(self) -> None:
        try:
            pipeline = ScanPipeline(self.config, self.db, self.store, log=self.log.emit)
            pipeline.scan_investigation(self.investigation_id)
            self.finished.emit("")
        except Exception as exc:
            self.finished.emit(f"Scan failed: {exc}")


class ScanPanel(QWidget):
    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.investigation_id: int | None = None
        self.thread: QThread | None = None
        self.worker: ScanWorker | None = None

        self.scan_button = QPushButton("Scan investigation")
        self.scan_button.clicked.connect(self.start_scan)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.scan_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.log_view, 1)

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.scan_button.setEnabled(investigation_id is not None)

    def start_scan(self) -> None:
        if self.investigation_id is None or self.thread is not None:
            return
        self.log_view.append("Starting scan.")
        self.scan_button.setEnabled(False)
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

    def scan_finished(self, message: str) -> None:
        if message:
            self.log_view.append(message)
 
    def scan_thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self.scan_button.setEnabled(self.investigation_id is not None)

    def shutdown(self, wait_ms: int = 5000) -> None:
        if self.thread is None or not self.thread.isRunning():
            return
        self.log_view.append("Waiting for scan thread to finish.")
        self.thread.quit()
        self.thread.wait(wait_ms)
