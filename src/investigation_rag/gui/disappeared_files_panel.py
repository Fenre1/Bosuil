from __future__ import annotations

from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.db.sqlite_db import SQLiteDB


class DisappearedFilesPanel(QWidget):
    def __init__(self, db: SQLiteDB):
        super().__init__()
        self.db = db
        self.investigation_id: int | None = None
        self.disappeared_ids: list[int] = []

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        inactive = QPushButton("Mark inactive")
        inactive.clicked.connect(lambda: self.apply_action("inactive"))
        keep = QPushButton("Keep active")
        keep.clicked.connect(lambda: self.apply_action("keep"))
        update_path = QPushButton("Update path")
        update_path.clicked.connect(self.update_path)

        buttons = QHBoxLayout()
        buttons.addWidget(refresh)
        buttons.addWidget(inactive)
        buttons.addWidget(keep)
        buttons.addWidget(update_path)
        buttons.addStretch(1)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Path", "Last seen", "Chunks", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(0)
        self.disappeared_ids = []
        if self.investigation_id is None:
            return
        rows = self.db.list_disappeared_files(self.investigation_id, status="pending")
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.disappeared_ids.append(int(row["id"]))
            self.table.setItem(row_index, 0, QTableWidgetItem(row["path"]))
            self.table.setItem(row_index, 1, QTableWidgetItem(row["last_seen_at"]))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row["chunk_count"])))
            self.table.setItem(row_index, 3, QTableWidgetItem(row["status"]))

    def selected_disappeared_id(self) -> int | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.disappeared_ids):
            return None
        return self.disappeared_ids[row]

    def apply_action(self, action: str) -> None:
        disappeared_id = self.selected_disappeared_id()
        if disappeared_id is None:
            return
        try:
            self.db.resolve_disappeared(disappeared_id, action)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Disappeared file", str(exc))

    def update_path(self) -> None:
        disappeared_id = self.selected_disappeared_id()
        if disappeared_id is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose replacement file")
        if not path:
            return
        try:
            self.db.resolve_disappeared(disappeared_id, "update_path", new_path=path)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Disappeared file", str(exc))
