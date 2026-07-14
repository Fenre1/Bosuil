from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractItemView,
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

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("GhostButton")
        self.refresh_button.clicked.connect(self.refresh)
        self.inactive_button = QPushButton("Mark inactive")
        self.inactive_button.setObjectName("GhostButton")
        self.inactive_button.clicked.connect(lambda: self.apply_action("inactive"))
        self.keep_button = QPushButton("Keep active")
        self.keep_button.setObjectName("GhostButton")
        self.keep_button.clicked.connect(lambda: self.apply_action("keep"))
        self.update_path_button = QPushButton("Update path")
        self.update_path_button.setObjectName("GhostButton")
        self.update_path_button.clicked.connect(self.update_path)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.inactive_button)
        buttons.addWidget(self.keep_button)
        buttons.addWidget(self.update_path_button)
        buttons.addStretch(1)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Path", "Last seen", "Chunks", "Status"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.update_action_buttons)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        self.update_action_buttons()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(0)
        self.disappeared_ids = []
        if self.investigation_id is None:
            self.update_action_buttons()
            return
        rows = self.db.list_disappeared_files(self.investigation_id, status="pending")
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.disappeared_ids.append(int(row["id"]))
            self.table.setItem(row_index, 0, QTableWidgetItem(row["path"]))
            self.table.setItem(row_index, 1, QTableWidgetItem(row["last_seen_at"]))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row["chunk_count"])))
            self.table.setItem(row_index, 3, QTableWidgetItem(row["status"]))
        self.update_action_buttons()

    def update_action_buttons(self) -> None:
        has_investigation = self.investigation_id is not None
        has_selection = self.selected_disappeared_id() is not None
        self.refresh_button.setEnabled(has_investigation)
        self.inactive_button.setEnabled(has_selection)
        self.keep_button.setEnabled(has_selection)
        self.update_path_button.setEnabled(has_selection)

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
