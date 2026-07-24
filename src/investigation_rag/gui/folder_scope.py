from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.db.sqlite_db import SQLiteDB


class FolderScope(QObject):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._folder_paths: list[str] | None = None

    def folder_paths(self) -> list[str] | None:
        if self._folder_paths is None:
            return None
        return list(self._folder_paths)

    def set_all(self) -> None:
        self.set_folder_paths(None)

    def set_folder_paths(self, folder_paths: list[str] | None) -> None:
        normalized = None if folder_paths is None else sorted(set(folder_paths), key=str.casefold)
        if normalized == self._folder_paths:
            return
        self._folder_paths = normalized
        self.changed.emit()


class FolderScopePanel(QWidget):
    def __init__(self, db: SQLiteDB, scope: FolderScope):
        super().__init__()
        self.db = db
        self.scope = scope
        self.investigation_id: int | None = None
        self._updating = False
        self.last_reload_timing: dict[str, float | int] = {}

        self.status_label = QLabel()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Folder"])
        self.tree.itemChanged.connect(self.on_item_changed)

        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(self.select_all)
        select_none_button = QPushButton("Select none")
        select_none_button.clicked.connect(self.select_none)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.reload)

        controls = QHBoxLayout()
        controls.addWidget(select_all_button)
        controls.addWidget(select_none_button)
        controls.addWidget(refresh_button)
        controls.addStretch(1)
        controls.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.tree, 1)

        self.scope.changed.connect(self.update_status)
        self.update_status()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.scope.set_all()
        self.reload()

    def reload(self) -> None:
        total_started = perf_counter()
        self._updating = True
        try:
            self.tree.clear()
            if self.investigation_id is None:
                self.last_reload_timing = {"total_seconds": perf_counter() - total_started}
                self.update_status()
                return
            query_started = perf_counter()
            roots = self.db.get_roots(self.investigation_id)
            folders = self.db.list_active_document_folders(self.investigation_id)
            query_seconds = perf_counter() - query_started
            populate_started = perf_counter()
            self.populate_tree(roots, folders)
            populate_seconds = perf_counter() - populate_started
            self.tree.expandToDepth(0)
            self.update_status()
            self.last_reload_timing = {
                "query_seconds": query_seconds,
                "populate_seconds": populate_seconds,
                "total_seconds": perf_counter() - total_started,
                "folder_count": len(folders),
            }
        finally:
            self._updating = False

    def populate_tree(self, roots: list[Path], folders: list[Path]) -> None:
        root_items: dict[str, QTreeWidgetItem] = {}
        items_by_path: dict[str, QTreeWidgetItem] = {}
        for root in roots:
            root_key = self.path_key(root)
            item = self.add_folder_item(self.tree, root.name or str(root), root)
            root_items[root_key] = item
            items_by_path[root_key] = item

        for folder in folders:
            folder_key = self.path_key(folder)
            if folder_key in items_by_path:
                continue
            root = self.best_root(folder, roots)
            if root is None:
                item = self.add_folder_item(self.tree, str(folder), folder)
                items_by_path.setdefault(folder_key, item)
                continue

            parent = root_items[self.path_key(root)]
            current = root
            try:
                relative_parts = folder.relative_to(root).parts
            except ValueError:
                relative_parts = ()
            for part in relative_parts:
                current = current / part
                key = self.path_key(current)
                if key not in items_by_path:
                    items_by_path[key] = self.add_folder_item(parent, part, current)
                parent = items_by_path[key]

        for index in range(self.tree.topLevelItemCount()):
            self.set_check_state_recursive(self.tree.topLevelItem(index), Qt.Checked)

    def add_folder_item(self, parent: QTreeWidget | QTreeWidgetItem, label: str, path: Path) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent)
        item.setText(0, label)
        item.setData(0, Qt.UserRole, str(path))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
        item.setCheckState(0, Qt.Checked)
        item.setToolTip(0, str(path))
        return item

    def on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        self._updating = True
        try:
            if item.checkState(0) != Qt.PartiallyChecked:
                self.set_check_state_recursive(item, item.checkState(0))
            self.update_parent_states(item.parent())
            self.scope.set_folder_paths(self.selected_folder_paths())
        finally:
            self._updating = False
        self.update_status()

    def selected_folder_paths(self) -> list[str] | None:
        selected: list[str] = []
        all_checked = True
        any_checked = False
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.checkState(0) != Qt.Checked:
                all_checked = False
            any_checked = self.collect_checked_paths(item, selected) or any_checked
        if all_checked:
            return None
        if not any_checked:
            return []
        return selected

    def collect_checked_paths(self, item: QTreeWidgetItem, selected: list[str]) -> bool:
        state = item.checkState(0)
        if state == Qt.Checked:
            selected.append(str(item.data(0, Qt.UserRole)))
            return True
        if state == Qt.Unchecked:
            return False
        any_checked = False
        for index in range(item.childCount()):
            any_checked = self.collect_checked_paths(item.child(index), selected) or any_checked
        return any_checked

    def select_all(self) -> None:
        self._updating = True
        try:
            for index in range(self.tree.topLevelItemCount()):
                self.set_check_state_recursive(self.tree.topLevelItem(index), Qt.Checked)
            self.scope.set_all()
        finally:
            self._updating = False
        self.update_status()

    def select_none(self) -> None:
        self._updating = True
        try:
            for index in range(self.tree.topLevelItemCount()):
                self.set_check_state_recursive(self.tree.topLevelItem(index), Qt.Unchecked)
            self.scope.set_folder_paths([])
        finally:
            self._updating = False
        self.update_status()

    def set_check_state_recursive(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        item.setCheckState(0, state)
        for index in range(item.childCount()):
            self.set_check_state_recursive(item.child(index), state)

    def update_parent_states(self, item: QTreeWidgetItem | None) -> None:
        while item is not None:
            states = [item.child(index).checkState(0) for index in range(item.childCount())]
            if all(state == Qt.Checked for state in states):
                item.setCheckState(0, Qt.Checked)
            elif all(state == Qt.Unchecked for state in states):
                item.setCheckState(0, Qt.Unchecked)
            else:
                item.setCheckState(0, Qt.PartiallyChecked)
            item = item.parent()

    def update_status(self) -> None:
        folder_paths = self.scope.folder_paths()
        if folder_paths is None:
            self.status_label.setText("All folders")
        elif not folder_paths:
            self.status_label.setText("No folders")
        elif len(folder_paths) == 1:
            self.status_label.setText("1 folder")
        else:
            self.status_label.setText(f"{len(folder_paths)} folders")

    @staticmethod
    def path_key(path: Path) -> str:
        # Manifest paths are already absolute. Resolving here performs
        # filesystem I/O for every file and parent, which is especially costly
        # when the evidence lives on a network share.
        return os.path.normpath(os.fspath(path)).casefold()

    @staticmethod
    def best_root(folder: Path, roots: list[Path]) -> Path | None:
        candidates = []
        for root in roots:
            try:
                folder.relative_to(root)
            except ValueError:
                continue
            candidates.append(root)
        if not candidates:
            return None
        return max(candidates, key=lambda root: len(root.parts))
