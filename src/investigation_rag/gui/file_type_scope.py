from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_FILE_TYPES = ("txt", "docx", "pdf", "msg")
FILE_TYPE_LABELS = {
    "txt": "TXT",
    "docx": "DOCX",
    "pdf": "PDF",
    "msg": "Email (.msg)",
}


class FileTypeScope(QObject):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._source_types: list[str] | None = None

    def source_types(self) -> list[str] | None:
        if self._source_types is None:
            return None
        return list(self._source_types)

    def set_all(self) -> None:
        self.set_source_types(None)

    def set_source_types(self, source_types: list[str] | None) -> None:
        normalized = normalize_source_types(source_types)
        if normalized == self._source_types:
            return
        self._source_types = normalized
        self.changed.emit()


class FileTypeScopePanel(QWidget):
    def __init__(self, scope: FileTypeScope):
        super().__init__()
        self.scope = scope
        self._updating = False
        self.checkboxes: dict[str, QCheckBox] = {}

        title = QLabel("File types")
        title.setObjectName("FieldLabel")
        self.status_label = QLabel()
        self.status_label.setObjectName("StatusLabel")

        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(self.select_all)
        select_none_button = QPushButton("Select none")
        select_none_button.clicked.connect(self.select_none)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(title)
        controls.addStretch(1)
        controls.addWidget(select_all_button)
        controls.addWidget(select_none_button)
        controls.addWidget(self.status_label)

        type_row = QHBoxLayout()
        type_row.setSpacing(14)
        for source_type in SUPPORTED_FILE_TYPES:
            checkbox = QCheckBox(FILE_TYPE_LABELS[source_type])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.on_type_changed)
            self.checkboxes[source_type] = checkbox
            type_row.addWidget(checkbox)
        type_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addLayout(controls)
        layout.addLayout(type_row)

        self.scope.changed.connect(self.update_status)
        self.update_status()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.select_all()

    def on_type_changed(self) -> None:
        if self._updating:
            return
        selected = [
            source_type
            for source_type, checkbox in self.checkboxes.items()
            if checkbox.isChecked()
        ]
        if len(selected) == len(SUPPORTED_FILE_TYPES):
            self.scope.set_all()
        else:
            self.scope.set_source_types(selected)
        self.update_status()

    def select_all(self) -> None:
        self._updating = True
        try:
            for checkbox in self.checkboxes.values():
                checkbox.setChecked(True)
            self.scope.set_all()
        finally:
            self._updating = False
        self.update_status()

    def select_none(self) -> None:
        self._updating = True
        try:
            for checkbox in self.checkboxes.values():
                checkbox.setChecked(False)
            self.scope.set_source_types([])
        finally:
            self._updating = False
        self.update_status()

    def update_status(self) -> None:
        source_types = self.scope.source_types()
        if source_types is None:
            self.status_label.setText("All file types")
        elif not source_types:
            self.status_label.setText("No file types")
        elif len(source_types) == 1:
            self.status_label.setText(f"{FILE_TYPE_LABELS.get(source_types[0], source_types[0])} only")
        else:
            self.status_label.setText(f"{len(source_types)} file types")


def normalize_source_types(source_types: list[str] | None) -> list[str] | None:
    if source_types is None:
        return None
    normalized = []
    seen = set()
    for source_type in source_types:
        value = source_type.strip().lower().lstrip(".")
        if not value or value not in SUPPORTED_FILE_TYPES or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return sorted(normalized)
