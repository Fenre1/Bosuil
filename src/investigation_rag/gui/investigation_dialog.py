from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class InvestigationDialog(QDialog):
    def __init__(
        self,
        parent=None,
        name: str = "",
        notes: str = "",
        roots: list[Path] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Investigation")

        self.name_edit = QLineEdit(name)
        self.notes_edit = QTextEdit(notes)
        self.notes_edit.setFixedHeight(90)
        self.roots_list = QListWidget()
        for root in roots or []:
            self.roots_list.addItem(str(root))

        add_root = QPushButton("Add root")
        add_root.clicked.connect(self.add_root)
        remove_root = QPushButton("Remove")
        remove_root.clicked.connect(self.remove_selected_root)
        root_buttons = QHBoxLayout()
        root_buttons.addWidget(add_root)
        root_buttons.addWidget(remove_root)
        root_buttons.addStretch(1)

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Notes", self.notes_edit)
        form.addRow("Roots", self.roots_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(root_buttons)
        layout.addWidget(buttons)

    def add_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose root folder")
        if folder:
            self.roots_list.addItem(folder)

    def remove_selected_root(self) -> None:
        for item in self.roots_list.selectedItems():
            self.roots_list.takeItem(self.roots_list.row(item))

    def values(self) -> tuple[str, str, list[Path]]:
        roots = [Path(self.roots_list.item(i).text()) for i in range(self.roots_list.count())]
        return self.name_edit.text().strip(), self.notes_edit.toPlainText().strip(), roots
