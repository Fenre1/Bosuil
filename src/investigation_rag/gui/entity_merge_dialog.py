from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QVBoxLayout,
)


class EntityMergeDialog(QDialog):
    def __init__(self, entities: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge entities")
        self.setMinimumWidth(420)

        explanation = QLabel(
            "All mentions from the selected entities will be combined under the final name and type."
        )
        explanation.setWordWrap(True)

        self.entities_list = QListWidget()
        self.entities_list.setMaximumHeight(130)
        for name, entity_type in entities:
            self.entities_list.addItem(f"{name}  —  {entity_type}")

        default_name, default_type = entities[0]
        self.name_edit = QLineEdit(default_name)
        self.name_edit.selectAll()
        self.type_combo = QComboBox()
        self.type_combo.setEditable(True)
        for entity_type in sorted({value for _, value in entities}, key=str.casefold):
            self.type_combo.addItem(entity_type)
        self.type_combo.setCurrentText(default_type)

        form = QFormLayout()
        form.addRow("Selected entities", self.entities_list)
        form.addRow("Final name", self.name_edit)
        form.addRow("Final type", self.type_combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Merge entities")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(explanation)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

    def accept(self) -> None:
        name, entity_type = self.values()
        if not name or not entity_type:
            QMessageBox.warning(self, "Merge entities", "Enter both a final name and a final type.")
            return
        super().accept()

    def values(self) -> tuple[str, str]:
        return self.name_edit.text().strip(), self.type_combo.currentText().strip()
