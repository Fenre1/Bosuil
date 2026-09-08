from __future__ import annotations

from collections.abc import Iterable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
    QWidgetAction,
)


class CheckableTypeFilter(QPushButton):
    """A compact dropdown whose options can be selected independently."""

    selectionChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._updating = False
        self._type_checkboxes: dict[str, QCheckBox] = {}
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.setAccessibleName("Entity types")
        self.set_entity_types([])

    def entity_types(self) -> list[str]:
        return list(self._type_checkboxes)

    def selected_types(self) -> set[str]:
        return {
            entity_type
            for entity_type, checkbox in self._type_checkboxes.items()
            if checkbox.isChecked()
        }

    def is_filter_active(self) -> bool:
        available = set(self._type_checkboxes)
        return bool(available) and self.selected_types() != available

    def set_entity_types(self, entity_types: Iterable[str]) -> None:
        values = sorted({str(value) for value in entity_types}, key=str.casefold)
        previous_available = set(self._type_checkboxes)
        previous_selected = self.selected_types()
        previously_all_selected = previous_selected == previous_available
        selected = set(values) if previously_all_selected else previous_selected.intersection(values)

        self._updating = True
        try:
            self._menu.clear()
            self._type_checkboxes.clear()
            self._all_checkbox = self._add_checkbox("All types", self._toggle_all)
            self._menu.addSeparator()
            for entity_type in values:
                checkbox = self._add_checkbox(entity_type, self._type_toggled)
                checkbox.setChecked(entity_type in selected)
                self._type_checkboxes[entity_type] = checkbox
            self._sync_summary()
        finally:
            self._updating = False

    def set_selected_types(self, entity_types: Iterable[str]) -> None:
        selected = {str(value) for value in entity_types}
        before = self.selected_types()
        self._updating = True
        try:
            for entity_type, checkbox in self._type_checkboxes.items():
                checkbox.setChecked(entity_type in selected)
            self._sync_summary()
        finally:
            self._updating = False
        if self.selected_types() != before:
            self.selectionChanged.emit()

    def select_all(self) -> None:
        self.set_selected_types(self._type_checkboxes)

    def _add_checkbox(self, label: str, slot) -> QCheckBox:
        row = QWidget(self._menu)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 5, 12, 5)
        checkbox = QCheckBox(label, row)
        checkbox.clicked.connect(slot)
        layout.addWidget(checkbox)
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(row)
        self._menu.addAction(action)
        return checkbox

    def _toggle_all(self, checked: bool) -> None:
        if self._updating:
            return
        before = self.selected_types()
        self._updating = True
        try:
            for checkbox in self._type_checkboxes.values():
                checkbox.setChecked(checked)
            self._sync_summary()
        finally:
            self._updating = False
        if self.selected_types() != before:
            self.selectionChanged.emit()

    def _type_toggled(self) -> None:
        if self._updating:
            return
        self._sync_summary()
        self.selectionChanged.emit()

    def _sync_summary(self) -> None:
        available_count = len(self._type_checkboxes)
        selected = self.selected_types()
        selected_count = len(selected)

        self._all_checkbox.setTristate(True)
        if selected_count == available_count:
            self._all_checkbox.setCheckState(Qt.Checked)
        elif selected_count == 0:
            self._all_checkbox.setCheckState(Qt.Unchecked)
        else:
            self._all_checkbox.setCheckState(Qt.PartiallyChecked)

        if not available_count or selected_count == available_count:
            summary = "All types"
        elif selected_count == 0:
            summary = "No types"
        elif selected_count == 1:
            summary = next(iter(selected))
        else:
            summary = f"{selected_count} types"
        self.setText(summary)
        if selected_count == available_count:
            tooltip = "All types"
        elif selected:
            tooltip = ", ".join(sorted(selected, key=str.casefold))
        else:
            tooltip = "No types selected"
        self.setToolTip(tooltip)
