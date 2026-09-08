import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidgetItem

from investigation_rag.gui.checkable_type_filter import CheckableTypeFilter
from investigation_rag.gui.entities_panel import EntitiesPanel


def application():
    return QApplication.instance() or QApplication([])


def test_all_types_are_selected_by_default():
    app = application()
    type_filter = CheckableTypeFilter()

    type_filter.set_entity_types(["person", "organization", "location"])

    assert type_filter.selected_types() == {"person", "organization", "location"}
    assert not type_filter.is_filter_active()
    assert type_filter.text() == "All types"
    app.processEvents()


def test_multiple_types_can_be_selected():
    app = application()
    type_filter = CheckableTypeFilter()
    type_filter.set_entity_types(["person", "organization", "location"])

    type_filter.set_selected_types(["person", "location"])

    assert type_filter.selected_types() == {"person", "location"}
    assert type_filter.is_filter_active()
    assert type_filter.text() == "2 types"
    app.processEvents()


def test_all_types_checkbox_selects_and_clears_every_type():
    app = application()
    type_filter = CheckableTypeFilter()
    type_filter.set_entity_types(["person", "organization"])

    type_filter._all_checkbox.click()

    assert type_filter.selected_types() == set()
    assert type_filter.text() == "No types"

    type_filter._all_checkbox.click()

    assert type_filter.selected_types() == {"person", "organization"}
    assert type_filter.text() == "All types"
    app.processEvents()


def test_selection_is_preserved_when_types_are_refreshed():
    app = application()
    type_filter = CheckableTypeFilter()
    type_filter.set_entity_types(["person", "organization", "location"])
    type_filter.set_selected_types(["person", "location"])

    type_filter.set_entity_types(["person", "location", "event"])

    assert type_filter.selected_types() == {"person", "location"}
    assert type_filter.text() == "2 types"
    app.processEvents()


def test_select_all_clears_the_type_filter():
    app = application()
    type_filter = CheckableTypeFilter()
    type_filter.set_entity_types(["person", "organization"])
    type_filter.set_selected_types([])

    type_filter.select_all()

    assert type_filter.selected_types() == {"person", "organization"}
    assert not type_filter.is_filter_active()
    assert type_filter.text() == "All types"
    app.processEvents()


def test_entities_panel_shows_rows_matching_any_selected_type():
    app = application()
    panel = EntitiesPanel(object())
    panel.investigation_id = 1
    panel.update_type_filter(["person", "organization", "location"])
    panel.table.setSortingEnabled(False)
    panel.table.setRowCount(3)
    for row, (name, entity_type) in enumerate(
        [("Alice", "person"), ("ACME", "organization"), ("Utrecht", "location")]
    ):
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, row + 1)
        panel.table.setItem(row, 0, name_item)
        panel.table.setItem(row, 1, QTableWidgetItem(entity_type))
        mentions_item = QTableWidgetItem()
        mentions_item.setData(Qt.DisplayRole, 1)
        panel.table.setItem(row, 2, mentions_item)

    panel.type_filter.set_selected_types(["person", "location"])

    assert [panel.table.isRowHidden(row) for row in range(3)] == [False, True, False]
    assert panel.status_label.text() == "2 of 3 unique entities"
    panel.shutdown()
    app.processEvents()
