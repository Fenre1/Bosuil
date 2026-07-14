from __future__ import annotations

from PyQt5.QtWidgets import QApplication


APP_STYLE = """
* {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

QMainWindow,
QDialog,
QWidget#AppRoot {
    background: #f5f7fb;
    color: #1f2937;
}

QWidget#AppHeader {
    background: #ffffff;
    border-bottom: 1px solid #d8dee8;
}

QWidget#ComposerBar {
    background: #ffffff;
    border-top: 1px solid #d8dee8;
}

QWidget#InvestigationPanel {
    background: #ffffff;
    border: 1px solid #d8dee8;
    border-radius: 8px;
}

QLabel#AppTitle {
    color: #0C2A61;
    font-size: 15pt;
    font-weight: 700;
}

QLabel#FieldLabel,
QLabel#StatusLabel {
    color: #5b6472;
    font-weight: 600;
}

QLineEdit,
QTextEdit,
QTextBrowser,
QComboBox {
    background: #ffffff;
    border: 1px solid #cfd6e4;
    border-radius: 6px;
    color: #111827;
    padding: 7px 9px;
    selection-background-color: #0C2A61;
    selection-color: #ffffff;
}

QTextEdit:focus,
QTextBrowser:focus,
QLineEdit:focus,
QComboBox:focus {
    border: 1px solid #0C2A61;
}

QTextEdit[readOnly="true"],
QTextBrowser {
    background: #fbfcfe;
}

QTextEdit#Composer {
    background: #ffffff;
    border: 1px solid #b9c4d6;
    padding: 9px 10px;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #c8d1df;
    border-radius: 6px;
    color: #1f2937;
    font-weight: 600;
    padding: 7px 13px;
}

QPushButton:hover {
    background: #eef4ff;
    border-color: #9ab6e8;
}

QPushButton:pressed {
    background: #dbeafe;
}

QPushButton:disabled {
    background: #eef1f5;
    border-color: #d8dee8;
    color: #9aa3af;
}

QPushButton#PrimaryButton {
    background: #0C2A61;
    border-color: #0C2A61;
    color: #ffffff;
}

QPushButton#PrimaryButton:hover {
    background: #143B82;
    border-color: #143B82;
}

QPushButton#PrimaryButton:pressed {
    background: #071C42;
}

QPushButton#GhostButton {
    background: transparent;
    border-color: transparent;
    color: #334155;
}

QPushButton#GhostButton:hover {
    background: #eef2f7;
    border-color: #d8dee8;
}

QPushButton[segment="true"] {
    border-radius: 6px;
    padding: 6px 12px;
}

QPushButton[segmentPosition="left"] {
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    margin-right: -1px;
}

QPushButton[segmentPosition="right"] {
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

QPushButton[segment="true"]:checked {
    background: #e5ecf7;
    border-color: #8fa4c7;
    color: #0C2A61;
}

QComboBox::drop-down {
    border: 0;
    width: 24px;
}

QTabWidget::pane {
    border: 0;
}

QTabBar::tab {
    background: transparent;
    color: #5b6472;
    font-weight: 600;
    min-width: 155px;
    padding: 9px 20px;
    margin: 0 2px;
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #111827;
    border-color: #d8dee8;
    border-bottom-color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background: #eef2f7;
    color: #334155;
}

QListWidget,
QTableWidget,
QTreeWidget {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #d8dee8;
    border-radius: 6px;
    gridline-color: #e5eaf1;
    color: #1f2937;
}

QListWidget::item,
QTableWidget::item,
QTreeWidget::item {
    padding: 7px 8px;
}

QListWidget::item:selected,
QTableWidget::item:selected,
QTreeWidget::item:selected {
    background: #dbeafe;
    color: #111827;
}

QHeaderView::section {
    background: #eef2f7;
    border: 0;
    border-bottom: 1px solid #d8dee8;
    color: #334155;
    font-weight: 700;
    padding: 7px 8px;
}

QSplitter::handle {
    background: #d8dee8;
}

QSplitter::handle:horizontal {
    width: 1px;
}

QSplitter::handle:vertical {
    height: 1px;
}

QScrollBar:vertical {
    background: #f5f7fb;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #c5cfdd;
    border-radius: 6px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: #9aa8bc;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}
"""


def apply_app_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = app.font()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(APP_STYLE)
