from __future__ import annotations

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication


APP_STYLE = """
QWidget {
    font-family: "Segoe UI";
    font-size: 9pt;
    color: #243746;
}
QMainWindow, QDialog, QWidget#AppRoot, QScrollArea#PageScroll {
    background: #f3f6f8;
}
QWidget#Sidebar { background: #142d3a; }
QLabel#AppTitle { color: #ffffff; font-size: 20pt; font-weight: 600; }
QLabel#SidebarCaption { color: #a4bbc7; font-size: 8pt; }
QLabel#AppIcon { background: #ffffff; border-radius: 10px; }
QListWidget#Navigation { background: transparent; border: none; outline: none; }
QListWidget#Navigation::item {
    color: #c2d1da; padding: 12px 10px; margin: 3px 0;
    border: 1px solid transparent; border-radius: 7px;
}
QListWidget#Navigation::item:hover { background: #203e4e; color: #ffffff; }
QListWidget#Navigation::item:selected { background: #285366; color: #ffffff; border-color: #397088; }
QListWidget#Navigation::item:focus { border-color: #83c5d5; }
QWidget#AppHeader { background: #ffffff; border-bottom: 1px solid #dfe7eb; }
QLabel#PageTitle { font-size: 18pt; font-weight: 600; color: #172f3e; }
QLabel#PageDescription { color: #647985; }
QLabel#HeaderInvestigationName { color: #486574; font-size: 8pt; }
QLabel#FieldLabel { color: #304c5c; font-weight: 600; }
QLabel#StatusLabel { color: #687d89; }
QLabel#EmptyState { background: #f3f8fa; color: #617b89; border-radius: 8px; padding: 18px; }
QWidget#InvestigationPanel {
    background: #ffffff; border: 1px solid #dfe7eb; border-radius: 10px;
}
QWidget#ComposerBar { background: transparent; }
QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff; border: 1px solid #ccd9e0; border-radius: 6px;
    padding: 6px 8px; selection-background-color: #d6edf0; selection-color: #183946;
}
QLineEdit:focus, QTextEdit:focus, QTextBrowser:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #248398;
}
QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    background: #edf1f4; color: #82939f; border-color: #dfe6eb;
}
QTextBrowser, QTextEdit[readOnly="true"] { background: #ffffff; }
QTextEdit#Composer { border: 1px solid #a9c5cf; padding: 10px; }
QPushButton, QToolButton {
    background: #ffffff; border: 1px solid #ccd9e0; border-radius: 6px;
    padding: 6px 10px; font-weight: 500;
}
QPushButton:hover, QToolButton:hover { background: #edf5f7; border-color: #8fb9c5; }
QPushButton:pressed, QToolButton:pressed { background: #d8eaf0; }
QPushButton:focus, QToolButton:focus { border-color: #248398; }
QPushButton#PrimaryButton { background: #176e80; border-color: #176e80; color: #ffffff; }
QPushButton#PrimaryButton:hover { background: #125a6b; border-color: #125a6b; }
QPushButton#PrimaryButton:pressed { background: #0e4a58; }
QPushButton#GhostButton { background: transparent; border-color: transparent; color: #466370; }
QPushButton#GhostButton:hover { background: #e9f1f4; border-color: #d3e2e8; }
QPushButton#GhostButton:focus { border-color: #248398; }
QPushButton:disabled, QPushButton#PrimaryButton:disabled, QPushButton#GhostButton:disabled, QToolButton:disabled {
    background: #edf1f4; border-color: #e0e7eb; color: #82939f;
}
QPushButton[segment="true"]:checked { background: #dceef1; border-color: #92bdc7; color: #155c6c; }
QPushButton[segmentPosition="left"] { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QPushButton[segmentPosition="right"] { border-top-left-radius: 0; border-bottom-left-radius: 0; }
QComboBox { padding-right: 24px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #ffffff; selection-background-color: #dceef1; selection-color: #243746; }
QTabWidget::pane { border: none; }
QTabBar::tab {
    background: transparent; color: #637984; padding: 9px 14px;
    border-bottom: 2px solid transparent; margin-right: 4px;
}
QTabBar::tab:selected { color: #176e80; border-bottom-color: #176e80; background: #eaf3f5; }
QTabBar::tab:hover:!selected { background: #edf2f5; }
QListWidget, QTableWidget, QTreeWidget {
    background: #ffffff; alternate-background-color: #f7fafb;
    border: 1px solid #dfe7eb; border-radius: 6px;
    gridline-color: #edf1f4; selection-background-color: #dceef1;
    selection-color: #183946;
}
QListWidget::item, QTreeWidget::item { padding: 5px 7px; }
QTableWidget::item { padding: 3px 6px; }
QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected { background: #dceef1; color: #183946; }
QHeaderView::section {
    background: #edf3f6; border: none; border-bottom: 1px solid #dfe7eb;
    color: #526b79; font-weight: 600; padding: 7px 8px;
}
QTableCornerButton::section { background: #edf3f6; border: none; }
QSplitter::handle { background: #e0e8ed; }
QSplitter::handle:hover { background: #8fb9c5; }
QSplitter::handle:horizontal { width: 5px; margin: 0 1px; }
QSplitter::handle:vertical { height: 5px; margin: 1px 0; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle { background: #bbcbd3; border-radius: 4px; }
QScrollBar::handle:vertical { min-height: 28px; }
QScrollBar::handle:horizontal { min-width: 28px; }
QScrollBar::handle:hover { background: #8ba7b5; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QProgressBar { border: 1px solid #dfe7eb; border-radius: 5px; background: #e9f0f3; text-align: center; min-height: 16px; }
QProgressBar::chunk { background: #70b7be; border-radius: 4px; }
QGroupBox { border: 1px solid #dfe7eb; border-radius: 8px; margin-top: 12px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QToolTip { background: #142d3a; color: #ffffff; border: none; padding: 6px; }
"""


def apply_app_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = app.font()
    font.setFamily("Segoe UI")
    font.setPointSize(9)
    app.setFont(font)
    # Explicit palette also covers native controls and unstyled child widgets
    # when Windows itself uses a dark theme.
    palette = QPalette()
    for role, color in (
        (QPalette.Window, "#f3f6f8"), (QPalette.WindowText, "#243746"),
        (QPalette.Base, "#ffffff"), (QPalette.AlternateBase, "#f7fafb"),
        (QPalette.Text, "#243746"), (QPalette.Button, "#ffffff"),
        (QPalette.ButtonText, "#243746"), (QPalette.Highlight, "#dceef1"),
        (QPalette.HighlightedText, "#183946"), (QPalette.PlaceholderText, "#718793"),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#82939f"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#82939f"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLE)
