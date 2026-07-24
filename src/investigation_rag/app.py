from __future__ import annotations

import sys

from investigation_rag.config import AppConfig
from investigation_rag.utils.logging import configure_logging
from investigation_rag.workspace import WorkspaceCatalog


def main() -> int:
    configure_logging()
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError("PyQt5 is required to run the desktop app.") from exc

    from investigation_rag.gui.main_window import MainWindow, app_icon
    from investigation_rag.gui.style import apply_app_style

    config = AppConfig.from_env()
    config.ensure_dirs()
    catalog = WorkspaceCatalog(config.app_data_dir)
    catalog.initialize()

    app = QApplication(sys.argv)
    apply_app_style(app)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = MainWindow(config, catalog)
    window.resize(1200, 780)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
