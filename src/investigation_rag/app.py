from __future__ import annotations

import sys

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB


def main() -> int:
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError("PyQt5 is required to run the desktop app.") from exc

    from investigation_rag.gui.main_window import MainWindow, app_icon
    from investigation_rag.gui.style import apply_app_style

    config = AppConfig.from_env()
    config.ensure_dirs()
    db = SQLiteDB(config.sqlite_path)
    db.initialize()
    store = LanceDBStore(config.lancedb_dir)

    app = QApplication(sys.argv)
    apply_app_style(app)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = MainWindow(config, db, store)
    window.resize(1200, 780)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
