from __future__ import annotations

import sys
import os
import ctypes
from importlib.util import find_spec

from investigation_rag.config import AppConfig
from investigation_rag.utils.logging import configure_logging
from investigation_rag.workspace import WorkspaceCatalog


def preload_torch_windows_dll() -> None:
    """Load c10 before Qt to avoid PyTorch's Windows DLL initialization conflict."""
    if sys.platform != "win32":
        return
    spec = find_spec("torch")
    if spec is None or spec.origin is None:
        return
    c10_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
    if os.path.isfile(c10_path):
        ctypes.CDLL(os.path.normpath(c10_path))


def main() -> int:
    configure_logging()
    preload_torch_windows_dll()
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
