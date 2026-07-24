import os
from pathlib import Path
from unittest.mock import patch

from investigation_rag.gui.folder_scope import FolderScopePanel


def test_path_key_is_lexical_and_does_not_touch_the_filesystem():
    path = Path(r"Z:\offline-share\Evidence\..\Evidence\Report.PDF")

    with patch.object(Path, "resolve", side_effect=AssertionError("resolve must not be called")):
        key = FolderScopePanel.path_key(path)

    assert key == os.path.normpath(os.fspath(path)).casefold()
