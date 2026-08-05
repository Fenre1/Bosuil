from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import investigation_rag.ingest.scanner as scanner
from investigation_rag.ingest.file_identity import get_file_identity
from investigation_rag.ingest.scanner import (
    normalize_inventory_path,
    scan_supported_files,
)


def test_scandir_inventory_is_recursive_metadata_only_and_lexical(tmp_path, monkeypatch):
    root = tmp_path / "case"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "note.TXT").write_text("note", encoding="utf-8")
    (nested / "document.pdf").write_bytes(b"pdf")
    (nested / "ignored.jpg").write_bytes(b"image")
    expected_stable_ids = {
        path.name: get_file_identity(path).stable_id
        for path in (root / "note.TXT", nested / "document.pdf")
    }

    resolved_paths: list[Path] = []
    real_resolve = Path.resolve

    def observed_resolve(path, *args, **kwargs):
        resolved_paths.append(path)
        return real_resolve(path, *args, **kwargs)

    def forbidden_rglob(*args, **kwargs):
        raise AssertionError("recursive Path traversal helper was used")

    monkeypatch.setattr(Path, "resolve", observed_resolve)
    monkeypatch.setattr(Path, "rglob", forbidden_rglob)

    progress = []
    inventory = scan_supported_files([root], directory_workers=1, progress=progress.append)

    assert {item.path.name for item in inventory.files} == {"note.TXT", "document.pdf"}
    assert {item.suffix for item in inventory.files} == {".txt", ".pdf"}
    assert {item.path.name: item.stable_id for item in inventory.files} == expected_stable_ids
    assert resolved_paths == []
    assert all(item.path.is_absolute() for item in inventory.files)
    assert all(
        item.normalized_path == normalize_inventory_path(item.path)
        for item in inventory.files
    )
    assert inventory.counters.directories_scanned == 2
    assert inventory.counters.entries_seen == 4
    assert inventory.counters.entries_visited == 4
    assert inventory.counters.supported_files == 2
    assert inventory.counters.supported_bytes == 7
    assert inventory.counters.unsupported_entries == 1
    assert progress[-1].directories_visited == 2
    assert progress[-1].entries_visited == 4
    assert progress[-1].supported_files == 2
    assert inventory.complete is True


def test_extension_filter_runs_before_file_stat(tmp_path, monkeypatch):
    root = tmp_path / "case"
    root.mkdir()
    calls: list[str] = []

    class FakeEntry:
        name = "ignored.jpg"
        path = str(root / name)

        def is_symlink(self):
            calls.append("is_symlink")
            return False

        def is_dir(self, *, follow_symlinks=True):
            calls.append("is_dir")
            return False

        def is_file(self, *, follow_symlinks=True):
            calls.append("is_file")
            raise AssertionError("unsupported entries must not be probed as files")

        def stat(self, *, follow_symlinks=True):
            calls.append("stat")
            raise AssertionError("unsupported entries must not be stat'ed")

    class FakeScandir:
        def __enter__(self):
            return iter((FakeEntry(),))

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(scanner.os, "scandir", lambda path: FakeScandir())

    inventory = scan_supported_files([root], directory_workers=1)

    assert inventory.files == ()
    assert inventory.counters.unsupported_entries == 1
    assert calls == ["is_symlink", "is_dir"]


def test_nested_and_duplicate_roots_are_scanned_once(tmp_path):
    root = tmp_path / "case"
    nested = root / "nested"
    nested.mkdir(parents=True)
    root_file = root / "root.txt"
    root_file.write_text("root", encoding="utf-8")
    (nested / "nested.txt").write_text("nested", encoding="utf-8")

    inventory = scan_supported_files(
        [nested, root, root_file, root],
        directory_workers=2,
    )

    assert {item.path.name for item in inventory.files} == {"root.txt", "nested.txt"}
    assert inventory.counters.roots_requested == 4
    assert inventory.counters.roots_scanned == 1
    assert inventory.counters.roots_deduplicated == 3
    assert inventory.counters.duplicate_paths == 0
    nested_result = next(result for result in inventory.root_results if result.root == nested)
    file_result = next(result for result in inventory.root_results if result.root == root_file)
    assert nested_result.covered_by == root
    assert nested_result.supported_files == 1
    assert file_result.covered_by == root
    assert file_result.supported_files == 1


def test_errors_and_completeness_are_reported_per_root(tmp_path, monkeypatch):
    incomplete_root = tmp_path / "incomplete"
    blocked = incomplete_root / "blocked"
    blocked.mkdir(parents=True)
    complete_root = tmp_path / "complete"
    complete_root.mkdir()
    (complete_root / "ok.txt").write_text("ok", encoding="utf-8")
    real_scandir = scanner.os.scandir

    def controlled_scandir(path):
        if normalize_inventory_path(path) == normalize_inventory_path(blocked):
            raise PermissionError(13, "blocked for test", os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(scanner.os, "scandir", controlled_scandir)

    inventory = scan_supported_files([incomplete_root, complete_root], directory_workers=2)

    incomplete = next(result for result in inventory.root_results if result.root == incomplete_root)
    complete = next(result for result in inventory.root_results if result.root == complete_root)
    assert incomplete.complete is False
    assert len(incomplete.errors) == 1
    assert incomplete.errors[0].operation == "scandir"
    assert incomplete.errors[0].path == blocked
    assert complete.complete is True
    assert complete.errors == ()
    assert inventory.complete is False
    assert inventory.counters.errors == 1


def test_cancellation_stops_discovery_and_marks_every_root_incomplete(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    for index in range(10):
        directory = root / f"directory-{index}"
        directory.mkdir()
        (directory / f"file-{index}.txt").write_text("content", encoding="utf-8")
    calls = 0

    def stop_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    inventory = scan_supported_files(
        [root],
        stop_requested=stop_requested,
        directory_workers=4,
    )

    assert inventory.cancelled is True
    assert inventory.complete is False
    assert inventory.root_results[0].complete is False
    assert inventory.counters.supported_files < 10


def test_directory_concurrency_is_bounded_and_configurable(tmp_path, monkeypatch):
    root = tmp_path / "case"
    root.mkdir()
    for index in range(6):
        (root / f"directory-{index}").mkdir()

    real_scan_directory = scanner._scan_directory
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def observed_scan_directory(job, cancel_event):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            return real_scan_directory(job, cancel_event)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(scanner, "_scan_directory", observed_scan_directory)

    scan_supported_files([root], directory_workers=2)

    assert maximum_active == 2
    with pytest.raises(ValueError, match="at least 1"):
        scan_supported_files([root], directory_workers=0)


def test_symlink_entries_are_not_followed_outside_the_root(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target_file = outside / "outside.txt"
    target_file.write_text("outside", encoding="utf-8")
    (outside / "nested.txt").write_text("nested", encoding="utf-8")
    try:
        (root / "file-link.txt").symlink_to(target_file)
        (root / "directory-link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    inventory = scan_supported_files([root], directory_workers=1)

    assert inventory.files == ()
    assert inventory.counters.skipped_links == 2
    assert inventory.counters.directories_scanned == 1
    assert inventory.complete is True


def test_link_entry_is_skipped_without_querying_its_target(tmp_path, monkeypatch):
    root = tmp_path / "case"
    root.mkdir()

    class FakeLink:
        name = "linked.txt"
        path = str(root / name)

        @staticmethod
        def is_symlink():
            return True

        @staticmethod
        def is_dir(*, follow_symlinks=True):
            raise AssertionError("link target must not be queried")

        @staticmethod
        def is_file(*, follow_symlinks=True):
            raise AssertionError("link target must not be queried")

        @staticmethod
        def stat(*, follow_symlinks=True):
            raise AssertionError("link target must not be queried")

    class FakeScandir:
        def __enter__(self):
            return iter((FakeLink(),))

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(scanner.os, "scandir", lambda path: FakeScandir())

    inventory = scan_supported_files([root], directory_workers=1)

    assert inventory.files == ()
    assert inventory.counters.skipped_links == 1
    assert inventory.complete is True


def test_cloud_reparse_metadata_is_not_treated_as_a_name_surrogate():
    cloud_placeholder_tag = 0x9000001A
    mount_point_tag = 0xA0000003

    assert scanner._is_name_surrogate(
        SimpleNamespace(st_reparse_tag=cloud_placeholder_tag)
    ) is False
    assert scanner._is_name_surrogate(
        SimpleNamespace(st_reparse_tag=mount_point_tag)
    ) is True


def test_missing_root_is_an_explicit_incomplete_error(tmp_path):
    missing = tmp_path / "missing"

    inventory = scan_supported_files([missing])

    assert inventory.files == ()
    assert inventory.complete is False
    assert inventory.root_results[0].exists is False
    assert inventory.root_results[0].complete is False
    assert inventory.errors[0].operation == "root_metadata"
    assert inventory.counters.errors == 1
