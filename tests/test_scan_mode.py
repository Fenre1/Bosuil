import pytest

from investigation_rag.ingest.scan_mode import ScanMode


def test_scan_mode_parse_accepts_enum_and_text():
    assert ScanMode.parse(ScanMode.FAST) is ScanMode.FAST
    assert ScanMode.parse(" VERIFY ") is ScanMode.VERIFY


def test_scan_mode_parse_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown scan mode"):
        ScanMode.parse("incremental")
