import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "entity_ground_truth"
    / "onveiligheid-door-extreme-regen.json"
)
PDF_PATH = Path(__file__).parents[2] / "test_pdf" / "onveiligheid-door-extreme-regen.pdf"


def normalized_pdf_text(value: str) -> str:
    value = re.sub(r"\u00ad\s*", "", value)
    value = value.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def test_ground_truth_has_all_entity_types_and_unique_rows():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    counts = Counter(item["type"] for item in payload["entities"])

    assert set(counts) == {"organization", "department", "location", "position", "person"}
    assert all(counts[entity_type] >= 8 for entity_type in counts)
    keys = [
        (normalized_pdf_text(item["text"]), item["type"])
        for item in payload["entities"]
    ]
    assert len(keys) == len(set(keys))


def test_ground_truth_entities_are_present_on_the_cited_pdf_pages():
    if not PDF_PATH.is_file():
        pytest.skip("The large local ground-truth PDF is not included in the source repository.")

    import fitz

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(PDF_PATH.read_bytes()).hexdigest() == payload["sha256"]
    with fitz.open(PDF_PATH) as document:
        assert len(document) == payload["page_count"]
        page_text = [normalized_pdf_text(page.get_text("text")) for page in document]

    missing = []
    for item in payload["entities"]:
        expected = normalized_pdf_text(item["text"])
        cited = [int(page_number) for page_number in item["evidence_pages"]]
        if not any(expected in page_text[page_number - 1] for page_number in cited):
            missing.append((item["text"], cited))
    assert missing == []
