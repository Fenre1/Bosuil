from __future__ import annotations

from pathlib import Path

from investigation_rag.extractors.base import BaseExtractor
from investigation_rag.extractors.docx_extractor import DocxExtractor
from investigation_rag.extractors.msg_extractor import MsgExtractor
from investigation_rag.extractors.pdf_extractor import PdfExtractor
from investigation_rag.extractors.txt_extractor import TxtExtractor


def get_extractor_for_path(path: str | Path) -> BaseExtractor:
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        return TxtExtractor()
    if suffix == ".docx":
        return DocxExtractor()
    if suffix == ".pdf":
        return PdfExtractor()
    if suffix == ".msg":
        return MsgExtractor()
    raise ValueError(f"Unsupported file type: {suffix}")
