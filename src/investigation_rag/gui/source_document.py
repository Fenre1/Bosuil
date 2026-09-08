"""Non-Qt source loading/rendering, executed in an isolated process.

The source bytes are verified against the indexed hash before using coordinates.
No document changes, network resources or persistent evidence copies are made.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import tempfile
from typing import Any

from investigation_rag.extractors.structure import blocks_html, text_blocks


_PAGE_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_DOCUMENT_CACHE: OrderedDict[tuple, Any] = OrderedDict()


@contextmanager
def source_bytes(path: str, expected_hash: str, chain: list[dict]):
    data = Path(path).read_bytes()
    if not expected_hash:
        raise ValueError("This chunk has no verified source reference. Rescan to enable Source view.")
    if sha256(data).hexdigest() != expected_hash:
        raise ValueError("The source has changed since indexing. Rescan to restore matching highlights; indexed text remains available.")
    suffix = Path(path).suffix.lower()
    for part in chain:
        import extract_msg
        message = extract_msg.Message(data)
        try:
            attachment = message.attachments[int(part["index"])]
            name = str(getattr(attachment, "longFilename", "") or getattr(attachment, "shortFilename", "") or "attachment")
            if name != part["name"]:
                raise ValueError("The attachment reference no longer matches the source message.")
            content = attachment.data
            data = content if isinstance(content, bytes) else content.exportBytes()
            suffix = Path(name).suffix.lower()
        finally:
            message.close()
    yield data, suffix


def load_source(path: str, metadata: dict[str, Any], page_number: int = 1) -> dict[str, Any]:
    try:
        with source_bytes(path, str(metadata.get("source_content_hash", "")), metadata.get("attachment_chain", [])) as (data, suffix):
            digest = sha256(data).hexdigest()
            if suffix == ".pdf":
                return render_pdf(data, digest, metadata.get("source_spans", []), page_number)
            from investigation_rag.extractors import get_extractor_for_path
            key = (digest, suffix)
            document = _DOCUMENT_CACHE.get(key)
            if document is None:
                with tempfile.TemporaryDirectory(prefix="bosuil-preview-") as folder:
                    temporary = Path(folder) / ("source" + suffix)
                    temporary.write_bytes(data)
                    extractor = get_extractor_for_path(temporary)
                    # Rendering the email body doesn't need attachment extraction.
                    if suffix == ".msg":
                        extractor.max_depth = 0
                    document = extractor.extract(temporary)
                if len(data) <= 8 * 1024 * 1024:
                    _DOCUMENT_CACHE[key] = document
                    while len(_DOCUMENT_CACHE) > 3:
                        _DOCUMENT_CACHE.popitem(last=False)
            else:
                _DOCUMENT_CACHE.move_to_end(key)
            spans = metadata.get("source_spans", [])
            selected_section = int(metadata.get("source_section", 0))
            pieces = []
            for index, section in enumerate(document.sections):
                blocks = section.blocks or text_blocks(section.text, f"s{index}")
                pieces.append(blocks_html(blocks, spans if index == selected_section else [], line_numbers=suffix == ".txt"))
            return {"kind": "text", "html": "".join(pieces), "status": "Source document · selected passage highlighted"}
    except FileNotFoundError:
        return {"kind": "error", "status": "Source file is unavailable. Indexed text remains available in Readable text."}
    except Exception as exc:
        return {"kind": "error", "status": f"Source preview unavailable: {exc}"}


def render_pdf(data: bytes, digest: str, spans: list[dict], page_number: int) -> dict[str, Any]:
    import fitz
    key = (digest, page_number)
    cached = _PAGE_CACHE.get(key)
    with fitz.open(stream=data, filetype="pdf") as document:
        page_number = max(1, min(page_number, len(document)))
        page = document[page_number - 1]
        scale = min(2.0, 2200 / max(page.rect.width, page.rect.height))
        matrix = fitz.Matrix(scale, scale)
        if cached is None:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            cached = {"png": pixmap.tobytes("png"), "width": pixmap.width, "height": pixmap.height}
            _PAGE_CACHE[key] = cached
            while len(_PAGE_CACHE) > 6:
                _PAGE_CACHE.popitem(last=False)
        else:
            _PAGE_CACHE.move_to_end(key)
        rectangles = []
        words = None
        for span in spans:
            if span.get("page_number") != page_number or not span.get("bbox"):
                continue
            rect = fitz.Rect(span["bbox"])
            selected = [rect]
            if span.get("partial"):
                if words is None:
                    words = page.get_text("words")
                line_words = sorted([word for word in words if rect.contains(fitz.Rect(word[:4]).tl + (fitz.Rect(word[:4]).br - fitz.Rect(word[:4]).tl) / 2)], key=lambda word: word[0])
                offset = 0
                matches = []
                for word in line_words:
                    end = offset + len(word[4])
                    if end > span.get("block_start", 0) and offset < span.get("block_end", end):
                        matches.append(fitz.Rect(word[:4]))
                    offset = end + 1
                if matches:
                    selected = matches
            for item in selected:
                transformed = item * page.rotation_matrix * matrix
                rectangles.append(list(transformed))
        return {"kind": "pdf", **cached, "rectangles": rectangles, "page": page_number,
                "page_count": len(document), "status": "Original PDF · selected passage highlighted"}
