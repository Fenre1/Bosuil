from __future__ import annotations

from pathlib import Path
from typing import Any

from investigation_rag.extractors.base import BaseExtractor, ExtractedDocument, ExtractedSection


class DocxExtractor(BaseExtractor):
    source_type = "docx"

    def extract(self, path: Path) -> ExtractedDocument:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("python-docx is required to extract .docx files") from exc

        document = Document(str(path))
        metadata = {"path": str(path), **_core_properties(document)}
        plain_parts: list[str] = []
        markdown_parts: list[str] = []
        for block in _iter_blocks(document):
            if _is_table(block):
                plain_table, markdown_table = _table_text(block)
                if plain_table:
                    plain_parts.append(plain_table)
                    markdown_parts.append(markdown_table)
                continue

            text = block.text.strip()
            if not text:
                continue
            plain_parts.append(text)
            markdown_parts.append(_paragraph_markdown(block))

        text = "\n\n".join(plain_parts).strip()
        markdown = "\n\n".join(markdown_parts).strip()
        return ExtractedDocument(
            source_type=self.source_type,
            sections=[ExtractedSection(text=text, source_type=self.source_type, display_markdown=markdown or None)],
            metadata=metadata,
        )


def _core_properties(document: object) -> dict[str, object]:
    properties = getattr(document, "core_properties", None)
    if properties is None:
        return {}
    field_map = {
        "title": "title",
        "author": "author",
        "subject": "subject",
        "keywords": "keywords",
        "comments": "comments",
        "category": "category",
        "last_modified_by": "last_modified_by",
        "revision": "revision",
        "created": "creation_date",
        "modified": "modification_date",
    }
    metadata: dict[str, object] = {}
    for attr, key in field_map.items():
        value = getattr(properties, attr, None)
        if value in (None, ""):
            continue
        metadata[key] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return metadata


def _iter_blocks(document: Any) -> list[Any]:
    try:
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        return list(getattr(document, "paragraphs", []))

    blocks = []
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            blocks.append(Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            blocks.append(Table(child, document))
    return blocks


def _is_table(block: Any) -> bool:
    return hasattr(block, "rows") and hasattr(block, "columns")


def _paragraph_markdown(paragraph: Any) -> str:
    text = paragraph.text.strip()
    style = paragraph.style.name if paragraph.style else ""
    style_name = style.lower()
    inline = _inline_markdown(paragraph) or _escape_markdown(text)

    if style_name.startswith("heading"):
        level = _heading_level(style_name)
        return f"{'#' * level} {inline}"
    if "list bullet" in style_name:
        return f"- {inline}"
    if "list number" in style_name:
        return f"1. {inline}"
    return inline


def _heading_level(style_name: str) -> int:
    digits = "".join(char for char in style_name if char.isdigit())
    if not digits:
        return 1
    return min(max(int(digits), 1), 6)


def _inline_markdown(paragraph: Any) -> str:
    parts: list[str] = []
    for run in paragraph.runs:
        text = run.text
        if not text:
            continue
        escaped = _escape_markdown(text)
        if run.bold and run.italic:
            parts.append(f"***{escaped}***")
        elif run.bold:
            parts.append(f"**{escaped}**")
        elif run.italic:
            parts.append(f"*{escaped}*")
        else:
            parts.append(escaped)
    return "".join(parts).strip()


def _table_text(table: Any) -> tuple[str, str]:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return "", ""

    plain = "\n".join("\t".join(cells) for cells in rows)
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---" for _ in range(width)]
    body = normalized[1:] or [["" for _ in range(width)]]
    markdown_rows = [header, separator, *body]
    markdown = "\n".join("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |" for row in markdown_rows)
    return plain, markdown


def _escape_markdown(text: str) -> str:
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")


def _escape_table_cell(text: str) -> str:
    return _escape_markdown(text).replace("|", "\\|")
