from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from investigation_rag.extractors.base import BaseExtractor, ExtractedDocument, ExtractedSection


class PdfExtractor(BaseExtractor):
    source_type = "pdf"

    def extract(self, path: Path) -> ExtractedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to extract .pdf files") from exc

        sections: list[ExtractedSection] = []
        with fitz.open(str(path)) as document:
            document_metadata = _clean_metadata(document.metadata or {})
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    layout = page.get_text("dict")
                    sections.append(
                        ExtractedSection(
                            text=text,
                            source_type=self.source_type,
                            page_number=index,
                            display_markdown=pdf_layout_to_markdown(layout),
                            metadata={**document_metadata, "page_number": index},
                        )
                    )
        return ExtractedDocument(
            source_type=self.source_type,
            sections=sections,
            metadata={"path": str(path), "page_count": len(sections), **document_metadata},
        )


def _clean_metadata(metadata: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    key_map = {
        "creationDate": "creation_date",
        "modDate": "modification_date",
    }
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        clean_key = key_map.get(str(key), str(key))
        cleaned[clean_key] = str(value).strip()
    return cleaned


@dataclass(frozen=True)
class PdfLine:
    text: str
    size: float
    bold: bool
    x0: float
    x1: float
    y0: float
    y1: float


def pdf_layout_to_markdown(layout: dict[str, Any]) -> str | None:
    raw_lines = _extract_lines(layout)
    if not raw_lines:
        return None
    page_width = _safe_float(layout.get("width"), 0.0) or max((line.x1 for line in raw_lines), default=0.0)
    page_height = _safe_float(layout.get("height"), 0.0) or max((line.y1 for line in raw_lines), default=0.0)
    lines = _merge_same_baseline_lines(_reflow_two_column_rows(raw_lines, page_width), page_width)
    body_size = _body_font_size(lines)

    markdown_lines: list[str] = []
    for index, line in enumerate(lines):
        text = _clean_line_text(line.text)
        if not text:
            continue
        if _is_page_artifact(text, line, page_width, page_height):
            continue
        previous_line = lines[index - 1] if index > 0 else None
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        if previous_line is not None and _needs_block_gap(line, previous_line):
            _append_blank(markdown_lines)

        rendered = _line_to_markdown(text, line, body_size, page_width, page_height, previous_line, next_line)
        if _merge_section_heading(markdown_lines, rendered):
            continue
        markdown_lines.append(rendered)

    markdown = "\n".join(markdown_lines).strip()
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown or None


def _extract_lines(layout: dict[str, Any]) -> list[PdfLine]:
    raw_lines: list[PdfLine] = []
    for block in layout.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            sizes = [_safe_float(span.get("size"), 0.0) for span in spans]
            flags = [int(span.get("flags", 0)) for span in spans]
            font_names = [str(span.get("font", "")).lower() for span in spans]
            bbox = line.get("bbox") or block.get("bbox") or (0, 0, 0, 0)
            raw_lines.append(
                PdfLine(
                    text=text,
                    size=max(sizes) if sizes else 0.0,
                    bold=any(_span_is_bold(flag, font_name) for flag, font_name in zip(flags, font_names)),
                    x0=_safe_float(bbox[0] if len(bbox) > 0 else 0.0, 0.0),
                    x1=_safe_float(bbox[2] if len(bbox) > 2 else 0.0, 0.0),
                    y0=_safe_float(bbox[1] if len(bbox) > 1 else 0.0, 0.0),
                    y1=_safe_float(bbox[3] if len(bbox) > 3 else 0.0, 0.0),
                )
            )
    return sorted(raw_lines, key=lambda item: (round(item.y0, 1), item.x0))


def _body_font_size(lines: list[PdfLine]) -> float:
    sizes = [round(line.size, 1) for line in lines if line.size > 0 and len(line.text) > 20]
    if len(sizes) < 3:
        sizes = [round(line.size, 1) for line in lines if line.size > 0]
    if not sizes:
        return 10.0
    counts = Counter(sizes)
    return sorted(counts, key=lambda size: (-counts[size], size))[0]


def _line_to_markdown(
    text: str,
    line: PdfLine,
    body_size: float,
    page_width: float,
    page_height: float,
    previous_line: PdfLine | None,
    next_line: PdfLine | None,
) -> str:
    bullet = _bullet_markdown(text)
    if bullet:
        return bullet
    heading_level = _heading_level(line, text, body_size, page_width, page_height, previous_line, next_line)
    if heading_level:
        return f"{'#' * heading_level} {_escape_markdown(text)}"
    return _escape_markdown(text)


def _heading_level(
    line: PdfLine,
    text: str,
    body_size: float,
    page_width: float,
    page_height: float,
    previous_line: PdfLine | None,
    next_line: PdfLine | None,
) -> int | None:
    if len(text) > 140:
        return None
    if text.lower().startswith("keywords:"):
        return None
    if re.match(r"^\d+\.\d+\s+\S", text):
        return 3
    if _looks_like_affiliation(text):
        return None
    if (
        previous_line is None
        and page_height
        and line.y0 < page_height * 0.2
        and line.size >= body_size * 1.05
        and _is_centered(line, page_width)
    ):
        return 1
    if re.match(r"^\d+[A-Z]", text):
        return None
    if _looks_like_sentence(text) and line.size < body_size * 1.45:
        return None
    if line.size >= body_size * 1.65:
        return 1
    if line.size >= body_size * 1.35:
        return 2
    if line.size >= body_size * 1.18 and line.bold and _has_heading_spacing(line, previous_line, next_line, body_size):
        return 3
    if re.match(r"^\d+\s+[A-Z]", text) and len(text) <= 80:
        return 3
    if _is_centered_heading(line, text, body_size, page_width, previous_line, next_line):
        return 3
    return None


def _bullet_markdown(text: str) -> str | None:
    bullet_match = re.match(r"^[\u2022\u2023\u25e6\u2043\u2219o]\s+(.+)$", text)
    if bullet_match:
        return f"- {_escape_markdown(bullet_match.group(1).strip())}"
    dash_match = re.match(r"^[-–]\s+(.+)$", text)
    if dash_match:
        return f"- {_escape_markdown(dash_match.group(1).strip())}"
    numbered_match = re.match(r"^(\d+[\.)])\s+(.+)$", text)
    if numbered_match:
        return f"{numbered_match.group(1)} {_escape_markdown(numbered_match.group(2).strip())}"
    return None


def _needs_block_gap(line: PdfLine, previous_line: PdfLine) -> bool:
    line_gap = line.y0 - previous_line.y0
    expected_gap = max(previous_line.size, line.size, 10.0) * 1.45
    return line_gap > expected_gap


def _merge_same_baseline_lines(lines: list[PdfLine], page_width: float = 0.0) -> list[PdfLine]:
    ordered = sorted(lines, key=lambda item: (round(item.y0, 1), item.x0))
    merged: list[PdfLine] = []
    for line in ordered:
        if not merged or abs(merged[-1].y0 - line.y0) > max(2.0, line.size * 0.25):
            merged.append(line)
            continue
        previous = merged[-1]
        if _crosses_table_column_boundary(previous, line, page_width):
            merged.append(line)
            continue
        if abs(previous.size - line.size) > max(0.8, min(previous.size, line.size) * 0.12):
            merged.append(line)
            continue
        gap = line.x0 - previous.x1
        if gap > max(previous.size, line.size) * 8:
            merged.append(line)
            continue
        text_gap = " " if gap > max(previous.size, line.size) * 0.6 else ""
        merged[-1] = PdfLine(
            text=f"{previous.text}{text_gap}{line.text}",
            size=max(previous.size, line.size),
            bold=previous.bold and line.bold,
            x0=min(previous.x0, line.x0),
            x1=max(previous.x1, line.x1),
            y0=min(previous.y0, line.y0),
            y1=max(previous.y1, line.y1),
        )
    return merged


def _reflow_two_column_rows(lines: list[PdfLine], page_width: float) -> list[PdfLine]:
    if not page_width:
        return lines
    split_x = page_width * 0.20
    row_groups = _same_baseline_groups(lines)
    two_column_groups = [
        group
        for group in row_groups
        if any(line.x0 < split_x for line in group) and any(line.x0 >= split_x for line in group)
    ]
    has_numbered_left_cell = any(
        _starts_table_row(_clean_line_text(line.text))
        for group in two_column_groups
        for line in group
        if line.x0 < split_x
    )
    if len(two_column_groups) < 3 or not has_numbered_left_cell:
        return lines

    output: list[PdfLine] = []
    current_left: list[PdfLine] = []
    current_right: list[PdfLine] = []

    def flush() -> None:
        if current_left:
            output.append(_combine_lines(current_left))
        output.extend(current_right)
        current_left.clear()
        current_right.clear()

    for group in row_groups:
        left = [line for line in group if line.x0 < split_x]
        right = [line for line in group if line.x0 >= split_x]
        left_text = " ".join(_clean_line_text(line.text) for line in left)

        if left and _starts_table_row(left_text):
            flush()
            current_left.extend(left)
            current_right.extend(right)
            continue
        if left and current_left and not _starts_table_row(left_text):
            current_left.extend(left)
            current_right.extend(right)
            continue
        if right and current_left:
            current_right.extend(right)
            continue

        flush()
        output.extend(group)

    flush()
    return sorted(output, key=lambda item: (round(item.y0, 1), item.x0))


def _same_baseline_groups(lines: list[PdfLine]) -> list[list[PdfLine]]:
    groups: list[list[PdfLine]] = []
    for line in sorted(lines, key=lambda item: (round(item.y0, 1), item.x0)):
        if not groups or abs(groups[-1][0].y0 - line.y0) > max(2.0, line.size * 0.25):
            groups.append([line])
        else:
            groups[-1].append(line)
    return groups


def _combine_lines(lines: list[PdfLine]) -> PdfLine:
    ordered = sorted(lines, key=lambda item: (item.y0, item.x0))
    text = " ".join(_clean_line_text(line.text) for line in ordered)
    return PdfLine(
        text=text,
        size=max(line.size for line in ordered),
        bold=any(line.bold for line in ordered),
        x0=min(line.x0 for line in ordered),
        x1=max(line.x1 for line in ordered),
        y0=min(line.y0 for line in ordered),
        y1=max(line.y1 for line in ordered),
    )


def _starts_table_row(text: str) -> bool:
    return bool(re.match(r"^(?:Domain / Subdomain|\d+(?:\.\d+)*\s*[A-Z])", text))


def _crosses_table_column_boundary(previous: PdfLine, line: PdfLine, page_width: float) -> bool:
    return bool(page_width and previous.x0 < page_width * 0.20 <= line.x0)


def _merge_section_heading(markdown_lines: list[str], rendered: str) -> bool:
    if not markdown_lines:
        return False
    current_match = re.match(r"^(#{1,6})\s+(.+)$", rendered)
    previous_match = re.match(r"^(#{1,6})\s+(\d+(?:\.\d+)*)$", markdown_lines[-1])
    if not current_match or not previous_match:
        return False
    if len(current_match.group(1)) != len(previous_match.group(1)):
        return False
    markdown_lines[-1] = f"{previous_match.group(1)} {previous_match.group(2)} {current_match.group(2)}"
    return True


def _is_centered_heading(
    line: PdfLine,
    text: str,
    body_size: float,
    page_width: float,
    previous_line: PdfLine | None,
    next_line: PdfLine | None,
) -> bool:
    if not page_width or len(text) > 90 or _looks_like_sentence(text):
        return False
    if abs(line.size - body_size) > max(1.0, body_size * 0.15):
        return False
    if not _is_centered(line, page_width):
        return False
    before = line.y0 - previous_line.y0 if previous_line is not None else body_size * 2
    after = next_line.y0 - line.y0 if next_line is not None else body_size * 2
    return before >= body_size * 1.4 and after >= body_size * 1.4


def _has_heading_spacing(
    line: PdfLine,
    previous_line: PdfLine | None,
    next_line: PdfLine | None,
    body_size: float,
) -> bool:
    before = line.y0 - previous_line.y0 if previous_line is not None else body_size * 2
    after = next_line.y0 - line.y0 if next_line is not None else body_size * 2
    threshold = body_size * 1.6
    return before >= threshold and after >= threshold


def _is_centered(line: PdfLine, page_width: float) -> bool:
    line_center = (line.x0 + line.x1) / 2
    page_center = page_width / 2
    return abs(line_center - page_center) <= page_width * 0.08


def _is_page_artifact(text: str, line: PdfLine, page_width: float, page_height: float) -> bool:
    if (
        re.fullmatch(r"\d+", text)
        and page_width
        and page_height
        and _is_centered(line, page_width)
        and (line.y0 > page_height * 0.82 or line.y0 < page_height * 0.08)
    ):
        return True
    if text.lower().startswith("arxiv:") and (line.x0 < page_width * 0.15 if page_width else True):
        return True
    return False


def _looks_like_affiliation(text: str) -> bool:
    return bool(
        re.match(
            r"^\d+\s*(?:MIT|School|Future|Computer Science|Department|University|Institute|Harmony)\b",
            text,
        )
    )


def _append_blank(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def _clean_line_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(\d)([A-Z][a-z])", r"\1 \2", text)


def _looks_like_sentence(text: str) -> bool:
    return text.endswith((".", "?", "!", ";", ":")) and len(text.split()) > 3


def _span_is_bold(flags: int, font_name: str) -> bool:
    return bool(flags & 16) or "bold" in font_name or "black" in font_name or "semibold" in font_name


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _escape_markdown(text: str) -> str:
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")
