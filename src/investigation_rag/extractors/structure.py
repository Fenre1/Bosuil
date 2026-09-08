"""Source blocks shared by extraction, chunking and the document viewer.

Offsets are Python character offsets in the unmodified block text. Formatting
runs use the same coordinates; slices never try to align Markdown by length.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html import escape
from html.parser import HTMLParser
import re
from typing import Any


@dataclass
class SourceBlock:
    block_id: str
    text: str
    kind: str = "paragraph"
    location: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    heading_level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceBlock":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})


def text_blocks(text: str, prefix: str = "text", *, kind: str = "paragraph") -> list[SourceBlock]:
    blocks = []
    for index, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\r?\n[ \t]*\r?\n|\Z)", text)):
        start, end = match.span()
        blocks.append(SourceBlock(
            f"{prefix}-{index}", match.group(), kind,
            {"char_start": start, "char_end": end,
             "line_start": text.count("\n", 0, start) + 1,
             "line_end": text.count("\n", 0, end) + 1},
        ))
    return blocks


def split_ranges(text: str, limit: int) -> list[tuple[int, int]]:
    """Partition text without losing characters, preferring sentences then words."""
    limit = max(1, limit)
    ranges = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            window = text[start:end]
            sentences = list(re.finditer(r"[.!?][\"'’”)]*\s+|\n", window))
            words = list(re.finditer(r"\s+", window))
            choices = sentences or words
            if choices:
                end = start + choices[-1].end()
        ranges.append((start, end))
        start = end
    return ranges


def end_line_offset(text: str, end: int) -> int:
    """A terminating newline belongs to its preceding line, not the next one."""
    return text.count("\n", 0, end) - int(end > 0 and text[end - 1] == "\n")


def inline_html(block: SourceBlock, start: int = 0, end: int | None = None,
                selected: list[tuple[int, int]] | None = None) -> str:
    end = len(block.text) if end is None else end
    boundaries = {start, end}
    runs = block.runs + [{"start": a, "end": b, "selected": True} for a, b in selected or []]
    for run in runs:
        boundaries.update((max(start, min(end, int(run["start"]))),
                           max(start, min(end, int(run["end"])))))
    points = sorted(boundaries)
    parts = []
    for a, b in zip(points, points[1:]):
        value = escape(block.text[a:b]).replace("\n", "<br>")
        active = [run for run in runs if int(run["start"]) <= a < int(run["end"])]
        if any(run.get("bold") for run in active):
            value = f"<b>{value}</b>"
        if any(run.get("italic") for run in active):
            value = f"<i>{value}</i>"
        if any(run.get("underline") for run in active):
            value = f"<u>{value}</u>"
        if any(run.get("selected") for run in active):
            value = f'<span style="background-color:#fde68a;color:#172033">{value}</span>'
        if selected and a == min(item[0] for item in selected):
            value = '<a name="selection"></a>' + value
        parts.append(value)
    return "".join(parts)


def blocks_html(blocks: list[SourceBlock], spans: list[dict[str, Any]] | None = None,
                *, excerpt: bool = False, line_numbers: bool = False) -> str:
    selections: dict[str, list[tuple[int, int]]] = {}
    for span in spans or []:
        selections.setdefault(span["block_id"], []).append((
            int(span.get("block_start", 0)), int(span.get("block_end", 0))))
    parts = []
    previous_table = None
    for block in blocks:
        selected = selections.get(block.block_id, [])
        if excerpt and not selected:
            continue
        ranges = selected if excerpt else [(0, len(block.text))]
        for start, end in ranges:
            value = inline_html(block, start, end, None if excerpt else selected)
            if block.kind == "table_row":
                # Cell boundaries are offsets in the original row, including tabs.
                cells = []
                for cell in block.location.get("cells", []):
                    a, b = max(start, cell["start"]), min(end, cell["end"])
                    cells.append(f'<td width="{100 // max(1, len(block.location.get("cells", [])))}%">' + (inline_html(block, a, b, None if excerpt else selected) if a <= b else "") + "</td>")
                table_id = block.location.get("table")
                row_html = "<tr>" + "".join(cells) + "</tr>"
                if previous_table == table_id and parts and parts[-1].endswith("</table>"):
                    parts[-1] = parts[-1][:-len("</table>")] + row_html + "</table>"
                    continue
                value = '<table border="1" cellspacing="0" cellpadding="5" width="100%">' + row_html + "</table>"
                previous_table = table_id
            elif block.kind == "heading":
                level = min(6, max(1, block.heading_level))
                value = f"<h{level}>{value}</h{level}>"
            elif block.kind == "list_item":
                marker = escape(str(block.location.get("list_marker", "•")))
                indent = 16 * (int(block.location.get("list_level", 0)) + 1)
                value = f'<p style="margin-left:{indent}px">{marker} {value}</p>'
            elif block.kind in {"pre", "message_header"} or line_numbers:
                if line_numbers:
                    # Render each original line with its source number.
                    first = int(block.location.get("line_start", 1)) + block.text.count("\n", 0, start)
                    rendered = []
                    offset = start
                    for number, line in enumerate(block.text[start:end].splitlines(keepends=True), first):
                        rendered.append(f'<span style="color:#64748b">{number:5}  </span>' + inline_html(block, offset, offset + len(line), None if excerpt else selected))
                        offset += len(line)
                    value = "".join(rendered)
                value = f'<pre style="white-space:pre-wrap">{value}</pre>'
            else:
                value = f"<p>{value}</p>"
            if block.kind != "table_row":
                previous_table = None
            if block.location.get("quote_depth", 0):
                value = f'<blockquote style="color:#475569">{value}</blockquote>'
            parts.append(value)
    return "".join(parts)


class HtmlSourceParser(HTMLParser):
    """Read email content into safe text/style blocks; no URLs or active HTML survive."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[SourceBlock] = []
        self.text = ""
        self.runs: list[dict[str, Any]] = []
        self.styles: list[str] = []
        self.hidden = 0
        self.kind = "paragraph"
        self.level = 0
        self.quote_depth = 0
        self.table = -1
        self.row = 0
        self.cells: list[dict[str, int]] = []
        self.in_row = False
        self.header = False
        self.lists: list[dict[str, Any]] = []
        self.list_marker = "•"

    def flush(self) -> None:
        if self.text.strip():
            self.blocks.append(SourceBlock(
                f"body-{len(self.blocks)}", self.text, self.kind,
                {"quote_depth": self.quote_depth, "table": self.table, "row": self.row,
                 "cells": list(self.cells), "is_header": self.header}, list(self.runs), self.level,
            ))
            self.blocks[-1].location.update(list_marker=self.list_marker, list_level=max(0, len(self.lists) - 1))
        self.text, self.runs, self.cells = "", [], []
        self.kind, self.level, self.header = "paragraph", 0, False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "head"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag in {"b", "strong", "i", "em", "u"}:
            self.styles.append(tag)
        if tag == "br":
            self.handle_data("\n")
        if tag in {"ul", "ol"}:
            self.flush()
            values = dict(attrs)
            try:
                start = int(values.get("start") or 1)
            except ValueError:
                start = 1
            self.lists.append({"ordered": tag == "ol", "next": start})
        if tag == "table":
            self.flush()
            self.table += 1
            self.row = 0
        if tag == "tr":
            self.flush()
            self.in_row = True
            self.kind = "table_row"
        if tag in {"td", "th"}:
            if self.cells:
                self.handle_data("\t")
            self.cells.append({"start": len(self.text), "end": len(self.text)})
            self.header |= tag == "th"
        if tag in {"p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}:
            if self.in_row:
                if self.text and not self.text.endswith(("\t", "\n")):
                    self.handle_data("\n")
                return
            self.flush()
            if tag == "li" and self.lists:
                current_list = self.lists[-1]
                self.list_marker = f"{current_list['next']}." if current_list["ordered"] else "•"
                current_list["next"] += 1
            if tag == "blockquote":
                self.quote_depth += 1
            self.kind = "list_item" if tag == "li" else "heading" if re.fullmatch(r"h[1-6]", tag) else "paragraph"
            self.level = int(tag[1]) if self.kind == "heading" else 0

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "head"}:
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if tag in self.styles:
            self.styles.reverse()
            self.styles.remove(tag)
            self.styles.reverse()
        if tag in {"ul", "ol"}:
            self.flush()
            if self.lists:
                self.lists.pop()
        if tag in {"td", "th"} and self.cells:
            self.cells[-1]["end"] = len(self.text)
        if tag == "tr":
            self.flush()
            self.in_row = False
            self.row += 1
        if not self.in_row and tag in {"p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
        if tag == "blockquote":
            self.quote_depth = max(0, self.quote_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.hidden:
            return
        start = len(self.text)
        self.text += data
        self.runs.append({"start": start, "end": len(self.text),
                          "bold": any(tag in self.styles for tag in ("b", "strong")),
                          "italic": any(tag in self.styles for tag in ("i", "em")),
                          "underline": "u" in self.styles})


def html_blocks(html: str) -> list[SourceBlock]:
    parser = HtmlSourceParser()
    parser.feed(html)
    parser.close()
    parser.flush()
    return parser.blocks
