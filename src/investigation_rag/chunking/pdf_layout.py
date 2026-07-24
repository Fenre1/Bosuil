from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from investigation_rag.chunking.base import Chunk
from investigation_rag.extractors.base import ExtractedDocument


PDF_CHUNKER_ALGORITHM_VERSION = "pdf-layout-v3"
PDF_HEADER_FOOTER_POLICY_VERSION = "pdf-furniture-v1"
PDF_HEADING_POLICY_VERSION = "pdf-headings-v1"
PDF_TABLE_POLICY_VERSION = "pdf-tables-v1"
PDF_FIGURE_POLICY_VERSION = "pdf-figures-v1"
PDF_DISPLAY_POLICY_VERSION = "pdf-display-v1"
DEFAULT_PDF_SOFT_MIN_CHARS = 1200
DEFAULT_PDF_OVERLAP_MODE = "sentence"


@dataclass(frozen=True)
class PdfChunkingConfig:
    target_chars: int = 2800
    max_chars: int = 4800
    soft_min_chars: int = DEFAULT_PDF_SOFT_MIN_CHARS
    overlap_mode: str = DEFAULT_PDF_OVERLAP_MODE
    overlap_chars: int = 240
    header_region: float = 0.13
    footer_region: float = 0.13
    min_recurring_ratio: float = 0.35

    def normalized(self) -> "PdfChunkingConfig":
        overlap_mode = self.overlap_mode if self.overlap_mode in {"none", "sentence", "paragraph"} else DEFAULT_PDF_OVERLAP_MODE
        target_chars = max(1, int(self.target_chars))
        max_chars = max(target_chars, int(self.max_chars))
        soft_min_chars = max(0, int(self.soft_min_chars))
        return PdfChunkingConfig(
            target_chars=target_chars,
            max_chars=max_chars,
            soft_min_chars=soft_min_chars,
            overlap_mode=overlap_mode,
            overlap_chars=max(0, int(self.overlap_chars)),
            header_region=max(0.05, min(0.25, float(self.header_region))),
            footer_region=max(0.05, min(0.25, float(self.footer_region))),
            min_recurring_ratio=max(0.1, min(0.8, float(self.min_recurring_ratio))),
        )


def pdf_chunker_version(config: PdfChunkingConfig) -> str:
    config = config.normalized()
    parts = [
        PDF_CHUNKER_ALGORITHM_VERSION,
        f"target={config.target_chars}",
        f"max={config.max_chars}",
        f"soft_min={config.soft_min_chars}",
        f"overlap={config.overlap_mode}",
        f"overlap_chars={config.overlap_chars}",
        f"header={config.header_region:.2f}",
        f"footer={config.footer_region:.2f}",
        PDF_HEADER_FOOTER_POLICY_VERSION,
        PDF_HEADING_POLICY_VERSION,
        PDF_TABLE_POLICY_VERSION,
        PDF_FIGURE_POLICY_VERSION,
        PDF_DISPLAY_POLICY_VERSION,
    ]
    return "|".join(parts)


@dataclass
class PdfLayoutBlock:
    block_id: str
    text: str
    page_number: int
    bbox: tuple[float, float, float, float]
    rel_bbox: tuple[float, float, float, float]
    page_width: float
    page_height: float
    font_size: float
    flags: int
    bold: bool
    italic: bool
    reading_order: int
    metadata: dict[str, Any] = field(default_factory=dict)
    is_recurring_header: bool = False
    is_recurring_footer: bool = False
    is_page_number: bool = False
    boilerplate_pattern_id: str | None = None
    boilerplate_score: dict[str, Any] | None = None
    boilerplate_reason: str = ""
    is_heading: bool = False
    heading_level: int | None = None
    heading_reason: str = ""
    content_type: str = "body"

    @property
    def x0_rel(self) -> float:
        return self.rel_bbox[0]

    @property
    def y0_rel(self) -> float:
        return self.rel_bbox[1]

    @property
    def x1_rel(self) -> float:
        return self.rel_bbox[2]

    @property
    def y1_rel(self) -> float:
        return self.rel_bbox[3]

    @property
    def x_center_rel(self) -> float:
        return (self.x0_rel + self.x1_rel) / 2

    @property
    def line_height_rel(self) -> float:
        return max(0.0, self.y1_rel - self.y0_rel)


@dataclass(frozen=True)
class LogicalUnit:
    text: str
    display_text: str
    pages: tuple[int, ...]
    heading_path: tuple[str, ...]
    content_type: str
    source_spans: tuple[dict[str, Any], ...]
    debug: tuple[str, ...] = ()
    contains_table: bool = False
    contains_list: bool = False
    contains_caption: bool = False
    contains_footnote: bool = False
    contains_figure: bool = False
    is_structurally_meaningful: bool = False
    is_continuation: bool = False
    fallback_rule: str | None = None


@dataclass
class PdfChunkDraft:
    units: list[LogicalUnit] = field(default_factory=list)
    boundary_reason: str = ""
    is_continuation: bool = False
    overlap_text: str = ""
    overlap_source_spans: tuple[dict[str, Any], ...] = ()

    @property
    def heading_path(self) -> tuple[str, ...]:
        for unit in self.units:
            if unit.heading_path:
                return unit.heading_path
        return ()

    @property
    def text(self) -> str:
        return "\n\n".join(unit.text for unit in self.units if unit.text).strip()

    @property
    def display_text(self) -> str:
        return "\n\n".join(unit.display_text for unit in self.units if unit.display_text).strip()

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for unit in self.units for page in unit.pages}))

    @property
    def content_types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(unit.content_type for unit in self.units))

    @property
    def source_spans(self) -> tuple[dict[str, Any], ...]:
        spans: list[dict[str, Any]] = []
        offset = 0
        first = True
        for unit in self.units:
            if not first:
                offset += 2
            for span in unit.source_spans:
                adjusted = dict(span)
                if "char_start" in adjusted:
                    adjusted["char_start"] = int(adjusted["char_start"]) + offset
                if "char_end" in adjusted:
                    adjusted["char_end"] = int(adjusted["char_end"]) + offset
                spans.append(adjusted)
            offset += len(unit.text)
            first = False
        return tuple(spans)

    @property
    def contains_table(self) -> bool:
        return any(unit.contains_table for unit in self.units)

    @property
    def contains_list(self) -> bool:
        return any(unit.contains_list for unit in self.units)

    @property
    def contains_caption(self) -> bool:
        return any(unit.contains_caption for unit in self.units)

    @property
    def contains_footnote(self) -> bool:
        return any(unit.contains_footnote for unit in self.units)

    @property
    def contains_figure(self) -> bool:
        return any(unit.contains_figure for unit in self.units)

    @property
    def fallback_rule(self) -> str | None:
        for unit in self.units:
            if unit.fallback_rule:
                return unit.fallback_rule
        return None


def has_pdf_layout(document: ExtractedDocument) -> bool:
    return document.source_type == "pdf" and any(section.metadata.get("layout_blocks") for section in document.sections)


def chunk_pdf_document(document: ExtractedDocument, config: PdfChunkingConfig) -> list[Chunk]:
    config = config.normalized()
    blocks = collect_layout_blocks(document)
    if not blocks:
        return []

    furniture_debug = mark_recurring_page_furniture(blocks, config)
    body_blocks = [block for block in blocks if not (block.is_recurring_header or block.is_recurring_footer or block.is_page_number)]
    body_size = _body_font_size(body_blocks)
    figure_debug = mark_figure_regions(body_blocks, body_size)
    heading_debug = classify_headings(body_blocks, body_size)
    units, unit_debug = build_logical_units(body_blocks, body_size)
    units = split_oversized_units(units, config.max_chars)
    drafts = pack_logical_units(units, config)
    chunks = drafts_to_chunks(drafts, document.metadata, config, furniture_debug, figure_debug, heading_debug, unit_debug)
    return chunks


def collect_layout_blocks(document: ExtractedDocument) -> list[PdfLayoutBlock]:
    blocks: list[PdfLayoutBlock] = []
    for section in sorted(document.sections, key=lambda item: item.page_number or 0):
        raw_blocks = section.metadata.get("layout_blocks")
        if not isinstance(raw_blocks, list):
            continue
        for index, payload in enumerate(raw_blocks):
            if isinstance(payload, dict):
                block = _block_from_payload(payload, section.page_number, index)
                if block.text:
                    blocks.append(block)
    grouped: dict[int, list[PdfLayoutBlock]] = defaultdict(list)
    for block in blocks:
        grouped[block.page_number].append(block)

    ordered: list[PdfLayoutBlock] = []
    for page_number in sorted(grouped):
        page_blocks = grouped[page_number]
        if _should_use_visual_page_order(page_blocks):
            ordered.extend(sorted(page_blocks, key=_visual_block_key))
        else:
            ordered.extend(sorted(page_blocks, key=lambda item: (item.reading_order, item.y0_rel, item.x0_rel)))
    return ordered


def _visual_block_key(block: PdfLayoutBlock) -> tuple[float, float, int]:
    return (round(block.y0_rel, 3), block.x0_rel, block.reading_order)


def _should_use_visual_page_order(blocks: list[PdfLayoutBlock]) -> bool:
    if len(blocks) < 8 or not any(_caption_kind(block.text) for block in blocks):
        return False
    ordered = sorted(
        [block for block in blocks if 0.06 <= block.y0_rel <= 0.92],
        key=lambda item: (item.reading_order, item.y0_rel, item.x0_rel),
    )
    for previous, current in zip(ordered, ordered[1:]):
        vertical_drop = previous.y0_rel - current.y0_rel
        likely_column_switch = current.x0_rel - previous.x0_rel > 0.18
        if vertical_drop >= 0.14 and not likely_column_switch:
            return True
    return False


def mark_recurring_page_furniture(blocks: list[PdfLayoutBlock], config: PdfChunkingConfig) -> dict[str, Any]:
    all_pages = sorted({block.page_number for block in blocks})
    candidates: list[tuple[PdfLayoutBlock, str, str]] = []
    for block in blocks:
        region = _candidate_region(block, config)
        if region is None:
            continue
        normalized = normalize_boilerplate_text(block.text)
        if not normalized:
            continue
        candidates.append((block, region, normalized))
        if _is_standalone_page_number(block.text) and _is_near_page_edge(block, config):
            block.is_page_number = True
            block.boilerplate_pattern_id = "standalone-page-number"
            block.boilerplate_reason = "standalone page number in page-edge region"
            block.boilerplate_score = {
                "policy": PDF_HEADER_FOOTER_POLICY_VERSION,
                "region": region,
                "normalized_text": normalized,
            }

    grouped: dict[tuple[str, str], list[PdfLayoutBlock]] = defaultdict(list)
    for block, region, normalized in candidates:
        if normalized == "#" and not block.is_page_number:
            continue
        grouped[(region, normalized)].append(block)

    patterns: list[dict[str, Any]] = []
    pattern_index = 0
    for (region, normalized), occurrences in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        recurring, score = _recurring_score(occurrences, all_pages, config)
        if not recurring:
            continue
        pattern_id = f"pdf-boilerplate-{pattern_index}"
        pattern_index += 1
        pages = sorted({block.page_number for block in occurrences})
        pattern = {
            "pattern_id": pattern_id,
            "region": region,
            "normalized_text": normalized,
            "pages": pages,
            "score": score,
            "reason": "repeated normalized page-edge text at consistent relative position",
        }
        patterns.append(pattern)
        for block in occurrences:
            if region == "header":
                block.is_recurring_header = True
            elif region == "footer":
                block.is_recurring_footer = True
            elif block.y0_rel < 0.5:
                block.is_recurring_header = True
            else:
                block.is_recurring_footer = True
            block.boilerplate_pattern_id = pattern_id
            block.boilerplate_reason = str(pattern["reason"])
            block.boilerplate_score = score

    marked = [
        {
            "block_id": block.block_id,
            "page_number": block.page_number,
            "text": block.text,
            "is_recurring_header": block.is_recurring_header,
            "is_recurring_footer": block.is_recurring_footer,
            "is_page_number": block.is_page_number,
            "boilerplate_pattern_id": block.boilerplate_pattern_id,
            "reason": block.boilerplate_reason,
        }
        for block in blocks
        if block.is_recurring_header or block.is_recurring_footer or block.is_page_number
    ]
    return {
        "policy": PDF_HEADER_FOOTER_POLICY_VERSION,
        "patterns": patterns,
        "marked_blocks": marked,
        "candidate_count": len(candidates),
    }


def mark_figure_regions(blocks: list[PdfLayoutBlock], body_size: float) -> dict[str, Any]:
    grouped: dict[int, list[PdfLayoutBlock]] = defaultdict(list)
    for block in blocks:
        grouped[block.page_number].append(block)

    marked_regions: list[dict[str, Any]] = []
    region_index = 0
    for page_number in sorted(grouped):
        page_blocks = sorted(grouped[page_number], key=_visual_block_key)
        caption_indices = [
            index
            for index, block in enumerate(page_blocks)
            if _caption_kind(block.text) in {"figure", "table"}
        ]
        for caption_index in caption_indices:
            caption = page_blocks[caption_index]
            kind = _caption_kind(caption.text)
            if kind not in {"figure", "table"} or caption.metadata.get("visual_region_id"):
                continue
            region_end = _visual_region_end(page_blocks, caption_index, body_size)
            if kind == "figure":
                marker_indexes = [
                    index
                    for index in range(caption_index + 1, region_end)
                    if _looks_like_figure_element(page_blocks[index], body_size)
                ]
                if not _figure_region_is_confident([page_blocks[index] for index in marker_indexes]):
                    continue
                content_type = "figure"
            else:
                marker_indexes = [
                    index
                    for index in range(caption_index + 1, region_end)
                    if _looks_like_table_region_block(page_blocks[index], body_size)
                ]
                marker_blocks = [page_blocks[index] for index in marker_indexes]
                if not _table_region_is_confident(page_blocks[caption_index + 1 : region_end], marker_blocks):
                    continue
                content_type = "table"

            last_marker_index = max(marker_indexes, default=caption_index)
            mark_end = _include_trailing_visual_note(page_blocks, last_marker_index + 1, region_end)
            region_blocks = page_blocks[caption_index:mark_end]
            if not region_blocks:
                continue

            region_id = f"pdf-visual-{region_index}"
            region_index += 1
            for block in region_blocks:
                block.content_type = content_type
                block.metadata["content_type"] = content_type
                block.metadata["visual_region_id"] = region_id
                block.metadata["visual_region_kind"] = content_type
                block.metadata["visual_region_reason"] = "caption followed by dense visual labels"

            marked_regions.append(
                {
                    "region_id": region_id,
                    "page_number": page_number,
                    "kind": content_type,
                    "caption": caption.text,
                    "block_count": len(region_blocks),
                    "marker_count": len(marker_indexes),
                    "y0_min": round(min(block.y0_rel for block in region_blocks), 4),
                    "y1_max": round(max(block.y1_rel for block in region_blocks), 4),
                }
            )

    return {
        "policy": PDF_FIGURE_POLICY_VERSION,
        "marked_regions": marked_regions,
        "marked_block_count": sum(int(region["block_count"]) for region in marked_regions),
    }


def normalize_boilerplate_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\b(?:page|p)\.?\s*\d+\b", "page #", normalized)
    normalized = re.sub(r"\b\d+\s*/\s*\d+\b", "#/#", normalized)
    normalized = re.sub(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b", "date", normalized)
    normalized = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "date", normalized)
    normalized = re.sub(r"\b\d+\b", "#", normalized)
    normalized = re.sub(r"[^\w\s#/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def classify_headings(blocks: list[PdfLayoutBlock], body_size: float) -> dict[str, Any]:
    debug: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        block.content_type = _classify_content_type(block)
        if block.content_type in {"table", "list", "caption", "footnote", "figure", "toc"}:
            continue
        previous_block = _previous_body_block(blocks, index)
        next_block = _next_body_block(blocks, index)
        level, reason = _heading_level(block, body_size, previous_block, next_block)
        if level is None:
            continue
        block.is_heading = True
        block.heading_level = level
        block.heading_reason = reason
        debug.append(
            {
                "block_id": block.block_id,
                "page_number": block.page_number,
                "text": block.text,
                "heading_level": level,
                "reason": reason,
            }
        )
    return {"policy": PDF_HEADING_POLICY_VERSION, "headings": debug, "body_font_size": body_size}


def build_logical_units(blocks: list[PdfLayoutBlock], body_size: float) -> tuple[list[LogicalUnit], dict[str, Any]]:
    units: list[LogicalUnit] = []
    heading_stack: list[tuple[int, str]] = []
    current_blocks: list[PdfLayoutBlock] = []
    current_content_type = "body"
    current_heading_path: tuple[str, ...] = ()
    debug: list[dict[str, Any]] = []

    def flush(reason: str) -> None:
        nonlocal current_blocks, current_content_type, current_heading_path
        if not current_blocks:
            return
        unit = _unit_from_blocks(current_blocks, current_content_type, current_heading_path, reason)
        if unit.text:
            units.append(unit)
        current_blocks = []
        current_content_type = "body"

    for block in blocks:
        if block.is_heading:
            flush("heading boundary")
            level = block.heading_level or 3
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, _clean_unit_text(block.text)))
            debug.append(
                {
                    "event": "heading_path",
                    "block_id": block.block_id,
                    "path": [text for _, text in heading_stack],
                }
            )
            continue

        block_type = _classify_content_type(block)
        block.content_type = block_type
        heading_path = tuple(text for _, text in heading_stack)
        if not current_blocks:
            current_blocks = [block]
            current_content_type = block_type
            current_heading_path = heading_path
            continue

        join, reasons = _should_join_logical_block(current_blocks[-1], block, current_content_type, block_type, body_size)
        if heading_path != current_heading_path:
            join = False
            reasons.append("heading path changed")
        if join:
            current_blocks.append(block)
            if block_type != "body":
                current_content_type = block_type
            if reasons:
                debug.append(
                    {
                        "event": "joined",
                        "previous_block_id": current_blocks[-2].block_id,
                        "block_id": block.block_id,
                        "reasons": reasons,
                    }
                )
            continue

        flush("; ".join(reasons) if reasons else "layout boundary")
        current_blocks = [block]
        current_content_type = block_type
        current_heading_path = heading_path

    flush("end of document")
    units = _merge_list_introductions(units)
    return units, {"unit_count": len(units), "events": debug}


def split_oversized_units(units: list[LogicalUnit], max_chars: int) -> list[LogicalUnit]:
    split_units: list[LogicalUnit] = []
    for unit in units:
        if len(unit.text) <= max_chars:
            split_units.append(unit)
            continue
        split_units.extend(_split_unit(unit, max_chars))
    return split_units


def pack_logical_units(units: list[LogicalUnit], config: PdfChunkingConfig) -> list[PdfChunkDraft]:
    if not units:
        return []

    section_lengths = _section_lengths(units)
    drafts: list[PdfChunkDraft] = []
    current = PdfChunkDraft()

    def flush(reason: str) -> None:
        nonlocal current
        if not current.units:
            return
        current.boundary_reason = reason
        drafts.append(current)
        current = PdfChunkDraft()

    for unit in units:
        if not current.units:
            current.units.append(unit)
            continue
        projected_len = len(current.text) + len(unit.text) + 2
        current_heading = current.heading_path
        strong_boundary = unit.heading_path != current_heading and not _is_weak_section_boundary(
            current_heading,
            unit.heading_path,
            section_lengths,
            config.soft_min_chars,
        )
        if strong_boundary:
            flush("strong section boundary")
        elif projected_len > config.max_chars:
            flush("maximum character budget")
        elif projected_len > config.target_chars and len(current.text) >= config.soft_min_chars:
            flush("target character budget")

        current.units.append(unit)

    flush("end of document")
    drafts = _merge_soft_min_drafts(drafts, config)
    for index, draft in enumerate(drafts):
        if index > 0 and draft.heading_path == drafts[index - 1].heading_path:
            draft.is_continuation = True
    return drafts


def drafts_to_chunks(
    drafts: list[PdfChunkDraft],
    document_metadata: dict[str, Any],
    config: PdfChunkingConfig,
    furniture_debug: dict[str, Any],
    figure_debug: dict[str, Any],
    heading_debug: dict[str, Any],
    unit_debug: dict[str, Any],
) -> list[Chunk]:
    section_counts: dict[str, int] = defaultdict(int)
    document_title = _document_title(document_metadata)
    chunks: list[Chunk] = []
    for index, draft in enumerate(drafts):
        previous = drafts[index - 1] if index else None
        if previous and previous.heading_path == draft.heading_path and config.overlap_mode != "none":
            overlap_text, overlap_spans = _structural_overlap(previous, config.overlap_mode, config.overlap_chars)
            if overlap_text and len(overlap_text) + len(draft.text) + 40 <= config.max_chars:
                draft.overlap_text = overlap_text
                draft.overlap_source_spans = overlap_spans

        heading_label = " > ".join(draft.heading_path)
        section_key = heading_label or ""
        chunk_index_within_section = section_counts[section_key]
        section_counts[section_key] += 1
        pages = draft.pages
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None
        prefix = _retrieval_prefix(document_title, heading_label)
        overlap_prefix = f"Previous context: {draft.overlap_text}\n\n" if draft.overlap_text else ""
        body_offset = len(prefix) + len(overlap_prefix)
        text = f"{prefix}{overlap_prefix}{draft.text}".strip()
        display_markdown = _chunk_display_markdown(document_title, draft)
        source_spans = _offset_source_spans(draft.source_spans, body_offset)
        metadata = dict(document_metadata)
        metadata.update(
            {
                "page_start": page_start,
                "page_end": page_end,
                "pages": list(pages),
                "heading_path": list(draft.heading_path),
                "heading_path_text": heading_label,
                "content_type": _combined_content_type(draft.content_types),
                "chunk_index_within_section": chunk_index_within_section,
                "is_continuation": draft.is_continuation,
                "contains_table": draft.contains_table,
                "contains_list": draft.contains_list,
                "contains_caption": draft.contains_caption,
                "contains_footnote": draft.contains_footnote,
                "contains_figure": draft.contains_figure,
                "source_spans": source_spans,
                "retrieval_prefix": prefix,
                "retrieval_prefix_char_count": len(prefix),
                "overlap_text": draft.overlap_text,
                "overlap_mode": config.overlap_mode,
                "overlap_source_spans": list(draft.overlap_source_spans),
                "fallback_split_rule": draft.fallback_rule,
                "pdf_chunk_debug": {
                    "boundary_reason": draft.boundary_reason,
                    "furniture_pattern_count": len(furniture_debug.get("patterns", [])),
                    "marked_furniture_count": len(furniture_debug.get("marked_blocks", [])),
                    "marked_visual_region_count": len(figure_debug.get("marked_regions", [])),
                    "marked_visual_block_count": figure_debug.get("marked_block_count", 0),
                    "heading_count": len(heading_debug.get("headings", [])),
                    "unit_count": unit_debug.get("unit_count", 0),
                },
            }
        )
        if page_start == page_end:
            metadata["page_number"] = page_start
        chunks.append(
            Chunk(
                chunk_index=index,
                text=text,
                source_type="pdf",
                page_number=page_start,
                display_markdown=display_markdown,
                metadata=metadata,
            )
        )
    return chunks


def _block_from_payload(payload: dict[str, Any], section_page_number: int | None, fallback_index: int) -> PdfLayoutBlock:
    page_number = _safe_int(payload.get("page_number"), section_page_number or 0)
    page_width = _safe_float(payload.get("page_width"), 0.0)
    page_height = _safe_float(payload.get("page_height"), 0.0)
    bbox = _float_tuple(payload.get("bbox"), 4)
    rel_bbox = _float_tuple(payload.get("rel_bbox") or payload.get("relative_bbox"), 4)
    if rel_bbox == (0.0, 0.0, 0.0, 0.0) and page_width > 0 and page_height > 0:
        rel_bbox = (
            bbox[0] / page_width,
            bbox[1] / page_height,
            bbox[2] / page_width,
            bbox[3] / page_height,
        )
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    raw_text = str(payload.get("text", ""))
    text = raw_text.strip() if str(metadata.get("content_type", "")).strip().lower() == "table" else _clean_unit_text(raw_text)
    return PdfLayoutBlock(
        block_id=str(payload.get("block_id") or payload.get("id") or f"p{page_number}-b{fallback_index}"),
        text=text,
        page_number=page_number,
        bbox=bbox,
        rel_bbox=rel_bbox,
        page_width=page_width,
        page_height=page_height,
        font_size=_safe_float(payload.get("font_size") or payload.get("size"), 0.0),
        flags=_safe_int(payload.get("flags"), 0),
        bold=bool(payload.get("bold", False)),
        italic=bool(payload.get("italic", False)),
        reading_order=_safe_int(payload.get("reading_order"), fallback_index),
        metadata=metadata,
    )


def _candidate_region(block: PdfLayoutBlock, config: PdfChunkingConfig) -> str | None:
    if not block.text:
        return None
    if _is_standalone_page_number(block.text) and _is_near_page_edge(block, config):
        return "page_number"
    if block.y0_rel <= config.header_region:
        return "header"
    if block.y1_rel >= 1.0 - config.footer_region or block.y0_rel >= 1.0 - config.footer_region:
        return "footer"
    return None


def _is_near_page_edge(block: PdfLayoutBlock, config: PdfChunkingConfig) -> bool:
    return block.y0_rel <= config.header_region * 1.2 or block.y1_rel >= 1.0 - config.footer_region * 1.2


def _is_standalone_page_number(text: str) -> bool:
    stripped = text.strip()
    return bool(
        re.fullmatch(r"(?:[-–—]\s*)?(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*[-–—])?", stripped, flags=re.IGNORECASE)
        or re.fullmatch(r"[ivxlcdm]{1,8}", stripped, flags=re.IGNORECASE)
    )


def _recurring_score(
    occurrences: list[PdfLayoutBlock],
    all_pages: list[int],
    config: PdfChunkingConfig,
) -> tuple[bool, dict[str, Any]]:
    pages = sorted({block.page_number for block in occurrences})
    if len(pages) < 2:
        return False, {"reason": "single occurrence", "pages": pages}
    if not _positions_are_consistent(occurrences):
        return False, {"reason": "inconsistent relative position", "pages": pages}
    if not _font_properties_are_consistent(occurrences):
        return False, {"reason": "inconsistent font properties", "pages": pages}

    first_page = all_pages[0] if all_pages else None
    pages_without_first = [page for page in pages if page != first_page]
    eligible_without_first = [page for page in all_pages if page != first_page]
    overall = _frequency_ok(len(pages), len(all_pages), config, allow_short_docs=True)
    without_first = _frequency_ok(len(pages_without_first), len(eligible_without_first), config, allow_short_docs=False)
    parity = _parity_frequency_ok(pages, all_pages, first_page, config)
    recurring = overall or without_first or parity
    return recurring, {
        "pages": pages,
        "occurrences": len(pages),
        "eligible_pages": len(all_pages),
        "ratio": round(len(pages) / max(1, len(all_pages)), 3),
        "without_first_ratio": round(len(pages_without_first) / max(1, len(eligible_without_first)), 3),
        "overall_threshold_met": overall,
        "without_first_threshold_met": without_first,
        "parity_threshold_met": parity,
        "position_band": _position_band(occurrences),
        "policy": PDF_HEADER_FOOTER_POLICY_VERSION,
    }


def _frequency_ok(count: int, eligible: int, config: PdfChunkingConfig, *, allow_short_docs: bool) -> bool:
    if eligible <= 1:
        return False
    if eligible <= 3:
        min_occurrences = 2
        ratio = 0.66 if allow_short_docs else 0.5
    elif eligible <= 5:
        min_occurrences = 2
        ratio = 0.4
    else:
        min_occurrences = 3
        ratio = config.min_recurring_ratio
    return count >= min_occurrences and (count / eligible) >= ratio


def _parity_frequency_ok(pages: list[int], all_pages: list[int], first_page: int | None, config: PdfChunkingConfig) -> bool:
    for parity in (0, 1):
        eligible = [page for page in all_pages if page != first_page and page % 2 == parity]
        if len(eligible) < 3:
            continue
        count = len([page for page in pages if page in eligible])
        if count >= 3 and (count / len(eligible)) >= max(0.6, config.min_recurring_ratio):
            return True
    return False


def _positions_are_consistent(blocks: list[PdfLayoutBlock]) -> bool:
    y_values = [block.y0_rel for block in blocks]
    x_values = [block.x0_rel for block in blocks]
    heights = [block.line_height_rel for block in blocks]
    return (
        _spread(y_values) <= 0.035
        and _spread(x_values) <= 0.12
        and (_spread(heights) <= 0.03 or max(heights, default=0.0) <= 0.025)
    )


def _font_properties_are_consistent(blocks: list[PdfLayoutBlock]) -> bool:
    sizes = [block.font_size for block in blocks if block.font_size > 0]
    if sizes and _spread(sizes) > max(1.5, statistics.mean(sizes) * 0.18):
        return False
    bold_ratio = sum(1 for block in blocks if block.bold) / len(blocks)
    italic_ratio = sum(1 for block in blocks if block.italic) / len(blocks)
    return bold_ratio in {0.0, 1.0} or 0.2 <= bold_ratio <= 0.8 or italic_ratio in {0.0, 1.0} or 0.2 <= italic_ratio <= 0.8


def _position_band(blocks: list[PdfLayoutBlock]) -> dict[str, float]:
    return {
        "y0_min": round(min(block.y0_rel for block in blocks), 4),
        "y0_max": round(max(block.y0_rel for block in blocks), 4),
        "x0_min": round(min(block.x0_rel for block in blocks), 4),
        "x0_max": round(max(block.x0_rel for block in blocks), 4),
    }


def _body_font_size(blocks: list[PdfLayoutBlock]) -> float:
    body_candidates = [
        round(block.font_size, 1)
        for block in blocks
        if block.font_size > 0 and len(block.text) > 20 and not _looks_like_list_or_table_noise(block.text)
    ]
    if len(body_candidates) < 3:
        body_candidates = [round(block.font_size, 1) for block in blocks if block.font_size > 0]
    if not body_candidates:
        return 10.0
    counts = Counter(body_candidates)
    return sorted(counts, key=lambda size: (-counts[size], size))[0]


def _heading_level(
    block: PdfLayoutBlock,
    body_size: float,
    previous_block: PdfLayoutBlock | None,
    next_block: PdfLayoutBlock | None,
) -> tuple[int | None, str]:
    text = _clean_unit_text(block.text)
    if not text or len(text) > 160:
        return None, ""
    if text.casefold().startswith("keywords:"):
        return None, ""
    if text.casefold().startswith("arxiv:"):
        return None, ""
    if text.endswith(":") and block.font_size < body_size * 1.18 and not block.bold:
        return None, ""
    if _looks_like_sentence(text) and block.font_size < body_size * 1.35:
        return None, ""
    appendix_match = re.match(r"^(appendix\s+[A-Z]|chapter\s+\d+)\b", text, flags=re.IGNORECASE)
    if appendix_match:
        return 1, "appendix/chapter numbering"
    numbered = re.match(r"^(\d+(?:\.\d+){0,5})\.?\s+\S", text)
    if numbered and len(text) <= 140:
        if _looks_like_year_numbering(numbered.group(1)):
            return None, ""
        depth = numbered.group(1).count(".") + 1
        return min(depth + 1, 4), "numbered heading pattern"
    if block.font_size >= body_size * 1.6 and len(text) <= 120:
        return 1, "large font relative to body"
    if block.font_size >= body_size * 1.35 and len(text) <= 130 and not _looks_like_sentence(text):
        return 2, "larger font relative to body"
    if block.bold and len(text) <= 120 and _has_heading_spacing(block, previous_block, next_block, body_size):
        return 3, "bold text with heading spacing"
    if _is_centered(block) and len(text) <= 90 and not _looks_like_sentence(text) and _has_heading_spacing(block, previous_block, next_block, body_size):
        return 3, "centered short text with heading spacing"
    return None, ""


def _classify_content_type(block: PdfLayoutBlock) -> str:
    explicit = str(block.metadata.get("content_type", "")).strip().lower()
    if explicit in {"body", "list", "table", "caption", "footnote", "figure", "toc", "references", "index", "callout"}:
        return explicit
    text = _clean_unit_text(block.text)
    if _caption_kind(text):
        return "caption"
    if block.y0_rel > 0.78 and re.match(r"^(?:\d+|[*†‡])\s+\S", text):
        return "footnote"
    if re.match(r"^(?:[\u2022\u2023\u25e6\u2043\u2219o]|[-–])\s+\S", text):
        return "list"
    if re.match(r"^(?:\d+[\.)]|[a-zA-Z][\.)])\s+\S", text):
        return "list"
    if re.search(r"\.{4,}\s*\d{1,4}$", text):
        return "toc"
    if _looks_like_table_row(text, block):
        return "table"
    return "body"


def _caption_kind(text: str) -> str | None:
    lowered = _clean_unit_text(text).casefold()
    if re.match(r"^(?:figure|fig\.)\s*\d+[.:]?\b", lowered):
        return "figure"
    if re.match(r"^table\s*\d+[.:]?\b", lowered):
        return "table"
    return None


def _visual_region_end(page_blocks: list[PdfLayoutBlock], caption_index: int, body_size: float) -> int:
    caption_block = page_blocks[caption_index]
    for index in range(caption_index + 1, len(page_blocks)):
        block = page_blocks[index]
        previous = page_blocks[index - 1]
        if _is_caption_title_continuation(caption_block, previous, block, body_size):
            continue
        if _caption_kind(block.text):
            return index
        if _is_strong_visual_region_stop(block, body_size):
            return index
    return len(page_blocks)


def _is_caption_title_continuation(
    caption_block: PdfLayoutBlock,
    previous_block: PdfLayoutBlock,
    current_block: PdfLayoutBlock,
    body_size: float,
) -> bool:
    if _caption_kind(current_block.text):
        return False
    vertical_gap = current_block.y0_rel - previous_block.y1_rel
    return (
        abs(current_block.x0_rel - caption_block.x0_rel) <= 0.08
        and vertical_gap <= 0.025
        and current_block.font_size >= body_size * 1.05
        and abs(current_block.font_size - caption_block.font_size) <= max(1.2, body_size * 0.15)
    )


def _is_strong_visual_region_stop(block: PdfLayoutBlock, body_size: float) -> bool:
    text = _clean_unit_text(block.text)
    if not text or len(text) > 180:
        return False
    if block.font_size >= body_size * 1.35 and not _looks_like_sentence(text):
        return True
    if block.bold and block.font_size >= body_size * 1.15 and _looks_like_heading_text(text):
        return True
    return False


def _looks_like_figure_element(block: PdfLayoutBlock, body_size: float) -> bool:
    text = _clean_unit_text(block.text)
    if not text or _caption_kind(text) or len(text) > 120:
        return False
    small_text = 0 < block.font_size <= body_size * 0.95
    if small_text and len(text) <= 90:
        return True
    if small_text and _has_figure_label_keyword(text):
        return True
    if block.bold and len(text) <= 90 and _is_all_capsish(text):
        return True
    return False


def _looks_like_table_region_block(block: PdfLayoutBlock, body_size: float) -> bool:
    text = _clean_unit_text(block.text)
    if not text or _caption_kind(text):
        return False
    if _looks_like_table_row(text, block):
        return True
    return 0 < block.font_size <= body_size * 1.05 and len(text) <= 120


def _figure_region_is_confident(marker_blocks: list[PdfLayoutBlock]) -> bool:
    if len(marker_blocks) < 4:
        return False
    x_spread = _spread([block.x_center_rel for block in marker_blocks])
    y_spread = _spread([block.y0_rel for block in marker_blocks])
    keyword_count = sum(1 for block in marker_blocks if _has_figure_label_keyword(block.text))
    return (x_spread >= 0.12 and y_spread >= 0.10) or (keyword_count >= 3 and y_spread >= 0.05)


def _table_region_is_confident(candidate_blocks: list[PdfLayoutBlock], marker_blocks: list[PdfLayoutBlock]) -> bool:
    explicit_rows = sum(1 for block in candidate_blocks if _looks_like_table_row(_clean_unit_text(block.text), block))
    if explicit_rows >= 2:
        return True
    row_groups = _same_relative_baseline_groups(marker_blocks)
    multi_cell_rows = [
        group
        for group in row_groups
        if len(group) >= 2 and _spread([block.x_center_rel for block in group]) >= 0.12
    ]
    return len(multi_cell_rows) >= 2


def _include_trailing_visual_note(page_blocks: list[PdfLayoutBlock], mark_end: int, region_end: int) -> int:
    if mark_end >= region_end:
        return mark_end
    index = mark_end
    while index < region_end:
        text = _clean_unit_text(page_blocks[index].text)
        if index == mark_end and not _is_visual_note_text(text):
            break
        if index > mark_end and page_blocks[index].y0_rel - page_blocks[index - 1].y1_rel > 0.035:
            break
        index += 1
    return index


def _is_visual_note_text(text: str) -> bool:
    return bool(re.match(r"^(?:note|notes?|source)[:.]\s+", text, flags=re.IGNORECASE))


def _has_figure_label_keyword(text: str) -> bool:
    lowered = _clean_unit_text(text).casefold()
    return bool(
        re.search(
            r"\b(?:records?|documents?|screened|included|excluded|identification|screening|eligibility|consultation|search|total|full-text|workflow|process|stage)\b",
            lowered,
        )
        or re.search(r"\bn\s*=\s*[\d,]+\b", lowered)
    )


def _is_all_capsish(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 4:
        return False
    uppercase = sum(1 for character in letters if character.isupper())
    return uppercase / len(letters) >= 0.8


def _same_relative_baseline_groups(blocks: list[PdfLayoutBlock]) -> list[list[PdfLayoutBlock]]:
    groups: list[list[PdfLayoutBlock]] = []
    for block in sorted(blocks, key=_visual_block_key):
        if not groups or abs(groups[-1][0].y0_rel - block.y0_rel) > max(0.006, block.line_height_rel * 0.35):
            groups.append([block])
        else:
            groups[-1].append(block)
    return groups


def _should_join_logical_block(
    previous: PdfLayoutBlock,
    current: PdfLayoutBlock,
    current_unit_type: str,
    current_block_type: str,
    body_size: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if previous.page_number == current.page_number:
        if current_unit_type == current_block_type == "table":
            return True, ["adjacent table row"]
        if current_unit_type == current_block_type == "figure":
            return True, ["adjacent figure line"]
        if current_unit_type == current_block_type == "list":
            return True, ["adjacent list item"]
        if current_unit_type == "body" and current_block_type == "list" and previous.text.rstrip().endswith(":"):
            return True, ["list introduction"]
        if current_unit_type != current_block_type:
            return False, ["content type changed"]
        if current_block_type in {"caption", "footnote", "toc"}:
            return True, [f"adjacent {current_block_type} line"]
        if not _font_compatible(previous, current, body_size):
            return False, ["font changed"]
        if not _same_column(previous, current):
            return False, ["column changed"]
        vertical_gap = current.y0_rel - previous.y1_rel
        gap_limit = max(0.008, (max(previous.font_size, current.font_size, body_size) / max(previous.page_height, 1.0)) * 1.7)
        if vertical_gap <= gap_limit:
            reasons.append("small same-page vertical gap")
            return True, reasons
        return False, ["same-page paragraph gap"]

    if current.page_number != previous.page_number + 1:
        return False, ["non-adjacent pages"]
    if current_unit_type not in {"body", "list", "table"} or current_block_type != current_unit_type:
        return False, ["cross-page content type changed"]
    if current_unit_type == "table":
        return True, ["table continues on next page"]
    if current_unit_type == "list":
        return True, ["list continues on next page"]
    if not _font_compatible(previous, current, body_size):
        return False, ["cross-page font changed"]
    if not _same_column(previous, current):
        return False, ["cross-page column changed"]
    if previous.y1_rel < 0.60 or current.y0_rel > 0.35:
        return False, ["not near opposing body page edges"]
    if _looks_like_heading_text(current.text):
        return False, ["next block looks like heading"]
    if _looks_like_standalone_unit(previous.text):
        return False, ["previous block looks complete"]
    if previous.text.rstrip().endswith("-") and re.match(r"^[a-z]", current.text.strip()):
        return True, ["hyphenated word continues across page"]
    if not previous.text.rstrip().endswith((".", "?", "!", ":", ";")):
        return True, ["previous block lacks terminal punctuation across page"]
    if re.match(r"^[a-z,;)]", current.text.strip()):
        return True, ["next block begins as prose continuation"]
    return False, ["cross-page punctuation suggests new paragraph"]


def _unit_from_blocks(
    blocks: list[PdfLayoutBlock],
    content_type: str,
    heading_path: tuple[str, ...],
    boundary_reason: str,
) -> LogicalUnit:
    text, source_spans, join_debug = _join_block_texts(blocks, content_type)
    pages = tuple(sorted({block.page_number for block in blocks}))
    content_type = _low_value_content_type(content_type, heading_path, text)
    has_caption = content_type == "caption" or any(_caption_kind(block.text) for block in blocks)
    return LogicalUnit(
        text=text,
        display_text=_display_text(text, content_type),
        pages=pages,
        heading_path=heading_path,
        content_type=content_type,
        source_spans=tuple(source_spans),
        debug=tuple([f"boundary: {boundary_reason}", *join_debug]),
        contains_table=content_type == "table",
        contains_list=content_type == "list",
        contains_caption=has_caption,
        contains_footnote=content_type == "footnote",
        contains_figure=content_type == "figure",
        is_structurally_meaningful=_is_meaningful_standalone(content_type, heading_path, text),
    )


def _join_block_texts(blocks: list[PdfLayoutBlock], content_type: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    output = ""
    spans: list[dict[str, Any]] = []
    debug: list[str] = []
    for index, block in enumerate(blocks):
        text = block.text.strip() if content_type == "table" else _clean_unit_text(block.text)
        if not text:
            continue
        if index == 0 or not output:
            delimiter = ""
        elif output.endswith("-") and re.match(r"^[a-z]", text):
            output = output[:-1]
            delimiter = ""
            debug.append(f"dehyphenated join before {block.block_id}")
        elif content_type in {"list", "table", "caption", "footnote", "figure", "toc"}:
            delimiter = "\n"
        else:
            delimiter = " "
        if delimiter:
            output += delimiter
        start = len(output)
        output += text
        spans.append(
            {
                "page_number": block.page_number,
                "block_id": block.block_id,
                "bbox": list(block.bbox),
                "rel_bbox": list(block.rel_bbox),
                "char_start": start,
                "char_end": len(output),
            }
        )
    return output.strip(), spans, debug


def _merge_list_introductions(units: list[LogicalUnit]) -> list[LogicalUnit]:
    merged: list[LogicalUnit] = []
    index = 0
    while index < len(units):
        current = units[index]
        if (
            current.content_type == "body"
            and current.text.rstrip().endswith(":")
            and index + 1 < len(units)
            and units[index + 1].content_type == "list"
            and current.heading_path == units[index + 1].heading_path
        ):
            next_unit = units[index + 1]
            joined_text = f"{current.text}\n{next_unit.text}"
            merged.append(
                LogicalUnit(
                    text=joined_text,
                    display_text=_display_text(joined_text, "list"),
                    pages=tuple(sorted({*current.pages, *next_unit.pages})),
                    heading_path=current.heading_path,
                    content_type="list",
                    source_spans=(*current.source_spans, *_offset_source_spans(next_unit.source_spans, len(current.text) + 1)),
                    debug=(*current.debug, *next_unit.debug, "merged list introduction"),
                    contains_table=False,
                    contains_list=True,
                    contains_caption=current.contains_caption or next_unit.contains_caption,
                    contains_footnote=current.contains_footnote or next_unit.contains_footnote,
                    contains_figure=current.contains_figure or next_unit.contains_figure,
                    is_structurally_meaningful=True,
                    is_continuation=next_unit.is_continuation,
                    fallback_rule=next_unit.fallback_rule,
                )
            )
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _split_unit(unit: LogicalUnit, max_chars: int) -> list[LogicalUnit]:
    if unit.content_type in {"table", "list", "figure"}:
        return _split_unit_by_lines(unit, max_chars)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", unit.text) if part.strip()]
    if len(paragraphs) > 1:
        return _split_by_text_parts(unit, paragraphs, "\n\n", "paragraph", max_chars)
    sentences = _split_sentences(unit.text)
    if len(sentences) > 1:
        return _split_by_text_parts(unit, sentences, " ", "sentence", max_chars)
    return _split_by_character_windows(unit, max_chars)


def _split_unit_by_lines(unit: LogicalUnit, max_chars: int) -> list[LogicalUnit]:
    lines = [line for line in unit.text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return _split_by_character_windows(unit, max_chars)
    header = lines[0] if unit.content_type == "table" else ""
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        projected = len("\n".join(current + [line]))
        if current and projected > max_chars:
            chunks.append("\n".join(current))
            current = [header, line] if header and line != header else [line]
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return _split_by_text_parts(unit, chunks, "\n", f"{unit.content_type}-row", max_chars)


def _split_by_text_parts(
    unit: LogicalUnit,
    parts: list[str],
    delimiter: str,
    rule: str,
    max_chars: int,
) -> list[LogicalUnit]:
    output: list[LogicalUnit] = []
    current: list[str] = []
    current_len = 0
    budget = max(max_chars, 1)
    for part in parts:
        if len(part) > budget:
            if current:
                output.append(_unit_slice(unit, delimiter.join(current), rule, len(output) > 0))
                current = []
                current_len = 0
            output.extend(_split_by_character_windows(_unit_slice(unit, part, rule, len(output) > 0), budget))
            continue
        projected = current_len + len(part) + (len(delimiter) if current else 0)
        if current and projected > budget:
            output.append(_unit_slice(unit, delimiter.join(current), rule, len(output) > 0))
            current = []
            current_len = 0
        current.append(part)
        current_len += len(part) + (len(delimiter) if current_len else 0)
    if current:
        output.append(_unit_slice(unit, delimiter.join(current), rule, len(output) > 0))
    return output


def _split_by_character_windows(unit: LogicalUnit, max_chars: int) -> list[LogicalUnit]:
    output: list[LogicalUnit] = []
    start = 0
    while start < len(unit.text):
        end = min(start + max_chars, len(unit.text))
        output.append(_unit_slice(unit, unit.text[start:end].strip(), "character", bool(output)))
        if end == len(unit.text):
            break
        start = max(end, start + 1)
    return output


def _unit_slice(unit: LogicalUnit, text: str, rule: str, continuation: bool) -> LogicalUnit:
    spans = _spans_for_text_slice(unit, text)
    pages = tuple(sorted({int(span["page_number"]) for span in spans if "page_number" in span})) or unit.pages
    return LogicalUnit(
        text=text,
        display_text=_display_text(text, unit.content_type),
        pages=pages,
        heading_path=unit.heading_path,
        content_type=unit.content_type,
        source_spans=tuple(spans),
        debug=(*unit.debug, f"split by {rule}"),
        contains_table=unit.contains_table,
        contains_list=unit.contains_list,
        contains_caption=unit.contains_caption,
        contains_footnote=unit.contains_footnote,
        contains_figure=unit.contains_figure,
        is_structurally_meaningful=unit.is_structurally_meaningful,
        is_continuation=continuation,
        fallback_rule=rule,
    )


def _section_lengths(units: list[LogicalUnit]) -> dict[tuple[str, ...], int]:
    lengths: dict[tuple[str, ...], int] = defaultdict(int)
    for unit in units:
        lengths[unit.heading_path] += len(unit.text)
    return lengths


def _is_weak_section_boundary(
    current_heading: tuple[str, ...],
    next_heading: tuple[str, ...],
    section_lengths: dict[tuple[str, ...], int],
    soft_min_chars: int,
) -> bool:
    if not current_heading or not next_heading:
        return False
    threshold = max(300, soft_min_chars // 2)
    return section_lengths[current_heading] < threshold or section_lengths[next_heading] < threshold


def _merge_soft_min_drafts(drafts: list[PdfChunkDraft], config: PdfChunkingConfig) -> list[PdfChunkDraft]:
    if not drafts or config.soft_min_chars <= 0:
        return drafts
    merged: list[PdfChunkDraft] = []
    index = 0
    while index < len(drafts):
        current = drafts[index]
        if (
            len(current.text) < config.soft_min_chars
            and index + 1 < len(drafts)
            and current.heading_path == drafts[index + 1].heading_path
            and len(current.text) + len(drafts[index + 1].text) + 2 <= config.max_chars
            and not any(unit.is_structurally_meaningful for unit in current.units)
        ):
            combined = PdfChunkDraft(
                units=[*current.units, *drafts[index + 1].units],
                boundary_reason="soft minimum merge",
                is_continuation=current.is_continuation,
            )
            merged.append(combined)
            index += 2
            continue
        if merged:
            previous = merged[-1]
            if (
                len(current.text) < config.soft_min_chars
                and current.heading_path == previous.heading_path
                and len(current.text) + len(previous.text) + 2 <= config.max_chars
                and not any(unit.is_structurally_meaningful for unit in current.units)
            ):
                previous.units.extend(current.units)
                previous.boundary_reason = "soft minimum merge"
                index += 1
                continue
        merged.append(current)
        index += 1
    return merged


def _structural_overlap(
    previous: PdfChunkDraft,
    mode: str,
    overlap_chars: int,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    text = previous.text
    if not text or overlap_chars <= 0:
        return "", ()
    if mode == "paragraph":
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        candidate = paragraphs[-1] if paragraphs else ""
    else:
        sentences = _split_sentences(text)
        candidate = sentences[-1] if sentences else ""
    if not candidate:
        return "", ()
    if len(candidate) > overlap_chars * 2:
        return "", ()
    spans = _spans_for_text_slice(
        LogicalUnit(
            text=previous.text,
            display_text=previous.display_text,
            pages=previous.pages,
            heading_path=previous.heading_path,
            content_type=_combined_content_type(previous.content_types),
            source_spans=previous.source_spans,
        ),
        candidate,
    )
    return candidate, tuple(spans)


def _retrieval_prefix(document_title: str, heading_label: str) -> str:
    lines = []
    if document_title:
        lines.append(f"Document: {document_title}")
    if heading_label:
        lines.append(f"Section: {heading_label}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def _chunk_display_markdown(document_title: str, draft: PdfChunkDraft) -> str | None:
    parts: list[str] = []
    if document_title:
        parts.append(f"# {_escape_markdown(document_title)}")
    for index, heading in enumerate(draft.heading_path):
        if document_title and index == 0 and heading.casefold() == document_title.casefold():
            continue
        level = min(2 + index, 6)
        parts.append(f"{'#' * level} {_escape_markdown(heading)}")
    if draft.overlap_text:
        parts.append(_display_overlap_markdown(draft.overlap_text))
    body = draft.display_text.strip()
    if body:
        parts.append(body)
    markdown = "\n\n".join(part for part in parts if part).strip()
    return markdown or None


def _display_overlap_markdown(text: str) -> str:
    escaped = _escape_markdown(_clean_unit_text(text))
    return f"> Previous context: {escaped}"


def _document_title(document_metadata: dict[str, Any]) -> str:
    for key in ("title", "document_title"):
        value = str(document_metadata.get(key, "")).strip()
        if value:
            return value
    path = str(document_metadata.get("path", "")).strip()
    if path:
        return re.sub(r"\.[^.\\/]+$", "", path.replace("\\", "/").rsplit("/", 1)[-1])
    return ""


def _offset_source_spans(spans: Iterable[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for span in spans:
        item = dict(span)
        if "char_start" in item:
            item["char_start"] = int(item["char_start"]) + offset
        if "char_end" in item:
            item["char_end"] = int(item["char_end"]) + offset
        adjusted.append(item)
    return adjusted


def _combined_content_type(content_types: Iterable[str]) -> str:
    unique = [content_type for content_type in dict.fromkeys(content_types) if content_type]
    if not unique:
        return "body"
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _display_text(text: str, content_type: str) -> str:
    if content_type == "list":
        return _list_markdown(text)
    if content_type == "table":
        return _fenced_text_markdown(text)
    if content_type == "figure":
        return f"**Figure**\n\n{_fenced_text_markdown(text)}"
    if content_type == "caption":
        return f"*{_escape_markdown(_clean_unit_text(text))}*"
    if content_type == "footnote":
        return "\n".join(f"> {_escape_markdown(line)}" for line in text.splitlines() if line.strip())
    return "\n\n".join(_escape_markdown(part) for part in _display_paragraphs(text))


def _display_paragraphs(text: str) -> list[str]:
    if "\n\n" in text:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return [_clean_unit_text(text)] if text.strip() else []


def _list_markdown(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _clean_unit_text(raw_line)
        if not line:
            continue
        bullet_match = re.match(r"^[\u2022\u2023\u25e6\u2043\u2219o]\s+(.+)$", line)
        dash_match = re.match(r"^[-–]\s+(.+)$", line)
        numbered_match = re.match(r"^(\d+)[\.)]\s+(.+)$", line)
        alpha_match = re.match(r"^([A-Za-z])[\.)]\s+(.+)$", line)
        if bullet_match:
            _append_list_separator(lines)
            lines.append(f"- {_escape_markdown(bullet_match.group(1).strip())}")
        elif dash_match:
            _append_list_separator(lines)
            lines.append(f"- {_escape_markdown(dash_match.group(1).strip())}")
        elif numbered_match:
            _append_list_separator(lines)
            lines.append(f"{numbered_match.group(1)}. {_escape_markdown(numbered_match.group(2).strip())}")
        elif alpha_match:
            _append_list_separator(lines)
            lines.append(f"- {_escape_markdown(alpha_match.group(1))}. {_escape_markdown(alpha_match.group(2).strip())}")
        elif lines:
            lines[-1] = f"{lines[-1]} {_escape_markdown(line)}"
        else:
            lines.append(_escape_markdown(line))
    return "\n".join(lines)


def _append_list_separator(lines: list[str]) -> None:
    if lines and lines[-1] and not _is_markdown_list_line(lines[-1]):
        lines.append("")


def _is_markdown_list_line(line: str) -> bool:
    return bool(line.startswith("- ") or re.match(r"^\d+\.\s+", line))


def _fenced_text_markdown(text: str) -> str:
    body = text.replace("```", "` ` `").strip()
    return f"```text\n{body}\n```"


def _escape_markdown(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("`", "\\`")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _spans_for_text_slice(unit: LogicalUnit, text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    start = unit.text.find(text)
    if start < 0:
        return [dict(span) for span in unit.source_spans]
    end = start + len(text)
    spans: list[dict[str, Any]] = []
    for span in unit.source_spans:
        span_start = int(span.get("char_start", 0))
        span_end = int(span.get("char_end", span_start))
        if span_end <= start or span_start >= end:
            continue
        adjusted = dict(span)
        adjusted["char_start"] = max(span_start, start) - start
        adjusted["char_end"] = min(span_end, end) - start
        spans.append(adjusted)
    return spans or [dict(span) for span in unit.source_spans]


def _low_value_content_type(content_type: str, heading_path: tuple[str, ...], text: str) -> str:
    heading_text = " ".join(heading_path).casefold()
    lowered = text.casefold()
    if "table of contents" in heading_text or re.search(r"\.{4,}\s*\d{1,4}$", lowered):
        return "toc"
    if any(label in heading_text for label in ("references", "bibliography")):
        return "references"
    if heading_path and heading_path[-1].casefold() == "index":
        return "index"
    return content_type


def _is_meaningful_standalone(content_type: str, heading_path: tuple[str, ...], text: str) -> bool:
    heading = heading_path[-1].casefold() if heading_path else ""
    if content_type in {"callout", "caption", "figure", "table"}:
        return True
    if heading in {"abstract", "executive summary", "summary", "conclusion", "warning"}:
        return True
    if re.match(r"^(?:requirement|shall|must)\b", text.strip(), flags=re.IGNORECASE):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text) if part.strip()]
    return sentences or [text.strip()]


def _font_compatible(previous: PdfLayoutBlock, current: PdfLayoutBlock, body_size: float) -> bool:
    if previous.font_size <= 0 or current.font_size <= 0:
        return True
    return abs(previous.font_size - current.font_size) <= max(1.0, body_size * 0.18)


def _same_column(previous: PdfLayoutBlock, current: PdfLayoutBlock) -> bool:
    return abs(previous.x0_rel - current.x0_rel) <= 0.06 and abs(previous.x1_rel - current.x1_rel) <= 0.22


def _has_heading_spacing(
    block: PdfLayoutBlock,
    previous_block: PdfLayoutBlock | None,
    next_block: PdfLayoutBlock | None,
    body_size: float,
) -> bool:
    before = 0.05 if previous_block is None else block.y0_rel - previous_block.y1_rel
    after = 0.05 if next_block is None else next_block.y0_rel - block.y1_rel
    line_height = max(body_size / max(block.page_height, 1.0), block.line_height_rel, 0.01)
    return before >= line_height * 0.9 and after >= line_height * 0.9


def _previous_body_block(blocks: list[PdfLayoutBlock], index: int) -> PdfLayoutBlock | None:
    for candidate in reversed(blocks[:index]):
        if candidate.page_number == blocks[index].page_number:
            return candidate
        if candidate.page_number < blocks[index].page_number:
            return candidate
    return None


def _next_body_block(blocks: list[PdfLayoutBlock], index: int) -> PdfLayoutBlock | None:
    for candidate in blocks[index + 1 :]:
        if candidate.page_number == blocks[index].page_number:
            return candidate
        if candidate.page_number > blocks[index].page_number:
            return candidate
    return None


def _looks_like_sentence(text: str) -> bool:
    return text.endswith((".", "?", "!", ";", ":")) and len(text.split()) > 3


def _looks_like_heading_text(text: str) -> bool:
    stripped = _clean_unit_text(text)
    return bool(
        len(stripped) <= 100
        and not _looks_like_sentence(stripped)
        and (
            re.match(r"^(?:\d+(?:\.\d+)*|appendix\s+[A-Z]|chapter\s+\d+)\b", stripped, flags=re.IGNORECASE)
            or stripped[:1].isupper()
        )
    )


def _looks_like_standalone_unit(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80 and re.match(r"^(?:figure|fig\.|table)\s+\d+", stripped, flags=re.IGNORECASE):
        return True
    if stripped.endswith((".", "?", "!")) and len(stripped.split()) <= 12:
        return True
    return False


def _looks_like_table_row(text: str, block: PdfLayoutBlock) -> bool:
    if "\t" in text or "|" in text:
        return True
    if re.search(r"\S\s{3,}\S", text):
        return True
    if block.metadata.get("span_count", 1) and int(block.metadata.get("span_count", 1)) >= 4 and len(text.split()) >= 4:
        return True
    return False


def _looks_like_list_or_table_noise(text: str) -> bool:
    return bool(re.match(r"^(?:[\u2022\u2023\u25e6\u2043\u2219o]|[-–]|\d+[\.)])\s+", text) or "\t" in text or "|" in text)


def _looks_like_year_numbering(value: str) -> bool:
    if "." in value or len(value) != 4:
        return False
    try:
        year = int(value)
    except ValueError:
        return False
    return 1800 <= year <= 2099


def _is_centered(block: PdfLayoutBlock) -> bool:
    return abs(block.x_center_rel - 0.5) <= 0.08


def _clean_unit_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _spread(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_tuple(value: Any, length: int) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        values = [_safe_float(item, 0.0) for item in value[:length]]
    else:
        values = []
    values.extend([0.0] * (length - len(values)))
    return tuple(values[:length])
