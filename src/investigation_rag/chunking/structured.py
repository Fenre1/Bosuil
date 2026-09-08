from __future__ import annotations

from dataclasses import replace
from html import escape
from typing import Any

from investigation_rag.chunking.base import Chunk
from investigation_rag.extractors.base import ExtractedDocument
from investigation_rag.extractors.structure import SourceBlock, blocks_html, end_line_offset, split_ranges, text_blocks


SOURCE_REFERENCE_VERSION = 1


def chunk_structured_document(document: ExtractedDocument, target: int, maximum: int) -> list[Chunk]:
    output: list[Chunk] = []
    for section_index, section in enumerate(document.sections):
        blocks = section.blocks or text_blocks(section.text, f"s{section_index}")
        heading_path: list[str] = []
        pieces: list[tuple[SourceBlock, int, int]] = []
        current_path: list[str] = []
        previous_key: tuple[Any, ...] | None = None

        def flush() -> None:
            nonlocal pieces
            if not pieces:
                return
            text = ""
            spans = []
            for block, start, end in pieces:
                if text:
                    text += "\n\n"
                spans.append({"block_id": block.block_id, "block_start": start, "block_end": end,
                              "char_start": len(text), "char_end": len(text) + end - start,
                              "section_index": section_index, "location": block.location})
                text += block.text[start:end]
            unique_blocks = list({block.block_id: block for block, _, _ in pieces}.values())
            metadata = {**document.metadata, **section.metadata,
                        "source_reference_version": SOURCE_REFERENCE_VERSION,
                        "source_spans": spans, "heading_path": list(current_path),
                        "heading_path_text": " > ".join(current_path),
                        "contains_table": any(block.kind == "table_row" for block in unique_blocks),
                        "contains_list": any(block.kind == "list_item" for block in unique_blocks),
                        "display_html": blocks_html(unique_blocks, spans, excerpt=True),
                        "_source_blocks": unique_blocks,
                        "source_section": section_index,
                        "is_continuation": pieces[0][1] > 0,
                        "quote_depth": pieces[0][0].location.get("quote_depth", 0)}
            first_block = pieces[0][0]
            if metadata["contains_table"]:
                headers = [block.text for block in blocks if block.kind == "table_row"
                           and block.location.get("table") == first_block.location.get("table")
                           and block.location.get("is_header") and block.block_id not in {b.block_id for b in unique_blocks}]
                metadata["table_header_context"] = "\n".join(headers)
            if document.source_type == "txt":
                first_span, last_span = spans[0], spans[-1]
                metadata["line_start"] = first_block.location.get("line_start", 1) + first_block.text.count("\n", 0, first_span["block_start"])
                last_block = pieces[-1][0]
                metadata["line_end"] = last_block.location.get("line_start", 1) + end_line_offset(last_block.text, last_span["block_end"])
            output.append(Chunk(len(output), text, section.source_type, section.page_number, metadata=metadata))
            pieces = []

        for block in blocks:
            if not block.text.strip():
                continue
            if block.kind == "heading":
                flush()
                level = max(1, block.heading_level)
                heading_path = heading_path[:level - 1] + [block.text.strip()]
            key = (block.kind == "table_row", block.location.get("table") if block.kind == "table_row" else None,
                   block.location.get("quote_depth", 0), block.kind == "message_header")
            if previous_key is not None and key != previous_key:
                flush()
            previous_key = key
            # Preformatted text also uses natural boundaries, retaining every space.
            for start, end in split_ranges(block.text, min(maximum, target) if block.kind == "pre" else maximum):
                projected = sum(b - a for _, a, b in pieces) + max(0, len(pieces)) * 2 + end - start
                heading_only = pieces and all(item.kind == "heading" for item, _, _ in pieces)
                if pieces and (projected > maximum or (projected > target and not heading_only)):
                    flush()
                current_path = list(heading_path)
                pieces.append((block, start, end))
        flush()
    return output


def slice_spans(spans: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    selected = []
    for span in spans:
        a, b = int(span.get("char_start", 0)), int(span.get("char_end", 0))
        if b <= start or a >= end:
            continue
        item = dict(span)
        item["char_start"], item["char_end"] = max(a, start) - start, min(b, end) - start
        if "block_start" in item:
            item["block_start"] = int(span["block_start"]) + max(0, start - a)
            item["block_end"] = int(span["block_start"]) + min(b, end) - a
        if start > a or end < b:
            item["partial"] = True
        selected.append(item)
    return selected


def finalize_chunks(chunks: list[Chunk], maximum: int, max_tokens: int, overlap_chars: int) -> list[Chunk]:
    """Budget the entire embedding input and retain mappings when further splitting."""
    from investigation_rag.ingest.searchable_text import embedding_context, estimate_embedding_tokens
    result = []
    previous: Chunk | None = None
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        source_blocks = metadata.pop("_source_blocks", None)
        chain = metadata.get("attachment_chain", [])
        metadata["source_document_id"] = "/".join(str(part["index"]) for part in chain) or "root"
        metadata["source_reference_version"] = SOURCE_REFERENCE_VERSION
        if (previous and chunk.source_type != "pdf" and not metadata.get("overlap_text")
                and overlap_chars and not metadata.get("contains_table")):
            pm = previous.metadata
            if (pm.get("source_document_id") == metadata["source_document_id"]
                    and pm.get("heading_path_text") == metadata.get("heading_path_text")
                    and pm.get("quote_depth") == metadata.get("quote_depth")
                    and not pm.get("contains_table")):
                import re
                sentences = re.split(r"(?<=[.!?])\s+", previous.text.strip())
                if sentences and len(sentences[-1]) <= overlap_chars and previous.text.rstrip().endswith((".", "!", "?")):
                    metadata["overlap_text"] = sentences[-1]
        context = embedding_context(metadata)
        # Context is auxiliary. Keep it bounded, leaving at least 2/3 for evidence.
        context_limit = min(1200, maximum // 3, max_tokens)
        if len(context.encode("utf-8")) > context_limit:
            metadata["embedding_context_shortened"] = True
            context = context.encode("utf-8")[:context_limit].decode("utf-8", errors="ignore").rsplit(" ", 1)[0]
        metadata["embedding_context"] = context
        available = max(1, maximum - len(context) - (2 if context else 0))
        ranges = split_ranges(chunk.text, available)
        bounded = []
        for start, end in ranges:
            pending = [(start, end)]
            while pending:
                a, b = pending.pop(0)
                value = context + ("\n\n" if context else "") + chunk.text[a:b]
                if estimate_embedding_tokens(value) > max_tokens and b - a > 1:
                    smaller = split_ranges(chunk.text[a:b], max(1, (b - a) // 2))
                    pending[0:0] = [(a + x, a + y) for x, y in smaller]
                else:
                    bounded.append((a, b))
        for start, end in bounded:
            cm = dict(metadata)
            text = chunk.text[start:end]
            cm["source_spans"] = slice_spans(metadata.get("source_spans", []), start, end)
            if len(bounded) > 1:
                cm["display_html"] = (blocks_html(source_blocks, cm["source_spans"], excerpt=True)
                                      if source_blocks else "<pre>" + escape(text) + "</pre>")
                cm["is_continuation"] = start > 0 or metadata.get("is_continuation", False)
            pages = sorted({span["page_number"] for span in cm["source_spans"] if span.get("page_number")})
            if pages:
                cm.update(pages=pages, page_start=pages[0], page_end=pages[-1])
            if chunk.source_type == "txt":
                cm["line_start"] = metadata.get("line_start", 1) + chunk.text.count("\n", 0, start)
                cm["line_end"] = metadata.get("line_start", 1) + end_line_offset(chunk.text, end)
            cm["embedding_token_estimate"] = estimate_embedding_tokens(context + ("\n\n" if context else "") + text)
            result.append(replace(chunk, chunk_index=len(result), text=text, metadata=cm,
                                  page_number=pages[0] if pages else chunk.page_number,
                                  display_markdown=chunk.display_markdown if len(bounded) == 1 else None))
        previous = result[-1] if result else None
    return result
