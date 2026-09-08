from __future__ import annotations

from html.parser import HTMLParser
import tempfile
import re
from pathlib import Path
from typing import Any

from investigation_rag.extractors.base import BaseExtractor, ExtractedDocument, ExtractedSection
from investigation_rag.extractors.structure import SourceBlock, html_blocks, text_blocks


class MsgExtractor(BaseExtractor):
    source_type = "msg"

    def __init__(self, depth: int = 0, max_depth: int = 3):
        self.depth = depth
        self.max_depth = max_depth

    def extract(self, path: Path) -> ExtractedDocument:
        try:
            import extract_msg
        except ImportError as exc:
            raise RuntimeError("extract-msg is required to extract .msg files") from exc

        msg = extract_msg.Message(str(path))
        try:
            metadata = {
                "path": str(path),
                "subject": _safe_str(getattr(msg, "subject", "")),
                "sender": _safe_str(getattr(msg, "sender", "")),
                "recipients": _safe_str(getattr(msg, "to", "")),
                "cc": _safe_str(getattr(msg, "cc", "")),
                "date": _safe_str(getattr(msg, "date", "")),
            }
            header = [
                f"Subject: {metadata['subject']}",
                f"From: {metadata['sender']}",
                f"To: {metadata['recipients']}",
                f"CC: {metadata['cc']}",
                f"Date: {metadata['date']}",
            ]
            body = _safe_str(getattr(msg, "body", ""))
            html_body = _safe_html_body(msg)
            display_body = html_to_markdown(html_body) if html_body else body
            searchable_body = body or markdown_to_plain_text(display_body)
            body_blocks = html_blocks(html_body) if html_body else []
            if not body_blocks:
                body_blocks = text_blocks(searchable_body, "body", kind="pre")
            source_blocks = [SourceBlock("message-header", "\n".join(header), "message_header"), *body_blocks]
            display_markdown = "\n".join(_message_header_markdown(metadata))
            if display_body:
                display_markdown = f"{display_markdown}\n\n{display_body.strip()}"
            section = ExtractedSection(
                text="\n\n".join(block.text for block in source_blocks),
                source_type=self.source_type,
                display_markdown=display_markdown,
                metadata=metadata,
                blocks=source_blocks,
            )
            attachments = self._extract_attachments(msg, metadata)
            return ExtractedDocument(
                source_type=self.source_type,
                sections=[section],
                metadata=metadata,
                attachments=attachments,
            )
        finally:
            close = getattr(msg, "close", None)
            if callable(close):
                close()

    def _extract_attachments(self, msg: Any, parent_metadata: dict[str, Any]) -> list[ExtractedDocument]:
        if self.depth >= self.max_depth:
            return []
        attachments: list[ExtractedDocument] = []
        for attachment_index, attachment in enumerate(getattr(msg, "attachments", []) or []):
            name = (
                _safe_str(getattr(attachment, "longFilename", ""))
                or _safe_str(getattr(attachment, "shortFilename", ""))
                or "attachment"
            )
            suffix = Path(name).suffix.lower()
            if suffix not in {".txt", ".docx", ".pdf", ".msg"}:
                continue
            data = getattr(attachment, "data", None)
            if not data:
                continue
            try:
                if not isinstance(data, bytes):
                    data = data.exportBytes()
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / Path(name).name
                    temp_path.write_bytes(data)
                    document = self._extract_supported_attachment(temp_path, suffix)
                    document.metadata.update(
                        {
                            "attachment_name": name,
                            "parent_email_subject": parent_metadata.get("subject", ""),
                            "parent_email_sender": parent_metadata.get("sender", ""),
                        }
                    )
                    _set_attachment_chain(document, {"index": attachment_index, "name": name})
                    for section in document.sections:
                        section.metadata.update(document.metadata)
                    attachments.append(document)
            except Exception as exc:
                failure_text = f"Attachment extraction failed for {name}: {exc}"
                attachments.append(
                    ExtractedDocument(
                        source_type="attachment_error",
                        sections=[
                            ExtractedSection(
                                text=failure_text,
                                source_type="attachment_error",
                                metadata={"attachment_name": name},
                            )
                        ],
                        metadata={"attachment_name": name, "error": str(exc)},
                    )
                )
        return attachments

    def _extract_supported_attachment(self, temp_path: Path, suffix: str) -> ExtractedDocument:
        if suffix == ".txt":
            from investigation_rag.extractors.txt_extractor import TxtExtractor

            return TxtExtractor().extract(temp_path)
        if suffix == ".docx":
            from investigation_rag.extractors.docx_extractor import DocxExtractor

            return DocxExtractor().extract(temp_path)
        if suffix == ".pdf":
            from investigation_rag.extractors.pdf_extractor import PdfExtractor

            return PdfExtractor().extract(temp_path)
        if suffix == ".msg":
            return MsgExtractor(depth=self.depth + 1, max_depth=self.max_depth).extract(temp_path)
        raise ValueError(f"Unsupported attachment suffix: {suffix}")


def _set_attachment_chain(document: ExtractedDocument, parent: dict[str, Any]) -> None:
    chain = [parent, *document.metadata.get("attachment_chain", [])]
    document.metadata["attachment_chain"] = chain
    for section in document.sections:
        section.metadata["attachment_chain"] = chain
    for child in document.attachments:
        _set_attachment_chain(child, parent)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_html_body(msg: Any) -> str:
    for attr in ("htmlBody", "html_body", "html"):
        value = getattr(msg, attr, None)
        if not value:
            continue
        if isinstance(value, bytes):
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    return value.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()
    return ""


def _message_header_markdown(metadata: dict[str, Any]) -> list[str]:
    rows = [
        ("Subject", metadata.get("subject", "")),
        ("From", metadata.get("sender", "")),
        ("To", metadata.get("recipients", "")),
        ("CC", metadata.get("cc", "")),
        ("Date", metadata.get("date", "")),
    ]
    return [f"**{label}:** {_escape_markdown(str(value))}" for label, value in rows if value]


def html_to_markdown(html: str) -> str:
    parser = _HtmlToMarkdown()
    parser.feed(html)
    parser.close()
    return parser.markdown()


def markdown_to_plain_text(markdown: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


class _HtmlToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.link_href: str | None = None
        self.list_stack: list[str] = []
        self.bold_depth = 0
        self.italic_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"p", "div", "section", "article", "tr"}:
            self._block_break()
        elif tag == "br":
            self.parts.append("\n")
        elif tag in {"strong", "b"}:
            self.bold_depth += 1
            self.parts.append("**")
        elif tag in {"em", "i"}:
            self.italic_depth += 1
            self.parts.append("*")
        elif tag in {"ul", "ol"}:
            self.list_stack.append(tag)
            self._block_break()
        elif tag == "li":
            self._block_break()
            self.parts.append("1. " if self.list_stack and self.list_stack[-1] == "ol" else "- ")
        elif tag == "a":
            self.link_href = _attr(attrs, "href")
            if self.link_href:
                self.parts.append("[")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._block_break()
            self.parts.append(f"{'#' * int(tag[1])} ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"strong", "b"} and self.bold_depth:
            self.parts.append("**")
            self.bold_depth -= 1
        elif tag in {"em", "i"} and self.italic_depth:
            self.parts.append("*")
            self.italic_depth -= 1
        elif tag == "a" and self.link_href:
            self.parts.append(f"]({self.link_href})")
            self.link_href = None
        elif tag in {"p", "div", "section", "article", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}:
            if tag in {"ul", "ol"} and self.list_stack:
                self.list_stack.pop()
            self._block_break()
        elif tag == "li":
            self.parts.append("\n")
        elif tag == "tr":
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_data(self, data: str) -> None:
        text = re.sub(r"\s+", " ", data)
        if text.strip():
            self.parts.append(_escape_markdown(text))

    def markdown(self) -> str:
        text = "".join(self.parts)
        lines = [line.strip(" |") for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")
        return cleaned.strip()

    def _block_break(self) -> None:
        current = "".join(self.parts)
        if current and not current.endswith("\n\n"):
            self.parts.append("\n\n" if not current.endswith("\n") else "\n")


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
    for key, value in attrs:
        if key.lower() == name and value:
            return value
    return None


def _escape_markdown(text: str) -> str:
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")
