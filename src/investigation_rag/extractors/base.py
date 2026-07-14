from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXTRACTOR_VERSION = "extractors-v5"


@dataclass
class ExtractedSection:
    text: str
    source_type: str
    page_number: int | None = None
    display_markdown: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "display_markdown": self.display_markdown,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractedSection":
        return cls(
            text=payload.get("text", ""),
            source_type=payload.get("source_type", ""),
            page_number=payload.get("page_number"),
            display_markdown=payload.get("display_markdown"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class ExtractedDocument:
    source_type: str
    sections: list[ExtractedSection]
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list["ExtractedDocument"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractedDocument":
        return cls(
            source_type=payload.get("source_type", ""),
            sections=[ExtractedSection.from_dict(item) for item in payload.get("sections", [])],
            metadata=payload.get("metadata", {}),
            attachments=[cls.from_dict(item) for item in payload.get("attachments", [])],
        )


class BaseExtractor:
    source_type = ""
    version = EXTRACTOR_VERSION

    def extract(self, path: Path) -> ExtractedDocument:
        raise NotImplementedError


def docs_from_cache(payload: dict[str, Any]) -> ExtractedDocument:
    return ExtractedDocument.from_dict(payload)
