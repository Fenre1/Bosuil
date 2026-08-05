from __future__ import annotations

from enum import Enum


class ScanMode(str, Enum):
    FAST = "fast"
    VERIFY = "verify"
    REBUILD = "rebuild"

    @classmethod
    def parse(cls, value: "ScanMode | str") -> "ScanMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown scan mode {value!r}; expected one of: {choices}.") from exc
