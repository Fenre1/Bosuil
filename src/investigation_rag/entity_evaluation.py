from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from investigation_rag.entities import DEFAULT_ENTITY_LABELS, ParsedEntity, parse_entity_output


@dataclass(frozen=True)
class EntityEvaluation:
    required_total: int
    names_found: int
    correct_type: int
    missed: tuple[str, ...]
    wrong_type: tuple[str, ...]
    unscored_extra: tuple[str, ...]

    @property
    def required_recall(self) -> float:
        return self.names_found / self.required_total if self.required_total else 1.0

    @property
    def type_accuracy(self) -> float:
        return self.correct_type / self.names_found if self.names_found else 0.0


def evaluate_required_entities(
    extracted: list[ParsedEntity],
    ground_truth: dict[str, Any],
) -> EntityEvaluation:
    by_name: dict[str, set[str]] = {}
    for entity in extracted:
        by_name.setdefault(evaluation_name(entity.name), set()).add(entity.entity_type)

    required_keys: set[str] = set()
    missed: list[str] = []
    wrong_type: list[str] = []
    names_found = 0
    correct_type = 0
    for item in ground_truth.get("entities", []):
        name = str(item["text"])
        key = evaluation_name(name)
        required_keys.add(key)
        actual_types = by_name.get(key)
        if not actual_types:
            missed.append(name)
            continue
        names_found += 1
        accepted = {
            str(value).strip().casefold()
            for value in item.get("accepted_types", [item["type"]])
        }
        if actual_types & accepted:
            correct_type += 1
        else:
            wrong_type.append(f"{name}: expected {sorted(accepted)}, got {sorted(actual_types)}")

    extras = [
        f"{entity.name} [{entity.entity_type}]"
        for entity in extracted
        if evaluation_name(entity.name) not in required_keys
    ]
    return EntityEvaluation(
        required_total=len(ground_truth.get("entities", [])),
        names_found=names_found,
        correct_type=correct_type,
        missed=tuple(missed),
        wrong_type=tuple(wrong_type),
        unscored_extra=tuple(extras),
    )


def evaluation_name(value: str) -> str:
    value = re.sub(r"\u00ad\s*", "", value)
    value = re.sub(r"[‐‑‒–—]", "-", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score simple Entity [Type] output against a required-entity reference set."
    )
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("model_output", type=Path)
    args = parser.parse_args(argv)

    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    output = args.model_output.read_text(encoding="utf-8")
    extracted = parse_entity_output(output, DEFAULT_ENTITY_LABELS)
    result = evaluate_required_entities(extracted, truth)

    print(f"Required entities: {result.required_total}")
    print(f"Names found: {result.names_found} ({result.required_recall:.1%} required recall)")
    print(f"Correctly typed: {result.correct_type} ({result.type_accuracy:.1%} of found names)")
    print(f"Missed: {len(result.missed)}")
    for item in result.missed:
        print(f"  - {item}")
    print(f"Wrong type: {len(result.wrong_type)}")
    for item in result.wrong_type:
        print(f"  - {item}")
    print(f"Additional extracted entities (not scored): {len(result.unscored_extra)}")
    return 0 if not result.missed and not result.wrong_type else 2


if __name__ == "__main__":
    raise SystemExit(main())
