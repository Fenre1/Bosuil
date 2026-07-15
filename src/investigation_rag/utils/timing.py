from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimingStep:
    label: str
    seconds: float


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


def format_timing_step(step: TimingStep) -> str:
    return f"{step.label}: {format_duration(step.seconds)}"


def format_timing_steps(steps: list[TimingStep]) -> list[str]:
    return [format_timing_step(step) for step in steps]
