"""The learning curve, drawn as a standalone SVG from a run's history file.

`runner/`'s live check-ups already read `metrics.<mode>.history.jsonl`; the report
draws the same records. SVG rather than a plotting library for the reason the rest
of the repo keeps torch out of the host environment: a report should be generatable
with no extra dependency at all, and an SVG drops straight into Markdown.

Two reference lines make the picture honest on its own:

* the **claim** the run targeted, so the gap is visible rather than described;
* the **chance** level the script recorded, so "it learned" is legible without
  knowing the dataset - a regression RMSE of 3.3 means nothing until you know that
  predicting the mean gives 8.7.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WIDTH = 640
HEIGHT = 260
PAD_L, PAD_R, PAD_T, PAD_B = 58, 96, 18, 34
# Deliberately literal, not theme tokens: this SVG is embedded in Markdown that may be
# read on GitHub, in an editor or in a browser, none of which pass a theme in.
INK = "#33363d"
MUTED = "#8b8f98"
GRID = "#e2e2de"
TRAIN = "#1f4f82"
EVAL = "#b4541f"
TARGET = "#1c6444"
CHANCE = "#9aa0a8"


@dataclass(frozen=True)
class Curve:
    """What one history file says, reduced to what the drawing needs."""

    steps: list[float]
    train: list[float]
    evals: list[float]
    metric: str
    unit: str
    higher_is_better: bool | None
    target: float | None
    chance: float | None
    kind: str

    @property
    def usable(self) -> bool:
        return len(self.steps) >= 2 and any(v is not None for v in self.evals)


def read_curve(path: Path) -> Curve | None:
    """Read `metrics.<mode>.history.jsonl`; None when it is absent or unusable.

    Scripts generated before the progress contract write no history at all, so a
    missing file is a normal outcome, not an error.
    """
    if not path.exists():
        return None
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("step"), int | float):
            records.append(record)
    if len(records) < 2:
        return None
    last = records[-1]
    curve = Curve(
        steps=[float(r["step"]) for r in records],
        train=[_number(r.get("train_metric")) for r in records],
        evals=[_number(r.get("eval_metric")) for r in records],
        metric=str(last.get("metric") or "metric"),
        unit=str(last.get("unit") or ""),
        higher_is_better=last.get("higher_is_better")
        if isinstance(last.get("higher_is_better"), bool)
        else None,
        target=_number(last.get("target_value")),
        chance=_number(last.get("chance_metric")),
        kind=str(last.get("kind") or "step"),
    )
    return curve if curve.usable else None


def _number(value: object) -> Any:
    return float(value) if isinstance(value, int | float) else None


def _ticks(low: float, high: float) -> list[float]:
    """Four evenly spaced labels, rounded to something a reader would write."""
    if high <= low:
        return [low]
    return [low + (high - low) * i / 3 for i in range(4)]


def _fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def render_curve(curve: Curve, title: str) -> str:
    """One SVG: train and eval series, with the claim and chance as reference lines."""
    values = [v for v in [*curve.train, *curve.evals] if v is not None]
    for reference in (curve.target, curve.chance):
        if reference is not None:
            values.append(reference)
    low, high = min(values), max(values)
    if high == low:
        low, high = low - 1, high + 1
    span = high - low
    low, high = low - span * 0.08, high + span * 0.08

    def x(step: float) -> float:
        first, last = curve.steps[0], curve.steps[-1]
        frac = 0.0 if last == first else (step - first) / (last - first)
        return PAD_L + frac * (WIDTH - PAD_L - PAD_R)

    def y(value: float) -> float:
        frac = (value - low) / (high - low)
        return HEIGHT - PAD_B - frac * (HEIGHT - PAD_T - PAD_B)

    def path(series: list[Any], colour: str) -> str:
        points = [(x(s), y(v)) for s, v in zip(curve.steps, series, strict=True) if v is not None]
        if len(points) < 2:
            return ""
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(points)
        )
        return (
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.8" '
            f'stroke-linejoin="round"/>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{_esc(title)}">',
        f"<title>{_esc(title)}</title>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
    ]
    for tick in _ticks(low, high):
        ty = y(tick)
        parts.append(
            f'<line x1="{PAD_L}" y1="{ty:.1f}" x2="{WIDTH - PAD_R}" y2="{ty:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{ty + 3.5:.1f}" text-anchor="end" font-size="10" '
            f'font-family="monospace" fill="{MUTED}">{_fmt(tick)}</text>'
        )
    for reference, colour, label in (
        (curve.chance, CHANCE, "chance"),
        (curve.target, TARGET, "the paper's claim"),
    ):
        if reference is None or not low <= reference <= high:
            continue
        ry = y(reference)
        parts.append(
            f'<line x1="{PAD_L}" y1="{ry:.1f}" x2="{WIDTH - PAD_R}" y2="{ry:.1f}" '
            f'stroke="{colour}" stroke-width="1.4" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{WIDTH - PAD_R + 6}" y="{ry + 3.5:.1f}" font-size="10" '
            f'font-family="sans-serif" fill="{colour}">{_esc(label)} {_fmt(reference)}</text>'
        )
    parts.append(path(curve.train, TRAIN))
    parts.append(path(curve.evals, EVAL))
    for series, colour, name in ((curve.train, TRAIN, "train"), (curve.evals, EVAL, "eval")):
        last = next((v for v in reversed(series) if v is not None), None)
        if last is None:
            continue
        parts.append(
            f'<circle cx="{x(curve.steps[-1]):.1f}" cy="{y(last):.1f}" r="3.2" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{WIDTH - PAD_R + 6}" y="{y(last) + 3.5:.1f}" font-size="10" '
            f'font-family="sans-serif" fill="{colour}">{name} {_fmt(last)}</text>'
        )
    axis_y = HEIGHT - PAD_B
    parts.append(
        f'<line x1="{PAD_L}" y1="{axis_y}" x2="{WIDTH - PAD_R}" y2="{axis_y}" '
        f'stroke="{INK}" stroke-width="1"/>'
    )
    for step in (curve.steps[0], curve.steps[-1]):
        parts.append(
            f'<text x="{x(step):.1f}" y="{axis_y + 14}" text-anchor="middle" font-size="10" '
            f'font-family="monospace" fill="{MUTED}">{_fmt(step)}</text>'
        )
    unit = f" ({curve.unit})" if curve.unit else ""
    parts.append(
        f'<text x="{PAD_L}" y="{HEIGHT - 6}" font-size="10" font-family="sans-serif" '
        f'fill="{MUTED}">{_esc(curve.kind)} · {_esc(curve.metric)}{_esc(unit)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
