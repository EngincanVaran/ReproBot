"""A learning curve as a self-contained SVG string: no plotting library, no external files."""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

WIDTH, HEIGHT = 640, 260
LEFT, RIGHT, TOP, BOTTOM = 56, 16, 20, 40


def _points(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(r["step"]), float(r["eval_metric"]))
        for r in records
        if isinstance(r.get("step"), int | float) and isinstance(r.get("eval_metric"), int | float)
    ]


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


def curve_svg(records: list[dict[str, Any]], *, metric: str, unit: str = "") -> str | None:
    """The evaluation metric over training, with the paper's value (and chance) as guides.

    Returns None when there is no real curve to draw: fewer than three points, or a
    history that is not per-epoch (a grid-search trace is not a learning curve).
    """
    points = _points(records)
    if len(points) < 3 or any(r.get("kind", "epoch") != "epoch" for r in records):
        return None
    target = next((float(r["target_value"]) for r in records if "target_value" in r), None)
    chance = next((float(r["chance_metric"]) for r in records if "chance_metric" in r), None)

    values = [y for _, y in points] + ([target] if target is not None else [])
    low, high = min(values), max(values)
    # Chance only widens the axis when the curve starts near it; a far-off line would
    # squash the part of the curve that matters.
    if chance is not None and chance <= 1.25 * high:
        high = max(high, chance)
    pad = (high - low) * 0.08 or 1.0
    low, high = max(0.0, low - pad), high + pad
    x_min, x_max = points[0][0], points[-1][0]

    def sx(x: float) -> float:
        return LEFT + (x - x_min) / ((x_max - x_min) or 1.0) * (WIDTH - LEFT - RIGHT)

    def sy(y: float) -> float:
        return TOP + (1 - (y - low) / (high - low)) * (HEIGHT - TOP - BOTTOM)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="{escape(metric)} over training" font-family="sans-serif" font-size="11">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
    ]
    for tick in _ticks(low, high):
        y = sy(tick)
        out.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="#ddd"/>'
        )
        out.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:.3g}</text>')
    for tick in _ticks(x_min, x_max):
        x = sx(tick)
        out.append(
            f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 16}" text-anchor="middle">{tick:.0f}</text>'
        )
    middle = (LEFT + WIDTH - RIGHT) / 2
    out.append(f'<text x="{middle:.0f}" y="{HEIGHT - 6}" text-anchor="middle">epoch</text>')

    for value, colour, label in ((target, "#2a7", "paper"), (chance, "#a44", "chance")):
        if value is not None and low <= value <= high:
            y = sy(float(value))
            out.append(
                f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" '
                f'stroke="{colour}" stroke-dasharray="5 4"/>'
            )
            out.append(
                f'<text x="{WIDTH - RIGHT - 4}" y="{y - 4:.1f}" text-anchor="end" fill="{colour}">'
                f"{label} {float(value):.3g}{escape(unit)}</text>"
            )
    line = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    out.append(f'<polyline points="{line}" fill="none" stroke="#25c" stroke-width="1.6"/>')
    out.append(
        f'<text x="{LEFT + 4}" y="{TOP + 10}" fill="#25c">{escape(metric)}{escape(unit)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)
