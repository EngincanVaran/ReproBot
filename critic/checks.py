"""Deterministic sanity checks on a run's reported numbers, beyond the verdict itself.

The verdict compares the reproduced value to the claim. These checks ask whether the
reproduced value is what the run says it is. Pure arithmetic: no API key, no Docker.

`eval_count_check` catches a script that reports one evaluation-set size but scored a
smaller one. A rate such as a test error of 3.826% over N images is `k / N` for a whole
number of mistakes `k`. If `value / 100 * N` is not whole for the reported N, but is whole
for a slightly smaller N, the script evaluated only that many samples - typically because
`drop_last` silently discarded the final partial batch. This was found on the WRN run,
where the script reported `num_eval_samples: 10000` but 3.8261% is exactly 382 / 9984.
"""

from __future__ import annotations

from typing import Any

SEARCH_BELOW = 512
COUNT_TOLERANCE = 1e-8


def _whole(count: float) -> bool:
    return abs(count - round(count)) < COUNT_TOLERANCE


def eval_count_check(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """A finding when the reported rate only fits a smaller evaluation set, else None.

    Returns None whenever it cannot tell: not a percentage, a value rounded before it was
    stored (nothing whole fits), or a value that is already whole for the reported N.
    Silence is deliberately the default, so a mean over seeds or a rounded figure is
    never flagged.
    """
    if not metrics:
        return None
    value, reported = metrics.get("value"), metrics.get("num_eval_samples")
    if metrics.get("unit") != "%" or not isinstance(value, int | float):
        return None
    if isinstance(reported, bool) or not isinstance(reported, int) or reported <= 0:
        return None
    if not 0 <= value <= 100 or _whole(value / 100 * reported):
        return None
    for implied in range(reported - 1, max(0, reported - SEARCH_BELOW), -1):
        if _whole(value / 100 * implied):
            dropped = reported - implied
            mistakes = round(value / 100 * implied)
            return {
                "kind": "eval_count_mismatch",
                "reported_eval_samples": reported,
                "implied_eval_samples": implied,
                "dropped": dropped,
                "message": (
                    f"{value:.6g}% of {reported} samples is not a whole number of mistakes "
                    f"({value / 100 * reported:.2f}), but it is exactly {mistakes} / {implied}: "
                    f"the script scored {dropped} fewer samples than it reports "
                    f"(often drop_last discarding the final partial batch)."
                ),
            }
    return None


def run_checks(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every check's findings for one run's metrics (empty when all is consistent)."""
    finding = eval_count_check(metrics)
    return [finding] if finding else []
