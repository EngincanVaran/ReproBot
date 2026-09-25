"""Judge one reproduced result against the paper's claim.

The verdict is explicit arithmetic, not a model reading numbers (project plan §2.5):
every verdict carries the numbers and the rule that produced it.

TOLERANCE
---------
A reproduction is never judged closer than the evidence allows. Three measurable
sources of uncertainty feed the tolerance:

1. **Reporting precision** - a paper that reports 96.9 cannot be matched closer than
   +/-0.05. Always applies, as a floor.
2. **Test-set noise** - for accuracy-like metrics, the binomial standard error of the
   claimed rate on the evaluation set's size. It stands in for how much a result of
   that size moves between training runs when nothing else is known.
3. **Measured run spread** - when several runs exist (a claim averaged over repeated
   runs, or extra seeds the Orchestrator ran), their standard deviation. When it is
   available it replaces the binomial proxy: it measures the real variation instead
   of approximating it.

    tolerance = max(2 x uncertainty, reporting precision)      (~95% at 2 sigma)

VERDICTS
--------
* ``pass`` - within tolerance, or better than the claim beyond it (``exceeds_claim``).
* ``fail`` - worse than the claim beyond tolerance, with an uncertainty estimate
  behind the tolerance. Recommends one guided retry.
* ``inconclusive`` - worse beyond tolerance, but with no uncertainty estimate at all
  (one run of a non-proportion metric such as RMSE): the gap may be one unlucky
  split. Recommends more seeds.
* ``not_evaluated`` - nothing comparable to judge: the run did not succeed, produced
  no number, stopped before the ``full`` stage, or targeted a claim that is missing.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from critic.claims import group_claims, group_for, reporting_precision

type Verdict = Literal["pass", "fail", "inconclusive", "not_evaluated"]
type Recommendation = Literal["none", "run_more_seeds", "guided_retry"]

# Uncertainty is doubled for the tolerance: roughly a 95% band under a normal model.
SIGMA_MULTIPLIER: float = 2.0
# Only a full run reproduces the paper's setup; cheaper stages are execution checks.
COMPARABLE_MODES: tuple[str, ...] = ("full",)

_LOWER_IS_BETTER_WORDS = ("error", "err", "loss", "rmse", "mse", "mae", "perplexity")
_HIGHER_IS_BETTER_WORDS = ("acc", "accuracy", "r2", "rsquared", "f1", "auc", "map", "precision")


@dataclass(frozen=True)
class Judgement:
    """A verdict on one claim, with every number that produced it."""

    verdict: Verdict
    reason: str
    recommendation: Recommendation = "none"
    claim_id: str | None = None
    metric: str | None = None
    dataset: str | None = None
    unit: str = ""
    claimed: float | None = None
    reproduced: float | None = None
    run_values: list[float] = field(default_factory=list)
    higher_is_better: bool | None = None
    gap: float | None = None
    relative_gap: float | None = None
    tolerance: float | None = None
    reporting_precision: float | None = None
    test_noise: float | None = None
    run_spread: float | None = None
    evidence: str = ""
    exceeds_claim: bool = False
    merged_claim_ids: list[str] = field(default_factory=list)
    metrics_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.claimed is None or self.reproduced is None:
            return f"{self.verdict}: {self.reason}"
        runs = f" (mean of {len(self.run_values)} runs)" if len(self.run_values) > 1 else ""
        tol = f" +/- {self.tolerance:.4g}" if self.tolerance is not None else ""
        return (
            f"{self.verdict}: {self.metric} reproduced {self.reproduced:.6g}{self.unit}{runs} "
            f"vs claimed {self.claimed:.6g}{self.unit}{tol}"
        )


def infer_higher_is_better(metric: str) -> bool | None:
    key = metric.lower().replace("-", "").replace(" ", "")
    if any(word in key for word in _LOWER_IS_BETTER_WORDS):
        return False
    if any(word in key for word in _HIGHER_IS_BETTER_WORDS):
        return True
    return None


def binomial_noise(claimed: float, unit: str, metric: str, n: int | None) -> float | None:
    """Standard error of an accuracy-like rate on `n` evaluation samples, in its unit."""
    if not n or n <= 0:
        return None
    if unit.strip() == "%":
        p, scale = claimed / 100.0, 100.0
    elif 0.0 <= claimed <= 1.0 and any(w in metric.lower() for w in ("acc", "error", "err")):
        p, scale = claimed, 1.0
    else:
        return None
    if not 0.0 <= p <= 1.0:
        return None
    return math.sqrt(p * (1.0 - p) / n) * scale


def run_uncertainty(values: list[float], claim_is_a_mean: bool) -> float | None:
    """Uncertainty of the gap from measured runs, or None with fewer than two.

    A claim that is itself a mean over the same number of runs is compared mean to
    mean (both carry std/sqrt(n)); a single-run claim is one draw from the spread,
    compared with our mean (std * sqrt(1 + 1/n)).
    """
    if len(values) < 2:
        return None
    spread = statistics.stdev(values)
    n = len(values)
    return spread * math.sqrt(2.0 / n) if claim_is_a_mean else spread * math.sqrt(1.0 + 1.0 / n)


def not_evaluated(reason: str, **known: Any) -> Judgement:
    return Judgement(verdict="not_evaluated", reason=reason, **known)


def judge(
    claims: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    *,
    runner_status: str,
    metrics_mode: str | None,
    extra_run_values: list[float] | None = None,
) -> Judgement:
    """Judge the claim a run targeted. `extra_run_values` are extra seeds' results."""
    if runner_status != "success":
        return not_evaluated(f"the run did not succeed (runner status {runner_status!r})")
    if not metrics or not isinstance(metrics.get("value"), int | float):
        return not_evaluated("the run produced no reproduced value to compare")
    if metrics_mode is not None and metrics_mode not in COMPARABLE_MODES:
        return not_evaluated(
            f"the reproduced value comes from the {metrics_mode!r} stage; only a full run "
            f"reproduces the paper's setup, so nothing comparable exists yet",
            metrics_mode=metrics_mode,
        )
    claim_id = str(metrics.get("claim_id") or "")
    groups = group_claims(claims)
    group = group_for(claim_id, groups)
    if group is None:
        return not_evaluated(f"the targeted claim {claim_id!r} is not among the paper's claims")

    claim = group.canonical
    claimed = float(claim["reported_value"])
    unit = str(claim.get("unit") or "")
    metric = str(claim.get("metric") or metrics.get("metric") or "")
    known: dict[str, Any] = {
        "claim_id": claim_id,
        "metric": metric,
        "dataset": claim.get("dataset"),
        "unit": unit,
        "claimed": claimed,
        "merged_claim_ids": group.claim_ids,
        "metrics_mode": metrics_mode,
    }

    run_values = [float(v) for v in (metrics.get("run_values") or []) if isinstance(v, int | float)]
    claim_is_a_mean = len(run_values) >= 2
    if not claim_is_a_mean:
        run_values = [float(metrics["value"])]
    run_values += [float(v) for v in (extra_run_values or [])]
    reproduced = statistics.fmean(run_values)

    hib = metrics.get("higher_is_better")
    higher_is_better = hib if isinstance(hib, bool) else infer_higher_is_better(metric)

    precision = reporting_precision(claimed)
    test_noise = binomial_noise(claimed, unit, metric, metrics.get("num_eval_samples"))
    spread = run_uncertainty(run_values, claim_is_a_mean)
    if spread is not None:
        uncertainty: float | None = spread
        evidence = f"measured spread of {len(run_values)} runs"
    elif test_noise is not None:
        uncertainty = test_noise
        evidence = "binomial noise of the evaluation set"
    else:
        uncertainty = None
        evidence = "reporting precision only (single run, no noise estimate)"
    tolerance = max(SIGMA_MULTIPLIER * uncertainty, precision) if uncertainty else precision

    gap = reproduced - claimed
    favourable = gap if higher_is_better is not False else -gap
    relative = gap / abs(claimed) if claimed else None
    known.update(
        reproduced=reproduced,
        run_values=run_values,
        higher_is_better=higher_is_better,
        gap=gap,
        relative_gap=relative,
        tolerance=tolerance,
        reporting_precision=precision,
        test_noise=test_noise,
        run_spread=spread,
        evidence=evidence,
    )

    if abs(gap) <= tolerance:
        return Judgement(
            verdict="pass",
            reason=f"the reproduced value is within tolerance of the claim ({evidence})",
            **known,
        )
    if higher_is_better is not None and favourable > 0:
        return Judgement(
            verdict="pass",
            reason=(
                "the reproduced value is better than the claim beyond tolerance - the claimed "
                "result is achievable with this implementation"
            ),
            exceeds_claim=True,
            **known,
        )
    if uncertainty is None:
        return Judgement(
            verdict="inconclusive",
            reason=(
                "the reproduced value is worse than the claim beyond reporting precision, but a "
                "single run of this metric carries no noise estimate - the gap may come from one "
                "unlucky split or seed"
            ),
            recommendation="run_more_seeds",
            **known,
        )
    return Judgement(
        verdict="fail",
        reason=f"the reproduced value is worse than the claim beyond tolerance ({evidence})",
        recommendation="guided_retry",
        **known,
    )


def guided_retry_feedback(judgement: Judgement, assumptions: list[str]) -> str:
    """Feedback for the one fidelity retry after a `fail`: the gap, plus the Coder's
    own unstated guesses as the only things it may revisit."""
    guesses = "\n".join(f"- {item}" for item in assumptions) or "- (none were recorded)"
    runs = f" (mean of {len(judgement.run_values)} runs)" if len(judgement.run_values) > 1 else ""
    return (
        f"The previous script ran healthily, but its result misses the paper's claim: "
        f"{judgement.metric} {judgement.reproduced:.6g}{judgement.unit}{runs} against the "
        f"claimed {judgement.claimed:.6g}{judgement.unit}, beyond a tolerance of "
        f"{judgement.tolerance:.4g} ({judgement.evidence}). "
        f"Revisit ONLY settings the paper does not state. The unstated choices you recorded "
        f"last time were:\n{guesses}\n"
        f"Never change a value the paper states. Record every setting you change as its own "
        f"assumptions entry beginning 'fidelity retry:', giving the old value, the new value "
        f"and why. Justify each change from the paper's text, its framework's or library's "
        f"defaults, or standard practice for this method - never from wanting the test "
        f"number to move."
    )
