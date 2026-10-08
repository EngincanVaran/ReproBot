"""Live check-ups: is a running training job actually learning?

An exit code says a script finished, never that it learned. Twice in phase 3 a
`full` run exited 0 with a broken model: Tang 2013's network collapsed into a
constant predictor, and a soft decision tree trained on a sign-flipped loss
drove its test accuracy to 0.15%. Both were caught by a person reading
`docker logs`, long before the runs would have ended. This module makes the
Runner do that reading itself.

The channel is a file, not free-text logs. Every generated script appends one
JSON object per epoch (or per repetition/fold for a classical estimator) to a
history file beside its metrics file - `metrics.<mode>.json` gets
`metrics.<mode>.history.jsonl` - and prints the same object to stdout after
the `REPROBOT_PROGRESS` marker. Scripts log their prose however they like; the
record's shape is fixed by coder/'s prompt, so parsing never depends on
wording.

Everything here is pure and daemon-free, so every rule is verified against the
real curves of past runs (see tests/test_checkups.py) without Docker.

The rules are deliberately conservative. A false halt kills a run that would
have reproduced a paper, which is worse than a late one, so:

* cheap stages (`probe`, `smoke`) only get the checks that are wrong at any
  length - non-finite values and a loss below its own lower bound;
* trend checks need persistence (several consecutive records), a warmup, and
  a margin measured against the chance level of a trivial predictor, which the
  script computes from its own data;
* no rule treats a late plateau as a problem - a run that is clearly better
  than chance and has stopped improving is simply converged;
* the one rule that judges silence instead of a curve (`no_first_record`, for a
  script stuck inside its very first fit) is armed only on the stages where
  silence is diagnostic, disarms for good the instant any record arrives, and
  measures SILENCE rather than elapsed time - a dataset still arriving is a sign
  of life, so the ~1707 s cold CIFAR-10 fetch never counts against a script.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

PROGRESS_MARKER: Final[str] = "REPROBOT_PROGRESS"
HISTORY_SUFFIX: Final[str] = ".history.jsonl"

# A trend rule only judges records at or after this step: max(3, 10% of the run).
WARMUP_MIN_STEPS: Final[int] = 3
WARMUP_FRACTION: Final[float] = 0.10
# ...and only when its condition holds for this many consecutive eligible records.
PERSISTENCE: Final[int] = 3
# Proportion metrics (accuracy, error rate) get a binomial margin of this many SEs.
SE_MULTIPLIER: Final[float] = 3.0
# "Meaningfully better than chance" = at least this share of the way from chance
# to a perfect score.
STUCK_FRACTION: Final[float] = 0.10
# The loss has "not moved" when its range over the first window is below this
# fraction of its magnitude.
NO_PROGRESS_RELATIVE_RANGE: Final[float] = 1e-3
# Diverged = loss this many times its best AND the metric has kept less than this
# share of its best improvement over chance.
DIVERGE_LOSS_FACTOR: Final[float] = 1.5
DIVERGE_ADVANTAGE_KEPT: Final[float] = 0.5
# Slack before a loss counts as below its declared lower bound.
LOWER_BOUND_TOLERANCE: Final[float] = 1e-3
# Most files `SilenceClock` will walk in the shared dataset mount before it stops
# counting. Real caches hold a handful of archives and extracted folders; the bound
# is there so an unpacked image dataset cannot turn a 5 s poll into a slow one.
CACHE_SCAN_MAX_ENTRIES: Final[int] = 20_000

# How long a stage may stay completely silent - not one progress record - before
# `no_first_record` halts it. Every other rule needs a record, so a script stuck
# inside its FIRST fit is otherwise invisible: on 2026-09-20 a Fashion-MNIST run
# targeted the RBF SVC claim, one LIBSVM fit on 60,000 x 784 rows is tens of
# minutes single-threaded and the paper repeats it five times, and fifteen minutes
# in the history file was still empty. A person had to kill it.
#
# The numbers are deliberately far above anything a healthy run has ever needed:
#
#   * the longest real wait before a first record is a COLD dataset fetch, measured
#     at 1707 s of a 1774 s probe (~170 MB of CIFAR-10 on a slow connection);
#   * the longest real first unit of work is the soft decision tree's first epoch,
#     ~105 s (40 epochs in a 70 min full run); Tang's is ~8 s (400 epochs, 54 min),
#     the Fashion-MNIST forest's first repetition ~40 s (5 in 3.3 min), and the SVM
#     guide's single fit 73 s end to end.
#
# So `full` waits 5400 s: 3.2x the cold fetch and ~50x the slowest real first epoch,
# while still turning the SVC incident's remaining hours into a bounded 90 minutes.
# `capped` waits 900 s: 1.7x the whole slowest measured capped stage (~525 s) and
# half its own 1800 s budget.
#
# 900 s is BELOW the measured cold fetch on purpose, and what makes it safe is that
# the clock measures silence, not elapsed time: `SilenceClock` restarts it while the
# shared dataset cache is growing, so a download never counts against the deadline.
# Without that, `--mode capped` on a cold cache - a documented way to run one stage,
# with no `probe` in front of it to warm the mount - would be halted mid-download and
# the Orchestrator would hand the Coder a script that was never at fault.
#
# `probe` and `smoke` have NO entry on purpose, and that is the conservative choice
# rather than an omission: probe is the stage that absorbs the cold fetch, whose
# worst case is a property of the network rather than of the script, and both stages
# can legitimately finish without writing a record at all (a probe is a couple of
# optimizer steps). Their short budgets - 2700 s and 900 s - already bound them, and
# a timeout stops the Orchestrator for a human instead of asking the Coder to
# regenerate a script that was never the problem.
DEFAULT_FIRST_RECORD_SECONDS: Final[dict[str, float]] = {
    "capped": 900.0,
    "full": 5400.0,
    "seed2": 5400.0,
    "seed3": 5400.0,
}

type RuleName = Literal[
    "non_finite",
    "loss_below_lower_bound",
    "worse_than_chance",
    "no_progress",
    "stuck_at_chance",
    "diverged",
    "no_first_record",
]

ALL_RULES: Final[tuple[RuleName, ...]] = (
    "non_finite",
    "loss_below_lower_bound",
    "worse_than_chance",
    "no_progress",
    "stuck_at_chance",
    "diverged",
    "no_first_record",
)

# Which rules may halt which stage. Probe and smoke are too short for trends, and
# too short to read anything into silence (see DEFAULT_FIRST_RECORD_SECONDS).
STAGE_RULES: Final[dict[str, tuple[RuleName, ...]]] = {
    "probe": ("non_finite", "loss_below_lower_bound"),
    "smoke": ("non_finite", "loss_below_lower_bound"),
    # `no_progress` is deliberately NOT here: a capped stage is a few dozen optimizer
    # steps, and a real model can leave its loss unchanged to four digits over that
    # while its predictions already improve (a soft decision tree whose leaves start
    # uniform sat at ln 10 = 2.303 with 30% accuracy, three times chance).
    "capped": (
        "non_finite",
        "loss_below_lower_bound",
        "worse_than_chance",
        "no_first_record",
    ),
    "full": ALL_RULES,
    # Extra seeds are the full run again with another seed, judged the same way.
    "seed2": ALL_RULES,
    "seed3": ALL_RULES,
}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProgressRecord:
    """One line of a script's learning-curve history."""

    step: int
    steps_total: int | None
    kind: str
    train_loss: float | None
    eval_loss: float | None
    train_metric: float | None
    eval_metric: float | None
    metric: str
    unit: str
    higher_is_better: bool | None
    chance_metric: float | None
    target_value: float | None
    loss_lower_bound: float | None
    num_eval_samples: int | None
    elapsed_seconds: float | None

    @property
    def metric_value(self) -> float | None:
        """The metric the rules judge: evaluation split when logged, else training."""
        return self.eval_metric if self.eval_metric is not None else self.train_metric

    @property
    def loss_value(self) -> float | None:
        return self.train_loss if self.train_loss is not None else self.eval_loss

    def describe(self) -> str:
        total = f"/{self.steps_total}" if self.steps_total else ""
        parts = [f"{self.kind} {self.step}{total}"]
        if self.train_loss is not None:
            parts.append(f"train_loss={_fmt(self.train_loss)}")
        if self.eval_loss is not None:
            parts.append(f"eval_loss={_fmt(self.eval_loss)}")
        value = self.metric_value
        if value is not None:
            split = "eval" if self.eval_metric is not None else "train"
            parts.append(f"{self.metric or 'metric'}({split})={_fmt(value)}{self.unit}")
        if self.chance_metric is not None:
            parts.append(f"chance={_fmt(self.chance_metric)}{self.unit}")
        return " ".join(parts)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def _num(value: object) -> float | None:
    """A JSON number, or the strings a script may use for non-finite values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "-inf", "infinity"}:
        return float(value.strip().lower().replace("infinity", "inf"))
    return None


def _int(value: object) -> int | None:
    number = _num(value)
    if number is None or not math.isfinite(number):
        return None
    return int(number)


def parse_record(line: str) -> ProgressRecord | None:
    """Parse one history line (or marker-prefixed stdout line). None if it is not one.

    Python's `json` accepts the bare `NaN` / `Infinity` tokens `json.dumps` writes
    for non-finite floats, which is exactly what a diverging script produces, so
    they are kept rather than rejected.
    """
    text = line.strip()
    if text.startswith(PROGRESS_MARKER):
        text = text[len(PROGRESS_MARKER) :].strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    step = _int(payload.get("step"))
    if step is None:
        return None
    higher = payload.get("higher_is_better")
    return ProgressRecord(
        step=step,
        steps_total=_int(payload.get("steps_total")),
        kind=str(payload.get("kind") or "epoch"),
        train_loss=_num(payload.get("train_loss")),
        eval_loss=_num(payload.get("eval_loss")),
        train_metric=_num(payload.get("train_metric")),
        eval_metric=_num(payload.get("eval_metric")),
        metric=str(payload.get("metric") or ""),
        unit=str(payload.get("unit") or ""),
        higher_is_better=higher if isinstance(higher, bool) else None,
        chance_metric=_num(payload.get("chance_metric")),
        target_value=_num(payload.get("target_value")),
        loss_lower_bound=_num(payload.get("loss_lower_bound")),
        num_eval_samples=_int(payload.get("num_eval_samples")),
        elapsed_seconds=_num(payload.get("elapsed_seconds")),
    )


def writes_progress_history(script_path: Path) -> bool:
    """Does this generated script implement the progress contract at all?

    A script written before the contract (the `Trainer`-era NIN and WRN, and Tang's
    and Wijaya's scripts still on disk) never writes a record however healthy it is,
    so judging its silence would halt a run that is doing exactly the right thing.
    The test is a literal in the generated text, the same cheap kind of check
    `coder/`'s own CLI-flag gate makes, and it fails toward "not armed": a script
    that builds the marker dynamically is left alone rather than halted.
    """
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return PROGRESS_MARKER in text or HISTORY_SUFFIX in text


def history_path_from_metrics(metrics_path: Path) -> Path:
    """`metrics.full.json` -> `metrics.full.history.jsonl`: the rule scripts follow."""
    name = metrics_path.name
    stem = name[: -len(".json")] if name.endswith(".json") else name
    return metrics_path.with_name(stem + HISTORY_SUFFIX)


class HistoryTail:
    """Reads a history file incrementally while the script is still appending to it.

    Only complete lines are consumed; a partially written last line waits for the
    next read. A missing file is simply "nothing yet".
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._pending = ""
        self.malformed_lines = 0

    def read_new(self) -> list[ProgressRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        text = self._pending + chunk
        lines = text.split("\n")
        self._pending = lines.pop()  # "" when the chunk ended on a newline
        records = []
        for line in lines:
            if not line.strip():
                continue
            record = parse_record(line)
            if record is None:
                self.malformed_lines += 1
            else:
                records.append(record)
        return records


class SilenceClock:
    """How long a stage has gone without any observable sign of life.

    `no_first_record` must not confuse "doing nothing" with "doing something that
    does not report yet", and the one thing that legitimately precedes a first
    record is a dataset download: the repo measured 1707 s of a 1774 s probe as
    CIFAR-10 arriving. Wall-clock time since launch would therefore halt an honest
    `--mode capped` run on a cold cache, since nothing warms the mount in front of
    it - so the clock restarts whenever the shared dataset mount changes size or
    file count, and the deadline is measured from the last such change.

    That mount is the only live signal available. The script's own stdout is not:
    `runner/docker_runner.py` captures both streams through a blocking
    `subprocess.run(capture_output=True)`, so not one line is readable until the
    process exits. Two consequences worth knowing: a download the script buffers
    entirely in memory (`pandas.read_csv(url)`) touches no file and so looks like
    silence, and a script re-downloading in a loop keeps resetting the clock until
    the stage's own timeout ends it.

    The walk is bounded (CACHE_SCAN_MAX_ENTRIES) and only runs while a stage is
    still silent, so the diagnosis can never become the slow part of the run.
    """

    def __init__(self, cache_dir: Path | None, now: float) -> None:
        self.cache_dir = cache_dir
        self._fingerprint = self._read_fingerprint()
        self._last_sign_of_life = now

    def _read_fingerprint(self) -> tuple[int, int]:
        """(entries, total bytes) under the dataset mount; (0, 0) when unreadable.

        Any change counts as life, in either direction: a file being appended to, a
        new one appearing, or an archive being deleted after extraction.
        """
        if self.cache_dir is None or not self.cache_dir.exists():
            return (0, 0)
        entries = 0
        total = 0
        try:
            for path in self.cache_dir.rglob("*"):
                entries += 1
                try:
                    total += path.stat().st_size
                except OSError:
                    # A temporary file of a download in flight, already gone.
                    continue
                if entries >= CACHE_SCAN_MAX_ENTRIES:
                    break
        except OSError:
            return (entries, total)
        return (entries, total)

    def silent_seconds(self, now: float) -> float:
        """Seconds since the last sign of life, restarting the clock on cache growth."""
        fingerprint = self._read_fingerprint()
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._last_sign_of_life = now
        return max(0.0, now - self._last_sign_of_life)


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Halt:
    """A check-up's decision to stop a run, with the evidence behind it."""

    rule: RuleName
    mode: str
    step: int
    kind: str
    message: str
    advice: str

    def feedback(self) -> str:
        """The text the Orchestrator hands the Coder when it regenerates."""
        return (
            f"The Runner's live check-up stopped the previous script during its "
            f"'{self.mode}' stage at {self.kind} {self.step}, because training was not "
            f"healthy ({self.rule}): {self.message} {self.advice} Keep every setting the "
            f"paper states unchanged; revisit only what the paper leaves unstated or what "
            f"the script got wrong."
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feedback"] = self.feedback()
        return payload


def warmup_step(record: ProgressRecord) -> int:
    if record.steps_total:
        return max(WARMUP_MIN_STEPS, math.ceil(WARMUP_FRACTION * record.steps_total))
    return WARMUP_MIN_STEPS


def advantage(value: float, chance: float, higher_is_better: bool | None) -> float:
    """How far `value` is past chance in the good direction (negative = worse)."""
    return value - chance if higher_is_better is not False else chance - value


def perfect_value(record: ProgressRecord) -> float | None:
    """The best possible score of the record's metric, when it is knowable."""
    if record.higher_is_better is False:
        return 0.0  # error rates and loss-like metrics
    if record.unit == "%":
        return 100.0
    chance = record.chance_metric
    if chance is not None and 0.0 <= chance <= 1.0:
        return 1.0
    return None


def proportion_se(record: ProgressRecord) -> float | None:
    """Binomial standard error of the chance level, for accuracy-like metrics only."""
    chance, n = record.chance_metric, record.num_eval_samples
    if chance is None or not n or n <= 0:
        return None
    if record.unit == "%":
        p = chance / 100.0
        scale = 100.0
    elif 0.0 <= chance <= 1.0 and any(
        word in record.metric.lower() for word in ("acc", "error", "err")
    ):
        p, scale = chance, 1.0
    else:
        return None
    if not 0.0 <= p <= 1.0:
        return None
    return math.sqrt(p * (1.0 - p) / n) * scale


def _judged(records: list[ProgressRecord]) -> list[ProgressRecord]:
    """Records that carry a finite metric and a chance level to compare it with."""
    return [
        r
        for r in records
        if r.metric_value is not None
        and math.isfinite(r.metric_value)
        and r.chance_metric is not None
        and math.isfinite(r.chance_metric)
    ]


def _persistent(records: list[ProgressRecord], condition: Callable[[ProgressRecord], bool]) -> bool:
    tail = records[-PERSISTENCE:]
    return len(tail) == PERSISTENCE and all(condition(r) for r in tail)


def rule_non_finite(records: list[ProgressRecord], mode: str) -> Halt | None:
    for record in records:
        for name in ("train_loss", "eval_loss", "train_metric", "eval_metric"):
            value = getattr(record, name)
            if value is not None and not math.isfinite(value):
                return Halt(
                    "non_finite",
                    mode,
                    record.step,
                    record.kind,
                    f"{name} became {value} ({record.describe()}).",
                    "The computation has gone numerically unstable: check the loss for a "
                    "logarithm or division that can reach zero, the input scaling, and "
                    "whether the step size is too large for this objective.",
                )
    return None


def rule_loss_below_lower_bound(records: list[ProgressRecord], mode: str) -> Halt | None:
    for record in records:
        bound = record.loss_lower_bound
        if bound is None:
            continue
        slack = LOWER_BOUND_TOLERANCE * max(1.0, abs(bound))
        for name in ("train_loss", "eval_loss"):
            value = getattr(record, name)
            if value is not None and math.isfinite(value) and value < bound - slack:
                return Halt(
                    "loss_below_lower_bound",
                    mode,
                    record.step,
                    record.kind,
                    f"{name} reached {_fmt(value)}, below the loss's own lower bound of "
                    f"{_fmt(bound)} ({record.describe()}).",
                    "A loss cannot go below its lower bound unless the objective being "
                    "optimised is not the intended one: check its sign and formula against "
                    "the paper's equation. A misprinted or mis-transcribed equation can yield "
                    "an objective that training pushes in the wrong direction.",
                )
    return None


def rule_worse_than_chance(records: list[ProgressRecord], mode: str) -> Halt | None:
    """Systematically worse than a trivial predictor - accuracy-like metrics only.

    Restricted to proportion metrics on purpose: an untrained regression network
    routinely scores worse than predicting the mean for a while, which is normal,
    whereas a classifier more than 3 SEs below chance is getting the task backwards.
    """
    eligible = [
        r for r in _judged(records) if proportion_se(r) is not None and r.step >= WARMUP_MIN_STEPS
    ]

    def worse(r: ProgressRecord) -> bool:
        se = proportion_se(r)
        assert se is not None and r.metric_value is not None and r.chance_metric is not None
        return advantage(r.metric_value, r.chance_metric, r.higher_is_better) < -SE_MULTIPLIER * se

    if not _persistent(eligible, worse):
        return None
    last = eligible[-1]
    assert last.metric_value is not None and last.chance_metric is not None
    return Halt(
        "worse_than_chance",
        mode,
        last.step,
        last.kind,
        f"{last.metric or 'the metric'} was {_fmt(last.metric_value)}{last.unit} over the last "
        f"{PERSISTENCE} {last.kind}s, clearly worse than the {_fmt(last.chance_metric)}"
        f"{last.unit} a trivial predictor scores ({last.describe()}).",
        "A model systematically worse than guessing has the task backwards: check that "
        "labels line up with inputs, that predictions are read from the right axis, and "
        "the sign of the loss.",
    )


def _stuck_margin(r: ProgressRecord) -> float:
    """How far past chance a metric must be to count as meaningfully learning."""
    assert r.chance_metric is not None
    perfect = perfect_value(r)
    span = abs(perfect - r.chance_metric) if perfect is not None else abs(r.chance_metric)
    se = proportion_se(r) or 0.0
    return max(SE_MULTIPLIER * se, STUCK_FRACTION * span)


def rule_no_progress(records: list[ProgressRecord], mode: str) -> Halt | None:
    """The loss has not moved over the opening window AND the model is not beating chance.

    A flat loss alone is not enough: a model can sharpen its predictions while its
    averaged loss stays put to four digits. Only when the metric also shows no real
    gain over a trivial predictor (or there is no metric to tell) does it halt.
    """
    epochs = [
        r
        for r in records
        if r.kind == "epoch" and r.loss_value is not None and math.isfinite(r.loss_value)
    ]
    if len(epochs) < WARMUP_MIN_STEPS:
        return None
    window = warmup_step(epochs[0])
    opening = [r for r in epochs if r.step <= window]
    if len(opening) < WARMUP_MIN_STEPS or epochs[-1].step < window:
        return None
    losses = [r.loss_value for r in opening if r.loss_value is not None]
    spread = max(losses) - min(losses)
    scale = max(abs(losses[0]), 1e-8)
    if spread / scale > NO_PROGRESS_RELATIVE_RANGE:
        return None
    last = opening[-1]
    judged = _judged([last])
    if judged:
        record = judged[0]
        assert record.metric_value is not None and record.chance_metric is not None
        gain = advantage(record.metric_value, record.chance_metric, record.higher_is_better)
        if gain >= _stuck_margin(record):
            return None
    return Halt(
        "no_progress",
        mode,
        last.step,
        last.kind,
        f"the loss stayed between {_fmt(min(losses))} and {_fmt(max(losses))} over the first "
        f"{len(opening)} {last.kind}s - training is not changing the model "
        f"({last.describe()}).",
        "Check that gradients reach the parameters (no detached tensors or frozen modules), "
        "that the optimizer steps the model's own parameters, and that the loss actually "
        "depends on the model's output.",
    )


def rule_stuck_at_chance(records: list[ProgressRecord], mode: str) -> Halt | None:
    """After warmup, still not meaningfully better than a trivial predictor."""
    judged = _judged(records)
    if not judged:
        return None
    eligible = [r for r in judged if r.step >= warmup_step(r)]

    margin = _stuck_margin

    def stuck(r: ProgressRecord) -> bool:
        assert r.metric_value is not None and r.chance_metric is not None
        if r.target_value is not None and math.isfinite(r.target_value):
            # A claim that itself sits near chance cannot be judged by this rule.
            target_gain = advantage(r.target_value, r.chance_metric, r.higher_is_better)
            if target_gain < 2 * margin(r):
                return False
        return advantage(r.metric_value, r.chance_metric, r.higher_is_better) < margin(r)

    if not _persistent(eligible, stuck):
        return None
    last = eligible[-1]
    assert last.metric_value is not None and last.chance_metric is not None
    return Halt(
        "stuck_at_chance",
        mode,
        last.step,
        last.kind,
        f"after {last.step} {last.kind}s {last.metric or 'the metric'} is "
        f"{_fmt(last.metric_value)}{last.unit}, not meaningfully better than the "
        f"{_fmt(last.chance_metric)}{last.unit} of a trivial predictor ({last.describe()}).",
        "The model is not learning beyond chance. Look first at settings the paper leaves "
        "unstated that control stability and capacity - step size, momentum, "
        "initialization, loss constants - and at activations that have gone dead.",
    )


def rule_diverged(records: list[ProgressRecord], mode: str) -> Halt | None:
    """Learned, then fell apart: loss well above its best AND most gains lost."""
    judged = [
        r
        for r in _judged(records)
        if r.kind == "epoch" and r.loss_value is not None and math.isfinite(r.loss_value)
    ]
    if len(judged) < PERSISTENCE + 1:
        return None
    flags: list[bool] = []
    best_loss = math.inf
    best_gain = -math.inf
    best_value: float | None = None
    for r in judged:
        assert r.loss_value is not None and r.metric_value is not None
        assert r.chance_metric is not None
        gain = advantage(r.metric_value, r.chance_metric, r.higher_is_better)
        diverged = (
            best_loss > 0
            and math.isfinite(best_loss)
            and best_gain > 0
            and r.loss_value > DIVERGE_LOSS_FACTOR * best_loss
            and gain < DIVERGE_ADVANTAGE_KEPT * best_gain
        )
        flags.append(diverged)
        if r.loss_value < best_loss:
            best_loss = r.loss_value
        if gain > best_gain:
            best_gain, best_value = gain, r.metric_value
    if len(flags) < PERSISTENCE or not all(flags[-PERSISTENCE:]):
        return None
    last = judged[-1]
    assert last.loss_value is not None and last.metric_value is not None
    return Halt(
        "diverged",
        mode,
        last.step,
        last.kind,
        f"the loss rose to {_fmt(last.loss_value)} (best {_fmt(best_loss)}) and "
        f"{last.metric or 'the metric'} fell back to {_fmt(last.metric_value)}{last.unit} "
        f"(best {_fmt(best_value) if best_value is not None else '?'}{last.unit}) for "
        f"{PERSISTENCE} {last.kind}s running ({last.describe()}).",
        "Training learned and then became unstable. The usual cause is a step that is too "
        "aggressive for this loss - learning rate, momentum or a loss-scaling constant - "
        "so change unstated settings before stated ones.",
    )


def rule_no_first_record(
    records: list[ProgressRecord],
    mode: str,
    silent_seconds: float,
    first_record_seconds: dict[str, float] | None = None,
) -> Halt | None:
    """Not one record yet, long after this stage should have reported something.

    The only rule that judges silence rather than a curve, because every other rule
    needs a record and a script stuck inside its first fit writes none - the real
    2026-09-20 Fashion-MNIST SVC run was invisible to all six of them. Five things
    keep it from killing a slow-but-honest run: the first record disarms it for the
    rest of the stage (this is about never starting, never about slowing down); it
    only ever sees a stage that is STILL RUNNING, since the caller passes a clock
    only while the container is alive; it is armed on `capped` and `full` only; its
    deadline sits multiples above the longest first unit of work any healthy run has
    needed; and it is given SILENCE rather than elapsed time, so a dataset download
    does not count against it (`SilenceClock`, DEFAULT_FIRST_RECORD_SECONDS).
    """
    if records:
        return None
    table = DEFAULT_FIRST_RECORD_SECONDS if first_record_seconds is None else first_record_seconds
    deadline = table.get(mode)
    if deadline is None or silent_seconds < deadline:
        return None
    return Halt(
        "no_first_record",
        mode,
        int(silent_seconds),
        "second",
        f"nothing happened for {int(silent_seconds)}s of this stage - no progress record, "
        f"and no growth in the shared dataset cache either (its deadline is "
        f"{int(deadline)}s): the script never finished a single epoch, repetition or fit, "
        f"so there is no learning curve to judge at all.",
        "Nothing can be diagnosed from a run that never reports, so the first unit of work "
        "has to become observable. Check that the stage's own data and epoch caps really "
        "apply before the first fit (a capped stage quietly training on the whole dataset "
        "looks exactly like this), that the dataset is read from --data-dir instead of being "
        "downloaded again, and that progress is recorded at the finest unit the method "
        "allows - when one indivisible fit costs more than the whole stage, the script must "
        "report from inside it rather than only after it.",
    )


# Every RuleName belongs to exactly one of these two registries: RULES judges a
# curve, TIMED_RULES judges how long the stage has been silent.
RULES: Final[dict[RuleName, Callable[[list[ProgressRecord], str], Halt | None]]] = {
    "non_finite": rule_non_finite,
    "loss_below_lower_bound": rule_loss_below_lower_bound,
    "worse_than_chance": rule_worse_than_chance,
    "no_progress": rule_no_progress,
    "stuck_at_chance": rule_stuck_at_chance,
    "diverged": rule_diverged,
}

type TimedRule = Callable[[list[ProgressRecord], str, float, dict[str, float] | None], Halt | None]

TIMED_RULES: Final[dict[RuleName, TimedRule]] = {
    "no_first_record": rule_no_first_record,
}


def evaluate(
    records: list[ProgressRecord],
    mode: str,
    silent_seconds: float | None = None,
    first_record_seconds: dict[str, float] | None = None,
) -> Halt | None:
    """Run the rules this stage allows, in order; the first that fires wins.

    `silent_seconds` is how long the stage has shown no sign of life at all (see
    `SilenceClock`), and the caller passes it only while the container is still
    alive. Without it the rules that judge the clock are skipped, which is what
    keeps them out of the post-exit pass: once a stage has finished, "it never
    reported anything" is no longer worth halting - a probe may legitimately exit
    before its first epoch, and the exit code and logs are the evidence from then on.
    """
    for name in STAGE_RULES.get(mode, ALL_RULES):
        timed = TIMED_RULES.get(name)
        if timed is not None:
            if silent_seconds is None:
                continue
            halt = timed(records, mode, silent_seconds, first_record_seconds)
        else:
            if not records:
                continue
            halt = RULES[name](records, mode)
        if halt is not None:
            return halt
    return None
