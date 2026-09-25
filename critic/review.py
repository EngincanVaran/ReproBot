"""Critic v2: an LLM review of the script and the run, behind deterministic guards.

The arithmetic verdict in `judge.py` says WHETHER the number matches. It cannot say
WHY not, or whether a passing script is even the paper's method. This module adds
the half of project plan §2.5 that needs a model: one Claude call (Opus 5 by
default, see MODEL below) that reads the
paper's extraction, the paper itself, the final script and the run's evidence (the
learning curve, the metrics, extra seeds) and returns

* **findings** - what the script does against what the paper states, each one
  classified (`matches_paper`, `deviates_from_paper`, `unstated_choice`,
  `implementation_bug`) and cited: a quote from the paper and the script lines;
* **hypotheses** - on a result that misses its claim, concrete ranked explanations
  with the change each implies (PaperBench's "Code Development" dimension, and the
  AutoReproduce-style "paper uses step decay at 82/123, script uses cosine" hint);
* a `method_fidelity` rating and a short curve assessment.

The model never decides the verdict. Its output passes three deterministic checks
before anything reaches the Coder:

1. **Numbers.** Every number in its prose must appear in the material it was given
   (the facts, the paper, the extraction or the script), up to the rounding it was
   written with. Integers up to 10 are exempt ("three branches", "2 hidden layers").
2. **Quotes.** A `paper_quote` must occur in the paper or its extraction (case and
   whitespace ignored; "..." may join fragments), unless it says "not stated".
3. **Script references.** Line numbers must exist and the snippet must occur in the
   script.

An item that fails a check is kept in the record with its problems listed, but is
marked unverified and never fed back to the Coder.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from critic.judge import Judgement
from reader.tooluse import as_list, recover_leaked_fields, request_tool_use

if TYPE_CHECKING:
    from anthropic import Anthropic

# Opus 5, not Sonnet 5, after a controlled test on 2026-09-14: on the same inputs (a Wijaya
# script with its train/validation unpacking swapped, so the model trained on 81 rows
# instead of 324) Sonnet 5 missed the bug in three reviews - even with the script's own log
# saying "81 train_fit, 324 val" - while Opus 5 found it, cited the log and lines 178-180,
# and proposed the exact fix. One call per judged full run; `--review-model` overrides.
MODEL = "claude-opus-5"
MAX_TOKENS = 16000
LOG = "critic:review"

FINDING_KINDS: tuple[str, ...] = (
    "matches_paper",
    "deviates_from_paper",
    "unstated_choice",
    "implementation_bug",
)
CHANGE_TYPES: tuple[str, ...] = (
    "fix_stated_deviation",
    "fix_implementation_bug",
    "revisit_unstated_choice",
    "run_more",
)
# Kinds that mean "the script is not the paper's method" - a correctness problem,
# fixable by a normal retry, never a tuning question.
STATED_PROBLEM_KINDS: frozenset[str] = frozenset({"deviates_from_paper", "implementation_bug"})
SMALL_INTEGER_LIMIT = 10
HISTORY_SAMPLES = 12
LOG_HEAD_LINES = 60
LOG_TAIL_LINES = 20
LOG_MAX_CHARS = 8000

REVIEW_TOOL: dict[str, Any] = {
    "name": "review_replication",
    "description": "Report an evidence-cited review of a replication script and its run.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "At most 120 words: is this the paper's method, and why the "
                "result lands where it does.",
            },
            "method_fidelity": {
                "type": "string",
                "enum": ["faithful", "minor_deviations", "major_deviations"],
            },
            "curve_assessment": {
                "type": "string",
                "description": "What the learning curve and metrics show (converged, "
                "overfitting, plateau above the claim, diverging, ...), citing numbers "
                "from RUN EVIDENCE only.",
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(FINDING_KINDS)},
                        "aspect": {
                            "type": "string",
                            "description": "e.g. architecture, loss, optimizer, learning "
                            "rate, epochs, batch size, data split, preprocessing, "
                            "initialisation, evaluation",
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "paper_quote": {
                            "type": "string",
                            "description": "An EXACT quote of at most 30 words from PAPER or "
                            "READER EXTRACTION supporting this finding, or 'not stated'.",
                        },
                        "script_lines": {"type": "array", "items": {"type": "integer"}},
                        "script_snippet": {
                            "type": "string",
                            "description": "EXACT code copied from those script lines.",
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "kind",
                        "aspect",
                        "severity",
                        "paper_quote",
                        "script_lines",
                        "script_snippet",
                        "explanation",
                    ],
                },
            },
            "hypotheses": {
                "type": "array",
                "description": "Ranked, most likely first. Required when the verdict is "
                "fail or inconclusive; may be empty on a pass.",
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "evidence": {"type": "string"},
                        "change": {
                            "type": "string",
                            "description": "The concrete change to the script it implies.",
                        },
                        "change_type": {"type": "string", "enum": list(CHANGE_TYPES)},
                    },
                    "required": ["hypothesis", "evidence", "change", "change_type"],
                },
            },
        },
        "required": ["summary", "method_fidelity", "curve_assessment", "findings", "hypotheses"],
    },
}

PROMPT = """You are the reviewing half of the Critic in ReproBot, a system that reproduces \
ML papers. A generated training script has run in a sandbox, and its result has already \
been judged against the paper's claim by arithmetic (VERDICT FACTS). You do not decide or \
restate that verdict. Your job is to explain it and to check the script against the paper.

RULES
1. Compare the SCRIPT with what the PAPER and READER EXTRACTION state: architecture, loss, \
optimizer, learning rate and schedule, epochs, batch size, data source, split, \
preprocessing, initialisation, regularisation and how the metric is evaluated. Report the \
important aspects as findings, including the ones that match.
2. TRACE THE DATA, line by line, before judging anything else. Which arrays does the model \
train on, which does it monitor, which produce the final metric, and how many rows does \
each hold? Compare those sizes with the paper's stated split, with the sample counts in \
RUN EVIDENCE and with the sizes the script itself logged in RUNNER LOG. Check the unpacking \
order of every split call (train_test_split returns train before test for each array), and \
check that preprocessing is fitted on training data only. A model trained on the wrong \
subset is an implementation_bug even when it runs well.
3. Classify each finding:
   - matches_paper: the paper states it and the script does it.
   - deviates_from_paper: the paper states X and the script does something else.
   - unstated_choice: the paper is silent and the script had to choose. Disclosed guesses \
belong here, even framework-default translations.
   - implementation_bug: wrong whatever the paper says - a loss optimised in the wrong \
direction, test data leaking into training, the metric computed on the wrong split, a \
literal implementation of a misprinted equation that cannot be minimised. Behaviour the \
paper itself specifies is never a bug: training for the stated number of epochs without \
early stopping, when the paper does exactly that, is correct even if the curve overfits.
4. Cite every finding: paper_quote is an EXACT quote (at most 30 words) copied from PAPER or \
READER EXTRACTION, or the words "not stated". script_lines are line numbers from SCRIPT and \
script_snippet is code copied exactly from those lines. Uncited findings are discarded.
5. Use the learning curve and metrics in RUN EVIDENCE. A curve that stalls above the claim, \
overfits, diverges or never beats chance is evidence about which hypothesis is right.
6. When the verdict is fail or inconclusive, give ranked hypotheses for the gap, each with \
the concrete script change it implies. Prefer stated deviations and bugs over unstated \
choices. Never propose a change whose only justification is moving the test number, never \
propose changing a value the paper states unless the script got it wrong, and never \
propose adding a technique the paper does not use (early stopping, checkpoint selection, \
extra regularisation) as a fix_stated_deviation or fix_implementation_bug.
7. Only use numbers that appear in the material below. Do not compute new ratios or \
percentages; the facts already include the gap and the relative gap.
8. Severity is about the likely effect on the reproduced result: high can plausibly \
explain the gap on its own, low is cosmetic.

--- VERDICT FACTS ---
{verdict}

--- RUN EVIDENCE ---
{evidence}

--- RUNNER LOG (the full stage's own output, per-epoch progress lines removed) ---
{log}

--- CODER BOOKKEEPING (what the Coder says it used and guessed) ---
{bookkeeping}

--- READER EXTRACTION ---
{extraction}

--- SCRIPT (line-numbered) ---
{script}

--- PAPER ---
{paper}
"""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    kind: str
    aspect: str
    severity: str
    paper_quote: str
    script_lines: list[int]
    script_snippet: str
    explanation: str
    verified: bool = True
    problems: list[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    rank: int
    hypothesis: str
    evidence: str
    change: str
    change_type: str
    verified: bool = True
    problems: list[str] = field(default_factory=list)


@dataclass
class Review:
    method_fidelity: str
    summary: str
    curve_assessment: str
    findings: list[Finding]
    hypotheses: list[Hypothesis]
    summary_problems: list[str] = field(default_factory=list)
    model: str = MODEL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stated_problems(self) -> list[Finding]:
        """Verified deviations and bugs that could matter: correctness, not tuning."""
        return [
            f
            for f in self.findings
            if f.verified and f.kind in STATED_PROBLEM_KINDS and f.severity in ("high", "medium")
        ]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = dict.fromkeys(FINDING_KINDS, 0)
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        counts["unverified"] = sum(not f.verified for f in self.findings) + sum(
            not h.verified for h in self.hypotheses
        )
        return counts


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.5g}")
    return value


def summarise_history(path: Path) -> dict[str, Any]:
    """A compact learning curve: evenly spaced records, the best eval point, the last."""
    if not path.exists():
        return {"available": False, "reason": f"no history file at {path.name}"}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    if not records:
        return {"available": False, "reason": f"{path.name} holds no records"}
    keys = ("step", "train_loss", "eval_loss", "train_metric", "eval_metric")
    last = records[-1]
    counts = {
        k: last.get(k)
        for k in ("num_train_samples", "num_eval_samples")
        if isinstance(last.get(k), int)
    }
    stride = max(1, len(records) // HISTORY_SAMPLES)
    sampled = records[::stride]
    if sampled[-1] is not records[-1]:
        sampled.append(records[-1])
    hib = last.get("higher_is_better")
    scored = [r for r in records if isinstance(r.get("eval_metric"), int | float)]
    best = None
    if scored and isinstance(hib, bool):
        best = (max if hib else min)(scored, key=lambda r: float(r["eval_metric"]))
    return {
        "available": True,
        "kind": last.get("kind"),
        "records": len(records),
        "steps_total": last.get("steps_total"),
        "metric": last.get("metric"),
        "higher_is_better": hib,
        "chance_metric": _round(last.get("chance_metric")),
        "target_value": _round(last.get("target_value")),
        "monitoring_split_sample_counts": counts,
        "note": "eval_metric in this curve is the script's own monitoring split; the final "
        "reproduced value in VERDICT FACTS is computed on the held-out evaluation set.",
        "sampled": [{k: _round(r.get(k)) for k in keys if k in r} for r in sampled],
        "best_eval": {k: _round(best.get(k)) for k in keys if k in best} if best else None,
    }


def build_facts(
    judgement: Judgement,
    metrics: dict[str, Any] | None,
    history: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The verdict facts and the run evidence the reviewer is given."""
    verdict = {
        k: _round(v)
        for k, v in judgement.to_dict().items()
        if k
        in (
            "verdict",
            "reason",
            "recommendation",
            "claim_id",
            "metric",
            "dataset",
            "unit",
            "claimed",
            "reproduced",
            "run_values",
            "higher_is_better",
            "gap",
            "relative_gap",
            "tolerance",
            "evidence",
            "exceeds_claim",
            "merged_claim_ids",
        )
    }
    verdict["run_values"] = [_round(v) for v in judgement.run_values]
    if judgement.relative_gap is not None:
        verdict["relative_gap_percent"] = _round(100.0 * judgement.relative_gap)
    metric_keys = (
        "value",
        "train_metric",
        "eval_metric",
        "train_loss",
        "eval_loss",
        "epochs_completed",
        "num_train_samples",
        "num_eval_samples",
        "num_runs",
        "run_values",
        "wall_clock_seconds",
    )
    evidence = {
        "final_metrics": {k: _round(v) for k, v in (metrics or {}).items() if k in metric_keys},
        "learning_curve": history,
    }
    return verdict, evidence


_PROGRESS_LINE = re.compile(r"REPROBOT_PROGRESS|\bEpoch\s+\d+\s*/\s*\d+", re.IGNORECASE)


def log_excerpt(logs_dir: Path | None, mode: str = "full") -> str:
    """The stage's own log lines (what the script said it did), without the per-epoch noise.

    Added after the first fix loop: the review missed a swapped train/validation split
    that the script's own log stated plainly ("81 train_fit, 324 val").
    """
    if logs_dir is None:
        return "(no runner log available)"
    lines: list[str] = []
    for stream in ("stderr", "stdout"):
        path = logs_dir / f"{mode}.{stream}.log"
        if path.is_file():
            kept = [
                line
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not _PROGRESS_LINE.search(line)
            ]
            lines += [f"[{stream}] {line}" for line in kept]
    if not lines:
        return "(no runner log available)"
    if len(lines) > LOG_HEAD_LINES + LOG_TAIL_LINES:
        omitted = len(lines) - LOG_HEAD_LINES - LOG_TAIL_LINES
        lines = [
            *lines[:LOG_HEAD_LINES],
            f"... {omitted} lines omitted ...",
            *lines[-LOG_TAIL_LINES:],
        ]
    return "\n".join(lines)[:LOG_MAX_CHARS]


def numbered(script: str) -> str:
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(script.splitlines(), start=1))


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


def _numbers_in(text: str) -> list[str]:
    return [m.group(0).replace(",", "").lstrip("+") for m in _NUMBER.finditer(text)]


def context_numbers(*texts: str) -> list[float]:
    values: set[float] = set()
    for text in texts:
        for token in _numbers_in(text):
            try:
                values.add(float(token))
            except ValueError:
                continue
    return sorted(values)


def _decimals(token: str) -> int:
    mantissa = token.lower().split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def _supported(token: str, known: list[float]) -> bool:
    try:
        value = float(token)
    except ValueError:
        return True
    if value.is_integer() and abs(value) <= SMALL_INTEGER_LIMIT and "." not in token:
        return True
    places = _decimals(token)
    step = 10.0**-places
    return any(abs(round(k, places) - value) <= step * 0.51 or abs(k - value) < 1e-9 for k in known)


_LINE_REFERENCE = re.compile(r"\blines?\b[\s\d,\-–and]*$", re.IGNORECASE)


def unsupported_numbers(text: str, known: list[float], script_lines: int = 0) -> list[str]:
    """Numbers in `text` that no known value explains at the precision they were written.

    "1,000" is read as one thousand, but "[237,239]" as two numbers: a comma-grouped
    token passes if either reading is supported. An integer right after "line"/"lines"
    passes if the script has that line.
    """
    missing = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0).lstrip("+")
        if (
            script_lines
            and raw.isdigit()
            and 1 <= int(raw) <= script_lines
            and _LINE_REFERENCE.search(text[max(0, match.start() - 40) : match.start()])
        ):
            continue
        if _supported(raw.replace(",", ""), known):
            continue
        if "," in raw and all(_supported(part, known) for part in raw.split(",")):
            continue
        missing.append(raw)
    return missing


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[`*_\"'\[\]]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_found(quote: str, sources: str) -> bool:
    fragments = [f.strip(" .,;:") for f in re.split(r"\.\.\.|…", _normalise(quote))]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in sources for f in fragments)


_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*\|\s?")


def snippet_found(snippet: str, normalised_script: str) -> bool:
    """Every quoted code line must occur in the script; the lines need not be adjacent.

    Reviewers stitch non-adjacent lines into one snippet (Opus 5 did on its first
    review) or elide with "..."; each real line still has to exist verbatim.
    """
    lines = [_LINE_NUMBER_PREFIX.sub("", line) for line in snippet.splitlines()]
    fragments = [
        fragment
        for line in lines
        for fragment in (f.strip(" .") for f in re.split(r"\.\.\.|…", _normalise(line)))
        if fragment
    ]
    return bool(fragments) and all(f in normalised_script for f in fragments)


def is_not_stated(quote: str) -> bool:
    return _normalise(quote).startswith(("not stated", "the paper does not state", "unstated"))


# Training techniques a correctness fix must never ADD unless the paper uses them. Seen on
# the first fix loop (2026-09-14): a review filed "trains the stated 1000 epochs with no
# early stopping" as an implementation bug, the citations were real, and the Coder added
# best-checkpoint restoring - moving the script away from the paper.
ADD_ON_TECHNIQUES: dict[str, tuple[str, ...]] = {
    "early stopping": ("early stopping", "early-stopping", "earlystopping", "patience"),
    "checkpoint selection": ("checkpoint", "best epoch", "best-epoch", "best weights",
                             "best model", "best-validation", "best validation", "restore best"),
    "dropout": ("dropout",),
    "weight decay": ("weight decay", "weight_decay", "l2 regularization", "l2 regularisation"),
    "gradient clipping": ("gradient clipping", "clip_grad", "grad clipping"),
    "learning-rate schedule": ("lr scheduler", "learning rate schedule", "learning-rate schedule",
                               "reducelronplateau", "cosine annealing", "step decay"),
    "data augmentation": ("augmentation",),
    "ensembling": ("ensemble",),
}  # fmt: skip
_REMOVAL = re.compile(r"\b(remove|drop|delete|disable|stop using|take out|without adding)\b")
_ABSENCE = re.compile(r"\b(no|without|lacks?|missing|never uses?|does not use)\b")


def foreign_techniques(text: str, paper_sources: str) -> list[str]:
    """Techniques named in `text` that the paper and its extraction never mention."""
    lowered = text.lower()
    foreign = []
    for technique, words in ADD_ON_TECHNIQUES.items():
        if any(w in lowered for w in words) and not any(w in paper_sources for w in words):
            foreign.append(technique)
    return foreign


def verify(
    review: Review,
    *,
    script: str,
    paper_sources: str,
    facts_text: str,
    paper_text: str | None = None,
) -> Review:
    """Mark every finding and hypothesis that fails a deterministic check.

    `paper_sources` (paper + extraction + Coder bookkeeping) backs quotes and numbers.
    `paper_text` - the paper alone - backs the add-on-technique check: the extraction
    and bookkeeping mention techniques as ABSENT ("dropout not mentioned"), which must
    not count as the paper using them.
    """
    paper_only = paper_text if paper_text is not None else paper_sources
    lines = script.splitlines()
    known = context_numbers(facts_text, paper_sources, script)
    count = len(lines)
    normalised_script = _normalise(script)

    for finding in review.findings:
        prose = f"{finding.explanation} {finding.aspect}"
        for token in unsupported_numbers(prose, known, count):
            finding.problems.append(f"number {token} is not in the material given")
        if not is_not_stated(finding.paper_quote) and not quote_found(
            finding.paper_quote, paper_sources
        ):
            finding.problems.append("paper_quote not found in the paper or its extraction")
        if finding.kind in ("matches_paper", "deviates_from_paper") and is_not_stated(
            finding.paper_quote
        ):
            finding.problems.append(f"a {finding.kind} finding needs a quote of what is stated")
        bad = [n for n in finding.script_lines if not 1 <= n <= len(lines)]
        if bad:
            finding.problems.append(f"script lines {bad} do not exist")
        snippet = _normalise(finding.script_snippet)
        if snippet and not snippet_found(finding.script_snippet, normalised_script):
            finding.problems.append("script_snippet not found in the script")
        if not finding.script_lines and not snippet:
            finding.problems.append("no script reference")
        if finding.kind in STATED_PROBLEM_KINDS and _ABSENCE.search(finding.explanation.lower()):
            for technique in foreign_techniques(finding.explanation, paper_only):
                finding.problems.append(
                    f"faults the script for lacking {technique}, which the paper never uses"
                )
        finding.verified = not finding.problems

    for hypothesis in review.hypotheses:
        prose = f"{hypothesis.hypothesis} {hypothesis.evidence} {hypothesis.change}"
        for token in unsupported_numbers(prose, known, count):
            hypothesis.problems.append(f"number {token} is not in the material given")
        if hypothesis.change_type != "run_more" and not _REMOVAL.search(hypothesis.change.lower()):
            for technique in foreign_techniques(hypothesis.change, paper_only):
                hypothesis.problems.append(
                    f"proposes adding {technique}, which the paper never uses"
                )
        hypothesis.verified = not hypothesis.problems

    for token in unsupported_numbers(f"{review.summary} {review.curve_assessment}", known, count):
        review.summary_problems.append(f"number {token} is not in the material given")
    return review


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #


def parse_review(payload: dict[str, object]) -> Review:
    findings = []
    for item in as_list(payload.get("findings")):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        findings.append(
            Finding(
                kind=kind if kind in FINDING_KINDS else "unstated_choice",
                aspect=str(item.get("aspect") or ""),
                severity=str(item.get("severity") or "low"),
                paper_quote=str(item.get("paper_quote") or ""),
                script_lines=[int(str(n)) for n in as_list(item.get("script_lines")) if _is_int(n)],
                script_snippet=str(item.get("script_snippet") or ""),
                explanation=str(item.get("explanation") or ""),
            )
        )
    hypotheses = []
    for rank, item in enumerate(as_list(payload.get("hypotheses")), start=1):
        if not isinstance(item, dict):
            continue
        change_type = str(item.get("change_type") or "")
        hypotheses.append(
            Hypothesis(
                rank=rank,
                hypothesis=str(item.get("hypothesis") or ""),
                evidence=str(item.get("evidence") or ""),
                change=str(item.get("change") or ""),
                change_type=change_type if change_type in CHANGE_TYPES else "run_more",
            )
        )
    return Review(
        method_fidelity=str(payload.get("method_fidelity") or "minor_deviations"),
        summary=str(payload.get("summary") or ""),
        curve_assessment=str(payload.get("curve_assessment") or ""),
        findings=findings,
        hypotheses=hypotheses,
    )


def _is_int(value: object) -> bool:
    try:
        int(str(value))
    except ValueError:
        return False
    return True


def review_run(
    client: Anthropic,
    *,
    judgement: Judgement,
    metrics: dict[str, Any] | None,
    reader_output: dict[str, Any],
    coder_output: dict[str, Any],
    script: str,
    paper_markdown: str,
    history_path: Path,
    logs_dir: Path | None = None,
    model: str = MODEL,
) -> Review:
    """One review of a judged run (Sonnet by default), verified before it is returned."""
    history = summarise_history(history_path)
    log_text = log_excerpt(logs_dir)
    verdict, evidence = build_facts(judgement, metrics, history)
    bookkeeping = {
        k: coder_output.get(k)
        for k in (
            "claim_targeted",
            "task_type",
            "model_family",
            "architecture_used",
            "dataset_used",
            "hyperparameters_used",
            "assumptions",
        )
    }
    extraction = {
        k: reader_output.get(k)
        for k in ("method_summary", "architecture_notes", "hyperparameters", "data_pipeline")
    }
    verdict_text = json.dumps(verdict, indent=2)
    evidence_text = json.dumps(evidence, indent=2)
    bookkeeping_text = json.dumps(bookkeeping, indent=2)
    extraction_text = json.dumps(extraction, indent=2)
    user_content = PROMPT.format(
        verdict=verdict_text,
        evidence=evidence_text,
        log=log_text,
        bookkeeping=bookkeeping_text,
        extraction=extraction_text,
        script=numbered(script),
        paper=paper_markdown,
    )
    logger.info(
        f"  [{LOG}] reviewing with {model}: verdict {judgement.verdict}, script "
        f"{len(script.splitlines())} lines, paper {len(paper_markdown)} chars, curve "
        f"{history.get('records', 0)} records, log {len(log_text)} chars"
    )
    payload = request_tool_use(
        client,
        log_prefix=LOG,
        model=model,
        max_tokens=MAX_TOKENS,
        tool=REVIEW_TOOL,
        user_content=user_content,
        required_keys=("summary", "method_fidelity", "findings"),
        may_be_empty_keys=("hypotheses",),
    )
    payload = recover_leaked_fields(payload, list(REVIEW_TOOL["input_schema"]["required"]), LOG)
    parsed = parse_review(payload)
    parsed.model = model
    review = verify(
        parsed,
        script=script,
        paper_sources=_normalise(f"{paper_markdown}\n{extraction_text}\n{bookkeeping_text}"),
        facts_text=f"{verdict_text}\n{evidence_text}\n{log_text}",
        paper_text=_normalise(paper_markdown),
    )
    log_review(review)
    return review


def log_review(review: Review) -> None:
    counts = review.counts()
    logger.info(
        f"  [{LOG}] method fidelity: {review.method_fidelity} - "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    logger.info(f"  [{LOG}] summary: {review.summary}")
    logger.info(f"  [{LOG}] curve: {review.curve_assessment}")
    for finding in review.findings:
        if finding.kind == "matches_paper" and finding.verified:
            continue
        mark = "ok" if finding.verified else "UNVERIFIED"
        level = "WARNING" if finding.kind in STATED_PROBLEM_KINDS else "INFO"
        logger.log(
            level,
            f"  [{LOG}] {finding.kind} ({finding.severity}, {mark}) {finding.aspect}: "
            f"{finding.explanation} [lines {finding.script_lines}]",
        )
        for problem in finding.problems:
            logger.warning(f"  [{LOG}]     check failed: {problem}")
    for hypothesis in review.hypotheses:
        mark = "ok" if hypothesis.verified else "UNVERIFIED"
        logger.info(
            f"  [{LOG}] hypothesis {hypothesis.rank} ({hypothesis.change_type}, {mark}): "
            f"{hypothesis.hypothesis} -> {hypothesis.change}"
        )
    for problem in review.summary_problems:
        logger.warning(f"  [{LOG}] summary check failed: {problem}")


# --------------------------------------------------------------------------- #
# Feedback to the Coder
# --------------------------------------------------------------------------- #


def _gap_line(judgement: Judgement) -> str:
    runs = f" (mean of {len(judgement.run_values)} runs)" if len(judgement.run_values) > 1 else ""
    return (
        f"The previous script ran healthily, but its result misses the paper's claim: "
        f"{judgement.metric} {judgement.reproduced:.6g}{judgement.unit}{runs} against the "
        f"claimed {judgement.claimed:.6g}{judgement.unit}, beyond a tolerance of "
        f"{judgement.tolerance:.4g} ({judgement.evidence})."
    )


def deviation_feedback(judgement: Judgement, review: Review) -> str:
    """Feedback for a correctness retry: the script is not what the paper states."""
    problems = "\n".join(
        f"- [{f.kind}, {f.severity}] {f.aspect}: {f.explanation}\n"
        f"  paper: {f.paper_quote}\n"
        f"  script lines {f.script_lines}: {f.script_snippet.strip()}"
        for f in review.stated_problems()
    )
    fixes = "\n".join(
        f"{h.rank}. {h.hypothesis} Change: {h.change}"
        for h in review.hypotheses
        if h.verified and h.change_type in ("fix_stated_deviation", "fix_implementation_bug")
    )
    return (
        f"{_gap_line(judgement)} An independent review of the script against the paper "
        f"found that it does not implement what the paper states:\n{problems}\n"
        + (f"Suggested fixes, most likely first:\n{fixes}\n" if fixes else "")
        + "Make the script do what the paper states for each of these. Keep every other "
        "setting as it was, including your earlier unstated choices. Record each fix as its "
        "own assumptions entry beginning 'deviation fix:', giving the old behaviour, the new "
        "behaviour and the paper text it follows."
    )


def unstated_feedback(judgement: Judgement, review: Review, assumptions: list[str]) -> str:
    """Feedback for the one guided fidelity retry: only unstated choices may change."""
    guesses = "\n".join(f"- {item}" for item in assumptions) or "- (none were recorded)"
    ranked = "\n".join(
        f"{h.rank}. {h.hypothesis} Evidence: {h.evidence} Change: {h.change}"
        for h in review.hypotheses
        if h.verified and h.change_type == "revisit_unstated_choice"
    )
    return (
        f"{_gap_line(judgement)} A review found no stated setting implemented wrongly. "
        + (f"Its ranked hypotheses about unstated choices:\n{ranked}\n" if ranked else "")
        + f"Revisit ONLY settings the paper does not state. The unstated choices you recorded "
        f"last time were:\n{guesses}\n"
        "Never change a value the paper states. Record every setting you change as its own "
        "assumptions entry beginning 'fidelity retry:', giving the old value, the new value "
        "and why. Justify each change from the paper's text, its framework's or library's "
        "defaults, or standard practice for this method - never from wanting the test number "
        "to move."
    )
