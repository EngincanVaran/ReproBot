"""Merge duplicate claims before judging against them.

A paper often states one result three times - in the prose, in a table and in a
figure caption - and the Reader extracts each occurrence as its own claim, at
whatever precision that occurrence used. Wijaya 2023 yields 14 claims for 8
distinct results: its test RMSE appears as c8 ("testing set") and c11 ("Proposed
NN, Table 3"), and its test R^2 as 0.911, 0.91 and 0.91 under three different metric
spellings. Judging against the less precise copy would widen the tolerance for no
reason, and a report listing all three would count one result three times.

Two claims are the same result when they share a dataset and a normalised metric
name, their values agree within the coarser of their two reporting precisions, their
variants do not name different splits (train vs test), and their variants do not
name different models. A variant's "model name" is what remains after removing split
words and words that only say where a number was printed ("Proposed NN, Table 3"):
Wijaya's "testing set" and "Proposed NN, Table 3" both reduce to nothing and merge,
while Fashion-MNIST's LinearSVC and LogisticRegression rows - both 0.917 on MNIST -
reduce to different names and stay apart. The group keeps the most precise value as
its canonical claim. Merging never edits the Reader's output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

# Spellings of one metric that papers and the Reader use interchangeably.
METRIC_SYNONYMS: dict[str, str] = {
    "rsquared": "r2",
    "rsquare": "r2",
    "r2": "r2",
    "r2score": "r2",
    "coefficientofdetermination": "r2",
}

_SPLIT_WORDS: dict[str, tuple[str, ...]] = {
    "train": ("train", "training"),
    "test": ("test", "testing"),
    "validation": ("val", "validation", "dev"),
}


def normalise_metric(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return METRIC_SYNONYMS.get(key, key)


def normalise_text(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def decimals_of(value: float) -> int:
    """Decimal places the number was reported with, read from its shortest repr.

    JSON drops trailing zeros (a reported 0.910 arrives as 0.91), so this is a
    lower bound on the paper's precision - the conservative direction.
    """
    exponent = Decimal(repr(float(value))).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def reporting_precision(value: float) -> float:
    """Half a unit of the last reported digit: 96.9 -> 0.05, 0.873 -> 0.0005."""
    return 0.5 * 10.0 ** (-decimals_of(value))


# Words that describe where a number was printed or which split it is on, not which
# model produced it.
_LOCATION_WORDS: frozenset[str] = frozenset(
    {
        "set", "split", "data", "the", "on", "of", "in", "a", "an", "and", "with", "for",
        "proposed", "our", "ours", "model", "method", "nn", "network", "final", "best",
        "table", "figure", "fig", "text", "prose", "section", "page", "appendix",
        "reported", "result", "results", "value", "values", "main",
    }
)  # fmt: skip


# A location reference together with its number: "Table 3", "Figure 6", "Sec. 4.2".
_LOCATION_REFERENCE = re.compile(
    r"\b(?:table|tab|figure|fig|section|sec|page|appendix|eq|equation)\.?\s*[a-z]?\d+(?:\.\d+)*"
)


def model_words(claim: dict[str, Any]) -> frozenset[str]:
    """The variant's words that could name a model, after dropping split/location words.

    Numbers count - "depth 40" and "depth 22" are different networks - except the ones
    that only say where a result was printed.
    """
    variant = _LOCATION_REFERENCE.sub(" ", str(claim.get("model_variant") or "").lower())
    words = re.findall(r"[a-z0-9_]+", variant)
    split_names = {name for names in _SPLIT_WORDS.values() for name in names}
    return frozenset(w for w in words if w not in _LOCATION_WORDS and w not in split_names)


def split_of(claim: dict[str, Any]) -> str | None:
    words = set(re.findall(r"[a-z]+", str(claim.get("model_variant") or "").lower()))
    found = [split for split, names in _SPLIT_WORDS.items() if words & set(names)]
    return found[0] if len(found) == 1 else None


@dataclass
class ClaimGroup:
    """One distinct result, with every claim that restates it."""

    canonical: dict[str, Any]
    members: list[dict[str, Any]] = field(default_factory=list)

    @property
    def claim_ids(self) -> list[str]:
        return [str(member.get("claim_id")) for member in self.members]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_claim_id": self.canonical.get("claim_id"),
            "metric": self.canonical.get("metric"),
            "dataset": self.canonical.get("dataset"),
            "reported_value": self.canonical.get("reported_value"),
            "unit": self.canonical.get("unit"),
            "model_variant": self.canonical.get("model_variant"),
            "merged_claim_ids": self.claim_ids,
        }


def _same_result(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if normalise_metric(str(a.get("metric", ""))) != normalise_metric(str(b.get("metric", ""))):
        return False
    if normalise_text(a.get("dataset")) != normalise_text(b.get("dataset")):
        return False
    if normalise_text(a.get("unit")) != normalise_text(b.get("unit")):
        return False
    va, vb = a.get("reported_value"), b.get("reported_value")
    if not isinstance(va, int | float) or not isinstance(vb, int | float):
        return False
    coarser = max(reporting_precision(float(va)), reporting_precision(float(vb)))
    if abs(float(va) - float(vb)) > coarser + 1e-12:
        return False
    split_a, split_b = split_of(a), split_of(b)
    if split_a and split_b and split_a != split_b:
        return False
    names_a, names_b = model_words(a), model_words(b)
    return not (names_a and names_b and names_a != names_b)


def group_claims(claims: list[dict[str, Any]]) -> list[ClaimGroup]:
    """Group claims that restate the same result; order follows first appearance."""
    groups: list[ClaimGroup] = []
    for claim in claims:
        for group in groups:
            if all(_same_result(claim, member) for member in group.members):
                group.members.append(claim)
                if decimals_of(float(claim.get("reported_value", 0.0))) > decimals_of(
                    float(group.canonical.get("reported_value", 0.0))
                ):
                    group.canonical = claim
                break
        else:
            groups.append(ClaimGroup(canonical=claim, members=[claim]))
    return groups


def group_for(claim_id: str, groups: list[ClaimGroup]) -> ClaimGroup | None:
    for group in groups:
        if claim_id in group.claim_ids:
            return group
    return None
