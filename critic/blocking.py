"""When a numeric pass may not be reported as a reproduction.

Kept apart from `review.py` and dependency-free, because `report/` recomputes it from a
stored review and must not import the API client.
"""

from __future__ import annotations

from typing import Any

# Kinds that mean "the script is not the paper's method" - a correctness problem,
# fixable by a normal retry, never a tuning question.
STATED_PROBLEM_KINDS: frozenset[str] = frozenset({"deviates_from_paper", "implementation_bug"})


def pass_blocking_reasons(review: dict[str, Any] | None) -> list[str]:
    """Why a numeric pass cannot stand as "reproduced", or [] when it can.

    The verdict is arithmetic and no model changes it; this only decides whether a pass
    may be REPORTED as a reproduction. A matching number from a script that is not the
    paper's method proves nothing about the paper - the first case was Pang & Lee, where
    a regenerated script invented 1,400 keyword-planted "reviews" and scored 92.8% against
    a claimed 82.9%. Deterministic over the stored review, so old state files re-render
    the same way: the review's own `major_deviations` rating, or any VERIFIED (citation-
    checked) high-severity deviation or bug. Medium findings and unverified ones never
    block - GCN's real pass carries a verified medium finding and must stand.
    """
    if not review:
        return []
    reasons = []
    if review.get("method_fidelity") == "major_deviations":
        reasons.append("the review rates the method as having major deviations from the paper")
    for finding in review.get("findings") or []:
        if (
            finding.get("verified", True)
            and finding.get("kind") in STATED_PROBLEM_KINDS
            and finding.get("severity") == "high"
        ):
            reasons.append(
                f"verified high-severity {str(finding.get('kind')).replace('_', ' ')} "
                f"({finding.get('aspect')}): {finding.get('explanation')}"
            )
    return reasons
