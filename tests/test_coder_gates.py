"""The Coder's claim gates, replayed from Breiman's Random Forests (2026-10-22).

The Reader's claims call hit max_tokens and returned no claims; the Coder then targeted
an invented `c_zipcode_forestRI`, and the Critic could only answer `not_evaluated`.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from anthropic import Anthropic

from coder.base import CodeWriter
from coder.pipeline import ClaimTargetError, CoderPipeline, NoClaimsError
from coder.script_writer import PROGRESS_MARKER, TrainingScript

PAPER = "2001-10 - Random Forests"
# Passes the syntax and progress gates, so only the claim gate can reject it.
SCRIPT = f'print("{PROGRESS_MARKER}")  # metrics.full.history.jsonl\n'


class StubWriter(CodeWriter[TrainingScript]):
    name: ClassVar[str] = "stub"

    def __init__(self, claim_targeted: str) -> None:
        self.claim_targeted = claim_targeted
        self.calls = 0

    def write(
        self,
        reader_output: dict[str, Any],
        paper_markdown: str,
        client: Anthropic,
        feedback: str | None = None,
    ) -> TrainingScript:
        self.calls += 1
        return TrainingScript(
            claim_targeted=self.claim_targeted,
            claim_selection_reasoning="",
            task_type="classification",
            model_family="classical",
            architecture_used="",
            dataset_used="",
            hyperparameters_used=[],
            assumptions=[],
            cli_flags_included=[],
            script_content=SCRIPT,
        )


def run(tmp_path: Path, claims: list[dict[str, Any]], writer: StubWriter) -> None:
    reader_json = tmp_path / f"{PAPER}.json"
    reader_json.write_text(json.dumps({"claims": {"claims": claims}}))
    markdown = tmp_path / f"{PAPER}.md"
    markdown.write_text("# Random Forests")
    client = Anthropic(api_key="unused")
    CoderPipeline(writer=writer).run(reader_json, markdown, tmp_path / "out", client)


def test_no_claims_stops_before_any_api_call(tmp_path: Path) -> None:
    writer = StubWriter("c1")
    with pytest.raises(NoClaimsError):
        run(tmp_path, [], writer)
    assert writer.calls == 0


def test_an_invented_claim_id_is_rejected(tmp_path: Path) -> None:
    claims = [{"claim_id": "c1", "metric": "test set error", "dataset": "zip-code"}]
    with pytest.raises(ClaimTargetError, match="c_zipcode_forestRI"):
        run(tmp_path, claims, StubWriter("c_zipcode_forestRI"))
    assert not (tmp_path / "out" / PAPER / "train.py").exists()


def test_a_real_claim_id_passes(tmp_path: Path) -> None:
    claims = [{"claim_id": "c1", "metric": "test set error", "dataset": "zip-code"}]
    run(tmp_path, claims, StubWriter("c1"))
    assert (tmp_path / "out" / PAPER / "train.py").exists()
