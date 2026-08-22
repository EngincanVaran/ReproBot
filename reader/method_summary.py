"""Extract a prose summary of this paper's method from its VLM Markdown.

Same input contract as the other `reader/` stages: already-extracted Markdown
from `ocr.vlm_extract` (`ocr/output/vlm/<paper>.md`), sent to Claude in one
tool-use call. This is the project plan's §1.2 `reader_output.method_summary`
field, and it is the only `reader/` stage whose output is prose rather than
records: its consumers are the Coder (which needs to know what it is building
before it reads `architecture_notes`' component list) and the eventual Report
Generator (which needs a human-readable description of the paper under
replication).

`problem` / `core_idea` / `novelty` are broken out as separate fields rather
than folded into `summary` because they are asked of the paper separately -
"what is new here" is a different question from "what does this do", and
keeping them apart stops the paragraph from being a rewrite of the abstract.
`summary` itself is the §1.2 field proper: one coherent paragraph.

**Deliberately narrow.** The prompt forbids result numbers, hyperparameters,
dataset preprocessing, and layer-level architecture specification, because
`claims.py`, `hyperparameters.py`, `data_pipeline.py`, and
`architecture_notes.py` each own one of those. That is not tidiness: every
stage's output is handed to `reader/validator.py` together, and a summary that
restates a number slightly differently from the stage that owns it produces a
cross-check flag about a disagreement that was never a real one.

This module is an importable extraction step, not a standalone script -
`reader/pipeline.py` is the entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from anthropic import Anthropic
from loguru import logger

from reader.base import Extractor
from reader.tooluse import as_list, request_tool_use

MODEL = "claude-sonnet-5"

PROMPT = """You are reading the Claude-VLM Markdown transcription of one ML paper. \
Your job is to write a faithful, self-contained prose summary of the method \
THIS paper proposes - what problem it attacks, what its central idea is, and \
what is new about it relative to the prior work it cites.

The material for this lives in the abstract, the introduction, the \
method/approach sections, and any "related work" contrast the paper draws \
between itself and prior approaches. Record one entry in `sources_examined` \
per section you drew on, with the transcribed page number it appears under \
(the nearest preceding "## Page N" heading), e.g. "Abstract (transcribed \
page 1)", "Section 1 Introduction (transcribed page 1)", "Section 3 Network \
In Network (transcribed page 3)".

Produce:
1. `problem`: the problem or limitation this paper attacks, stated as the \
paper itself frames it - including, where the paper says so, what is wrong \
with how prior work handles it.
2. `core_idea`: the paper's central contribution - the one idea that, if you \
removed it, there would be no paper. State the mechanism, not just the name.
3. `novelty`: what is genuinely NEW here relative to the prior work this \
paper cites, i.e. the contrast the paper itself draws against the specific \
approaches it names. If the paper's novelty is a combination or a \
substitution rather than a wholly new mechanism, say so plainly.
4. `summary`: ONE coherent paragraph (roughly 5-10 sentences) that reads as \
continuous prose, not as a bulleted restatement of the three fields above - \
problem, idea, and why it matters, in a form someone could read once to \
understand what the paper does. This is the headline field; the other three \
support it.

CRITICAL constraints:
- Stay SUMMARY-shaped, and stay in your lane. Other extraction stages own the \
rest of this paper and their output is cross-checked against yours. Do NOT \
include reported result numbers, accuracy/error rates, or any figure from a \
results table (the `claims` extraction owns those). Do NOT include training \
hyperparameters - learning rate, epochs, batch size, optimizer (the \
`hyperparameters` extraction owns those). Do NOT include dataset \
preprocessing, augmentation, or split detail (the `data_pipeline` extraction \
owns those). Do NOT restate the layer-by-layer architecture specification - \
filter counts, kernel sizes, per-block contents (the `architecture_notes` \
extraction owns those). Naming the architecture and describing its central \
mechanism in a sentence is correct and expected; enumerating its dimensions \
is not.
- Do NOT invent motivation, novelty, or significance the paper does not claim \
for itself, and do not import what you know about how this work was later \
received. Summarize what is on the page.
- Describe THIS paper's own method. Prior work appears only as the contrast \
the paper draws against it, never as the subject of the summary.

Call the `record_method_summary` tool with the results."""

METHOD_SUMMARY_TOOL: dict[str, Any] = {
    "name": "record_method_summary",
    "description": (
        "Record a prose summary of the paper's own proposed method - the problem it "
        "attacks, its core idea, its novelty relative to cited prior work, and one "
        "coherent summary paragraph - plus bookkeeping about which sections were read. "
        "No result numbers, hyperparameters, dataset detail, or layer-level "
        "architecture specification: other extraction stages own those."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sources_examined": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One entry per section drawn on, e.g. 'Abstract (transcribed page "
                    "1)', 'Section 3 Network In Network (transcribed page 3)'."
                ),
            },
            "problem": {
                "type": "string",
                "description": (
                    "The problem or limitation this paper attacks, as the paper itself frames it."
                ),
            },
            "core_idea": {
                "type": "string",
                "description": (
                    "The paper's central contribution - the mechanism, not just its name."
                ),
            },
            "novelty": {
                "type": "string",
                "description": (
                    "What is new relative to the prior work this paper cites, i.e. the "
                    "contrast the paper itself draws."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "ONE coherent paragraph (roughly 5-10 sentences) of continuous "
                    "prose covering problem, idea, and why it matters. No result "
                    "numbers, hyperparameters, dataset detail, or layer dimensions."
                ),
            },
        },
        "required": ["sources_examined", "problem", "core_idea", "novelty", "summary"],
    },
}


@dataclass
class MethodSummaryExtraction:
    problem: str
    core_idea: str
    novelty: str
    summary: str
    sources_examined: list[str]


class MethodSummaryExtractor(Extractor[MethodSummaryExtraction]):
    """Extracts a prose summary of a paper's own proposed method from its VLM
    Markdown."""

    name: ClassVar[str] = "method_summary"

    def extract(
        self, markdown_text: str, client: Anthropic, feedback: str | None = None
    ) -> MethodSummaryExtraction:
        """Send one paper's already-loaded VLM Markdown to Claude and summarize its
        own proposed method. No file I/O here - the pipeline owns reading input and
        writing output."""
        prompt = PROMPT
        if feedback:
            prompt = (
                f"{PROMPT}\n\n"
                f"A prior validation pass flagged this specific issue with your "
                f"previous attempt: {feedback}\nAddress it specifically in this "
                f"attempt."
            )

        payload = request_tool_use(
            client,
            log_prefix="method",
            model=MODEL,
            max_tokens=8192,
            tool=METHOD_SUMMARY_TOOL,
            user_content=f"{prompt}\n\n---\n\n{markdown_text}",
            # Every field is required *and* non-empty for a real paper: this is
            # the stage whose all-empty payload was seen live on Wide Residual
            # Networks, where the only tell was a summary of zero words.
            required_keys=("sources_examined", "problem", "core_idea", "novelty", "summary"),
        )

        sources_examined = [str(source) for source in as_list(payload.get("sources_examined"))]
        problem = str(payload.get("problem") or "")
        core_idea = str(payload.get("core_idea") or "")
        novelty = str(payload.get("novelty") or "")
        summary = str(payload.get("summary") or "")

        logger.info(f"  [method] sources examined ({len(sources_examined)}):")
        for source in sources_examined:
            logger.info(f"    - {source}")
        logger.info(f"  [method] problem: {problem}")
        logger.info(f"  [method] core_idea: {core_idea}")
        logger.info(f"  [method] novelty: {novelty}")
        logger.info(f"  [method] summary ({len(summary.split())} words): {summary}")
        if not summary:
            logger.error(
                "  [method] summary is EMPTY - this is the §1.2 field proper, so an "
                "empty value means the extraction failed rather than that there was "
                "nothing to say."
            )

        return MethodSummaryExtraction(
            problem=problem,
            core_idea=core_idea,
            novelty=novelty,
            summary=summary,
            sources_examined=sources_examined,
        )
