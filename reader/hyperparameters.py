"""Extract this paper's own training hyperparameters from its VLM Markdown.

Same input contract as `reader/claims.py`: already-extracted Markdown from
`ocr.vlm_extract` (`ocr/output/vlm/<paper>.md`), sent to Claude in one
tool-use call. Unlike claims, hyperparameters are more often described in
prose (an "Implementation Details" / "Training Details" section) than in a
dedicated table, so the prompt explicitly looks in both places rather than
assuming a table-only source. Baseline/prior-work configurations are
excluded, same as claims - only the paper's own training setup is kept.

`value` is deliberately a **string**, not a float: LR schedules and
optimizer descriptions ("step decay at epochs 82/123", "SGD with Nesterov
momentum 0.9") must survive verbatim rather than collapsing to a bare
number. See `docs/agent-log.md`'s "Research Agent - scope the next Reader
extraction slice" entry for the approved design this implements.

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
Your job is to extract this paper's OWN training hyperparameters as \
structured data - NOT the hyperparameters of a baseline/prior-work method \
it compares itself against or a method it merely cites.

Hyperparameters live in TWO kinds of places in this document, and you must \
check both:
1. A dedicated hyperparameters/training-details TABLE (e.g. captioned \
"Table N: Training hyperparameters" or similar).
2. PROSE sections describing training setup, most often titled something \
like "Implementation Details", "Training Details", "Experimental Setup", \
or "Training Procedure" - these are just as likely to hold the real values \
as any table, sometimes the only place they appear.

Steps:
1. Scan the whole document for both kinds of sources above. For each source \
you examine, record one entry in `sources_examined` describing it and the \
transcribed page number it appears under (the nearest preceding \
"## Page N" heading), e.g. "Table 3: Training hyperparameters (transcribed \
page 6)" or "Implementation Details prose section (transcribed page 4)".
2. Extract every distinct hyperparameter belonging to the paper's OWN \
training configuration: e.g. learning rate (and its full schedule if one is \
described, not just the initial value), batch size, number of epochs, \
optimizer (name and its own parameters like momentum/weight decay), weight \
initialization, data augmentation settings used during training, dropout \
rate, regularization strength, or any other training-setup detail the paper \
states as its own choice.
3. Do NOT extract hyperparameters that are explicitly attributed to a \
different, cited method (a baseline this paper compares against, not \
proposes) - only this paper's own training configuration.
4. If the paper describes more than one own-method training configuration \
(e.g. a base model vs. a model with data augmentation, using different \
epoch counts or schedules), emit separate hyperparameter entries for each, \
distinguished by `model_variant`.
5. CRITICAL: keep `value` as the exact string describing the value, \
verbatim or near-verbatim from the paper - do NOT collapse a schedule or a \
descriptive value down to a single bare number. Examples of correct \
`value` strings: "0.1, divided by 10 at epochs 82 and 123", "0.9 (Nesterov \
momentum)", "5 x 10^-4". A plain numeric value like batch size can still be \
a simple string, e.g. "128".
6. `unit` is optional and only for hyperparameters that have a genuine \
unit separate from their value (e.g. unit "epochs" for a value of "164"); \
omit it when the value string is already self-describing (e.g. an \
optimizer name) or has no natural unit.
7. `source` must cite exactly which table or prose section and transcribed \
page the hyperparameter came from, e.g. "Table 3, transcribed page 6" or \
"Implementation Details section, transcribed page 4".

Call the `record_hyperparameters` tool with the results."""

HYPERPARAMETER_TOOL: dict[str, Any] = {
    "name": "record_hyperparameters",
    "description": (
        "Record the paper's own training hyperparameters extracted from its "
        "hyperparameter table(s) and/or implementation-details prose, plus "
        "bookkeeping about what was examined."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sources_examined": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One entry per table or prose section looked at, e.g. 'Table 3: "
                    "Training hyperparameters (transcribed page 6)' or 'Implementation "
                    "Details prose section (transcribed page 4)'."
                ),
            },
            "hyperparameters": {
                "type": "array",
                "description": (
                    "Only the paper's own training configuration, after excluding "
                    "any baseline/prior-work hyperparameters."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "e.g. 'learning rate', 'batch size', 'optimizer'.",
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "Exact value as a string, verbatim or near-verbatim from "
                                "the paper - never collapse a schedule to a bare number, "
                                "e.g. '0.1, divided by 10 at epochs 82 and 123'."
                            ),
                        },
                        "unit": {
                            "type": "string",
                            "description": (
                                "Optional unit separate from the value string, e.g. "
                                "'epochs'. Omit when not applicable."
                            ),
                        },
                        "source": {
                            "type": "string",
                            "description": (
                                "e.g. 'Table 3, transcribed page 6' or 'Implementation "
                                "Details section, transcribed page 4'."
                            ),
                        },
                        "model_variant": {
                            "type": "string",
                            "description": (
                                "Optional configuration label when the paper describes "
                                "more than one own-method training configuration. Omit "
                                "if there is only one."
                            ),
                        },
                    },
                    "required": ["name", "value", "source"],
                },
            },
        },
        "required": ["sources_examined", "hyperparameters"],
    },
}


@dataclass
class Hyperparameter:
    name: str
    value: str
    source: str
    unit: str | None = None
    model_variant: str | None = None


@dataclass
class HyperparametersExtraction:
    hyperparameters: list[Hyperparameter]
    sources_examined: list[str]


def _resolve_sources_examined(
    *, reported: list[str], hyperparameters: list[Hyperparameter]
) -> list[str]:
    """Return the model's own `sources_examined`, or derive it when it is empty.

    This field is schema-required and the model has never populated it - empty
    on every paper of every run, while the extraction beside it was correct and
    fully source-grounded. Two ways to respond to that: keep insisting in the
    prompt, or notice that the field is *derivable* and stop asking.

    Every `Hyperparameter` already carries its own `source`, so the set of
    places examined is exactly the distinct sources of what was found. Deriving
    it is not a workaround for a flaky model; it is the correct place to compute
    a value that was always a function of data we already had. The one thing the
    derivation cannot know is a source that was searched and yielded nothing -
    hence `reported` still wins whenever the model does supply it.

    Order is stable (first appearance) rather than sorted, so the list reads in
    the order the paper presents them.
    """
    if reported:
        return reported
    seen: dict[str, None] = {}
    for hyperparameter in hyperparameters:
        if hyperparameter.source:
            seen.setdefault(hyperparameter.source, None)
    derived = list(seen)
    if derived:
        logger.info(
            f"  [hparams] model left 'sources_examined' empty (it always does); derived "
            f"{len(derived)} distinct source(s) from the extracted hyperparameters instead"
        )
    return derived


def _parse_hyperparameter(raw: object) -> Hyperparameter:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a hyperparameter object, got {raw!r}")
    unit = raw.get("unit")
    model_variant = raw.get("model_variant")
    return Hyperparameter(
        name=str(raw["name"]),
        value=str(raw["value"]),
        source=str(raw["source"]),
        unit=str(unit) if unit is not None else None,
        model_variant=str(model_variant) if model_variant is not None else None,
    )


class HyperparametersExtractor(Extractor[HyperparametersExtraction]):
    """Extracts a paper's own training hyperparameters from its VLM Markdown."""

    name: ClassVar[str] = "hyperparameters"

    def extract(
        self, markdown_text: str, client: Anthropic, feedback: str | None = None
    ) -> HyperparametersExtraction:
        """Send one paper's already-loaded VLM Markdown to Claude and extract its
        own training hyperparameters. No file I/O here - the pipeline owns reading
        input and writing output."""
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
            log_prefix="hparams",
            model=MODEL,
            max_tokens=8192,
            tool=HYPERPARAMETER_TOOL,
            user_content=f"{prompt}\n\n---\n\n{markdown_text}",
            required_keys=("hyperparameters",),
            # `sources_examined` is schema-required but the model has never once
            # populated it - empty on every paper, every run. Listing it as
            # required here made `tooluse` report a perfectly good extraction as
            # incomplete, which the validator then flagged, which cost a whole
            # retry pass per paper. See the derivation below for why that was
            # the wrong place to insist.
            may_be_empty_keys=("sources_examined",),
        )

        hyperparameters = [
            _parse_hyperparameter(raw) for raw in as_list(payload.get("hyperparameters"))
        ]
        sources_examined = _resolve_sources_examined(
            reported=[str(source) for source in as_list(payload.get("sources_examined"))],
            hyperparameters=hyperparameters,
        )

        logger.info(f"  [hparams] sources examined ({len(sources_examined)}):")
        for source in sources_examined:
            logger.info(f"    - {source}")
        logger.info(f"  [hparams] hyperparameters found: {len(hyperparameters)}")
        for hp in hyperparameters:
            variant = f" [{hp.model_variant}]" if hp.model_variant else ""
            unit = f" {hp.unit}" if hp.unit else ""
            logger.info(f"    {hp.name} = {hp.value}{unit}{variant} ({hp.source})")

        return HyperparametersExtraction(
            hyperparameters=hyperparameters,
            sources_examined=sources_examined,
        )
