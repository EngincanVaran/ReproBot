"""Extract this paper's own per-dataset data pipeline from its VLM Markdown.

Same input contract as `reader/claims.py` and `reader/hyperparameters.py`:
already-extracted Markdown from `ocr.vlm_extract` (`ocr/output/vlm/<paper>.md`),
sent to Claude in one tool-use call. Like hyperparameters, data-pipeline
details are often described in prose (a dedicated per-dataset preprocessing
paragraph, or general "Implementation Details" prose) rather than in a table,
so the prompt looks in both places rather than assuming a table-only source.

Unlike hyperparameters (one flat list, `model_variant`-tagged), each dataset
this paper trains/evaluates on gets its own first-class `DatasetPipeline`
entry - `dataset`, `dataset_source`, `normalization`, `augmentation`,
`split_convention`, `source` - all strings, never floats: this data is
routinely prose ("standard torchvision.datasets.CIFAR10-equivalent", "not
stated, cites Goodfellow et al. [8]"), and papers frequently defer exact
numbers to a cited work without restating them. The prompt explicitly
requires recording that deferral as the value rather than inventing
plausible-sounding numbers that aren't actually in the paper text.

`reference_urls` is paper-level, not per-dataset: GitHub/code-repo/dataset-
download URLs found anywhere in the paper text, collected once.

This module is an importable extraction step, not a standalone script -
`reader/pipeline.py` is the entry point. See `docs/agent-log.md`'s "Research
Agent - scope reader/data_pipeline.py" entry for the approved design this
implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from loguru import logger

from reader.base import Extractor

MODEL = "claude-sonnet-5"

PROMPT = """You are reading the Claude-VLM Markdown transcription of one ML paper. \
Your job is to extract this paper's OWN per-dataset data pipeline as \
structured data - NOT the data pipeline of a baseline/prior-work method it \
compares itself against or a method it merely cites.

Data-pipeline details live in TWO kinds of places in this document, and you \
must check both:
1. A dedicated per-dataset preprocessing description, often one paragraph \
or subsection per dataset (e.g. "4.1 CIFAR-10", "4.2 SVHN"), describing how \
that specific dataset is normalized, augmented, and split.
2. General "Implementation Details" / "Experimental Setup" prose that \
applies across datasets, or applies to one dataset only but isn't under a \
dataset-specific heading.

Steps:
1. Scan the whole document for every distinct dataset this paper trains or \
evaluates its OWN method on (e.g. CIFAR-10, CIFAR-100, SVHN, MNIST, \
ImageNet). For each dataset, examine both kinds of sources above; record one \
entry in `datasets_examined` per dataset describing where you looked and the \
transcribed page number(s) (the nearest preceding "## Page N" heading), e.g. \
"CIFAR-10: Section 4.2 preprocessing paragraph (transcribed page 5)" or \
"MNIST: no dedicated preprocessing paragraph found, checked Implementation \
Details (transcribed page 4) and Section 4.5 (transcribed page 7)".
2. For each dataset, extract:
   - `dataset_source`: how the dataset is obtained/loaded, e.g. "standard \
CIFAR-10 (60,000 32x32 color images, 10 classes)" or, if the paper gives no \
loader/library detail, note that plainly, e.g. "not stated - paper does not \
name a specific loader or library".
   - `normalization`: the paper's own stated normalization/whitening for \
this dataset (e.g. per-pixel mean subtraction, ZCA whitening, exact \
formula/values if given).
   - `augmentation`: the paper's own stated data augmentation for this \
dataset (e.g. random crop with padding, horizontal flip) - state "none" if \
the paper explicitly says no augmentation was used for this dataset, or \
"not stated" if the paper simply never says.
   - `split_convention`: how train/validation/test are split for this \
dataset (e.g. "standard 50,000/10,000 train/test split", "10,000 images \
from the training set held out as validation").
   - `source`: exactly which section/paragraph and transcribed page this \
dataset's pipeline info came from, e.g. "Section 4.2 CIFAR-10, transcribed \
page 5".
3. CRITICAL - do NOT invent or guess plausible-sounding numbers, formulas, \
or library names that are not actually present in the paper text. If the \
paper defers a detail to a cited work without giving the actual numbers or \
formula itself (e.g. "we follow the same split as [8]", "for details of the \
augmentation see [12]"), record that fact verbatim as the field's value, \
e.g. "not stated in this paper; cites [8] for exact parameters" - do not \
fill in what you think [8] probably says. Every field is a string; \
"not stated" (optionally with a citation) is always a valid, and often \
correct, value.
4. Do NOT extract data-pipeline details that are explicitly attributed to a \
different, cited baseline method this paper compares against, not proposes \
- only this paper's own data pipeline for each dataset it actually trains \
or evaluates on.
5. Separately, scan the WHOLE document (not just dataset sections) for any \
GitHub repository, code-release, or dataset-download URL mentioned in the \
paper's text (e.g. "our code and models are available at \
https://github.com/..."). Collect every distinct URL found into \
`reference_urls`, logging where each was found; return an empty list if \
none are found anywhere in the paper.

Call the `record_data_pipeline` tool with the results."""

DATA_PIPELINE_TOOL: dict[str, Any] = {
    "name": "record_data_pipeline",
    "description": (
        "Record the paper's own per-dataset data pipeline (source, normalization, "
        "augmentation, split convention) extracted from its dataset-specific "
        "preprocessing prose and/or implementation-details prose, plus any GitHub/"
        "dataset-download URLs found anywhere in the paper, and bookkeeping about "
        "what was examined."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "datasets_examined": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One entry per dataset looked at, describing where its pipeline "
                    "info was searched for and the transcribed page(s), e.g. "
                    "'CIFAR-10: Section 4.2 preprocessing paragraph (transcribed "
                    "page 5)'."
                ),
            },
            "datasets": {
                "type": "array",
                "description": (
                    "Only this paper's own data pipeline for each dataset it trains "
                    "or evaluates on, after excluding any baseline/prior-work pipeline "
                    "details."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "e.g. 'CIFAR-10'.",
                        },
                        "dataset_source": {
                            "type": "string",
                            "description": (
                                "How the dataset is obtained/loaded, as confirmed by the "
                                "paper text; note plainly if not stated, e.g. 'not stated - "
                                "paper does not name a specific loader or library'."
                            ),
                        },
                        "normalization": {
                            "type": "string",
                            "description": (
                                "The paper's own stated normalization/whitening for this "
                                "dataset. Never invent numbers not in the paper - use "
                                "'not stated, cites <ref>' when the paper defers to a "
                                "citation without giving the actual values."
                            ),
                        },
                        "augmentation": {
                            "type": "string",
                            "description": (
                                "The paper's own stated data augmentation for this "
                                "dataset, or 'none'/'not stated' as appropriate."
                            ),
                        },
                        "split_convention": {
                            "type": "string",
                            "description": (
                                "How train/validation/test are split for this dataset, "
                                "per the paper's own text."
                            ),
                        },
                        "source": {
                            "type": "string",
                            "description": ("e.g. 'Section 4.2 CIFAR-10, transcribed page 5'."),
                        },
                    },
                    "required": [
                        "dataset",
                        "dataset_source",
                        "normalization",
                        "augmentation",
                        "split_convention",
                        "source",
                    ],
                },
            },
            "reference_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Every distinct GitHub/code-repo/dataset-download URL found "
                    "anywhere in the paper text. Empty list if none are found."
                ),
            },
        },
        "required": ["datasets_examined", "datasets", "reference_urls"],
    },
}


@dataclass
class DatasetPipeline:
    dataset: str
    dataset_source: str
    normalization: str
    augmentation: str
    split_convention: str
    source: str


@dataclass
class DataPipelineExtraction:
    reference_urls: list[str]
    datasets: list[DatasetPipeline]


def _tool_input(message: Message) -> dict[str, object]:
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not call the record_data_pipeline tool")


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _parse_dataset(raw: object) -> DatasetPipeline:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dataset pipeline object, got {raw!r}")
    return DatasetPipeline(
        dataset=str(raw["dataset"]),
        dataset_source=str(raw["dataset_source"]),
        normalization=str(raw["normalization"]),
        augmentation=str(raw["augmentation"]),
        split_convention=str(raw["split_convention"]),
        source=str(raw["source"]),
    )


class DataPipelineExtractor(Extractor[DataPipelineExtraction]):
    """Extracts a paper's own per-dataset data pipeline from its VLM Markdown."""

    name: ClassVar[str] = "data_pipeline"

    def extract(
        self, markdown_text: str, client: Anthropic, feedback: str | None = None
    ) -> DataPipelineExtraction:
        """Send one paper's already-loaded VLM Markdown to Claude and extract its
        own per-dataset data pipeline. No file I/O here - the pipeline owns reading
        input and writing output."""
        prompt = PROMPT
        if feedback:
            prompt = (
                f"{PROMPT}\n\n"
                f"A prior validation pass flagged this specific issue with your "
                f"previous attempt: {feedback}\nAddress it specifically in this "
                f"attempt."
            )

        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            tools=[cast(ToolParam, DATA_PIPELINE_TOOL)],
            tool_choice=cast(ToolChoiceToolParam, {"type": "tool", "name": "record_data_pipeline"}),
            messages=[{"role": "user", "content": f"{prompt}\n\n---\n\n{markdown_text}"}],
        )
        payload = _tool_input(message)

        datasets_examined = [str(entry) for entry in _as_list(payload.get("datasets_examined"))]
        datasets = [_parse_dataset(raw) for raw in _as_list(payload.get("datasets"))]
        reference_urls = [str(url) for url in _as_list(payload.get("reference_urls"))]

        logger.info(f"  [data_pipeline] datasets examined ({len(datasets_examined)}):")
        for entry in datasets_examined:
            logger.info(f"    - {entry}")
        logger.info(f"  [data_pipeline] datasets extracted: {len(datasets)}")
        for ds in datasets:
            logger.info(f"    {ds.dataset} ({ds.source})")
            logger.info(f"      dataset_source: {ds.dataset_source}")
            logger.info(f"      normalization: {ds.normalization}")
            logger.info(f"      augmentation: {ds.augmentation}")
            logger.info(f"      split_convention: {ds.split_convention}")
        logger.info(f"  [data_pipeline] reference URLs found: {len(reference_urls)}")
        for url in reference_urls:
            logger.info(f"    - {url}")

        return DataPipelineExtraction(
            reference_urls=reference_urls,
            datasets=datasets,
        )
