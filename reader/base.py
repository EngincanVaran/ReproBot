"""Abstract base class for reader/ extraction stages.

Every extraction stage (claims, hyperparameters, and future ones like
`data_pipeline`) implements this interface so `reader/pipeline.py`'s
`ReaderPipeline` can run them uniformly - one `extract()` call per stage,
results keyed by `.name`, with an optional `feedback` string on retry after
a validation flag routes back to a specific stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from anthropic import Anthropic


class Extractor[ResultT](ABC):
    """One structured-extraction step over a paper's VLM Markdown."""

    name: ClassVar[str]

    @abstractmethod
    def extract(
        self, markdown_text: str, client: Anthropic, feedback: str | None = None
    ) -> ResultT:
        """Extract structured data from a paper's already-loaded VLM Markdown.

        `feedback` is set on a retry: a validation flag's description (or
        several, concatenated) that a prior validation pass raised against
        this stage's previous output, folded into the prompt so the model
        specifically addresses what was flagged, not a generic re-ask.
        """
        raise NotImplementedError
