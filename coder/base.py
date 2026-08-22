"""Abstract base class for coder/ code-generation stages.

Mirrors `reader/base.py`'s `Extractor` in spirit: every code-generation stage
(currently just the training-script writer; later ones might emit an eval
script, a Dockerfile, or a requirements file) implements `name:
ClassVar[str]` and a single `write()` call, so `coder/pipeline.py` can drive
them uniformly - call `.write()`, key results by `.name`, and on a retry hand
back a `feedback` string.

The two inputs are deliberately separate rather than pre-merged. `reader_output`
is the *authoritative* structured extraction (claims, hyperparameters,
data_pipeline - already validated by `reader/validator.py`); `paper_markdown`
is the raw VLM transcription that fills gaps the Reader has no stage for yet,
architecture description above all. Merging them here would throw away exactly
the precedence information the prompt needs to state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from anthropic import Anthropic


class CodeWriter[ResultT](ABC):
    """One code-generation step over a paper's Reader output plus its Markdown."""

    name: ClassVar[str]

    @abstractmethod
    def write(
        self,
        reader_output: dict[str, Any],
        paper_markdown: str,
        client: Anthropic,
        feedback: str | None = None,
    ) -> ResultT:
        """Generate code from one paper's already-loaded Reader output and Markdown.

        `reader_output` is the parsed `reader/output/<paper>.json` (authoritative
        for claims/hyperparameters/data_pipeline); `paper_markdown` is that same
        paper's `ocr/output/vlm/<paper>.md` (fills gaps the Reader has no
        extraction stage for, architecture especially).

        `feedback` is set on a retry: a description of what went wrong with the
        previous attempt, folded into the prompt so the model specifically
        addresses it rather than being generically re-asked. Nothing passes it
        yet - `coder/pipeline.py` is single-shot today. It exists because the
        planned Coder<->Runner loop will hand back a Runner error trace (a
        traceback from the sandboxed execution, or a metrics mismatch) exactly
        the way `reader/pipeline.py` already routes a validation flag into
        `Extractor.extract(feedback=...)`. Implementations should honour it now
        so that increment is a pipeline change, not a rewrite of every writer.
        """
        raise NotImplementedError
