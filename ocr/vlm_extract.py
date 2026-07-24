"""Extract Markdown from paper PDFs by sending rendered page images to Claude.

Requires the `vlm` extra: `uv sync --extra vlm`, plus an ANTHROPIC_API_KEY in
`.env` at the repo root (copy `.env.example`). No torch involved — pages are
rendered with pypdfium2 (pure Python, no system deps) and read by Claude's
native vision, one page per API call.

Usage:
    uv run python -m ocr.vlm_extract --input dataset --output ocr/output/vlm
    uv run python -m ocr.vlm_extract --input dataset/some-paper.pdf --max-pages 3
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "claude-sonnet-5"

PROMPT = """Transcribe this page of an academic paper into clean Markdown.

- Preserve headings, paragraph text, and reading order. If the page is laid
  out in two columns, transcribe the entire left column top-to-bottom first,
  then the entire right column top-to-bottom - never interleave lines across
  columns.
- Exclude page furniture: running headers, footers, page numbers, and
  arXiv/journal sidebar stamps (e.g. a vertical arXiv identifier printed in
  the margin). Do not transcribe these. Exception: still transcribe any
  in-text code or data repository URL (e.g. a GitHub/GitLab link) verbatim,
  even if it appears in a header, footer, or footnote.
- Render tables as Markdown tables. This is critical: transcribe EVERY row and
  EVERY column exactly as printed, including headers, row labels, units, and
  footnote markers. Preserve full numeric precision verbatim (e.g. "10.41",
  not "10.4" or "~10"); never round, summarize, abbreviate, merge, or omit any
  cell, row, or column, even if the table is large or dense. Downstream code
  parses exact numbers out of these tables, so a missing or altered digit is
  a real bug. If a cell is illegible, write [illegible] rather than guessing
  or skipping it. Immediately before or after the table, transcribe its
  caption text verbatim (e.g. "Table 1: Test set error rates..."); downstream
  code cites results by this exact caption text, so it must match the printed
  text exactly, not a paraphrase.
- Render math as LaTeX (inline $...$ or block $$...$$).
- Describe figures/diagrams in square brackets, e.g. [Figure 2: ...]. First
  transcribe the figure's caption text verbatim, then add a code-relevant
  description: include only detail relevant to reproducing the paper's
  method (architecture shapes, layer counts, hyperparameter values shown,
  axis/legend meaning), capture every numerical element visible in the
  figure, and cross-reference the caption rather than repeating it.
- Compact the bibliography/references section to one line per entry - author,
  year, and title only. Do not transcribe full reference details (venue,
  pages, DOI), and do not skip the bibliography entirely.
- Output only the transcribed content, no commentary or preamble."""


@dataclass
class OcrResult:
    pdf_path: Path
    markdown_path: Path


def _render_page_png(pdf_path: Path, page_index: int, scale: float) -> bytes:
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        bitmap = pdf[page_index].render(scale=scale)
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        pdf.close()


def _transcribe_page(client: Anthropic, png_bytes: bytes) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(png_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def run_vlm(
    pdf_path: Path,
    output_dir: Path,
    client: Anthropic,
    *,
    scale: float = 2.0,
    max_pages: int | None = None,
) -> OcrResult:
    """Render every page of a PDF and transcribe each one via Claude's vision."""
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{pdf_path.stem}.md"

    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)
    pdf.close()
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    sections: list[str] = []
    for page_index in range(n_pages):
        page_num = page_index + 1
        page_start = time.monotonic()
        png_bytes = _render_page_png(pdf_path, page_index, scale)
        text = _transcribe_page(client, png_bytes)
        elapsed = time.monotonic() - page_start
        has_table = "|" in text
        logger.info(
            f"  [page {page_num}/{n_pages}] {elapsed:.1f}s, "
            f"{len(text)} chars, table={'yes' if has_table else 'no'}"
        )
        sections.append(f"## Page {page_num}\n\n{text}")

    markdown_path.write_text("\n\n".join(sections), encoding="utf-8")
    return OcrResult(pdf_path=pdf_path, markdown_path=markdown_path)


def extract_dataset(
    input_dir: Path, output_dir: Path, *, scale: float = 2.0, max_pages: int | None = None
) -> None:
    """Run the VLM extractor over every PDF in input_dir, skipping already-extracted papers."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {input_dir}")
        return

    client = Anthropic()
    for pdf_path in pdf_paths:
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        if markdown_path.exists():
            logger.info(f"[skip]    {pdf_path.name} (already extracted)")
            continue

        logger.info(f"[extract] {pdf_path.name}")
        try:
            result = run_vlm(pdf_path, output_dir, client, scale=scale, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]   {pdf_path.name}: {exc}")
            continue
        logger.info(f"          -> {result.markdown_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("dataset"), help="PDF file or directory of PDFs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr/output/vlm"),
        help="Directory for extracted Markdown",
    )
    parser.add_argument("--scale", type=float, default=2.0, help="Page render scale (1.0 = 72dpi)")
    parser.add_argument(
        "--max-pages", type=int, default=None, help="Limit pages per paper (for cheap testing)"
    )
    args = parser.parse_args()

    if args.input.is_dir():
        extract_dataset(args.input, args.output, scale=args.scale, max_pages=args.max_pages)
    else:
        client = Anthropic()
        result = run_vlm(
            args.input, args.output, client, scale=args.scale, max_pages=args.max_pages
        )
        logger.info(f"-> {result.markdown_path}")


if __name__ == "__main__":
    main()
