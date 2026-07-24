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
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-sonnet-5"

PROMPT = """Transcribe this page of an academic paper into clean Markdown.

- Preserve headings, paragraph text, and reading order.
- Render tables as Markdown tables.
- Render math as LaTeX (inline $...$ or block $$...$$).
- Describe figures/diagrams briefly in square brackets, e.g. [Figure 2: ...].
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
        max_tokens=4096,
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
        png_bytes = _render_page_png(pdf_path, page_index, scale)
        text = _transcribe_page(client, png_bytes)
        sections.append(f"## Page {page_index + 1}\n\n{text}")

    markdown_path.write_text("\n\n".join(sections), encoding="utf-8")
    return OcrResult(pdf_path=pdf_path, markdown_path=markdown_path)


def extract_dataset(
    input_dir: Path, output_dir: Path, *, scale: float = 2.0, max_pages: int | None = None
) -> None:
    """Run the VLM extractor over every PDF in input_dir, skipping already-extracted papers."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}")
        return

    client = Anthropic()
    for pdf_path in pdf_paths:
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        if markdown_path.exists():
            print(f"[skip]    {pdf_path.name} (already extracted)")
            continue

        print(f"[extract] {pdf_path.name}")
        try:
            result = run_vlm(pdf_path, output_dir, client, scale=scale, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            print(f"[error]   {pdf_path.name}: {exc}")
            continue
        print(f"          -> {result.markdown_path}")


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
        print(f"-> {result.markdown_path}")


if __name__ == "__main__":
    main()
