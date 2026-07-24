"""Extract Markdown from paper PDFs using Docling.

Requires the `docling` extra: `uv sync --extra docling`. Docling does
learned layout/table/reading-order analysis (via docling-ibm-models, on
torch) like MinerU does, but its torch floor (>=2.2.2) still has Intel-macOS
wheels, unlike MinerU's (>=2.6.0) — see ocr/README.md for the comparison.

Usage:
    uv run python -m ocr.docling_extract --input dataset --output ocr/output/docling
    uv run python -m ocr.docling_extract --input dataset/some-paper.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter
from loguru import logger


@dataclass
class OcrResult:
    pdf_path: Path
    markdown_path: Path


def run_docling(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> OcrResult:
    """Convert a single PDF to Markdown via Docling's DocumentConverter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{pdf_path.stem}.md"

    result = converter.convert(pdf_path)
    markdown_path.write_text(result.document.export_to_markdown(), encoding="utf-8")

    return OcrResult(pdf_path=pdf_path, markdown_path=markdown_path)


def extract_dataset(input_dir: Path, output_dir: Path) -> None:
    """Run Docling over every PDF in input_dir, skipping papers already extracted."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {input_dir}")
        return

    converter = DocumentConverter()
    for pdf_path in pdf_paths:
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        if markdown_path.exists():
            logger.info(f"[skip]    {pdf_path.name} (already extracted)")
            continue

        logger.info(f"[extract] {pdf_path.name}")
        try:
            result = run_docling(pdf_path, output_dir, converter)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]   {pdf_path.name}: {exc}")
            continue
        logger.info(f"          -> {result.markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("dataset"), help="PDF file or directory of PDFs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr/output/docling"),
        help="Directory for extracted Markdown",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        extract_dataset(args.input, args.output)
    else:
        result = run_docling(args.input, args.output, DocumentConverter())
        logger.info(f"-> {result.markdown_path}")


if __name__ == "__main__":
    main()
