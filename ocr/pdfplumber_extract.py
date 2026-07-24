"""Extract text + tables from paper PDFs using pdfplumber.

Requires the `pdfplumber` extra: `uv sync --extra pdfplumber`. Pure Python,
no ML models, no torch — works everywhere, fastest to run. The tradeoff is
quality: no layout/reading-order model, no figure understanding, and table
detection is heuristic (rule-based, not learned). This is the naive baseline
to compare MinerU and Docling against, not a replacement for either.

Usage:
    uv run python -m ocr.pdfplumber_extract --input dataset --output ocr/output/pdfplumber
    uv run python -m ocr.pdfplumber_extract --input dataset/some-paper.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class OcrResult:
    pdf_path: Path
    markdown_path: Path


def _table_to_markdown(table: list[list[str | None]]) -> str:
    rows = [[cell or "" for cell in row] for row in table]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def run_pdfplumber(pdf_path: Path, output_dir: Path) -> OcrResult:
    """Extract per-page text and tables from a single PDF into one Markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{pdf_path.stem}.md"

    sections: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            sections.append(f"## Page {page_num}\n")
            if text := page.extract_text():
                sections.append(text)
            for table in page.extract_tables():
                sections.append("\n" + _table_to_markdown(table))

    markdown_path.write_text("\n\n".join(sections), encoding="utf-8")
    return OcrResult(pdf_path=pdf_path, markdown_path=markdown_path)


def extract_dataset(input_dir: Path, output_dir: Path) -> None:
    """Run pdfplumber over every PDF in input_dir, skipping papers already extracted."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}")
        return

    for pdf_path in pdf_paths:
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        if markdown_path.exists():
            print(f"[skip]    {pdf_path.name} (already extracted)")
            continue

        print(f"[extract] {pdf_path.name}")
        try:
            result = run_pdfplumber(pdf_path, output_dir)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            print(f"[error]   {pdf_path.name}: {exc}")
            continue
        print(f"          -> {result.markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("dataset"), help="PDF file or directory of PDFs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr/output/pdfplumber"),
        help="Directory for extracted Markdown",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        extract_dataset(args.input, args.output)
    else:
        result = run_pdfplumber(args.input, args.output)
        print(f"-> {result.markdown_path}")


if __name__ == "__main__":
    main()
