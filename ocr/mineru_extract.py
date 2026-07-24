"""Extract Markdown (+ images) from paper PDFs using MinerU.

Requires the `mineru` extra: `uv sync --extra mineru` (needs torch>=2.6.0 —
see ocr/README.md for the Intel-macOS caveat).

Usage:
    uv run python -m ocr.mineru_extract --input dataset --output ocr/output/mineru
    uv run python -m ocr.mineru_extract --input dataset/some-paper.pdf

MinerU is invoked as a subprocess (its documented, stable interface) rather
than through an internal Python API. See docs/notes/reader-agent-precedents.md
for why MinerU was chosen over a plain pdfplumber-only extraction as one of
our OCR backends.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OcrResult:
    """Markdown + image output of a single MinerU run on one PDF."""

    pdf_path: Path
    markdown_path: Path
    image_paths: list[Path]
    output_dir: Path


def run_mineru(pdf_path: Path, output_dir: Path, *, backend: str = "pipeline") -> OcrResult:
    """Run MinerU on a single PDF, returning its Markdown output and image assets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-b", backend],
        check=True,
    )

    try:
        markdown_path = next(output_dir.rglob("*.md"))
    except StopIteration as exc:
        raise RuntimeError(f"MinerU produced no Markdown output under {output_dir}") from exc

    image_paths = sorted(output_dir.rglob("images/*"))

    return OcrResult(
        pdf_path=pdf_path,
        markdown_path=markdown_path,
        image_paths=image_paths,
        output_dir=output_dir,
    )


def extract_dataset(input_dir: Path, output_dir: Path, *, backend: str = "pipeline") -> None:
    """Run MinerU over every PDF in input_dir, skipping papers already extracted."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}")
        return

    for pdf_path in pdf_paths:
        paper_output_dir = output_dir / pdf_path.stem
        if any(paper_output_dir.rglob("*.md")):
            print(f"[skip]    {pdf_path.name} (already extracted)")
            continue

        print(f"[extract] {pdf_path.name}")
        try:
            result = run_mineru(pdf_path, paper_output_dir, backend=backend)
        except subprocess.CalledProcessError as exc:
            print(f"[error]   {pdf_path.name}: MinerU exited with {exc.returncode}")
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
        default=Path("ocr/output/mineru"),
        help="Directory for MinerU output",
    )
    parser.add_argument(
        "--backend",
        default="pipeline",
        choices=["pipeline", "vlm"],
        help="MinerU backend (pipeline = local CPU/GPU models, vlm = VLM-based layout)",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        extract_dataset(args.input, args.output, backend=args.backend)
    else:
        result = run_mineru(args.input, args.output / args.input.stem, backend=args.backend)
        print(f"-> {result.markdown_path}")


if __name__ == "__main__":
    main()
