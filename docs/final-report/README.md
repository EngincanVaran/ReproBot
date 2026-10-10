# docs/final-report/ — the final project report

The final report, in single- and two-column variants. Both are **generated**; never edit a
`.tex` by hand, or the two variants will drift.

```
generate.py                      the body, the data, the TikZ figures, and build()
final_report_singlecolumn.tex    generated  (19 pages)
final_report_twocolumn.tex       generated  (14 pages)
final_report_*.pdf               built with pdflatex, twice
```

## Regenerate

Run from the **repository root** — the output path inside `generate.py` is repo-relative:

```bash
uv run python docs/final-report/generate.py
cd docs/final-report
pdflatex -interaction=nonstopmode final_report_singlecolumn.tex   # twice, for \ref
pdflatex -interaction=nonstopmode final_report_singlecolumn.tex
pdflatex -interaction=nonstopmode final_report_twocolumn.tex
pdflatex -interaction=nonstopmode final_report_twocolumn.tex
rm -f *.aux *.log *.out
```

`generate.py` needs nothing but the standard library. It is linted with the rest of the repo:

```bash
uv run ruff check docs/final-report/generate.py
uv run ruff format --check docs/final-report/generate.py
```

## How it is put together

Same pattern as `docs/progress-reports/third-progress-report/generate.py`:

- **The prose exists once**, in `BODY`. `build(twocolumn: bool)` prepends a different
  `\documentclass` and `geometry` block and substitutes the float environments, so the two
  variants cannot say different things.
- **`FIGENV` / `TABENV`** are placeholders. They become `figure` / `table` in the single-column
  build and `figure*` / `table*` in the two-column one, which is what lets a wide table span both
  columns. Write every wide float with them; use a plain `table` only when the content really
  fits one column (`tab:svmexact`, `tab:coverage`, `tab:review`, `tab:tolerance`,
  `tab:consistency`).
- **The figures are hand-emitted TikZ.** This TeX install has no `pgfplots`, so every coordinate
  is computed in Python from real run data and written out as a path. `lin`, `logmap`, `path`,
  `dots`, `axes` and `hline` are the shared helpers; the Tang / SVM-grid / random-forest /
  soft-tree panels are carried over unchanged from the third progress report, and
  `panel_spread` is new.
- **Every number traces to a file.** The constants at the top of `generate.py` name their source
  in a comment: `docs/notes/end-to-end/data/runs.json`, `docs/notes/consistency/data/runs.json`,
  `docs/notes/ablation/data/summary.json`, and the stage logs under `runner/output/` and
  `orchestrator/output/`. Nothing is rounded a second time or re-derived from memory.

## Traps hit while writing this, worth not re-deriving

**Wide tables overflow the margin silently.** `pdflatex` exits 0 with an `Overfull \hbox`
warning, and the table just runs into the margin. With fourteen tables the only workable
discipline is to check the log after every build:

```bash
grep -o "Overfull \\\\hbox ([0-9.]*pt" final_report_twocolumn.log | sort -t'(' -k2 -rn | head
```

Anything above ~6pt is visible on the page. Two fixes are enough: `\squeeze` (defined in the
preamble; drops `\tabcolsep` from 6pt to 4pt, worth about 0.1\textwidth on a seven-column table)
and trimming the declared column fractions. The sum of the `L{...}` widths plus
`2 × ncols × \tabcolsep` has to come in under `\textwidth`, and it is the `\tabcolsep` term people
forget.

**A long `\texttt{}` identifier cannot be hyphenated.** `\texttt{retry_budget_exhausted}` is wider
than any sensible table cell and `p{}` columns will not break it. Split it with `\allowbreak`
between the parts (`\texttt{retry\_}\allowbreak\texttt{budget\_}...`), which breaks without
inserting a hyphen that would misrepresent the identifier.

**Display math is wider than a two-column column.** The tolerance formula spelled out in words
(`\max(2 \times \text{uncertainty}, \text{reporting precision})`) overflows; it is written as
`\max(2u, p)` with the symbols defined in the sentence after it.

**Reference-line labels collide with data labels.** In `panel_spread` the per-point value labels
are rotated 90°, so a claim line labelled at the default left edge lands on top of them.
`claim_anchor` / `claim_x` exist to put each panel's annotation in whatever band of that panel is
empty: the right edge for the SVM panel, the gap between groups for the Wijaya panel.

**Build artefacts.** `*.log` is gitignored but `*.aux` and `*.out` are not. Delete them after a
build; the third progress report's directory holds only `generate.py`, the two `.tex` and the two
`.pdf`.

## What is in the report

Three figures and fourteen tables, built to be lifted into a deck one at a time:

| Label | What it holds |
|---|---|
| `fig:pipeline` | The seven stages and the two bounded feedback edges |
| `fig:curves` | Four learning curves at the papers' own full settings |
| `fig:spread` | Every reproduced value for Wijaya and the SVM guide, grouped by experiment |
| `tab:stages` | What each stage does, where it spends a model call, what it checks without one |
| `tab:fidelity` | The headline: 5 papers, claim vs reproduced, verdict, tolerance |
| `tab:svmexact` | The SVM guide's three reported settings, reproduced exactly |
| `tab:repeat` | 3 papers × 3 runs, with the named cause of each paper's spread |
| `tab:unstated` | The mechanism behind each spread — the report's main finding |
| `tab:coverage` | Claims extracted vs distinct vs tested, 9 papers |
| `tab:review` | The Critic's code review of the three run-1 scripts |
| `tab:tolerance` | Where each tolerance came from and how wide it is |
| `tab:experiments` | The four measurements and what each establishes |
| `tab:ablation` | Loop off against loop on, 5 papers |
| `tab:consistency` | 5 independent generations per paper, 2 papers |
| `tab:cpuset` | The 15-paper CPU set with target claims and status |
| `tab:gpuset` | The 8-paper CIFAR-10 set, per-paper status, zero fidelity verdicts |
| `tab:problems` | Engineering problem → solution |
| `tab:measurement` | Problems in *measuring* the system → solution |
