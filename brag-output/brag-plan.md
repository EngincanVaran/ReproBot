# ReproBot brag plan

**What it is:** a multi-agent pipeline that reads an ML paper PDF, writes and runs the
training script, and checks the reproduced number against the paper's claim.
**Who it's for:** ML researchers and reviewers who want a paper's number rerun, not trusted.
**What sets it apart:** it is a loop with a Critic that gives an arithmetic verdict and then
reads the generated code against the paper.
**Most impressive claim:** six papers, six model families, every Critic verdict a pass
(WRN-28-10 on CIFAR-10: 3.83% vs the paper's 4.00%).
**Visual hook:** a paper's results table with one number highlighted, then "Did anyone rerun it?"
**Real material shown:** the real WRN-28-10 learning curve from `report/output/`, the real
Critic loop on Wijaya (planted swapped split, 4.76 fail → 3.40 pass), the real results table.

**Angle:** papers print a number; ReproBot makes them prove it.
**Tone:** `default`, leaning dry. Confident, clean, no jokes that the project didn't earn.
**Identity (from `docs/handouts/`):** Archivo (wide, heavy) + DM Mono; near-black `#121211`,
cream `#F2F1EC`, grey `#807E77`, pass green `#4FC79B`, fail red `#FF5A6E`.
**Format:** landscape 1920×1080, 30 fps, 22 s. Music 120 BPM, D minor, written with the effects.

## Storyboard

| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0.0–3.0 | Hook | Paper card "Wide Residual Networks", results table, marker sweeps the 4.00 row. "The paper says 4.00%." then "Did anyone rerun it?" |
| 2 | 3.0–7.0 | Reveal | ReproBot wordmark. Pipeline chips light one by one: PDF → OCR → Reader → Coder → Runner → Critic → Report. "Reads an ML paper. Reruns it. Checks the number." |
| 3 | 7.0–12.0 | Highlight 1: the run | Terminal types `./reproduce.sh full`; the real 200-epoch test-error curve draws against the dashed "paper 4.00%" line; live counter lands on 3.83%; PASS stamp, ±0.39. |
| 4 | 12.0–15.5 | Highlight 2: the loop | "We planted a bug. The Critic found it." Attempt 1 RMSE 4.76 FAIL → Critic finding (swapped train/validation split, lines 178–180) → attempt 2 RMSE 3.40 PASS. |
| 5 | 15.5–19.0 | Highlight 3: the table | "6 papers. 6 model families. All pass." Six rows, paper says vs ReproBot, PASS badges popping in. |
| 6 | 19.0–22.0 | Outro | "Don't trust the number. Rerun it." Wordmark, repo URL, inzva AI Projects #10. |

Transitions: old content out, dip through the background, new content in. No crossfades.
