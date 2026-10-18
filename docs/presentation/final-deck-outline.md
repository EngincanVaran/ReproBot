# ReproBot — final presentation outline

inzva AI Projects #10 · Engincan Varan, Aleyna Kütük, Gül Korkut, Göktuğ Mert Özdoğan
20 slides · 20 minutes + questions · audience: technical, has not seen the project

**Rules this outline follows.** Bullets are three to six words; the detail lives in the
speaker notes. Every number carries an HTML comment naming the file it came from.
Anything not yet measured says `[needs measuring]`. The CIFAR-10 papers **execute** but
have **never been judged for fidelity** — no slide may imply otherwise.

---

## 1 — Title

- ReproBot
- Papers claim numbers
- We check them
- inzva AI Projects #10
- Varan · Kütük · Korkut · Özdoğan

> **Say:** ReproBot reads a machine learning paper as a PDF, writes the training script,
> runs it in a sandbox, and then does the thing nobody automates — compares the number it
> got against the number the paper printed. Twenty minutes: the problem, how it works, the
> five papers it reproduced, and where it is still weak.

---

## 2 — The problem

- A paper reports 94.45%
- Nobody re-measures it
- ~1 in 5 ships code
<!-- 16.7-21.2% of accepted ICLR/ICML/NeurIPS 2024 papers shipped a working implementation:
     docs/literature-review/Introduction_and_Literature_Review.md §1, citing Seo et al. 2025 -->
- Rest: reimplement from scratch
- Weeks of work each

> **Say:** Across ICLR, ICML and NeurIPS 2024 only 16.7 to 21.2 percent of accepted papers
> shipped a working implementation, so for four papers in five the published number is
> unverifiable without rebuilding the thing from scratch. That rebuild is mechanical in
> principle — read, implement, run, compare — and brutal in practice. ReproBot is an
> attempt to automate exactly that loop.

---

## 3 — Why it is hard

- Papers under-specify, silently
- Missing value → plausible guess
- Misprinted equation → faithful wrong code
- Unstated split → different test set
- None of these error
<!-- Frosst & Hinton state depth 8 but not lr, batch size, epochs, penalty or temperature:
     docs/progress-reports/third-progress-report/third_report_singlecolumn.tex §Frosst and Hinton
     soft tree Eq.3 is -log of a never-positive quantity: TODO.md "Findings worth keeping"
     Wijaya states 405/101 split sizes but not how rows were drawn: third report §Wijaya -->

> **Say:** Under-specification is the whole difficulty, and it fails quietly — the soft decision
> tree paper states its tree depth and then essentially nothing else, no learning rate, batch
> size, epoch count or penalty strength, and a guessed value does not crash, it trains a
> different network and reports a number for it with full confidence. Every line here is a real
> incident from this project: that same paper prints its loss as the logarithm of a quantity
> that is never positive, so a faithful implementation maximised cross-entropy and trained in
> exactly the wrong direction. The housing paper states its 405/101 split sizes but never how
> the rows were assigned, and on a 506-row dataset two different splits disagree by more than
> the gap being measured.

---

## 4 — What already exists

- PaperCoder: generates, never runs
- AutoP2C: runs, aggregate score
- AutoReproduce: runs, checks executability
- PaperBench: benchmark, not system
- Nobody checks the number
<!-- PaperCoder 45.1% PaperBench Code-Dev, no execution; AutoReproduce 48.5% Code-Dev,
     77-95% execution rate; PaperBench best agent 21.0%, 20 ICML papers, 8,316 rubric leaves,
     ~$400/paper: docs/project-plan/ReproBot_Project_Plan.md §0.1
     AutoP2C: 100% of 8 papers runnable, 99.5% average "relative performance" with no stated
     tolerance or pass/fail criterion: docs/literature-review/Introduction_and_Literature_Review.md §2.3 -->

> **Say:** The replication wave of 2025 divided the problem up neatly and left one piece out.
> PaperCoder produces high-quality code and never executes it. AutoP2C executes and reports
> an average relative performance of 99.5 percent — but with no tolerance and no claim-by-claim
> verdict, a paper reproduced at 89.8 percent and one at 122 percent are reported identically.
> AutoReproduce runs the code and stops at "it executes". None of them asks whether the number
> matches.

---

## 5 — What ReproBot adds

- They generate code
- ReproBot runs it
- Then checks the number
- Tolerance from measurement noise
- Mismatch routes back
<!-- Capability gap matrix and per-competitor differentiators:
     docs/literature-review/PaperAgent_LiteratureReview.md §4, §5 -->

> **Say:** The one-sentence difference: every related system's success criterion is "the code
> runs"; ours is "the number agrees, within a band we derived from how noisy the measurement
> actually is". And when it disagrees, that disagreement is not a report line — it is routed
> back into the code generator as a cited diagnosis. That loop is the contribution the
> literature review identified as the empty quadrant.

---

## 6 — The pipeline

- OCR: PDF → Markdown, figures included
- Reader: five structured fields
- Coder: one self-contained script
- Runner: Docker, four stages
- Critic: verdict, then review
- Report: claim table, curve, gap

> **Say:** Seven folders, one per stage, each runnable on its own. A vision model reads the
> PDF including its figures, because architecture often lives only in a figure caption. The
> Reader extracts method, architecture, claims, hyperparameters and data pipeline, and
> cross-checks them. The Coder writes one script — PyTorch for neural models, scikit-learn
> for classical ones. The Runner executes it in a container through probe, smoke, capped and
> full. Then the Critic judges, and the Report lays it out.

---

## 7 — It is a loop, not a line

- Script crashes → triaged, regenerated
- Training dies → halted mid-run
- Number misses → seeds or fix
- Budgets everywhere, no open-ended agent
- Plateau guard stops repetition
<!-- seven execution verdicts and the four pure decision functions: orchestrator/README.md
     "The six ways it stops" + "The Critic phase"
     plateau guard: SequenceMatcher(autojunk=False), threshold 0.98, real regenerations
     measured 0.22 / 0.25 / 0.27: orchestrator/README.md "Plateau guard", TODO.md -->

> **Say:** Three different failures route three different ways, and that separation is the
> design. A crash is triaged and fed back as a one-line fix. A run that stops learning is
> killed mid-training with the evidence. A number outside tolerance either gets more seeds
> or goes back to the Coder with a cited diagnosis. Everything is bounded: the routing is
> four pure Python functions, not a model deciding what to do next, and a plateau guard
> compares each regenerated script with its predecessor so the loop cannot burn its budget
> repeating itself.

---

## 8 — Results: five papers

| Paper | Model family | Claim | ReproBot | Verdict |
|---|---|---|---|---|
| Tang 2013 | MLP + L2-SVM loss | 0.87% error | **0.82%** | pass |
| Hsu/Chang/Lin | RBF SVM | 96.9% | **96.925%** | pass |
| Xiao 2017 | Random forest | 0.873 | **0.8773** | pass, exceeds |
| Frosst & Hinton | Soft decision tree | 94.45% | **95.11%** | pass, exceeds |
| Wijaya 2023 | Dense regression net | RMSE 3.02 | **3.33** | pass (wide band) |
<!-- all five rows: TODO.md "Results so far" and critic/README.md "The real results, judged"
     Wijaya 3.33 is the mean of 3.75 / 2.84 / 3.40 across three seeds
     SVM: 96.625% on the 2026-09-13 run, 96.925% on the 2026-09-14 rerun with check-ups -->

> **Say:** Five papers, five different model families, all run at the paper's own full
> settings on a laptop CPU. Four land within a fraction of a point of the claim and two
> actually beat it. The fifth, the housing network, passes only because three seeds span
> 2.84 to 3.75 — I will come back to why that matters. Every one of these verdicts was
> produced by the Critic and matches what a human had already concluded by hand.

---

## 9 — Why these five

- CIFAR-10 needs a GPU
- WRN full run: ~22 days
<!-- ~22 days per Wide Residual Networks full run on the dev CPU: TODO.md "Blockers",
     docs/progress-reports/third-progress-report/third_report_singlecolumn.tex §Replication Targets -->
- These train: 73 s – 70 min
<!-- runtimes 73 s (SVM) to 70 min (soft tree): third report §Cost, TODO.md results table -->
- Five families, deliberately different
- Loss function, Keras, LIBSVM, benchmark, under-specified

> **Say:** One full Wide Residual Networks run measured at about 22 days on this machine, so
> the CIFAR-10 set cannot produce a fidelity number without GPU compute we do not have. We
> picked five papers that train in 73 seconds to 70 minutes instead, and chose them to break
> the pipeline in different places: Tang's contribution is a loss function, Wijaya's model was
> written in Keras, the SVM guide is a full model-selection procedure, Fashion-MNIST's figure
> is a mean over five runs, and the soft tree states almost none of its settings.

---

## 10 — War story 1: a dead network reported success

- Tang's first full run
- Error ~89%, claim 0.87%
- Loss at ignore-the-input optimum
- Pipeline would have said success
- Now: live check-ups
<!-- collapse to ~89% error at epoch ~20; loss 3.599 predicted vs ~3.607 observed, always
     predicting digit "1" = 88.8% error; dead ReLUs; six-config ablation showed momentum 0.9
     and C=1.0 are only fatal together; C=0.1 trains cleanly:
     docs/agent-log.md "Tang's first full run collapses"; docs/notes/tang-2013-ablation/README.md -->

> **Say:** This was the project's first ever full-fidelity run. It learned through probe,
> smoke and capped, then collapsed around epoch 20 to roughly 89 percent error against a
> claimed 0.87. The giveaway is that the loss sat exactly on the optimum of a network that
> ignores its input — it had learned to always answer "1". Dead ReLUs. And the script exited
> zero, so the pipeline was about to file it as a success. An ablation showed two individually
> reasonable guesses, momentum 0.9 and SVM penalty C=1.0, are only fatal together.

---

## 11 — What that incident built

- Six rules watch training
- Reads curve every few seconds
- Kills container, returns evidence
- Orchestrator retries with it
- Sign-flipped tree: killed epoch 1/40
<!-- runner/checkups.py: six rules (non_finite, loss_below_lower_bound, worse_than_chance,
     no_progress, stuck_at_chance, diverged); live Docker kill 0.3 s after the epoch-1 record of
     a 40-epoch full run, and a probe halt 5 s in that regenerated to success:
     CLAUDE.md "runner/", docs/agent-log.md "Runner live check-ups" -->
- One false positive, fixed
<!-- no_progress halted a *correct* soft tree in capped three times (loss at ln 10 while accuracy
     was 3x chance); rule restricted to full and now requires no gain over chance: TODO.md -->

> **Say:** Every generated script now writes its learning curve to a structured file, and a
> watcher thread beside the container reads it live. Six rules — non-finite loss, loss below
> its mathematical lower bound, worse than chance, no progress, stuck at chance, diverged —
> can kill the run and hand the Orchestrator the evidence, which goes straight back to the
> Coder as feedback. A deliberately sign-flipped soft tree was killed 0.3 seconds after its
> first epoch record. It also produced a false positive on a correct script, which is why the
> rules are now conservative and every real curve is a regression test.

---

## 12 — War story 2: the reviewer model matters

- Injected bug: train/val swapped
- Trains on 81 rows, not 324
- Sonnet missed it, three times
- Blamed "no early stopping"
- Coder added it; drifted further
<!-- Sonnet 5 missed the swapped train_test_split unpacking live, then offline with a
     data-tracing prompt, then with the log line "81 train_fit, 324 val" in front of it; filed
     "no early stopping" as high severity; the Coder then added checkpoint selection the paper
     never uses, and the loop still ended pass at 2.96 on an unfaithful script:
     critic/README.md "The first Critic loops"; docs/agent-log.md "Critic v2" -->

> **Say:** To exercise the Critic's repair loop we injected one classic bug: the train and
> validation halves of a split were unpacked in the wrong order, so the model trains on 81
> rows instead of 324 while still learning normally and never tripping a check-up. Sonnet
> missed it three times — live, then offline with a data-tracing prompt, then with the runner
> log literally saying "81 train_fit, 324 val" in its context. Worse, it filed "no early
> stopping" as a high-severity bug, with real citations, so every guard passed and the Coder
> dutifully added checkpoint selection the paper never uses.

---

## 13 — The same bug, Opus

- Found the swap immediately
- Cited log line, lines 178–180
- 12 of 12 findings verified
- Fix retry → pass 3.40
- 11 minutes, no human
<!-- Opus 5: major_deviations, cited "81 train_fit, 324 val" and script lines 178-180 plus the
     scaler fitted on the wrong block, 12/12 verified; fix retry changed only those two things;
     result pass 3.40 (±1.05), review "faithful", 13/13 verified, 11 minutes, one retry:
     critic/README.md "The first Critic loops"; TODO.md -->
- Default reviewer changed

> **Say:** Opus found the swap on its first attempt, citing the log line and the exact script
> lines, plus a second consequence nobody had noticed — the scaler was being fitted on the
> wrong block. The fix retry changed those two things and nothing else, and the rerun passed
> at 3.40 with the review rating the script faithful. Eleven minutes, one retry, no human
> involved. That evidence is why the default reviewer is Opus. The lesson generalises: our
> deterministic guards prove a citation is real, never that the reasoning is right.

---

## 14 — How a verdict is made

- `max(2 × uncertainty, reporting precision)`
- Spread of runs, if any
- Else binomial test-set noise
- Else: no estimate → inconclusive
- No model can override
<!-- tolerance formula, uncertainty ladder and the four verdicts: critic/README.md "Tolerance"
     Tang ±0.186 (binomial, n=10,000); SVM guide ±0.548 (n=4,000); Fashion-MNIST ±0.0018
     (spread of 5 runs); soft tree ±0.458 (n=10,000); Wijaya ±1.07 (spread of 3 runs) -->

> **Say:** The verdict is arithmetic — no API call, no Docker, no network — because a model
> asked whether 0.82 and 0.87 are close will say yes one day and no the next. Tolerance is
> twice the measured uncertainty, floored at the paper's own reporting precision, because a
> paper that prints 96.9 cannot be matched more tightly than plus or minus 0.05. Uncertainty
> comes from the spread of real runs when we have several, from binomial test-set noise for
> accuracy-like metrics when we have one, and otherwise it does not exist — in which case a
> losing result is declared inconclusive and we run more seeds rather than calling it a failure.

---

## 15 — Wijaya: the honest pass

- One run: RMSE 3.75
- Inconclusive → two more seeds
- 2.84 and 3.40
- Mean 3.33, band ±1.07
<!-- full run 182 s gave 3.75 (inconclusive), seeds gave 2.84 and 3.40, mean 3.33 vs claimed
     3.02, tolerance ±1.07 = 35% of the claim: critic/README.md, TODO.md results table -->
- Band is 35% of claim

> **Say:** This is the verdict I want you to distrust, because the system distrusts it too.
> One run gave 3.75, which is worse than the claim but came with no noise estimate, so the
> Critic refused to call it a failure and ran two more seeds. Each seed also redraws the
> split the paper never specified, and they gave 2.84 and 3.40. The mean passes — but the
> tolerance is 35 percent of the claim, so the report prints an explicit warning: this means
> consistent with the paper, not matched. The spread is the actual finding. An earlier single
> run of 4.48 was never evidence of a failed replication.

---

## 16 — The report

- Both verdicts, execution and fidelity
- Every claim, tested or not
- Learning curve vs claim, chance
- Gap, tolerance, each seed
- Cited deviations and guesses
- Commands to rerun

> **Say:** The last stage renders the shared state as Markdown with no model call at all —
> every number is copied, and the curve is drawn as plain SVG. Three choices make it honest.
> The tolerance is printed wherever the verdict appears, with a warning when the band is wide.
> Claims the run did not test are listed as "not tested" rather than omitted — for the housing
> paper that is seven of eight distinct results. And only review findings that passed the
> citation checks appear, with a count of how many were dropped.
<!-- Wijaya: 14 extracted claims merge to 8 distinct results, 1 tested: report/README.md,
     critic/README.md "Duplicate claims" -->

---

## 17 — Ablation: does the loop earn its cost

**[pending — another agent is running this now]**
<!-- the loop-OFF arm is being run from docs/notes/ablation/run-ablation.sh; results will land
     under docs/notes/ablation/data/ (and orchestrator/output-ablation/). Fill this table from
     there — do not estimate it. -->

| Paper | Loop off (single-shot) | Loop on | Retries used |
|---|---|---|---|
| Tang 2013 | `[pending]` | `[pending]` | `[pending]` |
| Wijaya 2023 | `[pending]` | `[pending]` | `[pending]` |
| Hsu/Chang/Lin | `[pending]` | `[pending]` | `[pending]` |
| Xiao 2017 | `[pending]` | `[pending]` | `[pending]` |
| Frosst & Hinton | `[pending]` | `[pending]` | `[pending]` |

What we can say today, pending that table:
- Three defects repaired automatically
- Zero human involvement each
<!-- three real (not injected) runtime defects auto-repaired, one retry each: a wrong
     fetch_openml call (Wijaya), a label-sorted first-N subset (SVM guide), an in-place
     autograd op (soft tree): third report §Problems, TODO.md -->
- Prior art: single-shot 0%
<!-- MLR-Copilot ablation: single-prompt baseline 0% success on all 5 tasks / 8 trials,
     iterative loop up to 50%: docs/project-plan/ReproBot_Project_Plan.md §0.3 -->

> **Say:** This is the project plan's designated headline result and the measurement is
> running right now, so the table is a placeholder — I will not show you a number I do not
> have. What we can say is that three real defects, none of them planted, were repaired
> automatically with one retry each and no human. And the closest published ablation, from
> MLR-Copilot, found single-shot generation succeeded on zero percent of trials while the
> iterative loop reached fifty.

---

## 18 — Honest limits

- One claim per paper
- Two needed a human
- Wide band ≠ close match
- Five papers, not a benchmark
- CIFAR-10 runs, never judged
<!-- one claim per run: critic/README.md "Limits"; Tang needed manual C 1.0 -> 0.1 and the soft
     tree needed the misprinted Eq.3 fixed by hand: TODO.md results table, third report
     §Limitations; CIFAR-10 status: 8 PDFs -> 6 OCR'd -> 4 read -> 2 coded -> 2 executed at
     smoke only: TODO.md "Where the pipeline is" -->

> **Say:** Five things, plainly. One run targets one claim, so the housing paper's report shows
> one of eight results tested. Two of the five reproductions needed a human edit before their
> final run — Tang's unstated penalty and the soft tree's misprinted loss — though the Coder
> later derived that loss correctly on its own. A pass on a wide tolerance means consistent
> with, not matched. Five small papers is a demonstration, not a benchmark. And the CIFAR-10
> papers execute as far as a one-epoch smoke run — they have never been judged for fidelity,
> and we are not claiming they were.

---

## 19 — What is next

- Patch scripts, not regenerate
<!-- real one-line repairs rewrote 73-78% of the script (similarities 0.22, 0.25, 0.27):
     TODO.md "Loop controls" -->
- Judge every claim
<!-- Breiman states 13 per-dataset errors, Isolation Forest 9 AUCs; the four papers added
     2026-09-20 state 35+ claims between them: extra-papers/README.md, TODO.md -->
- GPU for CIFAR-10
- Record human corrections
- Merge the viewer

> **Say:** Four concrete things. Retries currently regenerate the whole script — a one-line
> fix rewrites three quarters of it — so patching is the next loop control. Multi-claim
> evaluation is the cheapest way to multiply the evidence: Breiman's Random Forests alone
> states thirteen per-dataset errors, and the four papers we added last month state over
> thirty-five claims between them. Then GPU compute to finally judge the CIFAR-10 set, and
> merging Mert's Streamlit viewer as the demo front end.

---

## 20 — Takeaway

- Running code is not enough
- The number is the claim
- Tolerance must be measured
- Watch training, not exit codes
- Five papers, honestly reported

> **Say:** If you take one thing away: in replication, "it runs" is not the result — the number
> is the result, and a number without a tolerance is not a comparison. Everything expensive we
> learned came from watching runs rather than waiting for them, and from refusing to let a
> model decide whether two numbers are close. Five papers, five model families, every caveat
> on the slide. Happy to take questions.

---
---

# Appendix A — figures

## Reusable as-is

| Figure | Path | Note |
|---|---|---|
| Coder stage diagram | `docs/handouts/coder-diagram.svg` (+ `.png`, `@2x.png`) | Built for slides already; predates scikit-learn support, so check the labels |
| Five-panel learning curves | `docs/progress-reports/third-progress-report/third_report_singlecolumn.pdf` (Fig. `fig:curves`) | Crop from the PDF, or regenerate — it is inline TikZ emitted by `docs/progress-reports/third-progress-report/generate.py`, computed from real run data |
| Retry loop walkthrough | same PDF, Fig. `fig:wijayarun` | The Wijaya auto-repair: attempt 1 fails probe, triage, attempt 2 clean to `full`. Ready for slide 7 |
| svmguide1 exactness table | same PDF, Table `tab:svm` | Paper vs scikit-learn at three settings, all exact — good backup slide |
| Problem → solution table | same PDF, Table `tab:problems` | Nine rows; trim to four for a slide |
| Reader coverage table | same PDF, Table `tab:reader` | 9 papers × claims/hyperparameters/gaps/flags |
| Stage one-pagers | `docs/handouts/01-ocr.pdf` … `05-summary.pdf` | Handouts for the room, not slides. All predate the generalization work (`TODO.md` backlog) |

## Needs making

| Figure | Why it does not exist yet |
|---|---|
| Seven-stage pipeline diagram | The third report's diagram is dated 13.09.2026 and shows five stages — `critic/` and `report/` are missing |
| Critic routing diagram | Exists only as ASCII in `critic/README.md`; draw pass / fail / inconclusive → accept / fix / seeds / guided retry |
| Results table as a slide | Data is in `TODO.md` and `critic/README.md`; no rendered version |
| Tolerance band chart | Five claims with their bands and reproduced points — the clearest way to show why Wijaya is different. `report/curves.py` draws curves, not this |
| A real `report.md` screenshot | `report/output/` is gitignored; regenerate with `uv run python -m report.pipeline --input orchestrator/output` |
| `curve.svg` for a slide | Same — generated per paper beside `report.md` |
| Ablation table | `[pending — another agent is running this now]` |
| Viewer screenshot | Streamlit dashboard lives on `origin/mert/runner-agent`, unmerged; needs someone to run it |

---

# Appendix B — the three questions to expect

**1. "How do you know the model isn't just recalling the paper from pretraining?"**

We do not fully, and we say so. The Coder's input precedence is explicit and three-level:
the Reader's extraction first, the paper's own text second, and the model's pretrained memory
third — permitted, but only when disclosed in the output's `assumptions`, because a code
generator cannot decline to answer the way an extractor can. The strongest evidence is the
choice of test: Network In Network was used deliberately instead of Wide ResNet, because a
wide ResNet would be rebuilt correctly from memory and prove nothing; the generated script
built a real mlpconv cascade with zero `nn.Linear` layers and excluded both decoy equations
by name. For the classical papers the question largely dissolves — scikit-learn's `SVC` wraps
LIBSVM, so reproducing the SVM guide exactly is the library doing the paper's method.
<!-- input precedence and the NIN test: CLAUDE.md "coder/"; SVC/LIBSVM exactness:
     third report Table tab:svm, TODO.md "Findings worth keeping" -->

**2. "Five small CPU papers — why should I believe this works on real papers?"**

You should not, yet, and that is the honest answer. Five papers is a demonstration that the
mechanism works end to end across five model families, not a benchmark result. The CIFAR-10
set is read and coded, and two papers execute to a smoke run, but no fidelity verdict exists
for any of them because one full Wide ResNet run measures at about 22 days on our CPU. What
the small papers do buy is real: they are the only way to get a *measured* fidelity number
at all without GPU compute, and the failures they surfaced — a dead network reporting success,
a misprinted equation, an unstated split — are not artefacts of being small.
<!-- CIFAR-10 pipeline depth and the 22-day figure: TODO.md -->

**3. "You use an LLM to review the code. How is that not the failure mode you criticise?"**

Two defences, and one admission. The defence: the verdict itself is arithmetic and no model
can change it, and every review finding must pass deterministic checks — each number must
appear in the material it was given, each paper quote must actually occur in the paper, each
script line and snippet must exist, and a proposed fix may not introduce a technique the paper
never mentions. The admission: those guards prove a citation is real, never that the reasoning
is right. Sonnet's misdiagnosis passed every guard, and the Coder acted on it. That is why the
reviewer model was changed to Opus on evidence, why the add-on-technique guard exists at all,
and why every new misdiagnosis becomes a new guard or a new test case.
<!-- guards and the Sonnet/Opus comparison: critic/README.md "Critic v2: the review" -->
