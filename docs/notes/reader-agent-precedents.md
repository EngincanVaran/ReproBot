# Paper-Reading Agents in AutoReproduce and AutoP2C

> Working note, not part of the polished literature-review stack (`docs/literature-review/`). Deep dive into *specifically* how AutoReproduce and AutoP2C parse the source PDF into something an LLM can code against — i.e. the closest existing precedent for ReproBot's own **Reader** agent. Read directly from the papers in `papers/`, not just their `docs/literature-review/Paper_Summaries.md` summaries.

## TL;DR

Both use **MinerU** (Wang et al., 2024a — an open-source PDF→Markdown OCR tool) as the very first step. That's where the similarity ends:

- **AutoReproduce**: MinerU *is* basically the whole reading pipeline. Its output (Markdown) is handed to a single general-purpose LLM ("Research Agent") that summarizes it in three passes. No separate VLM component is architected — visual-diagram understanding is an optional, ablated extra that leans on the backbone LLM's own native vision (they use Claude 3.5 Sonnet, which is multimodal) rather than a dedicated parsing stage.
- **AutoP2C**: MinerU is only step 1 of a **7-step pipeline**. It's followed by a dedicated LLM pass to repair MinerU's own OCR artifacts, a dedicated **VLM** pass specifically for figures/diagrams, a dedicated LLM pass for equations, a dedicated LLM pass for tables/hyperparameters, then an integration pass and a filtering pass. Each modality gets its own specialist call before everything is merged.

So: "do they just use MinerU?" — yes for the raw OCR step in both, but AutoP2C wraps that with genuinely separate per-modality parsing; AutoReproduce does not.

---

## 1. AutoReproduce's Research Agent

AutoReproduce (Zhao et al., 2025, Tsinghua) doesn't have a named "Reader" — reading is the first of its three pipeline phases (**Literature Review → Paper Lineage → Code Development**), handled by one of its two agent roles, the **Research Agent** (the other is the **Code Agent**). The Research Agent also does the citation-mining in phase 2 and reviews code in phase 3 — it's a generalist, not a parsing specialist.

```
Target paper PDF
      │
      ▼
┌─────────────────────────┐
│  MinerU (OCR)             │  PDF → Markdown, "significantly enhancing
│                            │  the fidelity of data preservation"
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────────────────┐
│  Research Agent (single LLM,             │
│  claude-3-5-sonnet-20240620 in the        │
│  main experiments)                        │
│                                            │
│  Hierarchical 3-stage summarization:      │
│   1. Overall Summary   — holistic overview│
│   2. Method Summary    — math formulations,│
│      implementation specifics             │
│   3. Experiment Summary — critical         │
│      settings needed for reproduction     │
│                                            │
│  [optional] enrich with "visual structure │
│  diagrams" — mentioned once, ablated as   │
│  "w/ Visual Diagram" in Table 3; no        │
│  separate VLM tool is named — this reads  │
│  as just also passing page images to the  │
│  same backbone LLM, which is natively     │
│  multimodal (Claude 3.5 Sonnet)]          │
└─────────────┬───────────────────────────┘
              ▼
     Overall / Method / Experiment summaries
     → feed into Paper Lineage + Code Development
```

**Why direct parsing (no MinerU) was rejected**: the paper explicitly motivates MinerU by saying "effective summarization hinges on the quality of text extraction, where direct parsing methods (Schmidgall et al., 2025 [Agent Laboratory]) often struggle with complex artifacts like mathematical formulas and tables."

**Paper Lineage is a second, separate reading process** — but it reads the *cited* papers, not the target paper:

```
Target paper's reference list
      │
      ▼
Research Agent selects top-k (default k=3) most relevant
cited papers, using citation-context analysis — papers used
as experimental baselines are prioritized
      │
      ▼
Download each via the ArXiv API → Research Agent summarizes
      │
      ▼
Research Agent also extracts each paper's linked GitHub repo
URL (if present) → Code Agent clones it via the GitHub API
      │
      ▼
Code Agent filters the repo down to relevant source files,
paired with the paper summary → <summary, code> tuples
      │
      ▼
K_lineage — domain-aligned reference exemplars, fed into
Code Development as grounding context
```

**Quantified impact of the OCR step** (Table 4 ablation, Claude-3.5-Sonnet backbone): removing MinerU (`w/o MinerU`) drops Mixed-Level score 70.14 → 58.42 and worsens Performance Gap 35.83% → 47.81% — one of the two largest single-component drops in the whole ablation (only `w/o Debug+Refine` hurts more, on Perf Gap specifically).

**Models/cost**: Research + Code Agent both run on `claude-3-5-sonnet-20240620` for the main results; average reproduction cost is **$1.87/paper** when using o3-mini as the backbone instead.

---

## 2. AutoP2C's Multimodal Content Parsing stage

AutoP2C (Lin et al., 2025) has four stages: (1) Repository Blueprint Extraction (offline, from unrelated established GitHub repos — not paper-specific), (2) **Multimodal Content Parsing** (the actual Reader-equivalent), (3) Hierarchical Task Decomposition, (4) Iterative Feedback-Driven Implementation. Stage 2 is a genuinely multi-step, per-modality pipeline:

```
Target paper PDF
      │
      ▼
┌─────────────────────────────────────┐
│ Step 1 — MinerU (OCR)                  │  P_raw = (T_raw, I_raw, M_raw, D_raw)
│ PDF → raw Markdown                     │  raw text / raw images / raw equations /
│                                         │  raw tables
└─────────────┬───────────────────────────┘
              │  artifacts: fragmented paragraphs, misplaced
              │  image references, misformed equations
              ▼
┌─────────────────────────────────────┐
│ Step 2 — LLM: structural restore       │  T_structured = LLM_restore(T_raw)
│ (repairs MinerU's own OCR artifacts)   │
└─────────────┬───────────────────────────┘
              │
   ┌──────────┼───────────────┬──────────────────┐
   ▼          ▼                ▼                  ▼
┌────────┐ ┌────────────┐ ┌────────────┐   (T_structured
│Step 3   │ │Step 4       │ │Step 5       │    passes through
│VLM:     │ │LLM:         │ │LLM:         │    unchanged)
│parse    │ │parse math   │ │parse tabular│
│images   │ │equations →  │ │data →       │
│(figures/│ │computational│ │hyperparams, │
│diagrams)│ │form         │ │configs,     │
│         │ │             │ │eval metrics │
│I_parsed │ │M_parsed     │ │D_parsed     │
└────┬────┘ └──────┬──────┘ └──────┬──────┘
     └─────────────┴────────────────┴──────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │ Step 6 — LLM: integrate    │  P_integrated = LLM_integrate(...)
        │ (merge modalities, strip   │
        │  cross-modal redundancy)   │
        └─────────────┬─────────────┘
                      ▼
        ┌─────────────────────────┐
        │ Step 7 — LLM: filter       │  P_distilled = LLM_filter(P_integrated)
        │ (keep only code-relevant   │
        │  content; drop lit review, │
        │  theoretical justification,│
        │  low-relevance references) │
        └─────────────┬─────────────┘
                      ▼
              P_distilled — single unified
              text representation, fed into
              Hierarchical Task Decomposition
```

**The VLM step (Step 3) is deliberately narrow-scoped, not "describe this image"** — the guiding prompt has three explicit parts:
1. Emphasize only code-relevant visual information (e.g. discard color, decorative elements) to avoid noise.
2. Comprehensively capture *all* numerical elements in the image, to prevent silent omission.
3. Cross-reference the image's own caption to ground the description accurately in the paper's text.

**Which model is actually the "VLM"?** AutoP2C doesn't use a separate open-source VLM like LLaVA — it uses **GPT-4o** for everything from repository blueprint extraction through hierarchical task decomposition (i.e., including this whole multimodal-parsing stage), citing "its advanced multimodal understanding capability." Coding then hands off to different models: **o1-mini** for initial per-file code generation, **o1** for validation, **o3-mini** for iterative refinement — a deliberate per-task model split, not one model doing everything (contrast with AutoReproduce's single Claude-3.5-Sonnet backbone throughout).

**Quantified impact, per modality** (Table 4 ablation, on one paper — "Convolutional...[34]"):

| Modalities present | Perf. | Rel. Perf. | COMP_class | COMP_func |
|---|---|---|---|---|
| Text + Image + Table (full) | 92.0% | 122.0% | 73.1% | 18.5% |
| Text + Table (image removed) | 70.1% | 92.9% | 59.8% | 4.2% |
| Text + Image (table removed) | 88.9% | 117.9% | 36.7% | 7.6% |

Removing **images** costs 21.9 absolute performance points — "architectural diagrams contain essential visual information about model structure, component relationships, and data flow that cannot be fully captured by textual descriptions alone." Removing **tables** costs less on raw performance but nearly halves class-completeness (73.1%→36.7%) and guts function-completeness (18.5%→7.6%) — unsurprising, since hyperparameters and exact configs live in tables, not prose.

**Overall ablation** (Table 3, all four AutoP2C stages): removing the Multimodal Content Parsing stage entirely (i.e., no dedicated per-modality parsing at all) drops average performance from 83.5% → 53.5%, class-completeness 61.6% → 38.0%, function-completeness 43.4% → 36.4% — the second-most damaging single removal after cutting the iterative feedback-driven implementation stage (which zeroes out executability completely, 0%).

---

## 3. Side-by-side

| | AutoReproduce | AutoP2C |
|---|---|---|
| PDF → text/table OCR | MinerU | MinerU |
| OCR-artifact cleanup pass | Implicit in MinerU's "high fidelity" claim; no explicit repair step named | Explicit dedicated LLM "restore" pass |
| Dedicated VLM for figures/diagrams | ✗ — optional, ablated, relies on backbone LLM's native vision (Claude 3.5 Sonnet) | ✓ — explicit VLM step (GPT-4o) with a 3-part scoped prompt |
| Dedicated equation parsing | ✗ — folded into general "Method Summary" | ✓ — explicit LLM pass, typeset → computational form |
| Dedicated table/hyperparameter parsing | ✗ — folded into general "Experiment Summary" | ✓ — explicit LLM pass, extracts hparams/configs/metrics |
| Modality integration + redundancy removal | N/A — one LLM call does everything inline | ✓ — explicit `integrate` then `filter` passes |
| Also reads *cited* papers? | ✓ — "Paper Lineage": mines top-k cited papers + their GitHub code for implicit domain knowledge | ✗ — no per-paper lineage; instead mines a one-time, paper-agnostic "blueprint" from unrelated popular ML repos |
| Backbone model(s) for the reading step | `claude-3-5-sonnet-20240620` (same model for everything) | `GPT-4o` (parsing/planning) → `o1-mini`/`o1`/`o3-mini` split across coding/validation/refinement |
| Ablation evidence the OCR step matters | `w/o MinerU`: Mixed-Level 70.14→58.42, Perf Gap 35.83%→47.81% | (not separately ablated — MinerU is folded into the whole "MM" stage ablation below) |
| Ablation evidence the *whole reading stage* matters | — | `w/o MM parsing`: avg. perf. 83.5%→53.5%, COMP_class 61.6%→38.0% |

---

## 4. Implications for ReproBot's Reader

The current plan (`docs/project-plan/ReproBot_Project_Plan.md`, §2.2) specs `pdfplumber` + `pdf2image` + Claude Sonnet 5 vision. Worth reconsidering in light of the above:

1. **Both systems independently converged on MinerU for the OCR front-end, and AutoReproduce's own ablation shows skipping a normalization step like it measurably hurts** (Mixed-Level −11.7pt, Perf Gap +12pt worse). `pdfplumber` is a layout/table extractor, not an OCR-and-Markdown-normalizer — it doesn't do the artifact-repair MinerU does. Worth prototyping MinerU (or at least comparing it against `pdfplumber`'s raw output) before locking in the text-extraction tool.
2. **AutoP2C's per-modality decomposition (separate passes for images, equations, tables, then integrate, then filter) is the more defensible design of the two** — its own ablation shows each modality contributes independently and non-redundantly (table 4), which validates *why* to keep them separate rather than one big "summarize the paper" prompt. This is a direct, stronger-evidence extension of what the project plan already flags for the *data pipeline* field specifically (§2.2's "give the Reader an explicit, separate extraction pass for 'data pipeline'") — the same argument applies to hyperparameters/equations/architecture, not just data processing.
3. **AutoReproduce's Paper Lineage (reading cited papers, not just the target paper) is still an open, un-adopted idea** for ReproBot per the existing literature review — nothing above changes that; it remains a plausible post-MVP extension rather than an M1 requirement.
4. **Neither system uses a distinct open-source VLM (e.g. LLaVA)** — both lean on their main LLM's native multimodality (Claude 3.5 Sonnet / GPT-4o) rather than a separate vision model. This matches the project plan's choice of Claude Sonnet 5's native vision rather than a bolted-on VLM, and is one place ReproBot's plan is already aligned with both precedents.
