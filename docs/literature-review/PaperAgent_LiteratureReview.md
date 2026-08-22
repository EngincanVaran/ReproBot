# ReproBot — Literature Review
> inzva AI Projects #10 | May 2026
> Covers all related works reviewed during the proposal phase, including landscape analysis, timeline, per-paper deep dives, and gap analysis.

---

## Table of Contents

1. [The Research Landscape](#1-the-research-landscape)
2. [How the Field Evolved — Timeline](#2-how-the-field-evolved--timeline)
3. [Per-Paper Deep Dives](#3-per-paper-deep-dives)
   - [MLAgentBench](#31-mlagentbench-huang-et-al-2023)
   - [PaperBench](#32-paperbench-starace-et-al-2025)
   - [PaperCoder](#33-papercoder-seo-et-al-2025)
   - [AutoP2C](#34-autop2c-lin-et-al-2025)
   - [AutoReproduce](#35-autoreproduce-zhao-et-al-2025)
   - [Agent Laboratory](#36-agent-laboratory-schmidgall-et-al-2025)
   - [The AI Scientist](#37-the-ai-scientist-lu-et-al-2024--2025)
   - [MLR-Copilot](#38-mlr-copilot-li-et-al-2024)
4. [Capability Gap Matrix](#4-capability-gap-matrix)
5. [Summary & Positioning](#5-summary--positioning)
6. [References](#6-references)

---

## 1. The Research Landscape

The field of automated ML research sits on two key axes:

- **Horizontal axis (Goal):** Generate new research ←→ Replicate existing papers
- **Vertical axis (Execution depth):** No execution ←→ Full verification loop

```
▲  Execution depth
│
│  Full              ReproBot ★
│  verification       (ours)
│  loop        ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
│                     AutoReproduce ●  AutoP2C ●
│                   ·  ·  ·  ·  ·  ·  ·  ·  ·
│                     PaperBench ◆
│  AI Scientist ●
│  Agent Lab ●
│  MLR-Copilot ●              ·  ·  PaperCoder ●
│  MLAgentBench ◆
│  No execution
└──────────────────────────────────────────────►
   Generate new research    Replicate existing papers
                                               Goal

Legend:
  ● System / framework
  ◆ Benchmark
  ★ Our work

  Purple  — New research automation (AI Scientist, Agent Lab, MLR-Copilot)
  Coral   — Replication systems (PaperCoder, AutoReproduce, AutoP2C)
  Gray    — Benchmarks (MLAgentBench, PaperBench)
  Amber   — ML experimentation (MLAgentBench)
  Teal    — Our work (ReproBot)
```

**Key observation:** ReproBot is the only system targeting the top-right quadrant — faithful replication *combined with* a full iterative verification loop. The quadrant was empty before this work.

### ReproBot's architecture at a glance

Every deep dive below is measured against this reference architecture:

```
PDF paper
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                      Orchestrator                        │
│                  (shared memory state)                   │
│                                                            │
│   Reader  ──►  Coder  ──►  Runner  ──►  Critic            │
│  (pdfplumber    (HuggingFace   (Docker      (metric vs.    │
│   + VLM;        Trainer-       sandbox;     claim;         │
│   claims,       based          captures     pass/retry/    │
│   datasets,     script)        metrics +    fail verdict)  │
│   hparams)                     traces)                    │
│                                     ▲              │       │
│                                     └── retry ──────┘       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Structured Markdown replication report
(claim-by-claim comparison + gap analysis)
```

Five named roles, not four: the **Orchestrator** with its **shared memory state** is a first-class architectural element in its own right — it is what turns Reader/Coder/Runner/Critic from independent tools into a coordinated loop with a retry budget and a confidence threshold for stopping. Evaluation scope is deliberately narrow at launch: a benchmark of **20 image-classification papers**, trading PaperBench's topic breadth (12 topics) for depth of verification within a single, well-understood domain.

---

## 2. How the Field Evolved — Timeline

The problem of automated ML research has developed in three distinct waves over two years.

### Wave 1 — Foundations (2023)

| Paper | Venue | Date | Type |
|---|---|---|---|
| MLAgentBench | arXiv / Stanford | Oct 2023 | Benchmark |

The first benchmark for LLM agents doing ML experimentation. Establishes the vocabulary for the field but exposes hard failure modes (planning, hallucination) that motivate all subsequent work.

---

### Wave 2 — Full research automation (2024–2025)

| Paper | Venue | Date | Type |
|---|---|---|---|
| The AI Scientist | arXiv / Sakana AI | Aug 2024 | New research |
| MLR-Copilot | arXiv | Aug 2024 | New research |
| Agent Laboratory | arXiv | Jan 2025 | New research |
| The AI Scientist-v2 | arXiv / Sakana AI | Apr 2025 | New research |

The community focuses on generating *new* research, not verifying existing work. AI Scientist-v2 achieves the first fully AI-generated paper accepted at an ICLR workshop. Note that AI Scientist-v2's Apr 2025 release date actually overlaps with Wave 3 below — it is grouped here by topic (new-research generation), not strict chronology. These systems are complementary to ReproBot rather than competing — they create papers; ReproBot verifies them.

---

### Wave 3 — Replication systems (2025)

| Paper | Venue | Date | Type |
|---|---|---|---|
| PaperBench | arXiv / OpenAI | Apr 2025 | Benchmark |
| PaperCoder (Paper2Code) | arXiv / ICLR 2026 | Apr 2025 | Replication |
| AutoP2C | arXiv | Apr 2025 | Replication |
| AutoReproduce | arXiv / Tsinghua | May 2025 | Replication |

Four replication-focused works appear within about eight weeks of each other (PaperBench Apr 2 → AutoReproduce May 27, 2025). PaperBench sets the gold-standard evaluation framework; PaperCoder masters code generation but skips execution; AutoP2C adds multimodal parsing and an execute-debug loop but stops at "runs and looks paper-aligned," not a numeric-tolerance verdict; AutoReproduce adds execution but does not verify numerical agreement with paper claims either. ReproBot enters to close the remaining gap.

---

### Our contribution

| Paper | Venue | Date | Type |
|---|---|---|---|
| **ReproBot** | inzva AI Projects #10 | 2025 | Replication + verification |

---

## 3. Per-Paper Deep Dives

---

### 3.1 MLAgentBench (Huang et al., 2023)

**Full title:** MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation
**Authors:** Qian Huang, Jian Vora, Percy Liang, Jure Leskovec
**Affiliation:** Stanford University
**Link:** https://arxiv.org/abs/2310.03302

#### Core idea

The first benchmark suite for LLM agents doing ML experimentation. Introduces 13 tasks ranging from improving model performance on CIFAR-10 to more recent research challenges like BabyLM. For each task, a ReAct-based agent can perform actions including reading and writing files, executing code, and inspecting outputs. Agents based on Claude v1–v3, GPT-4, Gemini-Pro, and Mixtral are benchmarked; Claude v3 Opus achieves the best average success rate at 37.5%.

#### Architecture

```
Task description (pre-structured, no PDF parsing)
        │
        ▼
┌─────────────────────┐
│   ReAct agent        │
│   (flat loop)        │
│   Read → Plan → Act  │
│   → Observe → Repeat │
└─────────┬───────────┘
          │
    ┌─────▼──────┐
    │ Code exec  │   bash shell, Python executor
    └─────┬──────┘
          │
    Results (no claim comparison)
```

#### What it does well
- Establishes reproducible evaluation methodology for ML agents
- Provides interpretable plans and actions
- Identifies the key failure modes of LLM-based agents: long-term planning and hallucination

#### Key limitations vs. ReproBot
- Does not accept PDF papers as input — tasks are pre-structured by humans
- No claim extraction or numerical verification
- No vision or figure parsing
- Goal is to *improve* model performance, not faithfully replicate a specific paper's results
- No structured output report

#### Feature summary

| Capability | MLAgentBench |
|---|---|
| PDF + VLM paper parsing | ✗ |
| Code generation | ~ (partial, task-specific) |
| Code execution | ✓ |
| Metric vs paper claim comparison | ✗ |
| Critic / iterative refinement loop | ~ (flat retry only) |
| Structured replication report | ✗ |

#### Relevance to ReproBot

**Foundational context.** MLAgentBench proves that LLM agents can do ML experimentation but exposes hard blockers: long-term planning failure and hallucination. ReproBot's structured pipeline directly addresses these by separating concerns under a central Orchestrator with shared memory — the Reader (pdfplumber + VLM) handles understanding, the Coder (HuggingFace `Trainer`-based scripts) handles implementation, the Runner (Docker sandbox) handles execution, and the Critic handles verification. No single agent is asked to do everything, and MLAgentBench's own finding — that flat ReAct agents struggle with long-horizon planning — is the direct motivation for giving the Orchestrator, not any one agent, responsibility for the retry loop.

---

### 3.2 PaperBench (Starace et al., 2025)

**Full title:** PaperBench: Evaluating AI's Ability to Replicate AI Research
**Authors:** Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung et al.
**Affiliation:** OpenAI
**Link:** https://arxiv.org/abs/2504.01848

#### Core idea

The gold-standard benchmark for paper replication. Consists of 20 Spotlight and Oral papers from ICML 2024, spanning 12 topics including deep reinforcement learning, robustness, and probabilistic methods. Each paper is accompanied by a manually created rubric co-developed with one of the original paper's authors, specifying all necessary outcomes for replication. The benchmark contains 8,316 individually gradable outcomes across the 20 papers.

Two baseline agents are provided:
- **BasicAgent** — flat ReAct loop with bash shell and Python executor
- **IterativeAgent** — extends BasicAgent with step-by-step reproduction within a fixed time budget

#### Architecture (BasicAgent)

```
Paper PDF + rubric
        │
        ▼
┌─────────────────────┐
│   BasicAgent         │
│   ReAct framework    │
│   Bash + Python      │
│   executor           │
└─────────┬───────────┘
          │
    Rubric-graded
    replication score
    (8,316 outcomes)
```

#### What it does well
- Defines the most rigorous evaluation framework in the field
- Rubrics co-authored with original paper authors — ground truth is reliable
- Provides direct comparability across all systems evaluated on it
- Covers a diverse set of 12 ML research topics

#### Key limitations vs. ReproBot
- A benchmark, not a system — defines evaluation, not the solution
- BasicAgent is a flat ReAct loop with no specialised roles or shared memory
- No explicit Critic that compares reproduced metrics against claimed numbers
- No structured replication report generated for the user
- No VLM-based figure or architecture diagram parsing

#### Feature summary

| Capability | PaperBench (BasicAgent) |
|---|---|
| PDF + VLM paper parsing | ✓ |
| Code generation | ✓ |
| Code execution | ✓ |
| Metric vs paper claim comparison | ~ (via rubric, external) |
| Critic / iterative refinement loop | ~ (IterativeAgent only) |
| Structured replication report | ✗ |

#### Relevance to ReproBot

**Primary evaluation target.** ReproBot should be evaluated on PaperBench to achieve direct comparison with PaperCoder, AutoReproduce, and the BasicAgent/IterativeAgent baselines. The rubric-based grading methodology also informs the design of ReproBot's Critic agent — the rubric's pass/fail criteria map naturally to our pass/retry/fail verdict logic.

**Scope trade-off worth stating explicitly:** PaperBench spans 20 papers across 12 topics (RL, robustness, probabilistic methods, etc.); ReproBot's own benchmark deliberately narrows to 20 image-classification papers. This is a legibility choice, not a limitation to hide — a single-domain benchmark makes it possible to define a much stricter "confidence threshold" for the Critic (e.g., top-1 accuracy within a fixed tolerance of the reported number) than PaperBench's cross-domain rubric grading can, at the cost of not yet demonstrating breadth. PaperBench remains the right benchmark for showing ReproBot generalizes beyond its launch domain.

---

### 3.3 PaperCoder (Seo et al., 2025)

**Full title:** Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning
**Authors:** Minju Seo, Jinheon Baek, Seongyun Lee, Sung Ju Hwang
**Link:** https://arxiv.org/abs/2504.17192
**Venue:** ICLR 2026

#### Core idea

A multi-agent LLM framework that transforms ML papers into functional code repositories. PaperCoder operates in three sequential stages, each with specialised agents:

1. **Planning** — constructs a high-level roadmap, designs system architecture with diagrams, identifies file dependencies, generates configuration files
2. **Analysis** — interprets implementation-specific details
3. **Coding** — produces modular, dependency-aware code

Evaluated on a custom Paper2Code benchmark (NeurIPS/ICML/ICLR 2024 papers) and on PaperBench, with strong performance on code quality metrics and author-confirmed correctness.

#### Architecture

```
PDF paper
    │
    ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Planning  │───►│ Analysis │───►│  Coding  │
│  agents   │    │  agents  │    │  agents  │
└──────────┘    └──────────┘    └──────────┘
                                      │
                               Code repository
                               (not executed)
```

#### What it does well
- Best-in-class code generation quality on PaperBench
- Architecture diagram generation during the planning phase
- Modular, file-dependency-aware code structure
- Accepted at ICLR 2026 — rigorous peer review validation

#### Key limitations vs. ReproBot
- **Single-pass only** — generates a code repo but never executes it
- Cannot verify whether the output actually reproduces the paper's claimed numbers
- No iterative refinement loop — if the generated code is wrong, there is no retry
- No Critic agent or replication report
- No structured metric comparison table
- No VLM-based figure parsing (partial in planning phase only)

#### Feature summary

| Capability | PaperCoder |
|---|---|
| PDF + VLM paper parsing | ✓ |
| Code generation | ✓ |
| Code execution | ✗ |
| Metric vs paper claim comparison | ✗ |
| Critic / iterative refinement loop | ✗ |
| Structured replication report | ~ (code repo only) |

#### Relevance to ReproBot

**Closest architectural cousin.** PaperCoder's Planning → Analysis → Coding decomposition directly inspires ReproBot's Reader → Coder pipeline. The key differentiator: ReproBot's Coder targets a single, constrained output format — a self-contained HuggingFace `Trainer`-based training script — rather than PaperCoder's general multi-file repository, which is precisely what makes downstream execution by the Runner and verification by the Critic tractable. ReproBot trades PaperCoder's general-purpose code generation for a narrower, runnable-by-construction target. A useful framing: PaperCoder answers "does the code look right?" — ReproBot answers "does the code *produce the right numbers*?"

---

### 3.4 AutoP2C (Lin et al., 2025)

**Full title:** AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers
**Authors:** Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, Mingjun Xiao
**Link:** https://arxiv.org/abs/2504.20115

#### Core idea

Defines "Paper-to-Code" (P2C) as its own task category — distinct from ordinary code generation, which converts a textual description into an isolated snippet — because turning a paper into a faithful repository requires fusing multiple modalities (prose, equations, diagrams, tables) and producing a multi-file, dependency-consistent codebase. AutoP2C runs four sequential stages: (1) mine a structural "blueprint" template from established GitHub repositories, (2) parse the target paper's multimodal content into a distilled representation (OCR text + VLM-parsed figures + parsed equations/tables), (3) hierarchically decompose that representation into a file/class/function-level plan, (4) generate the repository file-by-file with an iterative execute → localize-error → correct-error debugging loop, plus a validation step that cross-checks architecture/loss/optimizer choices against the paper.

#### Architecture

```
Established repos (blueprint)        Target paper PDF (multimodal parse:
        │                             OCR + VLM figures + equations + tables)
        ▼                                       │
┌─────────────────┐               ┌────────────────────────┐
│ Repo blueprint    │               │ Distilled representation │
│ template           │               └────────────┬────────────┘
└─────────┬─────────┘                            │
          └──────────────────┬────────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │ Hierarchical task decomposition │
              │ (files → classes/functions →    │
              │  dependency graph → per-file     │
              │  task descriptions)              │
              └───────────────┬────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │ Iterative feedback-driven        │
              │ implementation: generate file →   │
              │ validate vs. paper → execute →     │
              │ localize error → correct → repeat  │
              └───────────────┬────────────────┘
                              ▼
                  Executable code repository
```

#### What it does well
- 100% executability across all 8 benchmark papers vs. 12.5% for OpenAI o1 and DeepSeek-R1 used as single-pass reasoning-model baselines
- Actually parses figures/diagrams with a VLM (not just text) and demonstrates it matters: an ablation removing architecture diagrams alone costs 21.9 absolute performance points
- Reports "relative performance vs. original" (avg. 99.5%, two of 8 papers ≥100%) — a genuine quantitative comparison against the paper's own claimed numbers, not just an executability check
- Ablation shows the iterative feedback-driven implementation stage is load-bearing: removing it drops executability to 0%

#### Key limitations vs. ReproBot
- No explicit Critic verdict (pass/retry/fail) gated by a numeric tolerance — "relative performance" is reported as a single aggregate figure with no stated pass/fail criterion
- No claim-by-claim or rubric-style comparison (contrast with PaperBench's rubric or ReproBot's planned per-claim report)
- Benchmark is small — 8 papers, ablations on only 4 — no PaperBench-scale evaluation
- No structured, human-readable replication report; no human evaluation of output quality
- Python-only, by the authors' own stated future work

#### Feature summary

| Capability | AutoP2C |
|---|---|
| PDF + VLM paper parsing | ✓ |
| Code generation | ✓ |
| Code execution | ✓ |
| Metric vs paper claim comparison | ~ (aggregate relative performance, no tolerance/verdict) |
| Critic / iterative refinement loop | ~ (execute–localize–correct loop, but no claim-driven stop condition) |
| Structured replication report | ✗ |

#### Relevance to ReproBot

**Closest single system to ReproBot's full pipeline shape.** AutoP2C is the only prior work reviewed here that combines multimodal paper parsing (Reader-equivalent), execution (Runner-equivalent), and an iterative debug loop (Critic-adjacent) in one system — PaperCoder has the first without the second two, AutoReproduce has the second and a thin version of the third without genuine multimodal parsing. Its ablation evidence that removing VLM-based figure parsing costs ~22 points of absolute performance is independent validation of ReproBot's own decision to give the Reader a VLM component rather than relying on text extraction alone. The key differentiator to cite: AutoP2C's iterative loop terminates on "the code runs and looks architecturally aligned with the paper," while ReproBot's Critic terminates on "the reproduced metric matches the claimed number within a stated tolerance" — aggregate relative-performance reporting is not the same as a claim-by-claim pass/retry/fail verdict. AutoP2C also explicitly critiques generic multi-agent coding frameworks (MetaGPT, CodeAgent, CodeCoR) as text-only and unable to consume diagrams or tables — the same gap ReproBot's VLM-equipped Reader is built to close, and a useful independent citation for why paper replication specifically needs multimodal input handling.

---

### 3.5 AutoReproduce (Zhao et al., 2025)

**Full title:** AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage
**Authors:** Xuanle Zhao, Zilin Sang, Yuxuan Li, Qi Shi et al.
**Affiliation:** Tsinghua University
**Link:** https://arxiv.org/abs/2505.20662

#### Core idea

Introduces the **paper lineage** algorithm — the key technical innovation. Paper lineage systematically mines the cited literature of a source paper to recover implicit, unstated implementation details: common architecture choices, data processing conventions, and domain-specific practices that experienced practitioners know but papers rarely write down.

Built on top of paper lineage, AutoReproduce is a three-stage multi-agent pipeline:

1. **Literature review** — surveys existing work
2. **Paper lineage** — mines cited papers and their code repositories for implicit knowledge
3. **Code development** — generates code, validated by a sampling-based unit testing strategy

Evaluated on PaperBench and on a custom ReproduceBench (13 papers with verified ground-truth implementations and 5 evaluation metrics).

#### Architecture

```
PDF paper
    │
    ├──► Literature review agent
    │         │
    │         ▼
    │    Cited papers + repos
    │         │
    └──► Paper lineage algorithm
              │
              ▼
         Code development agent
              │
         ┌────▼────────────────┐
         │ Sampling-based unit  │
         │ testing (executability│
         │ validation only)     │
         └────────────────────-┘
              │
         Generated code
         (executed, not claim-verified)
```

#### What it does well
- Paper lineage is a genuinely novel and effective idea for recovering implicit knowledge
- Actually executes generated code and checks for runtime errors
- Outperforms PaperCoder and BasicAgent on both PaperBench and ReproduceBench
- Sampling-based unit testing catches errors early without full training runs

#### Key limitations vs. ReproBot
- **No explicit metric comparison against paper claims** — success is measured by executability, not numerical replication fidelity
- No Critic-style verdict (pass / retry / fail) driven by the gap between reproduced and reported numbers
- No VLM-based figure or architecture diagram parsing
- No structured replication report with a claim-by-claim comparison table
- ReproduceBench has only 13 papers vs PaperBench's 20

#### Feature summary

| Capability | AutoReproduce |
|---|---|
| PDF + VLM paper parsing | ✓ |
| Code generation | ✓ |
| Code execution | ✓ |
| Metric vs paper claim comparison | ✗ |
| Critic / iterative refinement loop | ~ (unit testing only) |
| Structured replication report | ✗ |

#### Relevance to ReproBot

**Most sophisticated direct competitor.** AutoReproduce is the system ReproBot most directly extends. The paper lineage concept is worth considering as an additional tool for ReproBot's Reader or Coder agents — recovering unstated implementation details from cited papers is complementary to ReproBot's core contribution, not competing with it. Both systems execute generated code in an isolated environment (AutoReproduce's sampling-based unit tests vs. ReproBot's Docker sandbox), but AutoReproduce stops at "did it run without errors?" while ReproBot's Runner writes captured metrics back to shared memory specifically so the Critic can diff them against the paper's stated numbers. ReproBot adds what AutoReproduce leaves open: explicit numerical claim verification and a Critic-driven iterative refinement loop tied to the gap between reproduced and reported metrics.

---

### 3.6 Agent Laboratory (Schmidgall et al., 2025)

**Full title:** Agent Laboratory: Using LLM Agents as Research Assistants
**Authors:** Samuel Schmidgall, Yusheng Su et al.
**Link:** https://arxiv.org/abs/2501.04227

#### Core idea

Takes a human research idea as input and produces a research report and code repository via a multi-agent pipeline with three phases:

1. **Literature Review** — surveys related work to ground the research idea
2. **Experimentation** — designs and runs new experiments
3. **Report Writing** — authors a structured research report

Human feedback can be injected at configurable frequency — from fully autonomous to heavily supervised. The system is compute-flexible, adapting to available GPU/CPU resources.

#### Key difference from ReproBot

Agent Laboratory's goal is to produce *new* research from a human-given idea. ReproBot's goal is to faithfully *replicate* an existing paper's reported results. These are different tasks with different success metrics — Agent Laboratory is complementary, not competing.

#### Feature summary

| Capability | Agent Laboratory |
|---|---|
| PDF + VLM paper parsing | ~ (for literature review only) |
| Code generation | ✓ |
| Code execution | ✓ |
| Metric vs paper claim comparison | ✗ |
| Critic / iterative refinement loop | ✓ |
| Structured replication report | ✓ (research report) |

#### Relevance to ReproBot

**Architectural inspiration.** Agent Laboratory demonstrates that a Literature Review → Experimentation → Report Writing decomposition is a proven multi-agent architecture for ML research tasks. ReproBot adapts this pattern to the replication setting: the Reader corresponds to literature understanding, the Coder + Runner to experimentation, and the report generator to report writing. The human-in-the-loop design also informs ReproBot's escalation logic: when the Critic issues a `fail` verdict, a human should be notified rather than silently giving up.

---

### 3.7 The AI Scientist (Lu et al., 2024 & 2025)

**Full title (v1):** The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
**Full title (v2):** The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search
**Authors:** Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha et al.
**Affiliation:** Sakana AI
**Links:** https://arxiv.org/abs/2408.06292 (v1) · https://arxiv.org/abs/2504.08066 (v2)

#### Core idea

The most ambitious system in the field. End-to-end autonomous scientist:

- Generates novel research ideas
- Writes code for experiments
- Executes computational studies
- Analyses and visualises results
- Authors a complete scientific paper
- Runs a simulated peer review

**v2 additions:** Eliminates reliance on human-authored code templates; introduces progressive agentic tree search for experiment management; integrates a VLM feedback loop for iterative figure quality refinement. One of the three fully AI-generated manuscripts submitted to a peer-reviewed ICLR workshop achieved scores above the average human acceptance threshold — the first fully AI-generated paper to pass peer review.

#### Key difference from ReproBot

AI Scientist creates new papers from scratch. ReproBot verifies existing ones. These systems are complementary — a natural future integration would have AI Scientist generate a paper and ReproBot automatically attempt replication, feeding the verification verdict back to the authors as a form of automated peer review.

#### Feature summary

| Capability | AI Scientist v1/v2 |
|---|---|
| PDF + VLM paper parsing | ✗ / ~ (v2 VLM for figures) |
| Code generation | ✓ |
| Code execution | ✓ |
| Metric vs paper claim comparison | ✗ |
| Critic / iterative refinement loop | ✓ (tree search in v2) |
| Structured replication report | ✓ (full paper) |

#### Relevance to ReproBot

**Upstream context and future integration.** Not a direct competitor. The VLM feedback loop in v2 independently validates ReproBot's design choice of using vision-language models to parse figures. The tree search strategy in v2 is worth exploring as a future enhancement to ReproBot's retry mechanism — rather than a linear retry budget, a tree search could explore multiple code rewrite strategies in parallel and select the most promising one.

---

### 3.8 MLR-Copilot (Li et al., 2024)

**Full title:** MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents
**Authors:** Ruochen Li, Teerth Patel, Qingyun Wang, Xinya Du
**Link:** https://arxiv.org/abs/2408.14033

#### Core idea

Released about two weeks after the original AI Scientist paper (Aug 2024), MLR-Copilot targets the same "generate new research" problem via a three-stage pipeline:

1. **Idea Generation** — an `IdeaAgent`, powered by an RL-tuned LLM, reads existing papers and proposes feasible research ideas and experiment plans.
2. **Implementation** — an `ExperimentAgent` converts the plan into executable code, retrieving prototype code and candidate models/datasets from HuggingFace rather than generating everything from scratch.
3. **Execution & Iteration** — experiments run with debugging cycles and optional human feedback injected to raise the odds of a successful run.

Evaluated on five ML research tasks, with human feedback shown to meaningfully improve implementation success.

#### Architecture

```
Existing papers
    │
    ▼
┌───────────┐    ┌────────────────┐    ┌───────────────────┐
│ IdeaAgent  │───►│ ExperimentAgent │───►│ Execution + debug  │
│ (RL-tuned  │    │ (retrieves      │    │ cycles + optional  │
│  LLM)      │    │  HF prototype   │    │ human feedback     │
│            │    │  code/models)   │    │                    │
└───────────┘    └────────────────┘    └───────────────────┘
                                                  │
                                        Research idea + code
                                        (no claim verification)
```

#### What it does well
- Retrieval of prototype code and HuggingFace models/datasets grounds implementation in working artifacts rather than generating everything blind — a pragmatic idea also relevant to ReproBot's Coder
- RL-tuning the idea-generation step is a distinctive choice among its contemporaries (AI Scientist, Agent Laboratory use prompting alone)
- Human feedback loop is optional and configurable, not mandatory

#### Key limitations vs. ReproBot
- Goal is generating *new* research ideas, not replicating a specific paper's claimed results — no source-of-truth numbers to verify against
- No PDF/VLM parsing of figures, tables, or architecture diagrams — works from paper text for idea grounding only
- No explicit metric-vs-claim comparison or Critic-style verdict
- No structured, claim-by-claim replication report — output is a research idea and code, not a gap analysis

#### Feature summary

| Capability | MLR-Copilot |
|---|---|
| PDF + VLM paper parsing | ✗ (text-based literature grounding only) |
| Code generation | ✓ (retrieval-augmented) |
| Code execution | ✓ |
| Metric vs paper claim comparison | ✗ |
| Critic / iterative refinement loop | ~ (debug cycles + optional human feedback) |
| Structured replication report | ✗ |

#### Relevance to ReproBot

**Minor architectural inspiration, same "new research" category as AI Scientist and Agent Laboratory.** MLR-Copilot's retrieval-augmented implementation step is the most directly transferable idea: rather than having the Coder generate a HuggingFace `Trainer` script purely from the Reader's extracted description, ReproBot could retrieve a matching prototype implementation or base model from HuggingFace first and adapt it — reducing hallucinated API calls and improving the odds the Runner's first attempt actually executes. Like AI Scientist and Agent Laboratory, MLR-Copilot is complementary rather than competing: it has no notion of a ground-truth claim to verify against, which is exactly the gap ReproBot's Critic fills.

*Design note, not a competitor:* CodeAgent (Zhang et al., 2024, ACL — arXiv:2401.07339) is a tool-integrated agent for general repo-level code generation (doc search → symbol navigation → code-interpreter testing → iterative debug), not a paper-replication system — it never touches a PDF or a claimed metric, so it sits outside this landscape's Generate-new/Replicate-paper axis entirely. It's worth one citation purely for the Coder/Runner's tool-use pattern (test-then-debug via a sandboxed interpreter), not for positioning.

---

## 4. Capability Gap Matrix

The table below maps every system across six key capabilities. ReproBot is the only system that covers all six.

| System | PDF + VLM parsing | Code generation | Code execution | Metric vs claim | Critic loop | Structured report |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MLAgentBench (Huang, 2023) | ✗ | ~ | ✓ | ✗ | ~ | ✗ |
| PaperBench / BasicAgent (Starace, 2025) | ✓ | ✓ | ✓ | ~ | ~ | ✗ |
| PaperCoder (Seo, 2025) | ✓ | ✓ | ✗ | ✗ | ✗ | ~ |
| AutoP2C (Lin, 2025) | ✓ | ✓ | ✓ | ~ | ~ | ✗ |
| AutoReproduce (Zhao, 2025) | ✓ | ✓ | ✓ | ✗ | ~ | ✗ |
| Agent Laboratory (Schmidgall, 2025) | ~ | ✓ | ✓ | ✗ | ✓ | ✓ |
| AI Scientist v1/v2 (Lu, 2024) | ✗ / ~ | ✓ | ✓ | ✗ | ✓ | ✓ |
| MLR-Copilot (Li, 2024) | ✗ | ✓ | ✓ | ✗ | ~ | ✗ |
| **ReproBot (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**Key:** ✓ = full support · ~ = partial · ✗ = not supported

### What the gap is

```
Execution
depth
  ▲
  │                          ┌─────────────────────────┐
  │  Full verification       │                         │
  │  loop                    │   ← Gap ReproBot        │
  │                          │     fills               │
  │                          │                         │
  │    · · · · · · · · · · · │ · · · · · · · · · · · · │
  │                          │                         │
  │  AutoReproduce ●         │                         │
  │  AutoP2C ●               │                         │
  │                          └─────────────────────────┘
  │    PaperBench ◆
  │
  │    PaperCoder ●
  │
  │  No execution
  └────────────────────────────────────────────────────►
     Code generation only      Execution + verification
```

The critical missing piece in every prior replication system is the explicit comparison of reproduced metrics against the paper's stated claims, combined with a Critic-driven iterative retry loop.

- **PaperCoder** generates code but never runs it
- **AutoP2C** runs the code and reports an aggregate relative-performance number, but has no numeric-tolerance pass/fail verdict or claim-by-claim comparison
- **AutoReproduce** runs the code but only checks executability, not numerical agreement
- **PaperBench BasicAgent** can run code but has no Critic that flags the gap and retries
- **ReproBot** closes the loop: run → compare → decide → retry if needed → report

---

## 5. Summary & Positioning

### The story in three acts

**Act 1 (2023) — Foundations.** MLAgentBench proves that LLM agents can do ML experimentation and identifies the hard problems: long-term planning and hallucination. The field has a benchmark but no dedicated solutions.

**Act 2 (2024–2025) — New research automation.** The community turns to the more exciting problem of *generating* new research. AI Scientist, MLR-Copilot, and Agent Laboratory demonstrate impressive capabilities but leave the replication problem untouched.

**Act 3 (2025) — Replication systems emerge.** PaperBench sets the evaluation standard. PaperCoder, AutoP2C, and AutoReproduce appear within about eight weeks of each other, each solving part of the problem. PaperCoder: code generation without execution. AutoP2C: multimodal parsing and execution with an aggregate performance number, but no numeric-tolerance verdict. AutoReproduce: execution without numerical claim verification. The gap remains.

**ReproBot:** Closes the gap. The Critic agent that explicitly asks "did we match the claimed number?" and uses the answer to drive a structured retry loop is the contribution no prior system provides.

### How to cite ReproBot against each competitor

| Competitor | One-sentence differentiator |
|---|---|
| MLAgentBench | MLAgentBench evaluates general ML experimentation; ReproBot specifically targets faithful replication of existing papers with explicit claim verification. |
| PaperBench BasicAgent | BasicAgent is a flat ReAct loop; ReproBot replaces it with four specialised agents, shared memory, and a Critic that drives metric-guided iteration. |
| PaperCoder | PaperCoder generates code but never executes it; ReproBot treats execution and numerical verification as first-class requirements. |
| AutoP2C | AutoP2C parses multimodal paper content and executes the result, reporting an aggregate relative-performance number; ReproBot's Critic gates a pass/retry/fail verdict on a claim-by-claim numeric tolerance instead of a single aggregate figure. |
| AutoReproduce | AutoReproduce checks code executability; ReproBot's Critic explicitly compares reproduced metrics against paper claims and iterates until they agree. |
| Agent Laboratory | Agent Laboratory generates new research from ideas; ReproBot replicates existing research from PDF papers — different task, different success metric. |
| AI Scientist | AI Scientist creates papers; ReproBot verifies them — they are complementary and could be integrated into a closed-loop autonomous research pipeline. |
| MLR-Copilot | MLR-Copilot generates and implements new research ideas with retrieval-augmented code; ReproBot has no notion of "new" — it faithfully reproduces one specific paper's already-published claims. |

---

## 6. References

| # | Citation | Link |
|---|---|---|
| 1 | Huang, Q., Vora, J., Liang, P., & Leskovec, J. (2023). MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation. *arXiv:2310.03302* | https://arxiv.org/abs/2310.03302 |
| 2 | Starace, G., Jaffe, O., Sherburn, D., Aung, J., et al. (2025). PaperBench: Evaluating AI's Ability to Replicate AI Research. *arXiv:2504.01848* | https://arxiv.org/abs/2504.01848 |
| 3 | Seo, M., Baek, J., Lee, S., & Hwang, S. J. (2025). Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning. *arXiv:2504.17192* (ICLR 2026) | https://arxiv.org/abs/2504.17192 |
| 4 | Lin, Z., Shen, Y., Cai, Q., Sun, H., Zhou, J., & Xiao, M. (2025). AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers. *arXiv:2504.20115* | https://arxiv.org/abs/2504.20115 |
| 5 | Zhao, X., Sang, Z., Li, Y., Shi, Q., et al. (2025). AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage. *arXiv:2505.20662* | https://arxiv.org/abs/2505.20662 |
| 6 | Schmidgall, S., Su, Y., et al. (2025). Agent Laboratory: Using LLM Agents as Research Assistants. *arXiv:2501.04227* | https://arxiv.org/abs/2501.04227 |
| 7 | Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv:2408.06292* | https://arxiv.org/abs/2408.06292 |
| 8 | Yamada, Y., Lange, R. T., Lu, C., et al. (2025). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. *arXiv:2504.08066* | https://arxiv.org/abs/2504.08066 |
| 9 | Li, R., Patel, T., Wang, Q., & Du, X. (2024). MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents. *arXiv:2408.14033* | https://arxiv.org/abs/2408.14033 |
| 10 | Zhang, K., Li, J., Li, G., Shi, X., & Jin, Z. (2024). CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-Level Coding Challenges. *arXiv:2401.07339* (ACL 2024) | https://arxiv.org/abs/2401.07339 |

---

*ReproBot literature review — inzva AI Projects #10*
*Generated as part of the project proposal and research documentation package.*
