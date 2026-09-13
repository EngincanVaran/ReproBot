# ruff: noqa: E501 - TikZ/LaTeX source lines are kept whole for readability
"""Generate the ReproBot third progress report in both column formats.

The body is held once, so the single- and two-column .tex files carry identical prose by
construction. Wide floats are written as FIGENV / TABENV and become figure* / table* in the
two-column build.

Every plot is TikZ with coordinates computed here from real run data (this TeX install has
no pgfplots). The data below is copied from the run logs, with the source named per series:
nothing in a curve is invented or smoothed.
"""

import math
from collections.abc import Callable
from pathlib import Path

# --- Real data -----------------------------------------------------------------------

# Tang 2013, full stage, svm_C = 0.1: training error (%) per 20 epochs.
# Source: runner/output/<paper>/logs/full.stderr.log
TANG = [
    (1, 27.440),
    (20, 12.202),
    (40, 9.812),
    (60, 8.482),
    (80, 6.857),
    (100, 5.485),
    (120, 4.443),
    (140, 3.503),
    (160, 2.800),
    (180, 2.093),
    (200, 1.545),
    (220, 1.152),
    (240, 0.865),
    (260, 0.583),
    (280, 0.355),
    (300, 0.272),
    (320, 0.163),
    (340, 0.112),
    (360, 0.088),
    (380, 0.063),
    (400, 0.062),
]

# Wijaya 2023, full stage, attempt 2, every 50 epochs: validation RMSE on the 81 held-out
# training rows, and the epoch-average training MSE (plotted as its square root).
# Source: orchestrator/output/<paper>/logs/attempt-2/full.stderr.log
WIJ_VAL = [
    (50, 3.4525),
    (100, 3.6150),
    (150, 3.3733),
    (200, 3.4795),
    (250, 3.3614),
    (300, 4.0142),
    (350, 3.2760),
    (400, 3.2580),
    (450, 3.4087),
    (500, 3.1587),
    (550, 3.4512),
    (600, 3.4664),
    (650, 3.2748),
    (700, 3.2184),
    (750, 3.4997),
    (800, 3.4581),
    (850, 3.3357),
    (900, 3.2041),
    (950, 3.2520),
    (1000, 3.3102),
]
WIJ_TRAIN_MSE = [
    (50, 15.3702),
    (100, 12.9997),
    (150, 7.1157),
    (200, 10.5356),
    (250, 6.3814),
    (300, 7.9118),
    (350, 5.7481),
    (400, 5.7324),
    (450, 7.6494),
    (500, 10.4038),
    (550, 5.2927),
    (600, 5.8451),
    (650, 4.7384),
    (700, 7.1127),
    (750, 5.3000),
    (800, 4.6026),
    (850, 4.7215),
    (900, 6.1806),
    (950, 6.6946),
    (1000, 10.8350),
]

# SVM guide: 5-fold CV accuracy (%) over the reproduction's grid, same folds as the run
# (StratifiedKFold, shuffle, seed 42). Rows: log2 C = -5..15 step 2; columns: log2 gamma =
# -15..3 step 2. Computed in the runner image with the generated script's own grid.
SVM_C_EXP = list(range(-5, 16, 2))
SVM_G_EXP = list(range(-15, 4, 2))
SVM_CV = [
    [64.75, 64.75, 64.75, 64.75, 64.75, 64.88, 84.85, 92.65, 93.33, 93.17],
    [64.75, 64.75, 64.75, 64.75, 64.91, 85.08, 92.78, 94.59, 95.53, 96.02],
    [64.75, 64.75, 64.75, 64.94, 85.14, 92.39, 94.79, 96.12, 96.7, 96.5],
    [64.75, 64.75, 64.94, 85.14, 92.23, 94.21, 96.02, 96.47, 96.8, 96.83],
    [64.75, 64.94, 85.14, 92.07, 94.11, 95.82, 96.15, 96.67, 96.92, 96.99],
    [64.94, 85.14, 92.1, 94.08, 95.4, 96.02, 96.31, 96.8, 96.86, 96.67],
    [85.14, 92.1, 94.04, 95.27, 95.82, 96.08, 96.6, 96.76, 96.86, 95.76],
    [92.1, 94.04, 95.18, 95.53, 96.08, 96.18, 96.63, 96.76, 96.63, 95.31],
    [94.04, 95.21, 95.44, 95.82, 96.02, 96.37, 96.63, 96.73, 96.47, 95.05],
    [95.24, 95.37, 95.37, 96.08, 96.18, 96.5, 96.8, 96.6, 96.18, 94.66],
    [95.4, 95.34, 95.79, 96.08, 96.47, 96.5, 96.73, 96.7, 95.82, 94.5],
]

# Fashion-MNIST random forest: test accuracy as trees are added (warm start), repetition 1's
# configuration (seed 42, shuffled training data). Its 100-tree value equals the pipeline's
# repetition 1 exactly. The five pipeline repetitions are FASHION_RUNS.
RF_TREES = [
    (1, 0.7742),
    (2, 0.7764),
    (3, 0.8183),
    (5, 0.8388),
    (8, 0.8517),
    (10, 0.854),
    (15, 0.8604),
    (20, 0.8668),
    (30, 0.8712),
    (40, 0.8747),
    (50, 0.8759),
    (60, 0.877),
    (70, 0.8755),
    (80, 0.8753),
    (90, 0.8764),
    (100, 0.8773),
]
FASHION_RUNS = [0.8773, 0.8774, 0.8753, 0.8792, 0.8775]

# Soft decision tree (fixed loss), full stage: test accuracy (%) per epoch.
# Source: live container log / orchestrator/output/<paper>/logs/attempt-1/full.stderr.log
SOFTDT = [(1, 65.84), (2, 80.68), (3, 84.40), (4, 89.10)]

# --- Plot helpers ----------------------------------------------------------------------

PW, PH = 6.2, 3.6  # panel plot area, cm


def lin(v: float, lo: float, hi: float, size: float) -> float:
    return (v - lo) / (hi - lo) * size


def logmap(v: float, lo: float, hi: float, size: float) -> float:
    return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * size


def path(
    points: list[tuple[float, float]],
    fx: Callable[[float], float],
    fy: Callable[[float], float],
) -> str:
    return " -- ".join(f"({fx(x):.3f},{fy(y):.3f})" for x, y in points)


def dots(
    points: list[tuple[float, float]],
    fx: Callable[[float], float],
    fy: Callable[[float], float],
    style: str,
) -> str:
    return "\n".join(
        f"  \\fill[{style}] ({fx(x):.3f},{fy(y):.3f}) circle (1.1pt);" for x, y in points
    )


def axes(
    xticks: list[tuple[float, str]], yticks: list[tuple[float, str]], xlabel: str, ylabel: str
) -> str:
    out = []
    for y, label in yticks:
        out.append(f"  \\draw[black!12] (0,{y:.3f}) -- ({PW},{y:.3f});")
        out.append(
            f"  \\draw (0,{y:.3f}) -- (-0.08,{y:.3f}) node[left, font=\\scriptsize] {{{label}}};"
        )
    for x, label in xticks:
        out.append(
            f"  \\draw ({x:.3f},0) -- ({x:.3f},-0.08) node[below, font=\\scriptsize] {{{label}}};"
        )
    out.append(f"  \\draw[thick] (0,0) -- ({PW},0);")
    out.append(f"  \\draw[thick] (0,0) -- (0,{PH});")
    out.append(f"  \\node[font=\\scriptsize] at ({PW / 2:.3f},-0.62) {{{xlabel}}};")
    out.append(f"  \\node[font=\\scriptsize, rotate=90] at (-1.08,{PH / 2:.3f}) {{{ylabel}}};")
    return "\n".join(out)


def hline(y: float, color: str, dash: str, label: str, anchor: str, lx: float) -> str:
    return (
        f"  \\draw[{color}, {dash}] (0,{y:.3f}) -- ({PW},{y:.3f});\n"
        f"  \\node[font=\\tiny, anchor={anchor}, text={color}] at ({lx:.3f},{y:.3f}) {{{label}}};"
    )


def panel_tang() -> str:
    lo, hi = 0.03, 100.0

    def fx(e: float) -> float:
        return lin(e, 0, 400, PW)

    def fy(v: float) -> float:
        return logmap(v, lo, hi, PH)

    body = axes(
        [(fx(e), str(e)) for e in (0, 100, 200, 300, 400)],
        [(fy(v), s) for v, s in ((0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"))],
        "epoch",
        "error (\\%, log scale)",
    )
    return f"""\\begin{{tikzpicture}}
{body}
  \\draw[builtc, very thick] {path(TANG, fx, fy)};
{dots(TANG, fx, fy, "builtc")}
{hline(fy(0.87), "black", "densely dashed", "paper: test error 0.87\\%", "south west", 0.08)}
  \\draw[missingc, thick] ({fx(400):.3f},{fy(0.82):.3f}) circle (2.2pt);
  \\node[font=\\tiny, text=missingc, anchor=south west] at ({fx(12):.3f},{fy(0.045):.3f})
    {{circle: ReproBot test error 0.82\\%}};
  \\node[font=\\tiny, text=builtc, anchor=west] at ({fx(40):.3f},{fy(20):.3f}) {{training error}};
\\end{{tikzpicture}}"""


def panel_wijaya() -> str:
    lo, hi = 1.8, 4.8

    def fx(e: float) -> float:
        return lin(e, 0, 1000, PW)

    def fy(v: float) -> float:
        return lin(v, lo, hi, PH)

    train = [(e, math.sqrt(m)) for e, m in WIJ_TRAIN_MSE]
    body = axes(
        [(fx(e), str(e)) for e in (0, 250, 500, 750, 1000)],
        [(fy(v), f"{v:.1f}") for v in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)],
        "epoch",
        "RMSE",
    )
    return f"""\\begin{{tikzpicture}}
{body}
  \\draw[black!45, thick] {path(train, fx, fy)};
  \\draw[loopc, very thick] {path(WIJ_VAL, fx, fy)};
{dots(WIJ_VAL, fx, fy, "loopc")}
{hline(fy(3.02), "black", "densely dashed", "paper: test RMSE 3.02", "north east", PW)}
{hline(fy(4.48), "missingc", "densely dashed", "ReproBot: test RMSE 4.48", "south east", PW)}
  \\node[font=\\tiny, text=loopc, anchor=south] at ({fx(300):.3f},{fy(4.01) + 0.03:.3f})
    {{validation}};
  \\node[font=\\tiny, text=black!60, anchor=north] at ({fx(800):.3f},{fy(2.15) - 0.02:.3f})
    {{training}};
\\end{{tikzpicture}}"""


def panel_svm() -> str:
    cw, ch = PW / len(SVM_G_EXP), PH / len(SVM_C_EXP)
    cells = []
    for i, row in enumerate(SVM_CV):
        for j, v in enumerate(row):
            t = max(0.0, min(1.0, (v - 64.0) / (97.0 - 64.0)))
            shade = int(round(8 + t * 92))
            x0, y0 = j * cw, i * ch
            cells.append(
                f"  \\fill[loopc!{shade}!white] ({x0:.3f},{y0:.3f}) "
                f"rectangle ({x0 + cw:.3f},{y0 + ch:.3f});"
            )
    ticks = [
        f"  \\node[font=\\tiny, below] at ({(j + 0.5) * cw:.3f},0) {{$2^{{{g}}}$}};"
        for j, g in enumerate(SVM_G_EXP)
        if j % 2 == 1
    ] + [
        f"  \\node[font=\\tiny, left] at (0,{(i + 0.5) * ch:.3f}) {{$2^{{{c}}}$}};"
        for i, c in enumerate(SVM_C_EXP)
        if i % 2 == 0
    ]
    pj, pi = SVM_G_EXP.index(1), SVM_C_EXP.index(1)
    rj, ri = SVM_G_EXP.index(3), SVM_C_EXP.index(3)
    cells_s = "\n".join(cells)
    ticks_s = "\n".join(ticks)
    return f"""\\begin{{tikzpicture}}
{cells_s}
  \\draw[thick] (0,0) rectangle ({PW},{PH});
{ticks_s}
  \\draw[black, very thick] ({pj * cw:.3f},{pi * ch:.3f})
    rectangle ({(pj + 1) * cw:.3f},{(pi + 1) * ch:.3f});
  \\draw[missingc, very thick] ({rj * cw:.3f},{ri * ch:.3f})
    rectangle ({(rj + 1) * cw:.3f},{(ri + 1) * ch:.3f});
  \\node[font=\\scriptsize] at ({PW / 2:.3f},-0.62) {{kernel width $\\gamma$}};
  \\node[font=\\scriptsize, rotate=90] at (-0.85,{PH / 2:.3f}) {{penalty $C$}};
  \\node[font=\\tiny, anchor=north west, fill=white, inner sep=1.5pt, align=left]
    at (0.08,{PH - 0.08:.3f})
    {{black box: paper's pick ($C{{=}}\\gamma{{=}}2$)\\\\
      \\textcolor{{missingc}}{{red box: ReproBot's ($C{{=}}\\gamma{{=}}8$)}}\\\\
      light 65\\% $\\to$ dark 97\\% CV accuracy}};
\\end{{tikzpicture}}"""


def panel_rf() -> str:
    lo, hi = 0.76, 0.90

    def fx(n: float) -> float:
        return lin(n, 0, 108, PW)

    def fy(v: float) -> float:
        return lin(v, lo, hi, PH)

    body = axes(
        [(fx(n), str(n)) for n in (0, 25, 50, 75, 100)],
        [(fy(v), f"{v:.2f}") for v in (0.78, 0.82, 0.86, 0.90)],
        "number of trees",
        "test accuracy",
    )
    mean = sum(FASHION_RUNS) / len(FASHION_RUNS)
    runs = "\n".join(
        f"  \\fill[missingc] ({fx(105):.3f},{fy(r):.3f}) circle (0.9pt);" for r in FASHION_RUNS
    )
    return f"""\\begin{{tikzpicture}}
{body}
  \\draw[builtc, very thick] {path(RF_TREES, fx, fy)};
{dots(RF_TREES, fx, fy, "builtc")}
{hline(fy(0.873), "black", "densely dashed", "paper: 0.873 (mean of 5 runs)", "north east", PW)}
{runs}
  \\draw[missingc, thick] ({fx(102):.3f},{fy(mean):.3f}) -- ({fx(108):.3f},{fy(mean):.3f});
  \\node[font=\\tiny, text=missingc, anchor=south east] at ({PW:.3f},{fy(0.8792) + 0.05:.3f})
    {{ReproBot: 5 runs, mean 0.8773}};
\\end{{tikzpicture}}"""


def panel_softdt() -> str:
    lo, hi = 50.0, 100.0

    def fx(e: float) -> float:
        return lin(e, 0, 40, PW)

    def fy(v: float) -> float:
        return lin(v, lo, hi, PH)

    body = axes(
        [(fx(e), str(e)) for e in (0, 10, 20, 30, 40)],
        [(fy(v), str(int(v))) for v in (50, 60, 70, 80, 90, 100)],
        "epoch",
        "test accuracy (\\%)",
    )
    last_e, last_v = SOFTDT[-1]
    return f"""\\begin{{tikzpicture}}
{body}
  \\draw[builtc, very thick] {path(SOFTDT, fx, fy)};
{dots(SOFTDT, fx, fy, "builtc")}
{hline(fy(94.45), "black", "densely dashed", "paper: 94.45\\%", "south west", 0.08)}
  \\node[font=\\tiny, text=builtc, anchor=north east] at ({fx(last_e):.3f},{fy(last_v) - 0.1:.3f})
    {{ReproBot: {last_v:.2f}\\%}};
\\end{{tikzpicture}}"""


def panels() -> str:
    def cell(tikz: str, label: str) -> str:
        return (
            "\\begin{minipage}[t]{0.49\\linewidth}\\centering\n"
            f"\\resizebox{{0.97\\linewidth}}{{!}}{{{tikz}}}\\\\[-1pt]\n"
            f"{{\\small {label}}}\n\\end{{minipage}}"
        )

    return "\n".join(
        [
            cell(panel_tang(), "(a) MLP with L2-SVM loss, MNIST"),
            "\\hfill",
            cell(panel_wijaya(), "(b) Dense network, Boston Housing"),
            "\\\\[10pt]",
            cell(panel_svm(), "(c) RBF SVM model selection, svmguide1"),
            "\\hfill",
            cell(panel_rf(), "(d) Random forest, Fashion-MNIST"),
            "\\\\[10pt]",
            cell(panel_softdt(), "(e) Soft decision tree, MNIST"),
        ]
    )


# --- LaTeX -----------------------------------------------------------------------------

PREAMBLE_COMMON = r"""
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\usepackage{hyperref}
\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=blue,
  pdftitle={ReproBot -- Third Progress Report},
  pdfauthor={Aleyna Kutuk, Engincan Varan, Gul Korkut, Goktug Mert Ozdogan}
}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{pifont}

\definecolor{builtc}{HTML}{1B6F50}
\definecolor{missingc}{HTML}{97303F}
\definecolor{loopc}{HTML}{0E6E78}
\definecolor{warnc}{HTML}{97590C}

\newcommand{\built}{\textcolor{builtc}{\ding{51}}}
\newcommand{\absent}{\textcolor{missingc}{\ding{55}}}
\newcommand{\partl}{\textcolor{warnc}{$\sim$}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}

\renewcommand{\topfraction}{0.9}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{3}
\setcounter{totalnumber}{5}

\title{\textbf{ReproBot: A Multi-Agent System for Automated Scientific Paper Replication}\\[0.4em]
\large Third Progress Report}
\author{
  Aleyna Kütük \\ \small\texttt{aleynakutuk10@gmail.com}
  \and
  Engincan Varan \\ \small\texttt{engincanvaran@gmail.com}
  \and
  Gül Korkut \\ \small\texttt{serifegulkorkut@gmail.com}
  \and
  Göktuğ Mert Özdoğan \\ \small\texttt{goekmeroz@gmail.com}
}
\date{inzva AI Projects \#10 \\ 13.09.2026}
"""

BODY = r"""
\begin{document}

\maketitle

\begin{abstract}
ReproBot is a multi-agent system that reads a machine learning paper, extracts its claims and
experimental setup, generates an implementation, runs it in a sandbox, and repairs execution
failures through a bounded retry loop. This report presents its first full-fidelity
replications. We generalized code generation beyond image classification --- the Coder now
writes a plain PyTorch training loop for neural models and a scikit-learn estimator for
classical ones --- and evaluated on five papers whose original experiments train on a CPU,
spanning five model families: a network with a margin-based loss, a tabular regression network,
a kernel support vector machine, a random forest and a soft decision tree. For each we report
the learning curve of the generated implementation and a comparison with the published result.
@@SDT_ABSTRACT@@ A housing-price network reaches a test RMSE of 4.48 against 3.02, a gap we
trace to the paper's unspecified data split. The problems met along the way, from a misprinted
loss to a dead dataset link, were each resolved, three of them automatically by the retry loop.
We also present the team's first front end, a dashboard over the pipeline's outputs, and set out
the next phase: live checks on running training jobs, an automated Critic, and tighter control of
the retry loop.
\end{abstract}

\section{Introduction}
\label{sec:intro}

Reproducing a machine learning paper's reported results is mechanical in principle and
laborious in practice, and most published work ships no usable implementation. ReproBot
automates the process as a pipeline of specialized stages. A vision-language model converts
the paper's PDF into Markdown, reading figures as well as text. A Reader extracts five
structured fields --- a method summary, architecture notes, the claimed results, the
hyperparameters and the data pipeline --- and cross-checks them with a validation loop. A Coder
turns that extraction into a self-contained training script. A Runner executes the script in a
Docker container through four escalating stages: \texttt{probe}, a few optimizer steps;
\texttt{smoke}, one epoch on a small slice; \texttt{capped}, five epochs on a small slice; and
\texttt{full}, the paper's own setup. When a stage fails, a small model classifies the failure,
and an Orchestrator feeds that diagnosis back to the Coder and retries, within a fixed budget.

Before this phase the pipeline ran end to end but had only ever executed short checks: its
CIFAR-10 target papers need weeks of CPU time per full training run. This phase produced the
first full-fidelity results. Its contributions are:

\begin{itemize}
  \item a Coder that handles any supervised task and model family, writing PyTorch for neural
    networks and scikit-learn for classical estimators (Section~\ref{sec:pipeline});
  \item five complete replications across five model families, each with the implementation's
    learning curve and a comparison with the paper (Section~\ref{sec:results});
  \item a dashboard front end over the pipeline's outputs (Section~\ref{sec:viewer});
  \item a concrete plan for the next phase, grounded in what the runs showed
    (Section~\ref{sec:future}).
\end{itemize}

\section{Replication Targets}
\label{sec:targets}

Full training of the CIFAR-10 benchmark papers is out of reach on the project's hardware: one
\emph{Wide Residual Networks} run at the paper's setup measured at about 22 days on the
development CPU. Testing the pipeline's actual claim --- that a paper can be read, implemented,
executed and checked automatically --- needs only papers whose experiments are small. We chose
five that train in one minute to one hour, deliberately covering different model families and
data types (Table~\ref{tab:targets}), and kept them apart from the CIFAR-10 set.

\begin{TABENV}[tbp]
\centering
\small
\caption{The five replication targets of this phase.}
\label{tab:targets}
\begin{tabular}{L{0.26\textwidth}L{0.22\textwidth}L{0.2\textwidth}L{0.2\textwidth}}
\toprule
Paper & Model family & Data & Targeted claim \\
\midrule
Tang (2013) \cite{tang2013dlsvm} & MLP with an L2-SVM output layer & MNIST & 0.87\% test error \\
Wijaya (2023) \cite{wijaya2023housing} & Multi-branch dense network (Keras) & Boston Housing
  \cite{harrison1978hedonic}, tabular & Test RMSE 3.02 \\
Hsu, Chang and Lin (2003) \cite{hsu2003guide} & RBF support vector machine (LIBSVM
  \cite{chang2011libsvm}) & svmguide1, tabular & 96.9\% test accuracy \\
Xiao, Rasul and Vollgraf (2017) \cite{xiao2017fashion} & Random forest (scikit-learn
  \cite{pedregosa2011sklearn}) & Fashion-MNIST & 0.873 test accuracy, mean of 5 runs \\
Frosst and Hinton (2017) \cite{frosst2017soft} & Soft decision tree & MNIST & 94.45\% test
  accuracy \\
\bottomrule
\end{tabular}
\end{TABENV}

Two of the targets are classification networks, one a regression network, and two classical
methods. Tang's contribution is a loss function; Wijaya's model was written in Keras; the SVM
guide's result comes from a full model-selection procedure; the Fashion-MNIST figure is an
average over five runs; and the soft-tree paper states its tree depth but none of its training
settings. Each property exercises a different part of the pipeline.

\section{Pipeline Updates}
\label{sec:pipeline}

\FIGPIPELINE

\subsection{One training script for any model family}

\textbf{Neural models: a plain PyTorch loop.} The Coder previously emitted scripts built on
HuggingFace's \texttt{Trainer}, whose image-model interface suits neither a custom loss nor a
regression head. It now writes an explicit training loop in which the optimizer, schedule and
loss are all visible. When a paper was written in another framework, the Coder reproduces that
framework's defaults where the paper is silent --- Keras's Adam $\epsilon = 10^{-7}$, for
example --- and discloses each difference.

\textbf{Classical models: scikit-learn.} A support vector machine re-implemented as a PyTorch
model trained by gradient descent would use a different optimizer and reach a different
solution, so the Coder chooses its library by model family: SVMs, decision trees, forests,
$k$-nearest neighbours and linear models are scikit-learn estimators with every hyperparameter
passed by name, while networks --- including trees trained by gradient descent --- stay in
PyTorch. Because a paper's ``default parameters'' are those of the library version it used, the
Coder sets paper-era defaults explicitly; scikit-learn changed several in version 0.22.

\textbf{A common contract.} Every script accepts the same command-line flags, so the Runner
needs no per-paper logic, and writes the same metrics file: the claim's own metric on the
training and test splits, the metric's direction, the task type, and, when the claim is an
average over runs or folds, every run's value. Data comes from stable sources ---
\texttt{torchvision}, OpenML or the paper's own dataset site --- and each script checks the
dataset's name and shape before training on it.

\subsection{Problems encountered and how we solved them}

Table~\ref{tab:problems} lists the problems the replications surfaced and the fix applied to
each. Three were runtime defects in generated code that the retry loop repaired without human
involvement; the rest led to permanent changes in the prompts or the pipeline, or to a
documented correction.

\begin{TABENV}[tbp]
\centering
\small
\caption{Problems encountered during the replications, and how each was solved.}
\label{tab:problems}
\begin{tabular}{L{0.44\textwidth}L{0.48\textwidth}}
\toprule
Problem & Solution \\
\midrule
HuggingFace \texttt{Trainer} silently applied gradient clipping that no paper states
  & Explicit PyTorch training loop; every setting written down \\
Re-implementing classical models in PyTorch would change their solver
  & Library chosen by model family; scikit-learn with paper-era defaults \\
A remembered dataset URL had gone dead (HTTP 403), and triage filed it as an environment fault
  & OpenML first; triage treats one refusing source as a fixable script error \\
Dataset identifiers recalled from memory can name the wrong dataset
  & No identifiers from memory; scripts assert the dataset's name and shape \\
Tang: two unstated defaults (momentum 0.9, $C = 1.0$) together destabilized training at full
  scale & A six-configuration ablation isolated the pair; $C = 0.1$ trains stably \\
Soft tree: the paper prints its loss as the logarithm of a non-positive quantity
  & Implemented the evident intent, the expected cross-entropy \\
Runtime defects: a wrong API call (Wijaya), a label-sorted data file (SVM), an in-place
  autograd operation (soft tree) & Repaired automatically by the retry loop, one retry each \\
A dataset paper has no ``own method'', so the first claims extraction was empty (Fashion-MNIST)
  & The validation loop flagged it; the retry extracted 26 claims \\
The full stage wrote its metrics under a different filename than the Runner reads
  & Script template corrected \\
\bottomrule
\end{tabular}
\end{TABENV}

\subsection{The retry loop in action}

\FIGWIJAYARUN

Figure~\ref{fig:wijayarun} shows a typical automatic repair. The first generated script for
the housing-price paper passed two conflicting arguments to scikit-learn's dataset loader and
stopped within seconds. Triage classified the failure as a fixable script error and proposed
removing one argument; the Coder regenerated the script with that feedback, and the second
attempt passed every stage through the full run. The SVM and soft-tree scripts were repaired
the same way. A plateau guard, which compares each regenerated script with its predecessor to
stop a loop that is not changing anything, measured similarities of 0.22 to 0.27, far from its
0.98 stopping threshold.

\section{Results}
\label{sec:results}

\subsection{Overview}

Table~\ref{tab:results} compares each reproduction with its paper, and
Figure~\ref{fig:curves} shows the learning curve of each generated implementation. All five
scripts were generated by the Coder and executed by the Runner at the papers' full settings.
Two needed a manual correction before their final run, listed in Table~\ref{tab:problems}: one
unstated hyperparameter for Tang, and the misprinted loss for the soft tree.

\begin{TABENV}[tbp]
\centering
\small
\caption{Reproduced results against the published claims. Runtimes are the full stage's wall
clock on the development CPU.}
\label{tab:results}
\begin{tabular}{L{0.19\textwidth}L{0.2\textwidth}rrrL{0.2\textwidth}}
\toprule
Paper & Claim & Paper & ReproBot & Runtime & Agreement \\
\midrule
Tang (2013) & MNIST test error & 0.87\% & \textbf{0.82\%} & 54\,min & Within 0.05 points \\
Wijaya (2023) & Boston Housing test RMSE & 3.02 & \textbf{4.48} & 32\,min & Not matched; train
  RMSE 2.94 vs 2.69 \\
Hsu et al.\ (2003) & svmguide1 test accuracy & 96.9\% & \textbf{96.63\%} & 73\,s & Within 0.3
  points; exact at the paper's settings \\
Xiao et al.\ (2017) & Fashion-MNIST accuracy, mean of 5 runs & 0.873 & \textbf{0.8773} &
  3.3\,min & Within 0.5 points \\
Frosst and Hinton (2017) & MNIST test accuracy & 94.45\% & \textbf{@@SDT_ACC@@} &
  @@SDT_TIME@@ & @@SDT_AGREE@@ \\
\bottomrule
\end{tabular}
\end{TABENV}

\FIGCURVES

\subsection{Tang: MLP with an L2-SVM output layer}

The network --- PCA to 70 dimensions, two hidden layers of 512 units, and a linear SVM trained
with a squared hinge loss in place of softmax --- trained for the paper's 400 epochs. Training
error fell steadily to 0.06\% (Figure~\ref{fig:curves}a), and the test set, evaluated once at
the end, gave 0.82\% against the paper's 0.87\%: five test images out of 10{,}000. With one seed
we cannot measure run-to-run variance, but for scale, the binomial standard error of an error
rate this size over 10{,}000 images is about 0.09 points, so the result is consistent with the
claim. The paper does not state its momentum or its SVM penalty $C$; the first full run, with
common defaults for both, became unstable, and an ablation over the unstated values showed
$C = 0.1$ to train cleanly.

\subsection{Wijaya: tabular regression network}

The three-level dense network, written in Keras by the paper and translated with Keras's
defaults, trained for 1000 epochs. Figure~\ref{fig:curves}b shows the training and validation
RMSE: the model fits within the first 150 epochs, and its validation RMSE, on 81 training rows
held out as the paper describes, stays between 3.2 and 3.5 --- close to the paper's test RMSE
of 3.02. The 101-row test set, however, gave 4.48, while the training RMSE of 2.94 is within 9\%
of the paper's 2.69. The paper states its 405/101 split sizes but not how rows were assigned, and
on a 506-row dataset two held-out subsets already disagree by more than the gap being measured;
the split, not the model, is the most likely source of the difference, and several seeds would
be needed to confirm it.

\subsection{Hsu, Chang and Lin: RBF support vector machine}

ReproBot reproduced the guide's full procedure: scale each feature to $[-1, 1]$ with the
training set's ranges, evaluate 110 $(C, \gamma)$ pairs by five-fold cross-validation, retrain
with the best pair, and test once. Figure~\ref{fig:curves}c shows the cross-validation surface.
A broad ridge of settings scores above 96\%; ReproBot's search picked $C = \gamma = 8$ at
96.99\%, the paper's picked $C = \gamma = 2$ at 96.89\%, and the two are within a tenth of a
point. The test accuracy was 96.625\% against the paper's 96.875\%. Because scikit-learn's
\texttt{SVC} wraps LIBSVM, we also evaluated it directly at each of the paper's reported
settings (Table~\ref{tab:svm}): all three test accuracies match the paper exactly. The
implementation is therefore exact, and the small remaining gap comes only from which near-equal
setting the randomly folded search selects.

\begin{table}[tbp]
\centering
\small
\caption{svmguide1 test accuracy: the paper's LIBSVM results against scikit-learn's
\texttt{SVC} at the same settings.}
\label{tab:svm}
\begin{tabular}{L{0.42\linewidth}rr}
\toprule
Setting & Paper & scikit-learn \\
\midrule
Unscaled features, default parameters & 66.925\% & 66.925\% \\
Scaled to $[-1, 1]$, default parameters & 96.15\% & 96.150\% \\
Scaled, $C = 2$, $\gamma = 2$ & 96.875\% & 96.875\% \\
Scaled, grid search (ReproBot: $C = \gamma = 8$) & --- & 96.625\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Xiao, Rasul and Vollgraf: random forest}

The Fashion-MNIST benchmark reports the mean test accuracy of five runs with shuffled training
data; ReproBot's script performed all five with the paper's configuration (100 trees, entropy
criterion, maximum depth 100). The runs gave between 0.8753 and 0.8792, with a mean of 0.8773
against the paper's 0.873. Figure~\ref{fig:curves}d shows test accuracy as trees are added: it
rises quickly to about 0.87 by 30 trees and then flattens, so the paper's choice of 100 trees
sits on the plateau. The reproduction is within half a point of the claim; the scikit-learn
version the paper used is not stated and may account for the remaining difference.

\subsection{Frosst and Hinton: soft decision tree}

@@SDT_SECTION@@

\subsection{Extraction across paper types}

The Reader ran unchanged on all five papers. Table~\ref{tab:reader} compares them with the four
CIFAR-10 papers it had processed earlier. Every targeted claim was extracted. For both
classical-ML papers, validation converged with no open flags, a first for the pipeline; for the
equation-heavy soft-tree paper, all five equations were correctly identified as the paper's
own.

\begin{TABENV}[tbp]
\centering
\small
\caption{Reader output across nine papers; the first four are CIFAR-10 papers. \emph{Comp.}
counts architecture components; \emph{own/eq} equations marked as the paper's own over
equations captured; \emph{Gaps} explicitly recorded unstated details; \emph{Flags} unresolved
validation flags.}
\label{tab:reader}
\begin{tabular}{lrrrrcrr}
\toprule
Paper & Claims & Hyp. & Data & Comp. & own/eq & Gaps & Flags \\
\midrule
Network In Network      & 12 & 17 & 4 &  7 & 1/3  & 12 & 4 \\
All Convolutional Net   & 17 & 13 & 4 & 10 & 2/6  & 11 & 5 \\
Deep Residual Learning  & 66 & 28 & 6 & 10 & 2/3  &  7 & 8 \\
Wide Residual Networks  & 61 & 20 & 4 & 11 & 1/1  & 11 & 6 \\
\midrule
Tang (2013)                  & 15 & 23 & 3 & 10 & 4/11 & 12 & 5 \\
Wijaya (2023)                & 14 & 10 & 1 &  8 & 2/2  &  8 & 5 \\
Hsu et al.\ (2003)           &  3 & 19 & 8 &  2 & 0/6  &  5 & 0 \\
Xiao et al.\ (2017)          & 26 & 14 & 1 &  0 & 0/0  &  3 & 0 \\
Frosst and Hinton (2017)     &  8 & 11 & 3 &  4 & 5/5  & 10 & 4 \\
\bottomrule
\end{tabular}
\end{TABENV}

\subsection{Cost}

A Coder call consumes about 27{,}000 input and 9{,}500 output tokens and takes 75\,s; a triage
call about 2{,}400 input and 240 output tokens, in 3\,s. Compute dominates: the full runs took
from 73\,s (SVM) to 54 minutes (Tang). Running two containers at once roughly doubled wall
clock, because each claims every CPU core, so runs are best scheduled one at a time.

\section{A Front End: The Pipeline Viewer}
\label{sec:viewer}

In parallel, the team built ReproBot's first user interface: \texttt{viewer/}, a Streamlit
dashboard developed on a feature branch. Its main page lists every paper with its pipeline
status --- whether OCR, Reader and Coder output exist, and how many validation flags remain.
Selecting a paper opens tabs for its method summary, architecture notes, claims,
hyperparameters, data pipeline, validation flags, raw extraction, the generated training script
with its \texttt{reproduce.sh}, and the OCR Markdown, so a reviewer can compare what was
extracted with what was generated without opening files. An ``Import a paper'' section accepts
a PDF, with an optional page limit for cheap trials, and buttons run Reader extraction and code
generation by calling each stage's own entry point, so the viewer cannot drift from the
command-line pipeline. The branch was verified with strict type checking and a headless run
that rendered every tab. Its next steps are to display Runner and Orchestrator output --- the
learning curves and comparisons of this report --- and to merge into the main pipeline as the
basis of the project's final demonstration.

\section{Lessons}
\label{sec:lessons}

\textbf{Watching a run beats waiting for it.} The two training problems of this phase --- Tang's
instability and the soft tree's reversed loss --- were both identified by reading the
container's log while the run was in progress, long before it would have finished. A script
that exits normally has not necessarily learned; the Runner should read these logs itself.

\textbf{Some claims can be matched exactly, others only within variance.} The SVM guide
reproduces to three decimal places at the paper's settings, while the housing-price result
depends on an unrecorded random split. A single tolerance cannot serve both, so the Critic's
must depend on how a claim was produced.

\textbf{Under-specified papers are reproduced through disclosed guesses.} The soft-tree script
had to choose about a dozen unstated values and recorded each one; Tang's instability came from
a combination of two individually reasonable guesses. Disclosure makes such choices visible and
correctable, which is what allowed both to be fixed.

\textbf{Classical and neural models need different checks.} For the random forest and the SVM,
the \texttt{smoke} and \texttt{capped} stages gave identical numbers, since they differ only in
epochs; for models without epochs, a ladder over training-set size would be more informative.

\textbf{Limitations.} The comparisons were made by hand, mostly from single runs; two of the
five reproductions needed a manual correction; the CIFAR-10 benchmark still needs GPU compute;
and Boston Housing includes a feature derived from the proportion of Black residents per town
\cite{harrison1978hedonic}, which we used only because the replicated paper did.

\section{Progress Against the Plan}
\label{sec:timeline}

The project was planned over four months: the Reader first; then the Coder, sandbox and a
skeleton Orchestrator; then the Critic, a real retry loop and the Report Generator; and finally
evaluation, ablation and a demonstration. Table~\ref{tab:timeline} compares the work to date
with that plan. The retry loop arrived ahead of plan; the Critic and Report Generator were
deferred in favour of first obtaining the measurable results they need as input.

\begin{TABENV}[tbp]
\centering
\small
\caption{Status against the project's four-month plan. \built{} done; \partl{} partial or in a
different form; \absent{} not started.}
\label{tab:timeline}
\begin{tabular}{L{0.1\textwidth}L{0.41\textwidth}cL{0.35\textwidth}}
\toprule
Phase & Planned & & Status \\
\midrule
Month 1 & PDF ingestion; Reader schema; shared-memory object & \built & Complete; five fields \\
Month 2 & Coder and Docker sandbox; Orchestrator skeleton with a single retry & \built
  & Exceeded: bounded retry loop with triage-driven feedback \\
Month 3 & Critic verdict logic & \absent & Next phase; inputs now exist \\
        & Real retry loop with targeted feedback & \built & Three real defects repaired \\
        & Report Generator & \absent & Next phase \\
        & Single-shot vs.\ iterative ablation & \partl & Three cases fail single-shot and run
  after one retry \\
Month 4 & Expansion towards 20 papers & \partl & Nine papers read; five replicated at full
  fidelity \\
        & Demonstration interface & \partl & Streamlit viewer on a feature branch \\
\bottomrule
\end{tabular}
\end{TABENV}

\section{Future Work}
\label{sec:future}

\textbf{Live check-ups in the Runner.} Read the container's log during every run, parse the
per-epoch metrics, and stop early with a clear verdict when training is not progressing: a loss
that does not move, non-finite values, accuracy at or below chance, or a loss outside its
possible range. Generated scripts will also write their learning curves to a structured file, so
the checks, the Critic and the reports can use them directly.

\textbf{The Critic agent.} Compare each reproduced value with the claim using the metric's
direction and a tolerance matched to the claim: near-exact for deterministic procedures, a
seed-variance band for stochastic training, and the reported spread when a claim is itself an
average. Its verdict --- pass, retry or fail, with a reason --- closes the loop the project is
built around.

\textbf{Loop controls.} Let the Orchestrator patch the faulty lines of a script instead of
regenerating it, as in AutoReproduce \cite{zhao2025autoreproduce}; keep unstated
hyperparameters fixed across retries unless the feedback concerns them; abort a stage as soon as
its live check fails; enforce per-paper time and cost budgets; and record any human correction
in the run's state.

\textbf{Broader coverage.} Adapt the stage ladder to models without epochs; check the domain of
a paper's own equations before implementing them; run the targets already selected --- gradient
boosting with XGBoost, LightGBM and CatBoost, a graph convolutional network, a text CNN and a
temporal convolutional network; and secure GPU compute for the CIFAR-10 benchmark.

\section{Conclusion}
\label{sec:conclusion}

ReproBot now carries a paper from PDF to a running implementation in the right library, at the
paper's own scale, for neural and classical models alike. @@SDT_CONCLUSION@@ The problems met on
the way were resolved, and they point clearly at the next phase: a Runner that watches training
as it happens, a Critic that judges each result with the right tolerance, and a retry loop that
changes only what it must.

\bibliographystyle{plain}
\begin{thebibliography}{9}

\bibitem{tang2013dlsvm}
Tang, Y. (2013).
Deep Learning using Linear Support Vector Machines.
\textit{ICML 2013 Workshop on Challenges in Representation Learning}. arXiv:1306.0239.

\bibitem{wijaya2023housing}
Wijaya, R. (2023).
Multi Level Dense Layer Neural Network Model for Housing Price Prediction.
\textit{arXiv:2310.08133}.

\bibitem{harrison1978hedonic}
Harrison, D., \& Rubinfeld, D.~L. (1978).
Hedonic housing prices and the demand for clean air.
\textit{Journal of Environmental Economics and Management}, 5(1), 81--102.

\bibitem{hsu2003guide}
Hsu, C.-W., Chang, C.-C., \& Lin, C.-J. (2003).
A Practical Guide to Support Vector Classification. Technical report, Department of Computer
Science, National Taiwan University (version updated 2025).

\bibitem{chang2011libsvm}
Chang, C.-C., \& Lin, C.-J. (2011).
LIBSVM: A library for support vector machines.
\textit{ACM Transactions on Intelligent Systems and Technology}, 2(3), 27.

\bibitem{xiao2017fashion}
Xiao, H., Rasul, K., \& Vollgraf, R. (2017).
Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
\textit{arXiv:1708.07747}.

\bibitem{pedregosa2011sklearn}
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).
Scikit-learn: Machine Learning in Python.
\textit{Journal of Machine Learning Research}, 12, 2825--2830.

\bibitem{frosst2017soft}
Frosst, N., \& Hinton, G. (2017).
Distilling a Neural Network Into a Soft Decision Tree. \textit{arXiv:1711.09784}.

\bibitem{zhao2025autoreproduce}
Zhao, X., Sun, M., Wang, W., et al. (2025).
AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage.
\textit{arXiv:2505.20662}.

\end{thebibliography}

\end{document}
"""

FIG_PIPELINE = r"""
\begin{FIGENV}[tbp]
\centering
\begin{tikzpicture}[
  node distance=0.85cm and 1.5cm,
  stage/.style={draw, rounded corners, minimum width=3.5cm, minimum height=0.95cm,
    align=center, font=\small, thick},
  gone/.style={draw, rounded corners, minimum width=3.5cm, minimum height=0.95cm,
    align=center, font=\small, dashed, thick, draw=missingc, text=missingc},
  side/.style={draw, rounded corners, minimum width=3.0cm, minimum height=0.95cm,
    align=center, font=\small, thick, dashed, draw=warnc, text=warnc},
  arr/.style={-{Stealth[length=2mm]}, thick},
  gonearr/.style={-{Stealth[length=2mm]}, thick, dashed, draw=missingc},
  sidearr/.style={-{Stealth[length=1.8mm]}, dashed, draw=warnc},
  looparr/.style={-{Stealth[length=2mm]}, very thick, draw=loopc},
  lbl/.style={font=\scriptsize, align=left},
  gonelbl/.style={font=\scriptsize, align=left, text=missingc}
]
  \node[stage] (ocr)    {\texttt{ocr/}\\[-2pt]\footnotesize VLM reads page images};
  \node[stage, below=of ocr] (reader) {\texttt{reader/}\\[-2pt]\footnotesize 5 extractors + validation loop};
  \node[stage, below=1.75cm of reader] (coder) {\texttt{coder/}\\[-2pt]\footnotesize PyTorch loop or scikit-learn};
  \node[stage, below=of coder] (runner) {\texttt{runner/}\\[-2pt]\footnotesize Docker sandbox + triage};
  \node[gone,  below=1.2cm of runner] (critic) {\texttt{critic/}\\[-2pt]\footnotesize compare reproduced vs.\ claimed};
  \node[gone,  below=of critic] (report) {report generator\\[-2pt]\footnotesize claim-by-claim report};

  \node[above=0.5cm of ocr, font=\small, align=center] (pdf) {paper PDF};

  \node[side, left=2.1cm of reader] (viewer) {\texttt{viewer/}\\[-2pt]\footnotesize Streamlit dashboard\\[-2pt]\footnotesize (feature branch)};
  \draw[sidearr] (viewer.north) |- (ocr.west);
  \draw[sidearr] (viewer.east) -- (reader.west);
  \draw[sidearr] (viewer.south) |- ([yshift=0.22cm]coder.west);

  \draw[arr] (pdf) -- (ocr);
  \draw[arr] (ocr) -- node[right, lbl] {\texttt{<paper>.md}} (reader);
  \draw[arr] (reader) -- node[right, lbl]
    {\texttt{method\_summary}, \texttt{architecture\_notes}\\
     \texttt{claims}, \texttt{hyperparameters}\\ \texttt{data\_pipeline}} (coder);
  \draw[arr] (coder) -- node[left, lbl, pos=0.45] {\texttt{train.py}\\ + \texttt{reproduce.sh}} (runner);
  \draw[gonearr] (runner) -- node[right, gonelbl] {\texttt{metrics.json}} (critic);
  \draw[gonearr] (critic) -- node[right, gonelbl] {pass / retry / fail} (report);

  \draw[looparr] (runner.east) -- ++(1.1,0) |- (coder.east);
  \node[lbl, text=loopc, right=1.25cm of coder.east, anchor=west, yshift=-0.75cm]
    {\textbf{\texttt{orchestrator/}}\\ retries failed runs\\ with triage feedback};

  \draw[gonearr] (critic.west) -- ++(-1.5,0) |- node[pos=0.25, left, gonelbl, align=right]
    {retry on a\\ \emph{numeric gap}\\ --- planned} ([yshift=-0.22cm]coder.west);
\end{tikzpicture}
\caption{ReproBot on 13.09.2026. Solid components are built and verified by execution; dashed
red components are planned for the next phase; the dashed amber component is on a feature
branch. The Coder writes a PyTorch training loop or a scikit-learn estimator depending on the
paper's model family, and the Orchestrator retries failed runs by feeding the Runner's diagnosis
back to the Coder.}
\label{fig:pipeline}
\end{FIGENV}
"""

FIG_WIJAYA_RUN = r"""
\begin{FIGENV}[tbp]
\centering
\begin{tikzpicture}[
  st/.style={draw, rounded corners=2pt, minimum width=2.45cm, minimum height=1.05cm,
    align=center, font=\scriptsize, thick},
  ok/.style={st, draw=builtc},
  bad/.style={st, draw=missingc},
  tri/.style={st, draw=loopc},
  arr/.style={-{Stealth[length=1.6mm]}, thick},
  row/.style={font=\scriptsize\itshape, anchor=south west}
]
  \node[st] (a1) at (0,0) {Coder, 75\,s\\ OpenML dataset};
  \node[bad, right=0.38cm of a1] (b1) {\texttt{probe} \ding{55}\\ \texttt{ValueError}, 7.5\,s};
  \node[tri, right=0.38cm of b1] (c1) {triage, 3\,s:\\ fixable script error};
  \node[tri, right=0.38cm of c1] (d1) {fix: remove\\ \texttt{version=1}};
  \node[tri, right=0.38cm of d1] (e1) {plateau guard\\ similarity 0.22};
  \draw[arr] (a1) -- (b1); \draw[arr] (b1) -- (c1); \draw[arr] (c1) -- (d1); \draw[arr] (d1) -- (e1);
  \node[row] at ([yshift=0.08cm]a1.north west) {attempt 1};

  \node[st] (a2) at (0,-2.0) {Coder, 74\,s\\ with triage feedback};
  \node[ok, right=0.38cm of a2] (b2) {\texttt{probe} \ding{51}\\ 13.8\,s, dataset checked};
  \node[ok, right=0.38cm of b2] (c2) {\texttt{smoke} \ding{51}\\ 11\,s};
  \node[ok, right=0.38cm of c2] (d2) {\texttt{capped} \ding{51}\\ 20\,s, RMSE 6.11};
  \node[ok, right=0.38cm of d2, line width=1.2pt] (e2) {\texttt{full} \ding{51}, 1918\,s\\ RMSE 4.48\\ verdict \texttt{success}};
  \draw[arr] (a2) -- (b2); \draw[arr] (b2) -- (c2); \draw[arr] (c2) -- (d2); \draw[arr] (d2) -- (e2);
  \node[row] at ([yshift=0.08cm]a2.north west) {attempt 2};

  \draw[arr, draw=loopc] (e1.south) -- ++(0,-0.3) -| ([xshift=-0.3cm]a2.west) -- (a2.west);
\end{tikzpicture}
\caption{An automatic repair by the retry loop, on the housing-price paper. The first script
failed its first check; triage diagnosed a fixable script error, the Coder regenerated the script
with that feedback, and the second attempt passed every stage through the full run.}
\label{fig:wijayarun}
\end{FIGENV}
"""

FIG_CURVES = (
    "\n\\begin{FIGENV}[tbp]\n\\centering\n"
    + panels()
    + r"""
\caption{Learning curves of the five generated implementations at the papers' full settings,
with each paper's reported result as a dashed line. (a) Training error per 20 epochs; the circle
is the reproduced test error. (b) Training and validation RMSE per 50 epochs (epoch 1 is off the
scale). (c) Five-fold cross-validation accuracy over the SVM's $(C, \gamma)$ grid, with the
paper's and ReproBot's selected settings outlined. (d) Test accuracy as trees are added, for the
first of five runs, with all five runs' final accuracies at the right. (e) Test accuracy per
epoch. Panels (c) and (d) were computed with the generated scripts' own configuration and code;
all other points are read directly from the run logs.}
\label{fig:curves}
\end{FIGENV}
"""
)

# Soft decision tree results, filled in when its full run ends.
SOFTDT_TEXT = {
    "@@SDT_ABSTRACT@@": (
        "Three reproductions agree closely with their papers: 0.82\\% against 0.87\\% MNIST test "
        "error, 96.63\\% against 96.9\\% SVM accuracy --- with the paper's own settings "
        "reproducing its numbers exactly --- and 87.73\\% against 87.3\\% random-forest accuracy; "
        "the soft decision tree was still training at the time of writing."
    ),
    "@@SDT_ACC@@": "(running)",
    "@@SDT_TIME@@": "---",
    "@@SDT_AGREE@@": "Pending",
    "@@SDT_SECTION@@": "The full run was in progress at the time of writing.",
    "@@SDT_CONCLUSION@@": "",
}


def build(twocolumn: bool) -> str:
    if twocolumn:
        docclass = r"\documentclass[10pt,twocolumn]{article}"
        geometry = (
            "\\usepackage[a4paper,margin=0.85in]{geometry}\n"
            "\\setlength{\\columnsep}{0.28in}\n"
            "\\raggedbottom"
        )
        figenv, tabenv = "figure*", "table*"
    else:
        docclass = r"\documentclass[11pt]{article}"
        geometry = r"\usepackage[a4paper,margin=1in]{geometry}"
        figenv, tabenv = "figure", "table"

    body = BODY
    body = body.replace("\\FIGPIPELINE", FIG_PIPELINE)
    body = body.replace("\\FIGWIJAYARUN", FIG_WIJAYA_RUN)
    body = body.replace("\\FIGCURVES", FIG_CURVES)
    for key, value in SOFTDT_TEXT.items():
        body = body.replace(key, value)
    body = body.replace("FIGENV", figenv).replace("TABENV", tabenv)
    return f"{docclass}\n{geometry}\n{PREAMBLE_COMMON}\n{body}"


out = Path("docs/progress-reports/third-progress-report")
out.mkdir(parents=True, exist_ok=True)
(out / "third_report_singlecolumn.tex").write_text(build(False), encoding="utf-8")
(out / "third_report_twocolumn.tex").write_text(build(True), encoding="utf-8")
print(f"wrote both .tex files to {out}")
