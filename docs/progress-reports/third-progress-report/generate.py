"""Generate the ReproBot third progress report in both column formats.

Same construction as the second report: the body is held once, so the single- and
two-column .tex files carry identical prose by construction. Wide floats are written
as FIGENV / TABENV and become figure* / table* in the two-column build.

The two training-curve plots are drawn from the real run logs below, with
coordinates computed here, because this TeX installation has no pgfplots.
"""

import math
from collections.abc import Callable
from pathlib import Path

# --- Real data, copied from the run logs -------------------------------------------

# Tang 2013, full stage, svm_C = 1.0 (as generated). Read live from the container;
# stopped at ~epoch 250. Source: docs/notes/tang-2013-ablation/README.md
TANG_C1 = [(1, 27.1), (20, 83.2), (40, 85.5), (100, 89.5), (140, 89.7), (200, 89.5), (240, 89.4)]

# Tang 2013, full stage, svm_C = 0.1. Source: runner/output/<paper>/logs/full.stderr.log
TANG_C01 = [
    (1, 27.440), (20, 12.202), (40, 9.812), (60, 8.482), (80, 6.857), (100, 5.485),
    (120, 4.443), (140, 3.503), (160, 2.800), (180, 2.093), (200, 1.545), (220, 1.152),
    (240, 0.865), (260, 0.583), (280, 0.355), (300, 0.272), (320, 0.163), (340, 0.112),
    (360, 0.088), (380, 0.063), (400, 0.062),
]

# Wijaya 2023, full stage, attempt 2: RMSE on the 81-row validation holdout.
# Source: orchestrator/output/<paper>/logs/attempt-2/full.stderr.log
WIJAYA_VAL = [
    (50, 3.4525), (100, 3.6150), (150, 3.3733), (200, 3.4795), (250, 3.3614),
    (300, 4.0142), (350, 3.2760), (400, 3.2580), (450, 3.4087), (500, 3.1587),
    (550, 3.4512), (600, 3.4664), (650, 3.2748), (700, 3.2184), (750, 3.4997),
    (800, 3.4581), (850, 3.3357), (900, 3.2041), (950, 3.2520), (1000, 3.3102),
]

# --- Plot geometry -----------------------------------------------------------------

TANG_W, TANG_H = 11.0, 5.2
TANG_YMIN, TANG_YMAX = 0.05, 100.0


def tang_x(epoch: float) -> float:
    return epoch / 400.0 * TANG_W


def tang_y(err: float) -> float:
    lo, hi = math.log10(TANG_YMIN), math.log10(TANG_YMAX)
    return (math.log10(err) - lo) / (hi - lo) * TANG_H


WIJ_W, WIJ_H = 11.0, 4.6
WIJ_YMIN, WIJ_YMAX = 2.6, 4.8


def wij_x(epoch: float) -> float:
    return epoch / 1000.0 * WIJ_W


def wij_y(rmse: float) -> float:
    return (rmse - WIJ_YMIN) / (WIJ_YMAX - WIJ_YMIN) * WIJ_H


def coords(
    points: list[tuple[float, float]],
    fx: Callable[[float], float],
    fy: Callable[[float], float],
) -> str:
    return " ".join(f"({fx(x):.3f},{fy(y):.3f})" for x, y in points)


def tang_plot() -> str:
    ticks_y = "\n".join(
        f"  \\draw (0,{tang_y(v):.3f}) -- (-0.1,{tang_y(v):.3f}) "
        f"node[left, font=\\scriptsize] {{{label}}};\n"
        f"  \\draw[gridc] (0,{tang_y(v):.3f}) -- ({TANG_W},{tang_y(v):.3f});"
        for v, label in [(0.1, "0.1"), (1, "1"), (10, "10"), (100, "100")]
    )
    ticks_x = "\n".join(
        f"  \\draw ({tang_x(e):.3f},0) -- ({tang_x(e):.3f},-0.1) "
        f"node[below, font=\\scriptsize] {{{e}}};"
        for e in (0, 100, 200, 300, 400)
    )
    floor = tang_y(88.8)
    stop = tang_x(250)
    return rf"""
\begin{{tikzpicture}}[gridc/.style={{draw=black!12}}]
{ticks_y}
{ticks_x}
  \draw[thick] (0,0) -- ({TANG_W},0);
  \draw[thick] (0,0) -- (0,{TANG_H});
  \node[rotate=90, font=\small] at (-1.0,{TANG_H / 2:.3f}) {{training error (\%, log scale)}};
  \node[font=\small] at ({TANG_W / 2:.3f},-0.85) {{epoch}};
  \draw[dashed, draw=warnc, thick] (0,{floor:.3f}) -- ({TANG_W},{floor:.3f});
  \node[font=\scriptsize, text=warnc, anchor=south east] at ({TANG_W},{floor:.3f})
    {{input-ignoring optimum, 88.8\%}};
  \draw[densely dotted, draw=missingc] ({stop:.3f},0) -- ({stop:.3f},{floor - 0.35:.3f});
  \node[font=\scriptsize, text=missingc, anchor=west, align=left] at ({stop + 0.08:.3f},{floor - 0.75:.3f})
    {{stopped by hand\\ at epoch $\sim$250}};
  \draw[missingc, very thick, mark=*, mark size=1.4pt]
    plot coordinates {{{coords(TANG_C1, tang_x, tang_y)}}};
  \draw[builtc, very thick, mark=square*, mark size=1.2pt]
    plot coordinates {{{coords(TANG_C01, tang_x, tang_y)}}};
  \draw[black, fill=white, thick] ({tang_x(400):.3f},{tang_y(0.82):.3f}) circle (2.4pt);
  \node[font=\scriptsize, anchor=east, align=right] at ({tang_x(392):.3f},{tang_y(0.82) + 0.42:.3f})
    {{final \emph{{test}} error 0.82\%\\ (claimed 0.87\%)}};
  \draw[missingc, very thick] (7.35,3.55) -- (7.95,3.55);
  \node[font=\scriptsize, anchor=west] at (8.0,3.55) {{$C = 1.0$ (as generated)}};
  \draw[builtc, very thick] (7.35,3.05) -- (7.95,3.05);
  \node[font=\scriptsize, anchor=west] at (8.0,3.05) {{$C = 0.1$ (one-line change)}};
\end{{tikzpicture}}"""


def wijaya_plot() -> str:
    ticks_y = "\n".join(
        f"  \\draw (0,{wij_y(v):.3f}) -- (-0.1,{wij_y(v):.3f}) "
        f"node[left, font=\\scriptsize] {{{v:.1f}}};\n"
        f"  \\draw[gridc] (0,{wij_y(v):.3f}) -- ({WIJ_W},{wij_y(v):.3f});"
        for v in (2.8, 3.2, 3.6, 4.0, 4.4, 4.8)
    )
    ticks_x = "\n".join(
        f"  \\draw ({wij_x(e):.3f},0) -- ({wij_x(e):.3f},-0.1) "
        f"node[below, font=\\scriptsize] {{{e}}};"
        for e in (0, 200, 400, 600, 800, 1000)
    )
    return rf"""
\begin{{tikzpicture}}[gridc/.style={{draw=black!12}}]
{ticks_y}
{ticks_x}
  \draw[thick] (0,0) -- ({WIJ_W},0);
  \draw[thick] (0,0) -- (0,{WIJ_H});
  \node[rotate=90, font=\small] at (-1.0,{WIJ_H / 2:.3f}) {{RMSE}};
  \node[font=\small] at ({WIJ_W / 2:.3f},-0.85) {{epoch}};
  \draw[dashed, draw=missingc, thick] (0,{wij_y(4.48):.3f}) -- ({WIJ_W},{wij_y(4.48):.3f});
  \node[font=\scriptsize, text=missingc, anchor=south west] at (0.1,{wij_y(4.48):.3f})
    {{reproduced \emph{{test}} RMSE, after epoch 1000: 4.48}};
  \draw[dashed, draw=black!75, thick] (0,{wij_y(3.02):.3f}) -- ({WIJ_W},{wij_y(3.02):.3f});
  \node[font=\scriptsize, text=black!75, anchor=north east] at ({WIJ_W},{wij_y(3.02):.3f})
    {{claimed test RMSE: 3.02}};
  \draw[loopc, very thick, mark=*, mark size=1.3pt]
    plot coordinates {{{coords(WIJAYA_VAL, wij_x, wij_y)}}};
  \node[font=\scriptsize, text=loopc, anchor=south] at ({wij_x(300):.3f},{wij_y(4.0142) + 0.05:.3f})
    {{validation RMSE (81 held-out training rows)}};
\end{{tikzpicture}}"""


# --- LaTeX ---------------------------------------------------------------------------

PREAMBLE_COMMON = r"""
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc,plotmarks}
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

% Float placement: this report carries more figures and tables than the second, and
% the default fractions strand wide floats at the end of the document.
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
The second progress report left ReproBot with a working five-stage pipeline and no
replication result: full-fidelity training on the CIFAR-10 targets was measured at roughly
22 days per paper on the available CPU, and no reproduced number had been compared against
any claim. This report covers the phase that changed both. We redirected the evaluation
towards papers whose original experiments train in minutes on a CPU, and, at the same
time, generalized the code-generation and execution stages beyond image classification: the
Coder now writes a plain PyTorch training loop for any supervised task, under a
task-agnostic metrics contract. Two new papers were carried through every stage at full
fidelity, producing the project's first replication measurements, compared by hand. The
reproduction of Tang's \emph{Deep Learning using Linear Support Vector Machines} reaches
0.82\% MNIST test error against a claimed 0.87\% --- but only after one hyperparameter the
paper never states was corrected by hand, following an ablation that diagnosed why the first
full run collapsed into a network that ignored its input. The reproduction of Wijaya's
housing-price model reaches a test RMSE of 4.48 against a claimed 3.02, a gap we cannot
attribute from a single seed because the paper does not specify its data split. Along the
way the retry loop repaired its first naturally occurring defect, and six prompt failures
observed in real runs were each converted into an explicit rule. We also report the
team's first front end, a Streamlit dashboard over the pipeline's outputs. The central
lesson of the phase is that every stage reported \emph{success} on a run that had
completely failed: nothing in ReproBot yet judges whether a number is plausible, which makes
the Critic the most important remaining component.
\end{abstract}

\section{Introduction}
\label{sec:intro}

The first progress report \cite{reprobot2026first} set out ReproBot's motivation and design,
and the second \cite{reprobot2026second} documented its implementation: PDF extraction, a
five-field structured Reader, a Coder, a Docker-sandboxed Runner, and an Orchestrator
running a Coder\,$\leftrightarrow$\,Runner retry loop. That report closed on two
limitations stated without qualification. No replication-fidelity claim was possible, since
the Critic did not exist and nothing compared a reproduced number with a paper's. And no
fidelity claim was even \emph{measurable}, since the only full training run the hardware
could support would have taken weeks.

This phase addressed the second limitation directly and the first by hand. Its three
threads are reported in turn. Section~\ref{sec:direction} describes a change of direction:
evaluating on papers small enough to train for real on a CPU, and a team decision that
ReproBot must reproduce more than one kind of paper. Section~\ref{sec:generalization}
describes the resulting changes to the Coder and Runner, and Section~\ref{sec:prompts} the
prompt failures that real runs exposed and the rules that now prevent them.
Section~\ref{sec:results} reports the measurements themselves, including a training
collapse, the ablation that explained it, and the first two comparisons against published
claims. Section~\ref{sec:viewer} describes the team's pipeline viewer.
Sections~\ref{sec:findings}--\ref{sec:conclusion} collect the engineering findings, the
methodology, progress against the original timeline, limitations, and next steps.

Since the second report, the repository has gained seven commits, which together added 933
lines and removed 280 in the pipeline stages, and a feature branch has added a 544-line
dashboard. Table~\ref{tab:status}
states, for each limitation the second report recorded, where it now stands.

\begin{TABENV}[tbp]
\centering
\small
\caption{The second report's stated limitations, revisited.}
\label{tab:status}
\begin{tabular}{L{0.34\textwidth}L{0.58\textwidth}}
\toprule
Stated on 22.08.2026 & Status on 13.09.2026 \\
\midrule
No fidelity claim is possible yet
  & Two claims measured at full fidelity and compared by hand
    (Section~\ref{sec:fidelity}); the Critic still does not exist \\
Full-fidelity runs infeasible ($\sim$22 days per CIFAR-10 paper)
  & Unchanged for CIFAR-10; sidestepped with two papers whose full runs took 32 and 54
    minutes on the same CPU \\
The evidence base is two papers
  & Four papers carried end to end: two beyond CIFAR-10, one of them tabular regression \\
The retry loop was demonstrated only on an injected fault
  & It repaired a naturally occurring defect (Section~\ref{sec:wijaya}) \\
Four of seven verdicts exercised only against test doubles
  & \texttt{environment\_error} occurred in a real run --- wrongly assigned, and the
    triage prompt corrected (Section~\ref{sec:prompts}); three remain unobserved \\
Retries regenerate rather than patch
  & Unchanged, and now measured: a one-argument fix produced a script only 22\% similar to
    its predecessor \\
Validation does not converge; two papers cannot be extracted
  & Unchanged \\
\bottomrule
\end{tabular}
\end{TABENV}

\section{A Change of Direction}
\label{sec:direction}

\subsection{The compute wall}

The CIFAR-10 papers chosen as ReproBot's first targets are the natural benchmark for the
proposal's image-classification scope, and they are unreachable on the hardware available to
the project. At the measured throughput of 5.3 training samples per second, one
\emph{Wide Residual Networks} run at the paper's own setup takes approximately 22 days. The
second report's reproduced numbers were therefore all drawn from stages that train for one
epoch on a few hundred images, and could say nothing about fidelity.

Securing GPU compute remains necessary for that benchmark. It is not, however, necessary to
test the claim ReproBot actually makes --- that a paper can be read, implemented, executed
and checked automatically --- which requires only a paper whose original experiment is
small. We therefore looked for papers whose headline result trains in minutes on a CPU,
while leaving the curated CIFAR-10 set untouched.

\subsection{More than one kind of paper}

The search surfaced a structural problem. The pipeline was, in practice, hardwired to image
classification: the Coder's prompt assumed \texttt{torchvision} datasets and the HuggingFace
\texttt{Trainer} image-model interface, the metrics contract was keyed on accuracy, and the
Runner's image carried no tabular libraries. An in-scope MNIST paper would have tested the
pipeline as it stood. A housing-price regression paper could not have run at all.

The team's decision was that this was a defect rather than a scope boundary: a system that
replicates only one task type is not a general replication system, and ReproBot should
handle several. CIFAR-10 remains the main evaluation set, but no stage may now assume
images. Both candidate papers were retained, precisely because they differ:

\begin{itemize}
  \item \textbf{Tang (2013)}, \emph{Deep Learning using Linear Support Vector Machines}
    \cite{tang2013dlsvm}. A classification paper whose contribution is a loss: the softmax
    output layer is replaced by a linear L2-SVM trained with a squared hinge loss. The
    targeted claim is 0.87\% MNIST test error. The setup --- PCA to 70 dimensions, two
    512-unit hidden layers, 400 epochs --- is stated in unusual detail, which makes the two
    values it omits all the more consequential (Section~\ref{sec:tang}).
  \item \textbf{Wijaya (2023)}, \emph{Multi Level Dense Layer Neural Network Model for
    Housing Price Prediction} \cite{wijaya2023housing}. A tabular regression paper in Keras,
    on the Boston Housing data \cite{harrison1978hedonic}: 506 rows, 13 features, a stated
    405/101 train/test split, 1000 epochs. The targeted claim is a test RMSE of 3.02. It
    tests three things CIFAR-10 cannot: a non-image data source, a regression metric where
    lower is better, and a paper written for a different framework.
\end{itemize}

Both were added under a separate directory, so the curated CIFAR-10 set remains exactly as
its curator left it.

\section{Generalizing the Coder and Runner}
\label{sec:generalization}

\FIGPIPELINE

\subsection{Dropping the HuggingFace \texttt{Trainer}}

The Coder previously emitted scripts built on HuggingFace's \texttt{Trainer}. Its model
contract --- image tensors in, logits out --- fits a custom margin loss or a regression head
poorly, and that alone argued for replacing it. The deciding reason, however, was that
\texttt{Trainer} applies defaults that no paper states, and this was demonstrated rather than
assumed.

When the generalized Coder regenerated its \emph{Network In Network} script as a regression
test, the new plain-PyTorch script kept the paper's architecture --- a genuine
\texttt{mlpconv} cascade with no fully-connected layer --- and passed its execution probe.
But its loss reached approximately 947, where the previous \texttt{Trainer}-based script had
reported 2.30. The cause was a default: the old script never set \texttt{max\_grad\_norm},
and therefore inherited \texttt{Trainer}'s gradient clipping at a norm of 1.0, which we
confirmed inside the container. The paper uses no clipping. Faithful to the paper and without
its unstated initialization, the same learning rate diverges.

We record this as a correction to the second report. Its \emph{Network In Network} runs
executed under a stabilizing mechanism the paper does not describe; their numbers were
already labelled as non-replication results, but the scripts were less faithful than that
report implied. The Coder is now forbidden from using \texttt{Trainer}, \texttt{accelerate},
Lightning or Keras, and writes an explicit training loop in which every mechanism is visible.

\subsection{Always PyTorch, whatever the paper used}

Wijaya's model was built in Keras. Rather than generate code for a second framework, the
Coder always writes PyTorch, and the prompt treats the source framework's defaults as part of
the method: where a paper is silent, it inherited its framework's behaviour, so a faithful
translation must reproduce that behaviour. The prompt names the Keras defaults that differ ---
Adam's $\epsilon = 10^{-7}$, Glorot initialization, batch normalization with momentum 0.99 and
$\epsilon = 10^{-3}$, and a \texttt{fit()} batch size of 32 --- and requires every remaining
difference to be disclosed. Section~\ref{sec:findings} reports that this rule was followed
only in part.

\subsection{A task-agnostic metrics contract}

Every generated script writes a fixed-shape metrics file that the Runner parses and a future
Critic will read. The second report's contract assumed accuracy. Its generalization is
summarized in Table~\ref{tab:contract}; the most consequential addition is
\texttt{higher\_is\_better}, since a bare value of 4.0 is meaningless without knowing
whether it is an error or an accuracy.

\begin{table}[tbp]
\centering
\small
\caption{Changes to the metrics contract. \texttt{metric} and \texttt{unit} are copied
verbatim from the targeted claim, so a comparison against the claim needs no conversion.
Files in the old shape still parse.}
\label{tab:contract}
\begin{tabular}{L{0.36\linewidth}L{0.54\linewidth}}
\toprule
Field & Change \\
\midrule
\texttt{train\_metric}, \texttt{eval\_metric} & Replace \texttt{train\_accuracy} and
  \texttt{eval\_accuracy}: the claim's own metric on each split \\
\texttt{higher\_is\_better} & New; the direction of the metric \\
\texttt{task\_type} & New; inferred by the Coder from the claim, e.g.\
  \texttt{classification} or \texttt{regression} \\
\texttt{value}, \texttt{metric}, \texttt{unit} & Unchanged; the headline number and its
  claim-verbatim label \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Data sourcing, and verifying what was fetched}

Image datasets still come from \texttt{torchvision}. Tabular data is fetched from OpenML,
pinned by dataset identifier, before any direct URL is considered, and every download lands
in the shared dataset cache the Runner mounts. The preference order was not a design choice
made in advance; Section~\ref{sec:prompts} describes the failure that produced it.

The generated script must also check, in code and before training, that the dataset it
fetched is the one it intended --- its name, row count and column count --- and raise
otherwise. The motivation is that a wrong identifier does not fail. It downloads a different
dataset that happens to carry that number, trains on it, and reports a plausible metric for
the wrong problem. Where a paper gives split sizes but not how the split was drawn, the
script uses a seeded random split and records the assumption.

\subsection{Runner changes}

The Runner's interface did not change: it still invokes only the generated
\texttt{reproduce.sh} and one of its four modes, which is what allowed two new task types to
be added without modifying its code. Its image gained pinned \texttt{pandas},
\texttt{scikit-learn} and \texttt{scipy}, installed in the same \texttt{pip} command as the
pinned \texttt{numpy}, so that a transitive upgrade fails the build rather than silently
moving a numerical dependency.

\section{Prompt Hardening from Real Failures}
\label{sec:prompts}

Running the generalized pipeline on real papers exposed six failures in the prompts of the
Coder and of the Runner's triage step. None was found by inspection; each surfaced in a run.
Table~\ref{tab:prompts} lists them with the rule that now addresses each. Every triage rule
was verified by re-running triage on the real log that exposed the problem.

\begin{TABENV}[tbp]
\centering
\small
\caption{Prompt failures observed in real runs, in the order they occurred, and the rule
that now addresses each. Rows 1--5 arose from the first tabular run; row 6 from its rerun.}
\label{tab:prompts}
\begin{tabular}{r L{0.33\textwidth} L{0.23\textwidth} L{0.31\textwidth}}
\toprule
& Observed failure & Consequence & Rule now in the prompt \\
\midrule
1 & The Coder fetched Boston Housing from a remembered CMU StatLib URL, which now returns
    HTTP~403 to every client
  & The first execution stage failed in 5\,s
  & Coder: OpenML by identifier first; a direct URL only for data found nowhere else \\
2 & Triage classified the 403 as \texttt{environment\_error}
  & The Orchestrator stopped with zero retries on a failure one regeneration would fix
  & Triage: a refusal from one source is recoverable; only a container with no network at
    all is environmental \\
3 & A re-triage suggested switching to ``a standard alternative regression dataset''
  & The fix would change what is being reproduced
  & Never change what is being reproduced \\
4 & A re-triage suggested \texttt{load\_boston}
  & That loader was removed from scikit-learn in version 1.2
  & Never suggest an API or source not certainly current; give one fix, not a menu \\
5 & A re-triage named Boston Housing as OpenML \texttt{data\_id=506}
  & Identifier 506 is a different dataset (Boston Housing is 531); training would
    silently proceed on it
  & Name no identifier absent from the log; the Coder asserts the dataset's name and shape
    in code \\
6 & The Coder's prompt twice said ``OpenML data\_id and version''; the script passed both
  & \texttt{fetch\_openml} rejects that combination with a \texttt{ValueError}
  & Pin by identifier alone, or by name together with a version \\
\bottomrule
\end{tabular}
\end{TABENV}

Two of these generalize beyond this project, and are drawn out in
Section~\ref{sec:findings}. Rows 1 and 5 share a pattern: in both, a model supplied an
identifier from memory that looked exactly as plausible as a correct one. Row 6 is the
converse, and the more uncomfortable: the defect was in our own prompt, whose descriptive
wording the model implemented literally.

\section{Results}
\label{sec:results}

\subsection{Extraction on the new papers}

The Reader required no changes for either paper. Table~\ref{tab:reader} extends the second
report's coverage table with both. Each targeted claim was extracted exactly, as was Tang's
distinction between the L1- and L2-SVM gradients.

\begin{TABENV}[tbp]
\centering
\small
\caption{Reader output across six papers; the first four rows are reproduced from the second
report. \emph{Comp.} is architecture components; \emph{own/eq} is equations marked as the
paper's own over equations captured; \emph{Gaps} is \texttt{unstated\_details} entries;
\emph{Flags} is unresolved validation flags after the three-pass cap.}
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
Tang (2013), DLSVM      & 15 & 23 & 3 & 10 & 4/11 & 12 & 5 \\
Wijaya (2023), housing  & 14 & 10 & 1 &  8 & 2/2  &  8 & 5 \\
\bottomrule
\end{tabular}
\end{TABENV}

Three weaknesses surfaced, all recorded as open defects. First, validation still does not
converge: both new papers end with five unresolved flags. Second, the same result is
extracted two or three times when a paper states it in prose, a table and a figure;
Wijaya's 14 claims describe eight distinct results, at mixed precision (0.911 and 0.91). A
Critic comparing against claims will need them deduplicated. Third, and most consequential,
the Reader records what a paper leaves unstated only for its \emph{architecture}. Tang's
paper states ``stochastic gradient descent with momentum'' without a momentum value, and an
L2-SVM objective without its penalty constant $C$. Neither omission was flagged anywhere ---
and they are exactly the two values that collapsed training (Section~\ref{sec:tang}).

\subsection{The retry loop repairs a real defect}
\label{sec:wijaya}

\FIGWIJAYARUN

The second report demonstrated the retry loop only against a deliberately reintroduced
fault. Wijaya's two orchestrated runs, shown in Figure~\ref{fig:wijayarun}, supply the first
naturally occurring case --- and, in the row before it, the failure that motivated the
triage rules of Section~\ref{sec:prompts}.

Before the prompts were hardened, the first execution stage failed on the dead URL, triage
called it environmental, and the loop stopped as designed for an environmental fault ---
correctly given its input, and wrongly in fact. After hardening, the regenerated script
fetched the right dataset but passed both an identifier and a version to
\texttt{fetch\_openml}, which raised. Triage classified the failure as recoverable and
proposed removing the \texttt{version} argument, which matches the traceback exactly. The
Coder regenerated with that feedback, the plateau guard measured a similarity of 0.22, and
the second attempt asserted the dataset's name and shape and passed every stage through
\texttt{full}. The whole repair cost two generation calls, one triage call, and under three
minutes.

\subsection{The first full run, and its collapse}
\label{sec:tang}

Tang's generated script passed \texttt{probe}, \texttt{smoke} and \texttt{capped} with
training error falling from 75\% to 31\% --- evidence, as far as those stages can provide
it, that the network learned. Its \texttt{full} run was ReproBot's first at a paper's own
scale. By epoch 20 its training error had risen to 83\%, and it then settled near 89.5\%
(Figure~\ref{fig:tang}). It was noticed by a team member reading the live training log. No
stage noticed, and had the run completed, the Runner would have reported \emph{success}:
the script exited normally and wrote a well-formed metrics file.

\FIGTANG

\textbf{Diagnosis.} For a one-vs-rest squared hinge loss, a network that ignores its input can
do no better than a constant score per class, $s_k = 2p_k - 1$, where $p_k$ is the frequency
of class $k$. The resulting loss is $\sum_k 4p_k(1-p_k)$, and the best such network always
predicts the most frequent digit. Evaluated at MNIST's class frequencies, both quantities
match the observed run (Table~\ref{tab:floor}). The network had died: early oversized
updates drove every hidden unit's pre-activation negative, so every ReLU output zero for
every input and no gradient reached the hidden layers; only the output bias still learned,
to exactly the input-ignoring optimum. A decaying learning rate cannot recover from this,
since a zero gradient stays zero at any step size.

\begin{table}[tbp]
\centering
\small
\caption{The collapsed run against a network that ignores its input. The small excess is
the L2 penalty and the input noise.}
\label{tab:floor}
\begin{tabular}{lrr}
\toprule
& Input-ignoring optimum & Observed \\
\midrule
Training loss  & 3.599 & $\sim$3.607 \\
Training error & 88.8\% & $\sim$89.5\% \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ablation.} The Coder had disclosed four values the paper does not state: momentum
0.9, $C = 1.0$, PyTorch's default initialization, and unwhitened PCA features. To isolate
the cause, we wrote a harness that imports the model, loss and data loading \emph{from the
generated script itself}, holds every paper-stated value fixed, and varies one of the four
guesses at a time for 30 epochs of the paper's schedule, measuring the fraction of hidden
units that output zero for every one of 2,000 probe images. Table~\ref{tab:ablation} reports
the six configurations.

\begin{TABENV}[tbp]
\centering
\small
\caption{Ablation over the four unstated values, each changed alone from the generated
configuration (A). \emph{Dead} is the fraction of second-layer hidden units that output
zero for all 2,000 probe images at epoch 30; first-layer units were never dead in any
configuration.}
\label{tab:ablation}
\begin{tabular}{llrrrl}
\toprule
& Configuration & Test @5 & Test @30 & Dead @30 & Outcome \\
\midrule
A & As generated (mom.\ 0.9, $C=1.0$) & 6.6\%  & 47.1\% & 0.71 & unstable \\
B & Momentum 0.5                        & 4.0\%  & 2.62\% & 0.38 & stable \\
C & Momentum 0.0                        & 4.2\%  & 2.66\% & 0.14 & stable \\
D & $C = 0.1$                           & 4.4\%  & 2.80\% & 0.04 & stable, healthiest \\
E & PCA whitened                        & 79.8\% & 90.3\% & 0.996 & total collapse \\
F & Init.\ $\mathcal{N}(0, 0.01)$     & 5.0\% & 4.2\% & 0.31 & stable, slower \\
\bottomrule
\end{tabular}
\end{TABENV}

The collapse requires momentum 0.9 and $C = 1.0$ \emph{together}: reducing either alone
stabilizes training. Momentum 0.9 amplifies the effective step roughly tenfold, and $C = 1.0$
scales the squared-hinge gradient tenfold relative to $C = 0.1$. Each value is a reasonable
default in isolation; the pair is too aggressive for this loss at the paper's learning rate
of 0.1. Whitening the PCA features enlarges the inputs --- the mean squared norm rises from
46 to 70 --- and collapses fastest, which supports the Coder's literal, unwhitened reading.
Configuration A degraded but did not fully collapse within 30 epochs while the real run did;
the two used different random streams for the input noise, so the collapse is
seed-dependent, which is precisely what makes it hard to catch with short runs.

We selected $C = 0.1$ (configuration D) as a one-line change to the existing script rather
than a regeneration, so that nothing else could change at the same time. It keeps the
momentum consistent with the paper's ``with momentum'', produced the healthiest network, and
concerns a constant the paper itself treats as tuned elsewhere.

\subsection{First fidelity measurements}
\label{sec:fidelity}

Both papers then completed \texttt{full} runs at their papers' stated settings.
Table~\ref{tab:fidelity} reports the results against the published claims.

\begin{TABENV}[tbp]
\centering
\small
\caption{ReproBot's first full-fidelity measurements, compared against the claims by hand.
One seed each. Runtimes are wall clock on the development CPU, with the two runs sharing it.}
\label{tab:fidelity}
\begin{tabular}{L{0.16\textwidth}L{0.15\textwidth}rrrL{0.25\textwidth}}
\toprule
Paper & Claim & Claimed & Reproduced & Runtime & Reading \\
\midrule
Tang (2013) & MNIST test error & 0.87\% & \textbf{0.82\%} & 3,223\,s
  & Consistent with the claim, after one unstated value was corrected by hand \\
Wijaya (2023) & Test RMSE & 3.02 & \textbf{4.48} & 1,918\,s
  & Not reproduced on this split \\
Wijaya (2023) & Train RMSE & 2.69 & 2.94 & ---
  & Close; see text \\
\bottomrule
\end{tabular}
\end{TABENV}

\textbf{Tang.} With $C = 0.1$, training error fell smoothly to 0.062\% with no sign of
collapse (Figure~\ref{fig:tang}), and the test set, evaluated once after the final epoch,
gave 0.82\%. The difference from the claim is five test images out of 10{,}000. With a single
seed we cannot measure run-to-run variance; as a reference scale, the binomial standard error
of a 0.85\% error rate over 10{,}000 samples is about 0.09 percentage points. We therefore
read the result as consistent with the claim, not as an improvement on it.

Two qualifications matter for how this result may be cited. It is not autonomous: a human
chose $C$. And although the ablation selected on network health --- dead units and distance
from the input-ignoring loss --- it also logged short-horizon test error, so the choice was
not made blind to the test set. The accurate statement is that \emph{the generated
implementation replicates the paper's claim once one hyperparameter the paper omits is
corrected}.

\FIGWIJAYA

\textbf{Wijaya.} The reproduced test RMSE of 4.48 is 48\% above the claim, while the
training RMSE of 2.94 is within 9\% of it. The training log qualifies the headline number
(Figure~\ref{fig:wijaya}). The generated script, following the paper's description, held out
20\% of the training rows for monitoring. On those 81 rows, which the model never trained
on, the RMSE stayed between 3.2 and 3.5 for most of training and ended at 3.31 --- close to
the claimed test value. Only the 101-row test set produced 4.48. Two held-out subsets of the
same 506-row dataset thus disagree by more than the gap being measured.

The paper states its split sizes but not how rows were assigned, so the script used a seeded
random split. On a dataset this small, with a target censored at its maximum value, which
rows fall into a 101-row test set plausibly dominates the result. A single seed cannot
separate an unfaithful implementation from an unlucky split, and we do not claim either. Two
further candidates remain untested: the Coder did not reproduce Keras's batch-normalization
and initialization defaults (Section~\ref{sec:findings}), and a constant learning rate left
the final epoch's training loss noisy.

\subsection{Cost}

One Coder call consumed approximately 27{,}000 input and 9{,}500 output tokens and took 75\,s;
one triage call, about 2{,}400 input and 240 output tokens, in 3\,s. Compute again dominated,
and in a way worth recording. Both full runs shared the CPU, and each container claimed
every core. Wijaya's 1000 epochs over 324 training rows took 1,906\,s --- about 1.9\,s per
epoch of 11 small batches, which is thread contention rather than model cost --- and Tang's
run took 54 minutes. The Runner deliberately leaves CPU limits unset, because an arbitrary
limit produces failures indistinguishable from crashes; the practical consequence is that
concurrent runs should be avoided.

\section{A Front End: The Pipeline Viewer}
\label{sec:viewer}

In parallel with the pipeline work, the team built ReproBot's first user interface:
\texttt{viewer/}, a dashboard written in Streamlit, developed on a feature branch. The
original proposal planned a Gradio demonstration for the final month; the viewer is an early
step towards it, and is already useful as a working tool.

\textbf{What it shows.} The main page lists every paper with a status table --- whether OCR,
Reader and Coder output exist, and how many validation flags remain. Selecting a paper opens
tabs for its method summary, architecture notes, claims, hyperparameters, data pipeline,
validation flags, raw extraction JSON, generated training script with its
\texttt{reproduce.sh}, and the raw OCR Markdown. A reviewer can therefore inspect what the
Reader extracted and what the Coder did with it side by side, without opening files.

\textbf{What it can run.} An ``Import a paper'' section accepts a PDF upload, with an
optional page limit so that a new paper can be tested cheaply before committing to the whole
document. Per-paper buttons then run Reader extraction or code generation. Two design
decisions make this safe. Every action calls the corresponding stage's own entry point,
unmodified, so the viewer cannot drift from the command-line pipeline. And uploaded papers
are stored in a separate directory rather than the curated dataset directory, which is
maintained by another team member.

\textbf{Status.} The branch was verified with strict type checking, the repository's
pre-commit hooks, and a headless Streamlit run that rendered every tab against synthetic
fixtures. It is not yet merged.
Two gaps remain before it is: it does not yet display Runner or Orchestrator output --- the
results this report is about --- and it predates the generalization of the Coder and Runner,
so its dependency changes need reconciling with the updated lock file before it can merge. Extending it to show a run's attempts, triage decisions and metrics, and
eventually the Critic's verdict, is the natural route to the planned demonstration.

\section{Engineering Findings}
\label{sec:findings}

\subsection{Execution success is not evidence of a result}

The second report argued that static checks cannot establish that generated code runs. This
phase established the next step: that code running to completion cannot establish that it
worked. Tang's collapsed run would have exited normally and been reported as a success. The
escalation ladder could not have caught it --- \texttt{capped} trains for about fifteen steps,
and the collapse needed thousands --- and \texttt{capped}'s stated purpose, checking that
training learns, is not enforced by anything: the Runner reads only the exit code.

The same pattern appeared at smaller scale throughout the phase. The \texttt{full} mode of
every generated \texttt{reproduce.sh} wrote its metrics to a different filename from the one
the Runner reads, so both fidelity results above reached the Runner only through its
fallback parser; this is now fixed. The OCR, Reader and Coder stages exit with status zero
even when every paper fails, which is how a missing image-library dependency went unnoticed
on the first OCR attempt of the phase. Every such seam is a place where failure can look like
success.

\subsection{Framework defaults are part of the method}

Two findings point the same way. HuggingFace \texttt{Trainer}'s default gradient clipping
made a script look stable that, written faithfully, diverges. And a paper written in Keras
inherits Keras's defaults wherever it is silent, so reproducing it faithfully in another
framework means reproducing those defaults. The Coder was instructed to do so, and followed
the instruction only partly: Wijaya's script matched Keras's Adam $\epsilon$ but used
PyTorch's batch-normalization momentum and $\epsilon$ and PyTorch's default initialization,
and disclosed the choice rather than following the rule. Disclosure makes a deviation
visible; it does not make it faithful.

\subsection{A remembered identifier is worse than a missing one}

A model that lacks an identifier fails visibly. A model that recalls one wrongly --- a dead
URL, a removed loader, a dataset number one digit away from correct --- produces an artifact
that looks correct and, for a dataset identifier, trains without complaint on the wrong data.
The defence that worked was not asking the model to be more careful, but removing the
opportunity: triage may not name identifiers absent from the log, and the generated script
must verify what it fetched before using it.

\subsection{Prompt wording is executed literally}

The one defect this phase introduced into generated code came from our own prompt, which
described OpenML datasets as pinned by ``data\_id and version''. The description was meant
loosely; the model passed both arguments. Prompts for code generators are read as
specifications, and descriptive phrasing in them needs the same care as an API contract.

\subsection{Unstated values fail jointly}

The Coder's disclosure discipline worked as designed: momentum 0.9 and $C = 1.0$ were both
listed as guesses. What it could not express was that the \emph{combination} was unsafe.
Reporting each unstated value in isolation is necessary but not sufficient; the values that
matter most may be the ones that interact, and finding that out required an ablation. This
argues for recording unstated hyperparameters as carefully as unstated architecture, and
for a cheap automated sensitivity probe over them.

\subsection{One seed cannot adjudicate a gap}

Wijaya's two held-out subsets disagreed by more than the gap between the reproduction and the
claim. A Critic that compares one run against one published number under a fixed tolerance
would declare this reproduction a failure --- and would declare a lucky split a success. A
tolerance must account for the variance of the quantity being compared, which in practice
means several seeds per claim.

\subsection{Regeneration rewrites, measurably}

The second report noted that retries regenerate whole scripts rather than patching them.
Wijaya's repair quantifies the cost: triage's fix was the removal of one keyword argument,
and the regenerated script shared only 22\% of its lines with its predecessor. The repair
succeeded, but almost everything else in the script also changed, unreviewed. The plateau
guard's threshold of 0.98 now has two observations --- 0.41 on the second report's injected
fault and 0.22 on this real one --- and neither came near it.

\begin{TABENV}[tbp]
\centering
\small
\caption{Status against the first report's staged timeline. \built{} done; \partl{} partial
or in a different form; \absent{} not started.}
\label{tab:timeline}
\begin{tabular}{L{0.1\textwidth}L{0.41\textwidth}cL{0.35\textwidth}}
\toprule
Phase & Planned & & Status \\
\midrule
Month 1 & PDF ingestion; Reader schema; shared-memory object & \built & Complete; five fields \\
Month 2 & Coder and Docker sandbox for a pilot set; Orchestrator skeleton with a single
  hard-coded retry & \built & Exceeded: a bounded retry loop with triage-driven feedback \\
Month 3 & Critic verdict logic & \absent & Not started; first real inputs now exist \\
        & Real retry loop with targeted feedback & \built & Repaired a real defect \\
        & Report Generator & \absent & Not started \\
        & Single-shot vs.\ iterative ablation & \partl & One observed case only: Wijaya
  fails single-shot and passes on the first retry \\
Month 4 & Expansion towards 20 papers & \partl & Four papers; classification and regression \\
        & Demonstration interface & \partl & Streamlit viewer on a feature branch \\
\bottomrule
\end{tabular}
\end{TABENV}

\section{Development Methodology}
\label{sec:methodology}

The phase continued the delegation pattern described in the second report, though most of
its work was investigative rather than constructive and was carried out directly by the
orchestrating session. One scoped implementation --- the generalization of the Coder and
Runner --- was delegated to a coding subagent.

That delegation produced the third instance, across the project, of a subagent's report
being wrong in a way only independent verification caught. The agent reported its
\emph{Network In Network} regression test as a clean pass. It had passed: the script executed
and the metrics file was well formed. The loss was approximately 947. The discrepancy was
found on review, and led directly to the hidden-default finding above. The practice of
treating an agent's report as a claim to check against the diff and the output remains the
most useful single habit we have adopted.

The two most consequential decisions of the phase were human: the direction change to
support multiple task types, and the choice of $C$ for Tang's rerun. The collapse itself was
spotted by a person reading a log. We state this plainly because an automated replication
system is exactly the kind of project in which human intervention can be quietly absorbed
into a result.

\section{Progress Against the Timeline}
\label{sec:timeline}

Table~\ref{tab:timeline} compares the phase's output with the staged timeline of the first
report. The retry loop arrived ahead of plan and in stronger form than planned. The Critic
and the Report Generator, both Month~3 deliverables, have not started, and the phase's effort
went instead to generalization and to obtaining the first measurable results --- a trade we
consider correct, since a Critic without measurable results would have had nothing to judge.


\section{Limitations}
\label{sec:limitations}

\textbf{Both fidelity results are single runs, compared by hand.} Neither has a variance
estimate, and no stage of the system performed either comparison.

\textbf{The positive result required a human.} Tang's reproduction matches its claim only
after a person chose one hyperparameter, informed by an ablation that had logged test error.

\textbf{The negative result is not attributed.} Wijaya's gap may stem from the unstated
split, from framework defaults the Coder did not reproduce, or from the implementation, and
one seed cannot distinguish them.

\textbf{The CIFAR-10 benchmark remains unmeasured.} GPU compute is still required for it, and
the two CPU-sized papers are not substitutes for the planned evaluation, only a way to test
the pipeline's claim end to end.

\textbf{The Reader does not record missing hyperparameters}, and claims are duplicated
across a paper's prose, tables and figures. Validation still does not converge, and two of
the eight CIFAR-10 papers still cannot be extracted.

\textbf{A note on the dataset.} Boston Housing includes a feature derived from the
proportion of Black residents in each town \cite{harrison1978hedonic}, which is among the
reasons scikit-learn deprecated and then removed its loader. We used it only because the paper
under replication did, and do not recommend it for any other purpose.

\section{Next Steps}
\label{sec:next}

\textbf{Build the Critic, with variance in mind.} Its verdict should remain explicit
arithmetic against a numeric tolerance using the claim's direction, as planned; Wijaya shows
that the tolerance must be informed by several seeds per claim rather than one run. Tang's
collapsed run is its first test case: a verdict of \emph{fail} at 89\% error against 0.87\%
is the minimum it must deliver.

\textbf{Make the Runner check that training learns.} Reading \texttt{train\_metric} at
\texttt{capped}, and comparing a run's loss against the trivial input-ignoring floor, would
have flagged Tang's collapse within twenty epochs. This is smaller than the Critic and
catches a different failure.

\textbf{Record and probe unstated hyperparameters.} Extend the Reader's gap-recording to
hyperparameters, and automate what the Tang ablation did by hand: short runs over candidate
values for the unstated ones, selected on training health and a validation split --- never
the test set.

\textbf{Patch rather than regenerate.} A line-range edit, as in AutoReproduce
\cite{zhao2025autoreproduce}, would keep a one-argument fix to one argument.

\textbf{Merge and extend the viewer} to show Runner, Orchestrator and Critic output, as the
basis of the final demonstration.

\textbf{Secure GPU compute} for the CIFAR-10 set, and build the Report Generator and a
single entry point from PDF to report, after fixing the stages that exit successfully on
failure.

\section{Conclusion}
\label{sec:conclusion}

This phase produced ReproBot's first replication measurements. One reproduction matches its
published claim to within five test images out of ten thousand; the other misses its claim
by a margin that a single run cannot explain. Getting there required generalizing the
pipeline beyond image classification, converting six real prompt failures into explicit
rules, and diagnosing a training collapse that every stage of the pipeline reported as a
success.

That last point is the phase's main result. ReproBot can now read a paper, write an
implementation, execute it at the paper's own scale, and repair its own execution failures.
What it cannot yet do is tell whether the number it produced is right. The two measurements
in this report were judged by people, and one of them needed a person to fix it first.
Building the component that makes those judgements automatically, honestly and with
appropriate uncertainty, is the work of the next phase.

\bibliographystyle{plain}
\begin{thebibliography}{9}

\bibitem{reprobot2026first}
Kütük, A., Varan, E., Korkut, G., \& Özdoğan, G.~M. (2026).
ReproBot: A Multi-Agent System for Automated Scientific Paper Replication --- First
Progress Report. \textit{inzva AI Projects \#10}.

\bibitem{reprobot2026second}
Kütük, A., Varan, E., Korkut, G., \& Özdoğan, G.~M. (2026).
ReproBot: A Multi-Agent System for Automated Scientific Paper Replication --- Second
Progress Report. \textit{inzva AI Projects \#10}.

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
  \node[stage, below=1.75cm of reader] (coder) {\texttt{coder/}\\[-2pt]\footnotesize plain PyTorch, any supervised task};
  \node[stage, below=of coder] (runner) {\texttt{runner/}\\[-2pt]\footnotesize Docker sandbox + triage\\[-2pt]\footnotesize\textcolor{warnc}{success = exit code 0}};
  \node[gone,  below=1.2cm of runner] (critic) {\texttt{critic/}\\[-2pt]\footnotesize compare reproduced vs.\ claimed};
  \node[gone,  below=of critic] (report) {report generator\\[-2pt]\footnotesize claim-by-claim report};

  \node[above=0.5cm of ocr, font=\small, align=center] (pdf)
    {\texttt{dataset/} --- 8 CIFAR-10 PDFs \quad $+$ \quad \texttt{extra-papers/} --- 2 CPU-sized PDFs};

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
  \draw[gonearr] (runner) -- node[right, gonelbl] {\texttt{metrics.<mode>.json}\\ \texttt{higher\_is\_better}} (critic);
  \draw[gonearr] (critic) -- node[right, gonelbl] {pass / retry / fail} (report);

  \draw[looparr] (runner.east) -- ++(1.1,0) |- (coder.east);
  \node[lbl, text=loopc, right=1.25cm of coder.east, anchor=west, yshift=-0.75cm]
    {\textbf{\texttt{orchestrator/}}\\ retries on \emph{crashes}\\ \texttt{triage.suggested\_fix}};

  \draw[gonearr] (critic.west) -- ++(-1.5,0) |- node[pos=0.25, left, gonelbl, align=right]
    {retry on a\\ \emph{numeric gap}\\ --- absent} ([yshift=-0.22cm]coder.west);
\end{tikzpicture}
\caption{ReproBot on 13.09.2026. Solid components are built and verified by execution;
dashed red components do not exist; the dashed amber component is on a feature branch. The
pipeline's shape is unchanged from the second report, but its inputs now include two
CPU-sized papers, one of them tabular regression, and the Coder writes a plain PyTorch loop
for any supervised task. The label inside the Runner marks this phase's central finding: a run is
reported as a success when its script exits normally, whatever number it produced.}
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
  old/.style={st, draw=black!45, text=black!60},
  arr/.style={-{Stealth[length=1.6mm]}, thick},
  row/.style={font=\scriptsize\itshape, anchor=south west}
]
  % row 0: before prompt hardening
  \node[old] (a0) at (0,0) {Coder\\ remembered URL};
  \node[old, right=0.38cm of a0] (b0) {\texttt{probe} \ding{55}\\ HTTP 403, 5\,s};
  \node[old, right=0.38cm of b0] (c0) {triage:\\ \texttt{environment\_error}};
  \node[old, right=0.38cm of c0] (d0) {verdict\\ \texttt{environment\_error}};
  \node[old, right=0.38cm of d0] (e0) {0 retries used};
  \draw[arr, black!45] (a0) -- (b0); \draw[arr, black!45] (b0) -- (c0);
  \draw[arr, black!45] (c0) -- (d0); \draw[arr, black!45] (d0) -- (e0);
  \node[row, text=black!60] at ([yshift=0.08cm]a0.north west) {before prompt hardening};

  % row 1: attempt 1
  \node[st] (a1) at (0,-2.0) {Coder, 75\,s\\ OpenML \texttt{data\_id=531}};
  \node[bad, right=0.38cm of a1] (b1) {\texttt{probe} \ding{55}\\ \texttt{ValueError}, 7.5\,s};
  \node[tri, right=0.38cm of b1] (c1) {triage, 3\,s:\\ \texttt{recoverable\_error}};
  \node[tri, right=0.38cm of c1] (d1) {fix: remove\\ \texttt{version=1}};
  \node[tri, right=0.38cm of d1] (e1) {plateau guard\\ similarity 0.22};
  \draw[arr] (a1) -- (b1); \draw[arr] (b1) -- (c1); \draw[arr] (c1) -- (d1); \draw[arr] (d1) -- (e1);
  \node[row] at ([yshift=0.08cm]a1.north west) {after hardening --- attempt 1};

  % row 2: attempt 2
  \node[st] (a2) at (0,-4.0) {Coder, 74\,s\\ with triage feedback};
  \node[ok, right=0.38cm of a2] (b2) {\texttt{probe} \ding{51}\\ 13.8\,s, dataset asserted};
  \node[ok, right=0.38cm of b2] (c2) {\texttt{smoke} \ding{51}\\ 11\,s};
  \node[ok, right=0.38cm of c2] (d2) {\texttt{capped} \ding{51}\\ 20\,s, RMSE 6.11};
  \node[ok, right=0.38cm of d2, line width=1.2pt] (e2) {\texttt{full} \ding{51}, 1918\,s\\ RMSE 4.48\\ verdict \texttt{success}};
  \draw[arr] (a2) -- (b2); \draw[arr] (b2) -- (c2); \draw[arr] (c2) -- (d2); \draw[arr] (d2) -- (e2);
  \node[row] at ([yshift=0.08cm]a2.north west) {attempt 2 --- one retry used};

  \draw[arr, draw=loopc] (e1.south) -- ++(0,-0.3) -| ([xshift=-0.3cm]a2.west) -- (a2.west);
\end{tikzpicture}
\caption{Wijaya's orchestrated runs. Top: before the prompt changes of
Section~\ref{sec:prompts}, a dead URL was classified as an environmental fault and the loop
stopped without retrying. Middle and bottom: after them, a genuine defect in the regenerated
script was triaged correctly, the Coder regenerated with the one-line fix as feedback, and
the retry passed every stage. The plateau guard's 0.22 is far below its 0.98 stopping
threshold --- and shows how much of the script changed to remove one argument.}
\label{fig:wijayarun}
\end{FIGENV}
"""

FIG_TANG = (
    r"""
\begin{FIGENV}[tbp]
\centering
"""
    + tang_plot()
    + r"""
\caption{Training error for Tang's reproduction at the paper's full setting, from the run
logs. As generated ($C = 1.0$, momentum 0.9), the network learned for a few epochs and then
collapsed onto the error of a network that ignores its input; the run was stopped by hand.
With the single change $C = 0.1$, training error fell smoothly to 0.062\%, and the test set,
evaluated once after the final epoch, gave 0.82\% against the claimed 0.87\%. The loss
values of the two runs are not comparable, since $C$ scales the loss, so error is plotted.}
\label{fig:tang}
\end{FIGENV}
"""
)

FIG_WIJAYA = (
    r"""
\begin{FIGENV}[tbp]
\centering
"""
    + wijaya_plot()
    + r"""
\caption{Wijaya's reproduction at the paper's full setting, from the run log. The model
trained on 324 rows; RMSE on 81 further held-out training rows (every 50 epochs; epoch 1, at
20.7, is off the scale) stayed near the claimed test RMSE throughout, while the 101-row test
set, evaluated once at the end, gave 4.48. Two held-out subsets of the same 506-row dataset
disagree by more than the gap being measured, which is why a single split cannot adjudicate
this reproduction.}
\label{fig:wijaya}
\end{FIGENV}
"""
)


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
    body = body.replace("\\FIGTANG", FIG_TANG)
    body = body.replace("\\FIGWIJAYA", FIG_WIJAYA)
    body = body.replace("FIGENV", figenv).replace("TABENV", tabenv)
    return f"{docclass}\n{geometry}\n{PREAMBLE_COMMON}\n{body}"


out = Path("docs/progress-reports/third-progress-report")
out.mkdir(parents=True, exist_ok=True)
(out / "third_report_singlecolumn.tex").write_text(build(False), encoding="utf-8")
(out / "third_report_twocolumn.tex").write_text(build(True), encoding="utf-8")
print(f"wrote both .tex files to {out}")
