"""Extract this paper's own model architecture from its VLM Markdown.

Same input contract as the other `reader/` stages: already-extracted Markdown
from `ocr.vlm_extract` (`ocr/output/vlm/<paper>.md`), sent to Claude in one
tool-use call. This is the `architecture_notes` field of the project plan's
§1.2 `reader_output`, and its consumer is the Coder, which has to emit a real
`nn.Module` - so this stage is graded on whether an engineer could write the
model from it, not on whether it reads well.

Three things make this stage different from its siblings:

1. **Figures are first-class source text.** `ocr/vlm_extract.py` renders every
   page image and transcribes figures in-line as bracketed `[Figure N: ...]`
   blocks (caption + a description of what the figure depicts). For
   architecture, that block is often where the structure actually lives -
   Network In Network's "three mlpconv layers + one global average pooling
   layer" is stated in the Figure 2 block, not in a table. The prompt
   explicitly directs the model at those blocks.

2. **`specification` is a string, never a number.** Same reasoning as
   `hyperparameters.value`, but sharper: "not stated" and prose formulas
   ("MLP with n layers, n is flexible") must survive verbatim rather than
   collapse into a plausible-looking integer.

3. **`unstated_details` is required, not optional.** Network In Network
   describes three mlpconv layers and global average pooling in prose and in
   Figure 2 but never gives filter counts or kernel sizes in the main text -
   it defers them to supplementary material this pipeline never sees. Same
   "deliberately does not guess" discipline `reader/data_pipeline.py`
   established, with a higher stake: code has to actually run, so a
   confidently-invented channel count produces a model that trains and
   reports a number for the *wrong network*, which is strictly worse than a
   gap that is labelled as a gap. The prompt says exactly that.

This module is an importable extraction step, not a standalone script -
`reader/pipeline.py` is the entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from anthropic import Anthropic
from loguru import logger

from reader.base import Extractor
from reader.tooluse import as_list, request_tool_use

MODEL = "claude-sonnet-5"

PROMPT = """You are reading the Claude-VLM Markdown transcription of one ML paper. \
Your job is to extract this paper's OWN model architecture as structured \
data, precise enough that an engineer could use it to write the model as \
code (e.g. a PyTorch `nn.Module`) - NOT the architecture of a \
baseline/prior-work method it compares itself against or a method it merely \
cites.

Architecture detail lives in THREE kinds of places in this document, and you \
must check all three:
1. METHOD/ARCHITECTURE PROSE sections (e.g. "3 Network In Network", \
"2 Wide residual networks"), including any equations they define.
2. ARCHITECTURE TABLES (e.g. "Table 1: Structure of wide residual networks", \
listing groups, output sizes, and per-group block contents).
3. FIGURE DESCRIPTIONS. This transcription was produced by a vision model \
that read every page image, so figures appear in-line as bracketed blocks \
beginning "[Figure N: ...]", holding the caption followed by a description of \
what the figure actually depicts. For architecture this is frequently where \
the real structure lives - how many stages there are, what feeds what, where \
pooling happens - when the prose only gestures at it. Read those blocks as \
first-class source text, not as decoration, and cite them as sources when you \
use them.

Steps:
1. Record one entry in `sources_examined` per section, table, or figure block \
you looked at, with the transcribed page number it appears under (the nearest \
preceding "## Page N" heading), e.g. "Section 3.1 MLP Convolution Layers \
(transcribed page 3)", "Figure 2 description block (transcribed page 4)", \
"Table 1: Structure of wide residual networks (transcribed page 4)".
2. `model_name`: exactly how the paper names its own model, e.g. "NIN", \
"WRN-n-k".
3. `overall_structure`: prose describing how the components compose end to \
end, in order, from the input image through to the output - the thing you \
would read once before writing the `forward()` method.
4. `components`: one entry per architecturally distinct building block the \
model is made of (e.g. "mlpconv layer", "BasicBlock B(3,3)", "global average \
pooling", "downsampling max-pooling layer", "initial convolution conv1"). For \
each: `name`; `role` (what it does and where it sits in the network); \
`specification` (the concrete build detail the paper gives for it - kernel \
sizes, channel counts, repetition counts, ordering of BN/ReLU/conv, whatever \
the paper actually states); `source` (which section/table/figure and \
transcribed page, e.g. "Figure 2, transcribed page 4" or "Table 1, \
transcribed page 4").
5. `key_equations`: copy, VERBATIM and in the LaTeX exactly as it appears in \
this document, every equation central to understanding how the method \
computes - keep the paper's own equation numbering (e.g. the `\\tag{2}`) if \
it is present. When a paper's whole contribution is a new layer, the equation \
defining that layer IS the implementation specification: copy it, do not \
paraphrase it, do not re-derive it, do not renumber it. \
DO include the baseline equations a paper contrasts itself against - they are \
useful context - but you MUST set `is_own_method` correctly on every entry, \
and give each a one-line `defines`. This matters more than it looks: a paper \
introducing a new layer typically prints the conventional formulation and a \
rival method's formulation right beside its own, so an unmarked list of \
equations is three plausible specifications with no way to tell which one to \
build. Marking them wrong is worse than omitting them.
6. `depth_or_scale`: any formula, naming convention, or scaling rule the \
paper gives for its model's depth or width (e.g. a "depth = 6N + 4" style \
relation, or "WRN-n-k, where n is the total number of convolutional layers \
and k the widening factor"). If the paper gives only a naming convention and \
no closed-form relation, say exactly that - do NOT derive or supply a formula \
the paper does not state.
7. `unstated_details`: CRITICAL, REQUIRED, and the most important field here. \
List explicitly every architectural detail that would be REQUIRED to write \
working code and that this paper does NOT actually state - e.g. per-layer \
filter counts / channel widths, kernel sizes, strides, padding, pooling \
window sizes, the number of units in a described MLP, weight initialization, \
whether a normalization layer is present. Include details the paper defers \
elsewhere without giving the numbers ("the detailed settings of the \
parameters are provided in the supplementary materials", "we follow [8]"): \
record that deferral as the gap, quoting the paper's own wording.
8. CRITICAL - do NOT invent, guess, or "fill in" any architectural value from \
your own knowledge of this architecture, from a later paper, or from what a \
typical implementation would use, even when you are confident you know the \
standard value. `specification` is a string precisely so that "not stated in \
the paper" and prose formulas survive verbatim instead of collapsing into a \
plausible-looking number; "not stated" is always a valid and often the \
correct value. Why this matters more here than in any other extraction: this \
output is consumed by a code generator, and code has to actually RUN. An \
invented-but-plausible number does not fail loudly - it produces a model that \
trains happily and reports a result for the WRONG network, which is far worse \
than a gap that is clearly labelled as a gap. Surface every gap loudly. \
Return an empty `unstated_details` list ONLY if the paper genuinely specifies \
every dimension needed to build the model.
9. Stay in your lane - other extraction stages own the rest of the paper. Do \
NOT include training hyperparameters (learning rate, epochs, batch size, \
optimizer), dataset preprocessing/augmentation, or reported result numbers \
here. A regularizer that is part of the MODEL DEFINITION does belong here, \
described by where it sits in the structure (e.g. "dropout applied on the \
outputs of all but the last mlpconv layer", "dropout inserted between the \
convolutions inside a residual block") - its rate belongs to hyperparameters, \
its placement belongs to you.

Call the `record_architecture_notes` tool with the results."""

ARCHITECTURE_NOTES_TOOL: dict[str, Any] = {
    "name": "record_architecture_notes",
    "description": (
        "Record the paper's own model architecture - overall structure, per-component "
        "specifications, key equations, depth/width scaling rule - as extracted from its "
        "method prose, architecture tables, and transcribed figure descriptions, together "
        "with an explicit list of the architectural details the paper does NOT state, and "
        "bookkeeping about what was examined."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sources_examined": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One entry per prose section, architecture table, or transcribed "
                    "figure block looked at, e.g. 'Figure 2 description block "
                    "(transcribed page 4)'."
                ),
            },
            "model_name": {
                "type": "string",
                "description": (
                    "Exactly how the paper names its own model, e.g. 'NIN', 'WRN-n-k'."
                ),
            },
            "overall_structure": {
                "type": "string",
                "description": (
                    "Prose describing how the components compose end to end, in order, "
                    "from input image to output - what you would read once before "
                    "writing the model's forward() method."
                ),
            },
            "components": {
                "type": "array",
                "description": (
                    "One entry per architecturally distinct building block of the "
                    "paper's own model, after excluding baseline/prior-work "
                    "architectures."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "e.g. 'mlpconv layer', 'BasicBlock B(3,3)', 'global "
                                "average pooling'."
                            ),
                        },
                        "role": {
                            "type": "string",
                            "description": (
                                "What this component does and where it sits in the network."
                            ),
                        },
                        "specification": {
                            "type": "string",
                            "description": (
                                "The concrete build detail the paper states for this "
                                "component (kernel sizes, channel counts, repetition "
                                "counts, BN/ReLU/conv ordering, ...). ALWAYS a string, "
                                "never a bare number: 'not stated in the paper' and "
                                "prose formulas must survive verbatim. Never invent a "
                                "value the paper does not give."
                            ),
                        },
                        "source": {
                            "type": "string",
                            "description": (
                                "e.g. 'Figure 2, transcribed page 4' or 'Section 3.1, "
                                "transcribed page 3'."
                            ),
                        },
                    },
                    "required": ["name", "role", "specification", "source"],
                },
            },
            "key_equations": {
                "type": "array",
                "description": (
                    "Equations central to understanding the method's computation, copied "
                    "VERBATIM in the LaTeX exactly as it appears in the document, keeping "
                    "the paper's own equation numbering. Include the baseline equations a "
                    "paper contrasts itself against, but mark them with "
                    "is_own_method=false - a consumer that cannot tell the proposed "
                    "computation from the one being argued against may implement the "
                    "wrong one."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "latex": {
                            "type": "string",
                            "description": "The equation verbatim, in the LaTeX as it appears.",
                        },
                        "label": {
                            "type": "string",
                            "description": (
                                "The paper's own number for it, e.g. '2'. "
                                "'not numbered' if it has none."
                            ),
                        },
                        "defines": {
                            "type": "string",
                            "description": (
                                "One line: what this equation specifies, e.g. 'the mlpconv "
                                "layer's per-patch MLP computation' or 'a conventional "
                                "linear convolution, shown for contrast'."
                            ),
                        },
                        "is_own_method": {
                            "type": "boolean",
                            "description": (
                                "true if this defines the paper's OWN proposed computation "
                                "(implement this one); false if it belongs to prior work or "
                                "a baseline the paper is contrasting itself against."
                            ),
                        },
                    },
                    "required": ["latex", "label", "defines", "is_own_method"],
                },
            },
            "depth_or_scale": {
                "type": "string",
                "description": (
                    "Any depth/width formula, naming convention, or scaling rule the "
                    "paper gives for its own model, e.g. 'depth = 6N + 4'. If the paper "
                    "gives only a naming convention and no closed-form relation, say "
                    "exactly that - never derive a formula the paper does not state."
                ),
            },
            "unstated_details": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "REQUIRED. Every architectural detail needed to write working code "
                    "that this paper does NOT state, including details deferred to "
                    "supplementary material or to a citation without giving numbers. "
                    "Quote the paper's own deferral wording where there is one. Empty "
                    "list ONLY if the paper genuinely specifies every dimension needed "
                    "to build the model."
                ),
            },
        },
        "required": [
            "sources_examined",
            "model_name",
            "overall_structure",
            "components",
            "key_equations",
            "depth_or_scale",
            "unstated_details",
        ],
    },
}


@dataclass
class ArchitectureComponent:
    name: str
    role: str
    specification: str
    source: str


@dataclass
class KeyEquation:
    """One equation from the paper, and crucially whose computation it defines.

    `is_own_method` is the load-bearing field. A paper introducing a new layer
    almost always prints the conventional formulation, and often a rival's,
    right next to its own - Network In Network prints all three within two
    pages. As a bare list of LaTeX strings these are indistinguishable, so a
    Coder reading them has three plausible specifications and no way to tell
    which one to build. That is worse than having no equations at all.
    """

    latex: str
    label: str
    defines: str
    is_own_method: bool


@dataclass
class ArchitectureNotesExtraction:
    model_name: str
    overall_structure: str
    components: list[ArchitectureComponent]
    key_equations: list[KeyEquation]
    depth_or_scale: str
    unstated_details: list[str]
    sources_examined: list[str]


def _parse_component(raw: object) -> ArchitectureComponent:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an architecture component object, got {raw!r}")
    return ArchitectureComponent(
        name=str(raw["name"]),
        role=str(raw["role"]),
        specification=str(raw["specification"]),
        source=str(raw["source"]),
    )


def _parse_equation(raw: object) -> KeyEquation:
    """Parse one equation entry, tolerating a bare LaTeX string.

    Older runs of this extractor emitted `key_equations` as plain strings, and a
    model can still return one. Rather than crash, keep the LaTeX and mark the
    entry `is_own_method=False` - the conservative reading, since an unmarked
    equation treated as the method's own is exactly the failure this field
    exists to prevent.
    """
    if isinstance(raw, str):
        logger.warning(
            "  [architecture] equation arrived as a bare string with no role; keeping the "
            "LaTeX but marking it is_own_method=false, which is the safe default"
        )
        return KeyEquation(
            latex=raw,
            label="not reported",
            defines="<not reported by the model>",
            is_own_method=False,
        )
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a key-equation object, got {raw!r}")
    return KeyEquation(
        latex=str(raw["latex"]),
        label=str(raw.get("label") or "not numbered"),
        defines=str(raw.get("defines") or "<not reported by the model>"),
        is_own_method=bool(raw.get("is_own_method", False)),
    )


class ArchitectureNotesExtractor(Extractor[ArchitectureNotesExtraction]):
    """Extracts a paper's own model architecture - and, explicitly, what the paper
    fails to specify about it - from its VLM Markdown."""

    name: ClassVar[str] = "architecture_notes"

    def extract(
        self, markdown_text: str, client: Anthropic, feedback: str | None = None
    ) -> ArchitectureNotesExtraction:
        """Send one paper's already-loaded VLM Markdown to Claude and extract its
        own model architecture. No file I/O here - the pipeline owns reading input
        and writing output."""
        prompt = PROMPT
        if feedback:
            prompt = (
                f"{PROMPT}\n\n"
                f"A prior validation pass flagged this specific issue with your "
                f"previous attempt: {feedback}\nAddress it specifically in this "
                f"attempt."
            )

        payload = request_tool_use(
            client,
            log_prefix="architecture",
            model=MODEL,
            max_tokens=8192,
            tool=ARCHITECTURE_NOTES_TOOL,
            user_content=f"{prompt}\n\n---\n\n{markdown_text}",
            required_keys=(
                "sources_examined",
                "model_name",
                "overall_structure",
                "components",
                "depth_or_scale",
            ),
            # Schema-required, but an empty value is a legitimate finding rather
            # than a malformed response: a paper can state no numbered equations,
            # and an empty `unstated_details` means it fully specified itself
            # (rare enough that the stage warns about it below on its own).
            may_be_empty_keys=("key_equations", "unstated_details"),
        )

        sources_examined = [str(source) for source in as_list(payload.get("sources_examined"))]
        model_name = str(payload.get("model_name") or "")
        overall_structure = str(payload.get("overall_structure") or "")
        components = [_parse_component(raw) for raw in as_list(payload.get("components"))]
        key_equations = [_parse_equation(raw) for raw in as_list(payload.get("key_equations"))]
        depth_or_scale = str(payload.get("depth_or_scale") or "")
        unstated_details = [str(detail) for detail in as_list(payload.get("unstated_details"))]

        logger.info(f"  [architecture] sources examined ({len(sources_examined)}):")
        for source in sources_examined:
            logger.info(f"    - {source}")
        logger.info(f"  [architecture] model_name: {model_name}")
        logger.info(f"  [architecture] overall_structure: {overall_structure}")
        logger.info(f"  [architecture] components found: {len(components)}")
        for component in components:
            logger.info(f"    {component.name} ({component.source})")
            logger.info(f"      role: {component.role}")
            logger.info(f"      spec: {component.specification}")
        own_equations = sum(1 for equation in key_equations if equation.is_own_method)
        logger.info(
            f"  [architecture] key equations captured: {len(key_equations)} "
            f"({own_equations} defining this paper's own method, "
            f"{len(key_equations) - own_equations} shown for contrast)"
        )
        for equation in key_equations:
            marker = "OWN     " if equation.is_own_method else "contrast"
            logger.info(f"    - [{marker}] eq {equation.label}: {equation.defines}")
            logger.info(f"                 {equation.latex[:100]}")
        if key_equations and not own_equations:
            logger.warning(
                "  [architecture] every captured equation is marked as prior-work contrast - "
                "either this paper genuinely defines none of its own, or the roles were "
                "assigned wrongly; a Coder will find no equation to implement here"
            )
        logger.info(f"  [architecture] depth_or_scale: {depth_or_scale}")
        logger.info(
            f"  [architecture] details the paper does NOT state ({len(unstated_details)}) "
            f"- these are gaps the Coder must fill or flag, not values to invent:"
        )
        for detail in unstated_details:
            logger.info(f"    - {detail}")
        if not unstated_details:
            logger.warning(
                "  [architecture] unstated_details is EMPTY - the paper is claimed to "
                "fully specify its architecture. Verify that, it is rare."
            )
        if not components:
            logger.error(
                "  [architecture] NO components extracted - every paper this stage runs "
                "on describes a model, so an empty component list means the extraction "
                "failed, not that the paper has no architecture."
            )

        return ArchitectureNotesExtraction(
            model_name=model_name,
            overall_structure=overall_structure,
            components=components,
            key_equations=key_equations,
            depth_or_scale=depth_or_scale,
            unstated_details=unstated_details,
            sources_examined=sources_examined,
        )
