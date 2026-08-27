"""Turn one CodePlan into a single self-contained training script.

Input is `MethodTranslator`'s output (`coder/models.py`'s `CodePlan`), not
reader/'s JSON or the paper text - Code Synthesizer never re-reasons about
which claim to target or which architecture to use, it only implements the
plan it's handed. Output is plain Python source text; this module never
executes or imports the generated code, matching the platform-trap
discipline documented in CLAUDE.md - `coder/` writes code, `runner/` (not
yet built) is the only stage that ever runs it, inside Docker.

This module is an importable step, not a standalone script - `coder/
pipeline.py` is the entry point that runs `MethodTranslator` then this then
`DependencyResolver`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from loguru import logger

from coder.models import CodePlan, SynthesizedScript

MODEL = "claude-sonnet-5"

PROMPT = """You are writing ONE self-contained Python training script implementing the \
reproduction plan given to you as JSON (a CodePlan: target_claim, \
model_choice, dataset_config, training_config, notes).

Requirements:

1. SELF-CONTAINED: a single script, no other project files, no CLI \
arguments required - hardcode the plan's values. It will run standalone \
inside a Docker container with torch, torchvision, transformers, datasets, \
evaluate, and numpy available. Do not reference any file outside itself.

2. REPRODUCIBILITY: set a fixed random seed (torch, numpy, and python's \
`random`) near the top. Select device as `cuda` if available, else `cpu`.

3. DATASET: load `dataset_config.hf_dataset_id` via the `datasets` library. \
Implement `dataset_config.normalization` and `dataset_config.augmentation` \
as torchvision transforms - augmentation applies to the train split only, \
normalization applies to both. If the description doesn't give exact \
numeric normalization statistics, use the standard published per-channel \
mean/std for that dataset.

4. MODEL: if `model_choice.hf_model_id` is present, load it via \
`transformers.AutoModelForImageClassification.from_pretrained(...)`, \
adapting the classifier head to the dataset's number of classes (pass \
`num_labels=...` and `ignore_mismatched_sizes=True`). If `hf_model_id` is \
absent, define a plain `torch.nn.Module` implementing \
`model_choice.architecture_description`.

5. TRAINING: use `transformers.Trainer` with `TrainingArguments` built from \
`training_config` (batch size, epochs, weight decay). If \
`training_config.optimizer` is not AdamW (Trainer's default), construct the \
specified optimizer (e.g. SGD with momentum) and an LR scheduler matching \
`training_config.lr_schedule` yourself, and pass both to \
`Trainer(optimizers=(optimizer, scheduler), ...)`.

6. METRIC: implement `compute_metrics` so the reported number is directly \
comparable to `target_claim.metric`/`target_claim.unit` - e.g. if the \
metric is an error rate, report `100 - accuracy`, not raw accuracy.

7. RESULT LINE: at the very end of the script, after evaluation completes, \
print EXACTLY ONE line of the form:
   REPROBOT_RESULT <json>
   where <json> is a JSON object: {"metric": "<target_claim.metric>", \
"value": <float>, "unit": "<target_claim.unit>"}. This is the only way the \
result reaches the rest of the pipeline - never omit it, never print more \
than one such line.

8. CAVEAT BANNER: if `model_choice.is_exact_reproduction` is false, print a \
one-line warning near the start of the script quoting \
`model_choice.caveats`, so anyone reading the run's logs sees the \
architecture substitution immediately.

Call the `record_script` tool with the result."""

SCRIPT_TOOL: dict[str, Any] = {
    "name": "record_script",
    "description": "Record the generated self-contained training script.",
    "input_schema": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "Full Python script content, ready to save as a .py file and run.",
            },
            "notes": {
                "type": "string",
                "description": (
                    "Optional implementation-level caveats or workarounds, e.g. how the "
                    "optimizer/schedule was fit into transformers.Trainer's API."
                ),
            },
        },
        "required": ["script"],
    },
}


def _tool_input(message: Message) -> dict[str, object]:
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not call the record_script tool")


class CodeSynthesizer:
    """Writes one self-contained training script implementing a CodePlan."""

    def synthesize(
        self, code_plan: CodePlan, client: Anthropic, feedback: str | None = None
    ) -> SynthesizedScript:
        """No file I/O here - the pipeline owns writing the script out."""
        prompt = PROMPT
        if feedback:
            prompt = (
                f"{PROMPT}\n\n"
                f"A prior attempt's feedback flagged this specific issue: {feedback}\n"
                f"Address it specifically in this attempt."
            )

        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            tools=[cast(ToolParam, SCRIPT_TOOL)],
            tool_choice=cast(ToolChoiceToolParam, {"type": "tool", "name": "record_script"}),
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n---\n\n{json.dumps(asdict(code_plan), indent=2)}",
                }
            ],
        )
        payload = _tool_input(message)
        script = str(payload["script"])
        notes = payload.get("notes")

        logger.info(f"  [code_synthesizer] script generated: {len(script.splitlines())} lines")
        if notes:
            logger.info(f"  [code_synthesizer] notes: {notes}")

        return SynthesizedScript(script=script, notes=str(notes) if notes else None)
