"""Translate reader/'s structured extraction into one concrete CodePlan.

Input is one paper's already-loaded `reader/output/<paper>.json` (a plain
dict - `claims`, `hyperparameters`, `data_pipeline`, `validation`), not the
paper's Markdown or PDF - Method Translator never re-reads the paper itself,
it only reasons over what Reader already extracted. Two judgment calls
happen here that reader/'s extractors don't have to make: which single
claim (out of potentially many table rows) to target for reproduction, and
which model architecture to actually build.

v1 deliberately does not attempt to reproduce a paper's exact custom
architecture from a text description - there is no `architecture_notes`
extraction in reader/ yet (see CLAUDE.md), and precisely translating prose
into a correct novel `nn.Module` is failure-prone. Instead, this picks an
existing Hugging Face architecture that is a reasonable stand-in for the
paper's method family (e.g. a ResNet variant for a paper proposing a novel
residual-network variant), and marks `is_exact_reproduction=False` with a
caveat explaining the substitution. This trades reproduction fidelity for a
pipeline that reliably produces a runnable script - see coder/README.md.

This module is an importable step, not a standalone script - `coder/
pipeline.py` is the entry point that loads reader/'s output, runs
`MethodTranslator.translate()`, and passes the resulting CodePlan to
`CodeSynthesizer`.
"""

from __future__ import annotations

import json
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from loguru import logger

from coder.models import CodePlan, DatasetConfig, ModelChoice, TargetClaim, TrainingConfig

MODEL = "claude-sonnet-5"

PROMPT = """You are turning one ML paper's already-extracted structured data (its \
own reported claims, training hyperparameters, and dataset/preprocessing \
info) into a concrete plan for reproducing ONE of its results in code.

You are given, as JSON: the paper's title, its `claims` (own reported \
results - there are usually several, from different tables/configurations), \
its `hyperparameters`, its `data_pipeline` info (per-dataset preprocessing/ \
augmentation/split), and any `validation_flags` a prior cross-check pass \
raised (uncertainties or gaps - use judgment about whether they affect your \
choices below).

Make these decisions:

1. PICK ONE TARGET CLAIM to reproduce, out of the `claims` list. Prefer \
CIFAR-10 over CIFAR-100 if the paper reports both. Among multiple \
configurations for that dataset, pick whichever the paper treats as its \
flagship/best result for its own proposed method (not an ablation variant), \
typically the lowest error / highest accuracy the paper highlights in its \
abstract or introduction.

2. PICK A MODEL ARCHITECTURE. Do NOT attempt to reconstruct the paper's \
exact custom architecture from prose - that is out of scope here. Instead:
   - If a well-known Hugging Face Hub model id is a reasonable architectural \
stand-in for the paper's method family (e.g. a ResNet variant for a paper \
proposing a novel residual network), give its `hf_model_id` and set \
`is_exact_reproduction` to false, explaining the substitution in `caveats`.
   - If nothing on the Hub is a reasonable stand-in, omit `hf_model_id` \
entirely and instead describe, in `architecture_description`, a simple, \
clearly-buildable custom architecture in the spirit of the paper's method \
(e.g. "a ~10-layer CNN with batch normalization and residual connections, \
channel widths doubling every stage") - concrete enough for someone to \
write a working `nn.Module` from it, without claiming architectural \
fidelity to the paper. Always set `is_exact_reproduction` to false and \
explain in `caveats` what was substituted and why.

3. ASSEMBLE THE DATASET CONFIG for the target claim's dataset, using the \
matching entry in `data_pipeline`. Give a real Hugging Face `datasets` hub \
id (e.g. "cifar10", "uoft-cs/cifar100").

4. ASSEMBLE THE TRAINING CONFIG from `hyperparameters`, filtering to what's \
relevant to the target claim's model_variant when hyperparameters are \
variant-specific. Put anything relevant that doesn't fit optimizer/ \
learning_rate/lr_schedule/batch_size/epochs/weight_decay into \
`other_hyperparameters` rather than dropping it. If a value genuinely isn't \
available, use a standard reasonable default and say so in `notes` - never \
invent a value and present it as if the paper stated it.

5. Use `notes` for anything else Code Synthesizer should know (e.g. a \
relevant validation flag, an assumption you made).

Call the `record_code_plan` tool with the results."""

CODE_PLAN_TOOL: dict[str, Any] = {
    "name": "record_code_plan",
    "description": "Record the concrete reproduction plan translated from reader/'s output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "target_claim": {
                "type": "object",
                "description": "The single claim (from the input claims list) chosen to reproduce.",
                "properties": {
                    "claim_id": {"type": "string"},
                    "metric": {"type": "string"},
                    "dataset": {"type": "string"},
                    "reported_value": {"type": "number"},
                    "unit": {"type": "string"},
                    "model_variant": {"type": "string", "description": "Optional."},
                },
                "required": ["claim_id", "metric", "dataset", "reported_value", "unit"],
            },
            "model_choice": {
                "type": "object",
                "properties": {
                    "hf_model_id": {
                        "type": "string",
                        "description": (
                            "Hugging Face Hub model id, e.g. 'microsoft/resnet-50'. "
                            "Omit entirely if no suitable stand-in exists."
                        ),
                    },
                    "architecture_description": {
                        "type": "string",
                        "description": (
                            "What to build: either why hf_model_id is a reasonable "
                            "stand-in, or a concrete custom architecture description "
                            "if hf_model_id is omitted."
                        ),
                    },
                    "is_exact_reproduction": {
                        "type": "boolean",
                        "description": "Always false in v1 - no exact reproduction path exists.",
                    },
                    "caveats": {
                        "type": "string",
                        "description": "What was substituted vs. the paper's real architecture.",
                    },
                },
                "required": ["architecture_description", "is_exact_reproduction"],
            },
            "dataset_config": {
                "type": "object",
                "properties": {
                    "hf_dataset_id": {"type": "string"},
                    "normalization": {"type": "string"},
                    "augmentation": {"type": "string"},
                    "split_convention": {"type": "string"},
                },
                "required": [
                    "hf_dataset_id",
                    "normalization",
                    "augmentation",
                    "split_convention",
                ],
            },
            "training_config": {
                "type": "object",
                "properties": {
                    "optimizer": {"type": "string"},
                    "learning_rate": {"type": "string"},
                    "lr_schedule": {"type": "string", "description": "Optional."},
                    "batch_size": {"type": "string"},
                    "epochs": {"type": "string"},
                    "weight_decay": {"type": "string", "description": "Optional."},
                    "other_hyperparameters": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Any other relevant hyperparameters, name -> value.",
                    },
                },
                "required": ["optimizer", "learning_rate", "batch_size", "epochs"],
            },
            "notes": {"type": "string", "description": "Optional guidance for Code Synthesizer."},
        },
        "required": ["target_claim", "model_choice", "dataset_config", "training_config"],
    },
}


def _tool_input(message: Message) -> dict[str, object]:
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not call the record_code_plan tool")


def _parse_target_claim(raw: object) -> TargetClaim:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a target_claim object, got {raw!r}")
    model_variant = raw.get("model_variant")
    return TargetClaim(
        claim_id=str(raw["claim_id"]),
        metric=str(raw["metric"]),
        dataset=str(raw["dataset"]),
        reported_value=float(raw["reported_value"]),
        unit=str(raw["unit"]),
        model_variant=str(model_variant) if model_variant is not None else None,
    )


def _parse_model_choice(raw: object) -> ModelChoice:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a model_choice object, got {raw!r}")
    hf_model_id = raw.get("hf_model_id")
    caveats = raw.get("caveats")
    return ModelChoice(
        hf_model_id=str(hf_model_id) if hf_model_id else None,
        architecture_description=str(raw["architecture_description"]),
        is_exact_reproduction=bool(raw.get("is_exact_reproduction", False)),
        caveats=str(caveats) if caveats else None,
    )


def _parse_dataset_config(raw: object) -> DatasetConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dataset_config object, got {raw!r}")
    return DatasetConfig(
        hf_dataset_id=str(raw["hf_dataset_id"]),
        normalization=str(raw["normalization"]),
        augmentation=str(raw["augmentation"]),
        split_convention=str(raw["split_convention"]),
    )


def _parse_training_config(raw: object) -> TrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a training_config object, got {raw!r}")
    lr_schedule = raw.get("lr_schedule")
    weight_decay = raw.get("weight_decay")
    other = raw.get("other_hyperparameters")
    return TrainingConfig(
        optimizer=str(raw["optimizer"]),
        learning_rate=str(raw["learning_rate"]),
        lr_schedule=str(lr_schedule) if lr_schedule else None,
        batch_size=str(raw["batch_size"]),
        epochs=str(raw["epochs"]),
        weight_decay=str(weight_decay) if weight_decay else None,
        other_hyperparameters={str(k): str(v) for k, v in other.items()}
        if isinstance(other, dict)
        else {},
    )


class MethodTranslator:
    """Picks a target claim and model/dataset/training config, producing one
    CodePlan for CodeSynthesizer to turn into a script."""

    def translate(
        self, reader_output: dict[str, Any], client: Anthropic, feedback: str | None = None
    ) -> CodePlan:
        """No file I/O here - the pipeline owns reading reader/'s output and
        writing the CodePlan out as part of the combined coder output."""
        prompt = PROMPT
        if feedback:
            prompt = (
                f"{PROMPT}\n\n"
                f"A prior attempt's feedback flagged this specific issue: {feedback}\n"
                f"Address it specifically in this attempt."
            )

        context = {
            "paper": reader_output.get("paper"),
            "claims": reader_output.get("claims", {}).get("claims", []),
            "hyperparameters": reader_output.get("hyperparameters", {}).get("hyperparameters", []),
            "data_pipeline": reader_output.get("data_pipeline", {}),
            "validation_flags": reader_output.get("validation", {}).get("flags", []),
        }
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=[cast(ToolParam, CODE_PLAN_TOOL)],
            tool_choice=cast(ToolChoiceToolParam, {"type": "tool", "name": "record_code_plan"}),
            messages=[
                {"role": "user", "content": f"{prompt}\n\n---\n\n{json.dumps(context, indent=2)}"}
            ],
        )
        payload = _tool_input(message)

        target_claim = _parse_target_claim(payload["target_claim"])
        model_choice = _parse_model_choice(payload["model_choice"])
        dataset_config = _parse_dataset_config(payload["dataset_config"])
        training_config = _parse_training_config(payload["training_config"])
        notes = payload.get("notes")

        logger.info(
            f"  [method_translator] target claim: {target_claim.claim_id} - "
            f"{target_claim.metric} = {target_claim.reported_value}{target_claim.unit} "
            f"on {target_claim.dataset}"
            + (f" [{target_claim.model_variant}]" if target_claim.model_variant else "")
        )
        logger.info(
            f"  [method_translator] model choice: "
            f"{model_choice.hf_model_id or '(custom architecture)'} "
            f"(exact_reproduction={model_choice.is_exact_reproduction})"
        )
        if model_choice.caveats:
            logger.info(f"  [method_translator] caveats: {model_choice.caveats}")
        logger.info(f"  [method_translator] dataset: {dataset_config.hf_dataset_id}")
        logger.info(
            f"  [method_translator] training config: optimizer={training_config.optimizer}, "
            f"lr={training_config.learning_rate}, batch_size={training_config.batch_size}, "
            f"epochs={training_config.epochs}"
        )

        return CodePlan(
            target_claim=target_claim,
            model_choice=model_choice,
            dataset_config=dataset_config,
            training_config=training_config,
            notes=str(notes) if notes else None,
        )
