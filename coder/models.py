"""Data shapes threaded between coder/'s sequential stages.

`CodePlan` is Method Translator's output and Code Synthesizer's input;
`Dependencies` is Dependency Resolver's output. Kept in one shared module
because, unlike reader/'s per-extractor dataclasses (each owned by exactly
one extractor), these are genuinely produced by one stage and consumed by
another.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TargetClaim:
    """The one flagship result Method Translator picked to reproduce, out of
    potentially many rows in reader/'s claims output."""

    claim_id: str
    metric: str
    dataset: str
    reported_value: float
    unit: str
    model_variant: str | None


@dataclass
class ModelChoice:
    """Which architecture Code Synthesizer should build. v1 favors an
    existing Hugging Face architecture over reproducing the paper's custom
    one from text - see coder/README.md for why - so this is deliberately
    NOT a literal layer-by-layer spec."""

    hf_model_id: str | None
    architecture_description: str
    is_exact_reproduction: bool
    caveats: str | None


@dataclass
class DatasetConfig:
    hf_dataset_id: str
    normalization: str
    augmentation: str
    split_convention: str


@dataclass
class TrainingConfig:
    optimizer: str
    learning_rate: str
    lr_schedule: str | None
    batch_size: str
    epochs: str
    weight_decay: str | None
    other_hyperparameters: dict[str, str] = field(default_factory=dict)


@dataclass
class CodePlan:
    """Method Translator's output: everything Code Synthesizer needs to
    write one self-contained training script, with no further paper text
    to consult."""

    target_claim: TargetClaim
    model_choice: ModelChoice
    dataset_config: DatasetConfig
    training_config: TrainingConfig
    notes: str | None


@dataclass
class SynthesizedScript:
    """Code Synthesizer's output: the full training script plus any
    implementation-level notes (e.g. a workaround needed to fit the paper's
    optimizer/schedule into transformers.Trainer's API)."""

    script: str
    notes: str | None


@dataclass
class Dependency:
    package: str
    version: str


@dataclass
class Dependencies:
    """Dependency Resolver's output: pinned pip requirements for the
    generated script, derived deterministically from its imports."""

    requirements: list[Dependency]
