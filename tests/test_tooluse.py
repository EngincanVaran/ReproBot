"""Replays of the two real tool-payload leaks this repo has hit.

Both are the same failure: the model closed one field's delimiter and then wrote the
remaining tool fields into it as text, leaving them absent from the payload.
"""

from typing import Any

from reader.tooluse import recover_leaked_fields

CODER_FIELDS = ("architecture_used", "dataset_used", "hyperparameters_used")
CRITIC_FIELDS = ("curve_assessment", "findings")


def test_coder_wide_resnet_leak() -> None:
    # Wide Residual Networks, first two runs: two fields serialized inside a third.
    payload: dict[str, Any] = {
        "architecture_used": (
            "WRN-28-10 rebuilt from architecture_notes.</architecture_used>\n"
            '<parameter name="dataset_used">CIFAR-10 (torchvision)</dataset_used>\n'
            '<parameter name="hyperparameters_used">[{"name": "optimizer", "value": "SGD"}]'
        )
    }
    recovered = recover_leaked_fields(payload, CODER_FIELDS, "t")
    assert recovered["architecture_used"] == "WRN-28-10 rebuilt from architecture_notes."
    assert recovered["dataset_used"] == "CIFAR-10 (torchvision)"
    assert recovered["hyperparameters_used"] == [{"name": "optimizer", "value": "SGD"}]


def test_a_properly_emitted_field_survives_a_leaked_duplicate() -> None:
    payload: dict[str, Any] = {
        "architecture_used": (
            "real value</architecture_used>\n"
            '<parameter name="dataset_used">leaked copy</dataset_used>'
        ),
        "dataset_used": "properly emitted",
    }
    assert recover_leaked_fields(payload, CODER_FIELDS, "t")["dataset_used"] == "properly emitted"


def test_a_payload_with_no_leak_is_returned_unchanged() -> None:
    payload: dict[str, Any] = {"architecture_used": "a plain value", "dataset_used": "another"}
    assert recover_leaked_fields(payload, CODER_FIELDS, "t") == payload
