from critic.history import build_records, claim_target, parse_trainer_log

LOG = """\
some progress text that is not a dict
{'loss': 1.0, 'grad_norm': 2.0, 'learning_rate': 0.1, 'epoch': 0.5}
{'loss': 0.8, 'grad_norm': 2.0, 'learning_rate': 0.1, 'epoch': 1.0}
{'eval_loss': 0.9, 'eval_accuracy': 60.0, 'epoch': 1.0}
{'loss': 0.4, 'grad_norm': 2.0, 'learning_rate': 0.02, 'epoch': 2.0}
{'eval_loss': 0.5, 'eval_accuracy': 80.0, 'epoch': 2.0}
{'train_runtime': 12.0, 'train_loss': 0.7, 'epoch': 2.0}
{'eval_loss': 0.5, 'eval_accuracy': 80.0, 'epoch': 2.0}
"""


def test_a_log_becomes_one_record_per_epoch() -> None:
    evals, losses = parse_trainer_log(LOG)
    assert sorted(evals) == [1, 2]
    assert losses == {1: [1.0, 0.8], 2: [0.4]}
    metrics = {"metric": "Test error", "num_train_samples": 50000, "num_eval_samples": 10000}
    first, last = build_records(evals, losses, metrics, target=4.0, chance=90.0)
    assert first["eval_metric"] == 40.0  # error = 100 - accuracy
    assert first["train_loss"] == 0.9
    assert last["steps_total"] == 2
    assert last["higher_is_better"] is False
    assert last["target_value"] == 4.0


def test_an_accuracy_metric_is_kept_as_accuracy() -> None:
    evals, losses = parse_trainer_log(LOG)
    (first, _) = build_records(
        evals, losses, {"metric": "Top-1 accuracy"}, target=None, chance=None
    )
    assert first["eval_metric"] == 60.0
    assert "target_value" not in first  # nothing is invented


def test_claim_target_reads_both_claim_shapes() -> None:
    claim = {"claim_id": "c1", "reported_value": 4.0}
    assert claim_target({"claims": {"claims": [claim]}}, "c1") == 4.0
    assert claim_target({"claims": [claim]}, "c1") == 4.0
    assert claim_target({"claims": [claim]}, "c9") is None
