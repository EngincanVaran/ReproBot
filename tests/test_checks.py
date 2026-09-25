from critic.checks import eval_count_check

WRN = {"value": 3.8261217948717956, "unit": "%", "num_eval_samples": 10000}


def test_the_real_wrn_run_is_flagged() -> None:
    finding = eval_count_check(WRN)
    assert finding is not None
    assert finding["kind"] == "eval_count_mismatch"
    assert finding["implied_eval_samples"] == 9984  # 78 batches of 128
    assert finding["dropped"] == 16


def test_a_consistent_count_is_not_flagged() -> None:
    assert eval_count_check({"value": 3.83, "unit": "%", "num_eval_samples": 10000}) is None
    assert eval_count_check({"value": 4.0, "unit": "%", "num_eval_samples": 10000}) is None
    assert (
        eval_count_check({"value": 382 / 9984 * 100, "unit": "%", "num_eval_samples": 9984}) is None
    )


def test_it_stays_silent_when_it_cannot_tell() -> None:
    assert eval_count_check(None) is None
    assert eval_count_check({"value": 3.826, "unit": "%", "num_eval_samples": 10000}) is None
    assert eval_count_check({**WRN, "unit": "RMSE"}) is None
    assert eval_count_check({**WRN, "num_eval_samples": None}) is None
    assert eval_count_check({**WRN, "value": 250.0}) is None
