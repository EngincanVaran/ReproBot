"""Replay real learning curves through the Runner's live check-ups.

Every curve below comes from an actual ReproBot run or its ablation, not from a
synthetic shape chosen to pass. The two failure modes of phase 3 must halt early;
every healthy run must never halt - a false halt kills a reproduction.
"""

import json
from pathlib import Path

import pytest

from runner.checkups import (
    HANG_FLOOR_SECONDS,
    HANG_GAP_MULTIPLIER,
    PROGRESS_MARKER,
    ArrivalClock,
    HistoryTail,
    ProgressRecord,
    check_hang,
    evaluate,
    hang_limit,
    history_path_from_metrics,
    parse_record,
)

ABLATION = Path(__file__).resolve().parents[1] / "docs/notes/tang-2013-ablation/ablation.json"


def rec(
    step: int,
    total: int | None,
    *,
    loss: float | None = None,
    eval_metric: float | None = None,
    train_metric: float | None = None,
    metric: str = "test error",
    unit: str = "%",
    higher_is_better: bool = False,
    chance: float | None = None,
    n: int | None = 10000,
    kind: str = "epoch",
    bound: float | None = 0.0,
    target: float | None = None,
) -> ProgressRecord:
    return ProgressRecord(
        step=step,
        steps_total=total,
        kind=kind,
        train_loss=loss,
        eval_loss=None,
        train_metric=train_metric,
        eval_metric=eval_metric,
        metric=metric,
        unit=unit,
        higher_is_better=higher_is_better,
        chance_metric=chance,
        target_value=target,
        loss_lower_bound=bound,
        num_eval_samples=n,
        elapsed_seconds=None,
    )


def first_halt(records: list[ProgressRecord], mode: str = "full") -> tuple[str, int] | None:
    """Feed records one at a time, as the watcher sees them; return (rule, step)."""
    for i in range(1, len(records) + 1):
        halt = evaluate(records[:i], mode)
        if halt is not None:
            return halt.rule, halt.step
    return None


# --------------------------------------------------------------------------- #
# Tang 2013 - six-configuration ablation, 30 epochs each (MNIST, test error %)
# --------------------------------------------------------------------------- #


def ablation_records(name: str) -> list[ProgressRecord]:
    configs = {c["name"]: c for c in json.loads(ABLATION.read_text(encoding="utf-8"))}
    return [
        rec(r["epoch"], 400, loss=r["train_loss"], eval_metric=r["test_err_clean"], chance=88.65)
        for r in configs[name]["history"]
    ]


def test_tang_as_generated_config_halts_when_it_diverges() -> None:
    assert first_halt(ablation_records("A_as_generated")) == ("diverged", 12)


def test_tang_whitened_config_halts_on_collapse() -> None:
    assert first_halt(ablation_records("E_pca_whitened")) == ("diverged", 6)


@pytest.mark.parametrize("name", ["B_momentum_0.5", "C_momentum_0.0", "D_C_0.1", "F_init_N0.01"])
def test_tang_stable_configs_never_halt(name: str) -> None:
    assert first_halt(ablation_records(name)) is None


def test_tang_real_full_run_collapse_halts() -> None:
    # The real C=1.0 full run, read live from the container (sparse: every 20-60 epochs).
    points = [(1, 1.888, 27.1), (20, 3.554, 83.2), (40, 3.781, 85.5), (100, 3.611, 89.5)]
    records = [rec(e, 400, loss=loss, train_metric=err, chance=88.8) for e, loss, err in points]
    assert first_halt(records) is not None


def test_tang_healthy_full_run_never_halts() -> None:
    errors = [27.44, 12.202, 9.812, 8.482, 6.857, 5.485, 4.443, 3.503, 2.8, 2.093, 1.545]
    records = [
        rec(1 + 20 * i, 400, loss=0.18 / (1 + i), train_metric=err, chance=88.8)
        for i, err in enumerate(errors)
    ]
    assert first_halt(records) is None


# --------------------------------------------------------------------------- #
# Frosst & Hinton - soft decision tree (MNIST, test accuracy %)
# --------------------------------------------------------------------------- #

BROKEN_FULL = [(1, -1.5632, 0.28), (2, -2.9972, 0.17), (3, -3.0053, 0.16), (4, -3.0077, 0.15)]
BROKEN_CAPPED = [(1, -0.8340, 5.08), (2, -0.8341, 5.47), (3, -0.8341, 8.20), (4, -0.8341, 7.03)]


def softtree(
    points: list[tuple[int, float, float]], total: int, *, n: int, bound: float | None
) -> list[ProgressRecord]:
    return [
        rec(
            e,
            total,
            loss=loss,
            eval_metric=acc,
            metric="test accuracy",
            higher_is_better=True,
            chance=11.35,
            n=n,
            bound=bound,
            target=94.45,
        )
        for e, loss, acc in points
    ]


def test_softtree_reversed_loss_halts_at_first_epoch_of_full() -> None:
    assert first_halt(softtree(BROKEN_FULL, 40, n=10000, bound=0.0)) == (
        "loss_below_lower_bound",
        1,
    )


def test_softtree_reversed_loss_halts_in_capped_before_the_long_run() -> None:
    records = softtree(BROKEN_CAPPED, 5, n=256, bound=0.0)
    assert first_halt(records, "capped") == ("loss_below_lower_bound", 1)


def test_softtree_without_declared_bound_is_left_to_the_full_stage() -> None:
    # Capped is too short to call a flat loss "no progress" (see the next test).
    records = softtree(BROKEN_CAPPED, 5, n=256, bound=None)
    assert first_halt(records, "capped") is None


def test_flat_loss_with_learning_predictions_never_halts() -> None:
    # Real capped run of a correctly generated soft tree (2026-09-14): leaves start
    # uniform, so the loss sat at ln 10 while accuracy was already 3x chance. An
    # earlier version of the rules halted this - a false positive caught end to end.
    points = [(1, 2.3026, 30.0), (2, 2.3025, 29.6), (3, 2.3025, 29.6), (4, 2.3024, 29.6)]
    for mode, total in (("capped", 5), ("full", 4)):
        assert first_halt(softtree(points, total, n=250, bound=0.0), mode) is None


def test_flat_loss_at_chance_halts_in_full() -> None:
    points = [(e, 2.3026, 10.2) for e in range(1, 6)]
    assert first_halt(softtree(points, 40, n=10000, bound=0.0)) == ("no_progress", 4)


def test_softtree_without_declared_bound_halts_on_worse_than_chance() -> None:
    points = [*BROKEN_FULL, (5, -3.0080, 0.15)]
    halt = first_halt(softtree(points, 40, n=10000, bound=None))
    assert halt == ("worse_than_chance", 5)


def test_softtree_fixed_full_run_never_halts() -> None:
    acc = [
        65.84,
        80.68,
        84.40,
        89.10,
        90.64,
        91.74,
        93.58,
        93.72,
        94.14,
        93.85,
        94.50,
        94.27,
        94.39,
        94.38,
        94.47,
        94.39,
        94.49,
        94.41,
        94.41,
        94.73,
        94.54,
        94.85,
        94.98,
        94.54,
        95.06,
        94.81,
        94.95,
        94.95,
        95.10,
        94.94,
        94.68,
        94.89,
        95.08,
        95.16,
        95.02,
        94.53,
        94.98,
        95.11,
        95.07,
        95.11,
    ]
    loss = [
        1.3531,
        0.7373,
        0.5396,
        0.4452,
        0.3876,
        0.3485,
        0.2956,
        0.2573,
        0.2438,
        0.2322,
        0.2201,
        0.2096,
        0.1990,
        0.1906,
        0.1831,
        0.1764,
        0.1719,
        0.1652,
        0.1608,
        0.1521,
        0.1486,
        0.1466,
        0.1418,
        0.1383,
        0.1355,
        0.1305,
        0.1298,
        0.1266,
        0.1200,
        0.1168,
        0.1143,
        0.1140,
        0.1116,
        0.1099,
        0.1054,
        0.1037,
        0.1027,
        0.1011,
        0.0982,
        0.0954,
    ]
    points = [(i + 1, loss[i], acc[i]) for i in range(40)]
    assert first_halt(softtree(points, 40, n=10000, bound=0.0)) is None


# --------------------------------------------------------------------------- #
# Wijaya, Fashion-MNIST random forest, SVM guide
# --------------------------------------------------------------------------- #


def test_wijaya_noisy_regression_never_halts() -> None:
    val = [
        3.4525,
        3.6150,
        3.3733,
        3.4795,
        3.3614,
        4.0142,
        3.2760,
        3.2580,
        3.4087,
        3.1587,
        3.4512,
        3.4664,
        3.2748,
        3.2184,
        3.4997,
        3.4581,
        3.3357,
        3.2041,
        3.2520,
        3.3102,
    ]
    mse = [
        15.3702,
        12.9997,
        7.1157,
        10.5356,
        6.3814,
        7.9118,
        5.7481,
        5.7324,
        7.6494,
        10.4038,
        5.2927,
        5.8451,
        4.7384,
        7.1127,
        5.3000,
        4.6026,
        4.7215,
        6.1806,
        6.6946,
        10.8350,
    ]
    points = [(1, 531.36, 20.72)] + [
        (50 * (i + 1), m, v) for i, (m, v) in enumerate(zip(mse, val, strict=True))
    ]
    records = [
        rec(e, 1000, loss=m, eval_metric=v, metric="RMSE", unit="", chance=9.2, n=81, target=3.02)
        for e, m, v in points
    ]
    assert first_halt(records) is None


def test_random_forest_repetitions_never_halt() -> None:
    records = [
        rec(
            i + 1,
            5,
            kind="repetition",
            eval_metric=v,
            metric="Test Accuracy",
            unit="",
            higher_is_better=True,
            chance=0.1,
            bound=None,
            target=0.873,
        )
        for i, v in enumerate([0.8773, 0.8774, 0.8753, 0.8792, 0.8775])
    ]
    assert first_halt(records) is None


def test_svm_single_fit_never_halts() -> None:
    record = rec(
        1,
        1,
        kind="fit",
        eval_metric=96.625,
        metric="Accuracy",
        higher_is_better=True,
        chance=50.0,
        n=4000,
        bound=None,
        target=96.9,
    )
    assert first_halt([record]) is None


# --------------------------------------------------------------------------- #
# Hang - records stop arriving. Timings below are measured, not invented.
# --------------------------------------------------------------------------- #

# Seconds between consecutive records of the SVM guide's real `full` run (192
# records: two cross-validation grids, then the final fit), from the history
# file's `elapsed_seconds`. The script restarted that clock for the second grid,
# which is why the rule runs on the watcher's clock instead.
SVM_FULL_GAPS = [
    *(0.44, 0.43, 0.44, 0.47, 0.44, 0.44, 0.38, 0.29, 0.25, 0.28, 0.42, 0.44, 0.44, 0.44, 0.45),
    *(0.4, 0.29, 0.21, 0.16, 0.19, 0.5, 0.43, 0.44, 0.44, 0.41, 0.29, 0.2, 0.15, 0.12, 0.15, 0.44),
    *(0.49, 0.43, 0.38, 0.29, 0.21, 0.14, 0.11, 0.11, 0.13, 0.43, 0.46, 0.39, 0.29, 0.2, 0.19),
    *(0.11, 0.11, 0.13, 0.16, 0.45, 0.61, 0.34, 0.21, 0.15, 0.13, 0.12, 0.12, 0.12, 0.14, 0.39),
    *(0.29, 0.2, 0.21, 0.13, 0.13, 0.12, 0.12, 0.15, 0.19, 0.29, 0.2, 0.16, 0.13, 0.15, 0.13, 0.14),
    *(0.17, 0.27, 0.32, 0.21, 0.15, 0.13, 0.14, 0.13, 0.2, 0.2, 0.38, 0.63, 0.61, 0.15, 0.14, 0.14),
    *(0.14, 0.16, 0.24, 0.42, 1.03, 1.7, 1.18, 0.14, 0.14, 0.15, 0.17, 0.32, 0.53, 1.06, 3.38),
    *(5.33, 2.08, 0.1, 0.11, 0.12, 0.13, 0.13, 0.14, 0.17, 0.16, 0.17, 0.11, 0.12, 0.12, 0.12),
    *(0.13, 0.14, 0.14, 0.15, 0.17, 0.11, 0.12, 0.12, 0.13, 0.13, 0.13, 0.15, 0.19, 0.16, 0.12),
    *(0.11, 0.13, 0.12, 0.13, 0.14, 0.15, 0.15, 0.16, 0.12, 0.11, 0.13, 0.13, 0.14, 0.13, 0.15),
    *(0.16, 0.17, 0.11, 0.12, 0.13, 0.13, 0.13, 0.13, 0.15, 0.16, 0.21, 0.12, 0.12, 0.13, 0.13),
    *(0.13, 0.14, 0.14, 0.16, 0.18, 0.11, 0.12, 0.12, 0.13, 0.14, 0.14, 0.15, 0.16, 0.17, 0.12),
    *(0.12, 0.12, 0.14, 0.13, 0.14, 0.16, 0.16, 0.17, 0.1),
]

# WRN-28-10's real `full` run: 200 epochs in 3.4 h on an RTX 4090. Its history has
# no `elapsed_seconds`, so this is the run's mean epoch, not a per-epoch reading.
WRN_GPU_EPOCH_SECONDS = 3.4 * 3600 / 200
# The same network measured on the dev machine's CPU: ~22 days for 200 epochs.
WRN_CPU_EPOCH_SECONDS = 22 * 86400 / 200


def epoch_run(
    epochs_done: int, total: int, seconds_per_epoch: float
) -> tuple[list[ProgressRecord], ArrivalClock, float]:
    """A healthy run that has reported `epochs_done` epochs; returns (records, clock, now)."""
    clock = ArrivalClock(0.0)
    records = []
    now = 0.0
    for epoch in range(1, epochs_done + 1):
        now += seconds_per_epoch
        clock.note(now)
        records.append(rec(epoch, total, loss=1.0 / epoch, eval_metric=40.0 / epoch, chance=90.0))
    return records, clock, now


def test_svm_real_cadence_never_hangs() -> None:
    assert len(SVM_FULL_GAPS) == 192
    clock = ArrivalClock(0.0)
    records: list[ProgressRecord] = []
    now = 0.0
    for i, gap in enumerate(SVM_FULL_GAPS):
        # Asked just before each record lands: the longest silence that record allowed.
        assert check_hang(records, "full", clock, now + gap) is None
        now += gap
        clock.note(now)
        records.append(
            rec(i + 1, 191, kind="fold", eval_metric=96.0, higher_is_better=True, chance=50.0)
        )
    assert clock.arrivals == 192
    assert clock.longest_gap == pytest.approx(5.33)


def test_svm_real_cadence_is_why_the_limit_uses_the_longest_gap_and_a_floor() -> None:
    """The slowest real fold took 35x the median one, so a limit built on the typical
    gap would have killed this run. Built on the longest gap alone it would have been
    a few seconds wide; at that scale the floor is what governs."""
    ordered = sorted(SVM_FULL_GAPS)
    median = ordered[len(ordered) // 2]
    assert max(SVM_FULL_GAPS) > HANG_GAP_MULTIPLIER * median
    assert HANG_GAP_MULTIPLIER * max(SVM_FULL_GAPS) < HANG_FLOOR_SECONDS
    assert hang_limit(max(SVM_FULL_GAPS)) == HANG_FLOOR_SECONDS


def test_wrn_gpu_run_hangs_only_past_the_floor() -> None:
    records, clock, now = epoch_run(50, 200, WRN_GPU_EPOCH_SECONDS)
    assert HANG_GAP_MULTIPLIER * WRN_GPU_EPOCH_SECONDS < HANG_FLOOR_SECONDS
    assert check_hang(records, "full", clock, now + HANG_FLOOR_SECONDS) is None
    halt = check_hang(records, "full", clock, now + HANG_FLOOR_SECONDS + 1)
    assert halt is not None
    assert (halt.rule, halt.step, halt.kind) == ("hang", 50, "epoch")
    assert "1801 s" in halt.message and "hang" in halt.feedback()


def test_wrn_cpu_slow_epochs_raise_the_limit_above_the_floor() -> None:
    records, clock, now = epoch_run(3, 200, WRN_CPU_EPOCH_SECONDS)
    limit = hang_limit(clock.longest_gap)
    assert limit == pytest.approx(HANG_GAP_MULTIPLIER * WRN_CPU_EPOCH_SECONDS)
    assert limit > 86400  # past the full stage's own timeout: such a run is never hang-halted
    assert check_hang(records, "full", clock, now + 3 * WRN_CPU_EPOCH_SECONDS) is None


def test_slow_start_counts_as_a_gap() -> None:
    # The measured cold CIFAR-10 download: 1707 s before the script did anything.
    clock = ArrivalClock(0.0)
    clock.note(1707.0 + WRN_GPU_EPOCH_SECONDS)
    clock.note(1707.0 + 2 * WRN_GPU_EPOCH_SECONDS)
    assert hang_limit(clock.longest_gap) == pytest.approx(10 * (1707.0 + WRN_GPU_EPOCH_SECONDS))


@pytest.mark.parametrize("mode", ["probe", "smoke", "capped"])
def test_hang_is_left_to_the_timeout_in_cheap_stages(mode: str) -> None:
    records, clock, now = epoch_run(50, 200, WRN_GPU_EPOCH_SECONDS)
    assert check_hang(records, mode, clock, now + 10 * HANG_FLOOR_SECONDS) is None


@pytest.mark.parametrize("mode", ["seed2", "seed3"])
def test_hang_applies_to_extra_seed_runs(mode: str) -> None:
    records, clock, now = epoch_run(50, 200, WRN_GPU_EPOCH_SECONDS)
    assert check_hang(records, mode, clock, now + HANG_FLOOR_SECONDS + 1) is not None


def test_hang_needs_two_arrivals() -> None:
    records, clock, now = epoch_run(1, 200, WRN_GPU_EPOCH_SECONDS)
    assert check_hang(records, "full", clock, now + 10 * HANG_FLOOR_SECONDS) is None
    assert check_hang([], "full", ArrivalClock(0.0), 10 * HANG_FLOOR_SECONDS) is None


def test_silence_after_the_final_step_is_left_to_the_timeout() -> None:
    records, clock, now = epoch_run(200, 200, WRN_GPU_EPOCH_SECONDS)
    assert check_hang(records, "full", clock, now + 10 * HANG_FLOOR_SECONDS) is None


def test_hang_is_never_judged_after_exit() -> None:
    """`evaluate` is what runs after the process exits, and it has no clock."""
    records, _, _ = epoch_run(50, 200, WRN_GPU_EPOCH_SECONDS)
    assert evaluate(records, "full") is None


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #


def test_non_finite_loss_halts_in_any_stage() -> None:
    for mode in ("probe", "smoke", "capped", "full"):
        assert first_halt([rec(1, 10, loss=float("nan"))], mode) == ("non_finite", 1)


def test_cheap_stages_ignore_trend_rules() -> None:
    records = ablation_records("E_pca_whitened")
    assert first_halt(records, "probe") is None
    assert first_halt(records, "smoke") is None


def test_claim_near_chance_disables_stuck_rule() -> None:
    # A claim that itself barely beats chance must not be judged "stuck".
    records = [
        rec(
            e,
            20,
            loss=1.0 / e,
            eval_metric=52.0,
            metric="accuracy",
            higher_is_better=True,
            chance=50.0,
            n=200,
            target=53.0,
        )
        for e in range(1, 21)
    ]
    assert first_halt(records) is None


def test_parse_record_accepts_marker_nan_and_rejects_noise() -> None:
    line = (
        f'{PROGRESS_MARKER} {{"step": 3, "steps_total": 40, "kind": "epoch", "train_loss": NaN, '
        f'"eval_metric": 93.5, "metric": "test accuracy", "unit": "%", '
        f'"higher_is_better": true, "chance_metric": 11.35}}'
    )
    record = parse_record(line)
    assert record is not None
    assert record.step == 3 and record.higher_is_better is True
    assert record.train_loss is not None and record.train_loss != record.train_loss  # NaN
    assert parse_record("Epoch 3/40 - train_loss=0.5") is None
    assert parse_record('{"no_step": 1}') is None


def test_history_tail_reads_only_complete_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.full.history.jsonl"
    tail = HistoryTail(path)
    assert tail.read_new() == []
    path.write_text('{"step": 1, "train_loss": 1.0}\n{"step": 2, "tra', encoding="utf-8")
    assert [r.step for r in tail.read_new()] == [1]
    with path.open("a", encoding="utf-8") as handle:
        handle.write('in_loss": 0.9}\nnot json\n')
    assert [r.step for r in tail.read_new()] == [2]
    assert tail.malformed_lines == 1


def test_history_path_from_metrics() -> None:
    assert history_path_from_metrics(Path("/w/metrics.full.json")).name == (
        "metrics.full.history.jsonl"
    )
