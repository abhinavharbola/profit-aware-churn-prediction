import numpy as np
import pandas as pd
import pytest
from src.evaluation.profit_optimizer import (
    find_optimal_threshold,
    evaluate_random_baseline,
    evaluate_default_baseline,
    compute_expected_profit,
    compute_avg_monthly_spend,
    _evaluate,
)


def _synthetic_case():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.95, 0.85, 0.55, 0.90, 0.60, 0.40, 0.30, 0.20, 0.10, 0.05])
    avg_monthly_spend = pd.Series([100.0] * 10)
    return y_true, y_prob, avg_monthly_spend


def test_threshold_sweep_covers_full_configured_range():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()
    _, results_df = find_optimal_threshold(y_true, y_prob, avg_monthly_spend)

    assert results_df["threshold"].min() == 0.01
    assert results_df["threshold"].max() == 0.90


def test_argmax_selection_matches_manual_recomputation():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()
    optimal_threshold, results_df = find_optimal_threshold(y_true, y_prob, avg_monthly_spend)

    max_row = results_df.loc[results_df["net_profit"].idxmax()]
    assert optimal_threshold == max_row["threshold"]
    assert results_df["net_profit"].max() == results_df["net_profit"].max()

    for _, row in results_df.iterrows():
        assert row["net_profit"] <= results_df["net_profit"].max()


def test_higher_threshold_reduces_or_keeps_interventions():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()
    _, results_df = find_optimal_threshold(y_true, y_prob, avg_monthly_spend)

    sorted_df = results_df.sort_values("threshold")
    interventions = sorted_df["total_interventions"].values
    assert all(interventions[i] >= interventions[i + 1] for i in range(len(interventions) - 1))


def test_false_positive_only_case_yields_negative_profit():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.95, 0.90, 0.85])
    avg_monthly_spend = pd.Series([50.0, 50.0, 50.0])

    result = evaluate_default_baseline(y_true, y_prob, avg_monthly_spend, threshold=0.5)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 3
    assert result["net_profit"] == -3 * 10.0


def test_random_baseline_intervention_count_matches_fraction():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()
    result = evaluate_random_baseline(y_true, y_prob, avg_monthly_spend, fraction=0.2)
    assert result["total_interventions"] == 2


def test_random_baseline_is_reproducible_and_does_not_leak_global_state():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()

    np.random.seed(123)
    state_before = np.random.get_state()[1].copy()

    result_a = evaluate_random_baseline(y_true, y_prob, avg_monthly_spend, fraction=0.3)
    state_after_a = np.random.get_state()[1]

    assert np.array_equal(state_before, state_after_a)

    result_b = evaluate_random_baseline(y_true, y_prob, avg_monthly_spend, fraction=0.3)
    assert result_a == result_b


def test_random_baseline_follows_configured_seed_not_a_hardcoded_one():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()
    n = len(y_true)
    fraction = 0.3
    n_target = int(n * fraction)

    for seed in (42, 7, 2024):
        expected_indices = np.random.RandomState(seed).choice(n, size=n_target, replace=False)
        expected_predictions = np.zeros(n)
        expected_predictions[expected_indices] = 1
        expected = _evaluate(y_true, expected_predictions, avg_monthly_spend)

        actual = evaluate_random_baseline(y_true, y_prob, avg_monthly_spend, fraction=fraction, random_seed=seed)
        assert actual == expected

    default_result = evaluate_random_baseline(y_true, y_prob, avg_monthly_spend, fraction=fraction)
    explicit_42_result = evaluate_random_baseline(
        y_true, y_prob, avg_monthly_spend, fraction=fraction, random_seed=42
    )
    assert default_result == explicit_42_result


def test_compute_expected_profit_formula():
    profit = compute_expected_profit(prob=0.5, avg_monthly_spend=100.0)
    expected = 0.5 * 0.15 * (100.0 * 3) - 10.0
    assert profit == expected


def test_avg_monthly_spend_scales_down_from_annual_total():
    monetary_total = 1200.0
    result = compute_avg_monthly_spend(monetary_total, observation_days=365)
    assert result == pytest.approx(monetary_total / (365 / 30.44))
    assert result < monetary_total


def test_revenue_saved_no_longer_collapses_to_full_total():
    monetary_total = 1200.0
    avg_spend = compute_avg_monthly_spend(monetary_total, observation_days=365)
    revenue_saved_3mo = avg_spend * 3
    assert revenue_saved_3mo < monetary_total
    assert revenue_saved_3mo == pytest.approx(monetary_total * 3 / (365 / 30.44))


def test_avg_monthly_spend_handles_series_input():
    totals = pd.Series([1200.0, 600.0, 0.0])
    result = compute_avg_monthly_spend(totals, observation_days=365)
    assert list(result.round(4)) == list((totals / (365 / 30.44)).round(4))


def test_round_decimals_param_preserves_small_thresholds():
    y_true, y_prob, avg_monthly_spend = _synthetic_case()
    fine_thresholds = np.array([0.0001, 0.0005, 0.001, 0.002, 0.005])

    _, coarse_results = find_optimal_threshold(
        y_true, y_prob, avg_monthly_spend, thresholds=fine_thresholds
    )
    _, fine_results = find_optimal_threshold(
        y_true, y_prob, avg_monthly_spend, thresholds=fine_thresholds, round_decimals=4
    )

    assert coarse_results["threshold"].nunique() == 1
    assert fine_results["threshold"].nunique() == len(fine_thresholds)
