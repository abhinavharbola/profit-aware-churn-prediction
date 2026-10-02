import pandas as pd
from src.data.temporal import generate_windows


def _make_transactions():
    rows = []
    start = pd.Timestamp("2010-01-01")
    for day_offset in range(0, 500, 5):
        date = start + pd.Timedelta(days=day_offset)
        customer = "A" if day_offset % 10 == 0 else "B"
        rows.append({"customer_id": customer, "invoicedate": date, "is_return": False})
    return pd.DataFrame(rows)


def test_windows_generated_for_sufficient_span():
    df = _make_transactions()
    windows = generate_windows(df, observation_days=200, prediction_days=60, slide_days=30)
    assert len(windows) > 0


def test_observation_data_stays_inside_observation_window():
    df = _make_transactions()
    windows = generate_windows(df, observation_days=200, prediction_days=60, slide_days=30)

    for window in windows:
        obs_data = window["obs_data"]
        assert (obs_data["invoicedate"] < window["obs_end"]).all()
        assert (obs_data["invoicedate"] >= window["obs_start"]).all()
        assert window["obs_end"] < window["pred_end"]


def test_churn_label_zero_when_customer_purchases_in_prediction_window():
    df = _make_transactions()
    windows = generate_windows(df, observation_days=200, prediction_days=60, slide_days=30)

    window = windows[0]
    for cust_id, label in window["churn_labels"].items():
        pred_mask = (df["invoicedate"] >= window["obs_end"]) & (df["invoicedate"] < window["pred_end"])
        purchased_in_pred = cust_id in set(df[pred_mask]["customer_id"])
        expected_label = 0 if purchased_in_pred else 1
        assert label == expected_label


def test_no_windows_when_span_too_short():
    df = _make_transactions()
    windows = generate_windows(df, observation_days=1000, prediction_days=200, slide_days=30)
    assert windows == []


def test_return_in_prediction_window_does_not_count_as_a_purchase():
    rows = [
        {"customer_id": 1, "invoicedate": pd.Timestamp("2010-01-10"), "is_return": False},
        {"customer_id": 1, "invoicedate": pd.Timestamp("2010-04-05"), "is_return": True},
        {"customer_id": 2, "invoicedate": pd.Timestamp("2010-01-10"), "is_return": False},
        {"customer_id": 2, "invoicedate": pd.Timestamp("2010-04-05"), "is_return": False},
        {"customer_id": 3, "invoicedate": pd.Timestamp("2010-06-30"), "is_return": False},
    ]
    df = pd.DataFrame(rows)
    windows = generate_windows(df, observation_days=60, prediction_days=60, slide_days=30)

    labels = windows[0]["churn_labels"]
    assert labels[1] == 1
    assert labels[2] == 0


def test_customer_with_only_returns_in_observation_window_is_not_scored():
    rows = [
        {"customer_id": 1, "invoicedate": pd.Timestamp("2010-01-10"), "is_return": True},
        {"customer_id": 2, "invoicedate": pd.Timestamp("2010-01-10"), "is_return": False},
        {"customer_id": 3, "invoicedate": pd.Timestamp("2010-06-30"), "is_return": False},
    ]
    windows = generate_windows(pd.DataFrame(rows), observation_days=60, prediction_days=60, slide_days=30)
    assert set(windows[0]["churn_labels"]) == {2}
