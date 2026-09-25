import pandas as pd
from src.data.temporal import generate_windows


def _make_transactions():
    rows = []
    start = pd.Timestamp("2010-01-01")
    for day_offset in range(0, 500, 5):
        date = start + pd.Timedelta(days=day_offset)
        customer = "A" if day_offset % 10 == 0 else "B"
        rows.append({"customer_id": customer, "invoicedate": date})
    return pd.DataFrame(rows)


def test_windows_generated_for_sufficient_span():
    df = _make_transactions()
    windows = generate_windows(df, observation_days=200, prediction_days=60, slide_days=30)
    assert len(windows) > 0


def test_no_window_transaction_crosses_prediction_boundary():
    df = _make_transactions()
    windows = generate_windows(df, observation_days=200, prediction_days=60, slide_days=30)

    for window in windows:
        obs_data = window["obs_data"]
        obs_end = window["obs_end"]
        pred_end = window["pred_end"]

        assert (obs_data["invoicedate"] < obs_end).all()
        assert (obs_data["invoicedate"] >= window["obs_start"]).all()
        assert obs_end <= pred_end


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
