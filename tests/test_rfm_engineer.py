import pandas as pd
import pytest
from src.features.rfm_engineer import compute_rfm_features, build_current_snapshot, TREND_BUCKETS


def _row(customer_id, invoice, date, revenue, stockcode, is_return=False):
    return {
        "customer_id": customer_id, "invoice": invoice, "invoicedate": pd.Timestamp(date),
        "revenue": revenue, "stockcode": stockcode, "is_return": is_return,
    }


def _make_obs_df():
    return pd.DataFrame([
        _row(1, "I1", "2010-01-01", 100.0, "P1"),
        _row(1, "I2", "2010-02-01", 50.0, "P2"),
        _row(1, "I2", "2010-02-01", 25.0, "P3"),
        _row(2, "I3", "2010-01-15", 200.0, "P1"),
    ])


REFERENCE = pd.Timestamp("2010-03-01")


def _cust(rfm, customer_id):
    return rfm[rfm["customer_id"] == customer_id].iloc[0]


def test_recency_is_days_since_last_purchase():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 1)["recency"] == (REFERENCE - pd.Timestamp("2010-02-01")).days


def test_frequency_counts_unique_invoices():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 1)["frequency"] == 2
    assert _cust(rfm, 2)["frequency"] == 1


def test_monetary_total_and_avg_computed_per_row_not_per_invoice():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 1)["monetary_total"] == 175.0
    assert _cust(rfm, 1)["monetary_avg"] == pytest.approx(175.0 / 3)


def test_unique_products_counts_distinct_stockcodes():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 1)["unique_products"] == 3


def test_returns_reduce_monetary_total_but_not_frequency_or_recency():
    df = pd.concat([
        _make_obs_df(),
        pd.DataFrame([_row(1, "C9", "2010-02-20", -40.0, "P1", is_return=True)]),
    ], ignore_index=True)
    rfm = compute_rfm_features(df, REFERENCE)
    cust1 = _cust(rfm, 1)
    assert cust1["monetary_total"] == 135.0
    assert cust1["frequency"] == 2
    assert cust1["recency"] == (REFERENCE - pd.Timestamp("2010-02-01")).days
    assert cust1["monetary_avg"] == pytest.approx(175.0 / 3)


def test_customer_with_only_returns_is_not_featurised():
    df = pd.concat([
        _make_obs_df(),
        pd.DataFrame([_row(7, "C1", "2010-02-20", -40.0, "P1", is_return=True)]),
    ], ignore_index=True)
    rfm = compute_rfm_features(df, REFERENCE)
    assert set(rfm["customer_id"]) == {1, 2}


def test_interpurchase_uses_invoice_dates_not_line_items():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 1)["interpurchase_mean"] == 31.0
    assert _cust(rfm, 1)["interpurchase_std"] == 0.0


def test_interpurchase_fallback_for_single_invoice_customer():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 2)["interpurchase_mean"] == 365
    assert _cust(rfm, 2)["interpurchase_std"] == 0.0


def test_spend_trend_zero_fills_empty_buckets():
    reference = pd.Timestamp("2010-12-31")
    df = pd.DataFrame([
        _row(1, "I1", reference - pd.Timedelta(days=5), 100.0, "P1"),
        _row(2, "I2", reference - pd.Timedelta(days=5 + 30 * (TREND_BUCKETS - 1)), 100.0, "P1"),
    ])
    rfm = compute_rfm_features(df, reference)
    assert _cust(rfm, 1)["spend_trend"] > 0
    assert _cust(rfm, 2)["spend_trend"] < 0


def test_spend_trend_is_zero_for_flat_spend():
    reference = pd.Timestamp("2010-12-31")
    rows = [
        _row(1, f"I{i}", reference - pd.Timedelta(days=5 + 30 * i), 50.0, "P1")
        for i in range(TREND_BUCKETS)
    ]
    rfm = compute_rfm_features(pd.DataFrame(rows), reference)
    assert _cust(rfm, 1)["spend_trend"] == pytest.approx(0.0, abs=1e-9)


def test_product_diversity_is_unique_products_per_invoice():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert _cust(rfm, 1)["product_diversity"] == pytest.approx(3 / 2)


def test_seasonal_dropoff_flags_customer_inactive_in_last_90_days():
    df = pd.DataFrame([_row(1, "I1", "2010-01-01", 50.0, "P1")])
    rfm = compute_rfm_features(df, pd.Timestamp("2010-05-01"))
    assert _cust(rfm, 1)["seasonal_dropoff"] == 1


def test_seasonal_dropoff_zero_when_customer_active_recently():
    df = pd.DataFrame([_row(1, "I1", "2010-04-20", 50.0, "P1")])
    rfm = compute_rfm_features(df, pd.Timestamp("2010-05-01"))
    assert _cust(rfm, 1)["seasonal_dropoff"] == 0


def test_seasonal_dropoff_computable_regardless_of_calendar_month():
    for month in range(1, 13):
        reference_date = pd.Timestamp(year=2010, month=month, day=15)
        df = pd.DataFrame([_row(1, "I1", reference_date - pd.Timedelta(days=120), 50.0, "P1")])
        rfm = compute_rfm_features(df, reference_date)
        assert _cust(rfm, 1)["seasonal_dropoff"] == 1


def test_no_customer_leaks_across_window_rows():
    rfm = compute_rfm_features(_make_obs_df(), REFERENCE)
    assert set(rfm["customer_id"]) == {1, 2}
    assert len(rfm) == rfm["customer_id"].nunique()


def test_current_snapshot_references_the_day_after_the_last_transaction():
    df = pd.DataFrame([
        _row(1, "I1", "2010-06-01 09:00", 10.0, "P1"),
        _row(2, "I2", "2011-06-01 14:00", 20.0, "P1"),
    ])
    snapshot = build_current_snapshot(df)
    assert (snapshot["obs_end"] == pd.Timestamp("2011-06-02")).all()
    assert set(snapshot["customer_id"]) == {2}
