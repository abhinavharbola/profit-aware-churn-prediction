import pandas as pd
from src.features.rfm_engineer import compute_rfm_features


def _make_obs_df():
    return pd.DataFrame([
        {"customer_id": 1, "invoice": "I1", "invoicedate": pd.Timestamp("2010-01-01"), "revenue": 100.0, "stockcode": "P1"},
        {"customer_id": 1, "invoice": "I2", "invoicedate": pd.Timestamp("2010-02-01"), "revenue": 50.0, "stockcode": "P2"},
        {"customer_id": 1, "invoice": "I2", "invoicedate": pd.Timestamp("2010-02-01"), "revenue": 25.0, "stockcode": "P3"},
        {"customer_id": 2, "invoice": "I3", "invoicedate": pd.Timestamp("2010-01-15"), "revenue": 200.0, "stockcode": "P1"},
    ])


def test_recency_is_days_since_last_purchase():
    df = _make_obs_df()
    reference_date = pd.Timestamp("2010-03-01")
    rfm = compute_rfm_features(df, reference_date)

    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    assert cust1["recency"] == (reference_date - pd.Timestamp("2010-02-01")).days


def test_frequency_counts_unique_invoices():
    df = _make_obs_df()
    reference_date = pd.Timestamp("2010-03-01")
    rfm = compute_rfm_features(df, reference_date)

    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    assert cust1["frequency"] == 2

    cust2 = rfm[rfm["customer_id"] == 2].iloc[0]
    assert cust2["frequency"] == 1


def test_monetary_total_and_avg_computed_per_row_not_per_invoice():
    df = _make_obs_df()
    reference_date = pd.Timestamp("2010-03-01")
    rfm = compute_rfm_features(df, reference_date)

    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    assert cust1["monetary_total"] == 175.0
    assert cust1["monetary_avg"] == 175.0 / 3


def test_unique_products_counts_distinct_stockcodes():
    df = _make_obs_df()
    reference_date = pd.Timestamp("2010-03-01")
    rfm = compute_rfm_features(df, reference_date)

    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    assert cust1["unique_products"] == 3


def test_seasonal_dropoff_flags_customer_inactive_in_last_90_days():
    df = pd.DataFrame([
        {"customer_id": 1, "invoice": "I1", "invoicedate": pd.Timestamp("2010-01-01"), "revenue": 50.0, "stockcode": "P1"},
    ])
    reference_date = pd.Timestamp("2010-05-01")
    rfm = compute_rfm_features(df, reference_date)

    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    assert cust1["seasonal_dropoff"] == 1


def test_seasonal_dropoff_zero_when_customer_active_recently():
    df = pd.DataFrame([
        {"customer_id": 1, "invoice": "I1", "invoicedate": pd.Timestamp("2010-04-20"), "revenue": 50.0, "stockcode": "P1"},
    ])
    reference_date = pd.Timestamp("2010-05-01")
    rfm = compute_rfm_features(df, reference_date)

    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    assert cust1["seasonal_dropoff"] == 0


def test_seasonal_dropoff_computable_regardless_of_calendar_month():
    for month in range(1, 13):
        reference_date = pd.Timestamp(year=2010, month=month, day=15)
        df = pd.DataFrame([{
            "customer_id": 1,
            "invoice": "I1",
            "invoicedate": reference_date - pd.Timedelta(days=120),
            "revenue": 50.0,
            "stockcode": "P1",
        }])
        rfm = compute_rfm_features(df, reference_date)
        cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
        assert cust1["seasonal_dropoff"] == 1


def test_no_customer_leaks_across_window_rows():
    df = _make_obs_df()
    reference_date = pd.Timestamp("2010-03-01")
    rfm = compute_rfm_features(df, reference_date)

    assert set(rfm["customer_id"]) == {1, 2}
    assert len(rfm) == rfm["customer_id"].nunique()
