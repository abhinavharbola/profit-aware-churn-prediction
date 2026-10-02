import numpy as np
import pandas as pd
from config import OBSERVATION_WINDOW_DAYS

TREND_BUCKET_DAYS = 30
TREND_BUCKETS = OBSERVATION_WINDOW_DAYS // TREND_BUCKET_DAYS

FEATURE_ORDER = [
    "customer_id", "recency", "frequency", "monetary_total", "monetary_avg",
    "unique_products", "spend_30d", "spend_90d", "interpurchase_mean",
    "interpurchase_std", "spend_trend", "product_diversity", "seasonal_dropoff",
]


def compute_rfm_features(obs_df, reference_date):
    df = obs_df
    purchases = df[~df["is_return"]]

    rfm = purchases.groupby("customer_id").agg(
        last_purchase=("invoicedate", "max"),
        frequency=("invoice", "nunique"),
        monetary_avg=("revenue", "mean"),
        unique_products=("stockcode", "nunique"),
    )
    rfm["recency"] = (reference_date - rfm.pop("last_purchase")).dt.days

    net_revenue = df.groupby("customer_id")["revenue"].sum()
    rfm["monetary_total"] = net_revenue.reindex(rfm.index).fillna(0.0)

    for days, name in ((30, "spend_30d"), (90, "spend_90d")):
        recent = df[df["invoicedate"] >= reference_date - pd.Timedelta(days=days)]
        rfm[name] = recent.groupby("customer_id")["revenue"].sum().reindex(rfm.index).fillna(0.0)

    invoices = purchases.groupby(["customer_id", "invoice"], as_index=False)["invoicedate"].min()
    invoices = invoices.sort_values(["customer_id", "invoicedate"])
    invoices["gap"] = invoices.groupby("customer_id")["invoicedate"].diff().dt.days
    gaps = invoices.groupby("customer_id")["gap"].agg(["mean", "std"])
    rfm["interpurchase_mean"] = gaps["mean"].reindex(rfm.index).fillna(OBSERVATION_WINDOW_DAYS)
    rfm["interpurchase_std"] = gaps["std"].reindex(rfm.index).fillna(0.0)

    days_ago = (reference_date - df["invoicedate"]).dt.days
    bucket = TREND_BUCKETS - 1 - (days_ago // TREND_BUCKET_DAYS)
    in_range = (bucket >= 0) & (bucket < TREND_BUCKETS)
    trend_df = pd.DataFrame({
        "customer_id": df["customer_id"],
        "bucket": bucket,
        "revenue": df["revenue"],
    })[in_range]
    spend_matrix = (
        trend_df.groupby(["customer_id", "bucket"])["revenue"].sum()
        .unstack("bucket")
        .reindex(index=rfm.index, columns=range(TREND_BUCKETS))
        .fillna(0.0)
    )
    x_centered = np.arange(TREND_BUCKETS, dtype=float)
    x_centered = x_centered - x_centered.mean()
    rfm["spend_trend"] = spend_matrix.to_numpy(dtype=float) @ x_centered / (x_centered ** 2).sum()

    rfm["product_diversity"] = rfm["unique_products"] / rfm["frequency"]

    recent_start = reference_date - pd.Timedelta(days=90)
    prior_start = reference_date - pd.Timedelta(days=180)
    purchase_dates = purchases["invoicedate"]
    recent_custs = purchases.loc[
        (purchase_dates >= recent_start) & (purchase_dates < reference_date), "customer_id"
    ].unique()
    prior_custs = purchases.loc[
        (purchase_dates >= prior_start) & (purchase_dates < recent_start), "customer_id"
    ].unique()
    rfm["seasonal_dropoff"] = (
        rfm.index.isin(prior_custs) & ~rfm.index.isin(recent_custs)
    ).astype(int)

    return rfm.reset_index()[FEATURE_ORDER]


def build_feature_matrix(windows):
    all_features = []

    for i, window in enumerate(windows):
        reference_date = window["obs_end"]

        features = compute_rfm_features(window["obs_data"], reference_date)
        features["churn"] = features["customer_id"].map(window["churn_labels"])
        features["window_id"] = i
        features["obs_end"] = reference_date

        all_features.append(features)

    full_df = pd.concat(all_features, ignore_index=True)
    full_df["churn"] = full_df["churn"].astype(int)

    return full_df


def build_current_snapshot(df, observation_days=OBSERVATION_WINDOW_DAYS):
    reference_date = df["invoicedate"].max().normalize() + pd.Timedelta(days=1)
    obs_start = reference_date - pd.Timedelta(days=observation_days)
    obs_df = df[(df["invoicedate"] >= obs_start) & (df["invoicedate"] < reference_date)]

    snapshot = compute_rfm_features(obs_df, reference_date)
    snapshot["obs_end"] = reference_date
    return snapshot
