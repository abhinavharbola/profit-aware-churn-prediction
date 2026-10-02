from datetime import timedelta
from config import OBSERVATION_WINDOW_DAYS, PREDICTION_WINDOW_DAYS, SLIDE_INTERVAL_DAYS


def generate_windows(df, observation_days=OBSERVATION_WINDOW_DAYS,
                     prediction_days=PREDICTION_WINDOW_DAYS,
                     slide_days=SLIDE_INTERVAL_DAYS):
    date_min = df["invoicedate"].min()
    date_max = df["invoicedate"].max()
    purchases = df[~df["is_return"]]

    window_start = date_min
    windows = []

    while window_start + timedelta(days=observation_days + prediction_days) <= date_max:
        obs_start = window_start
        obs_end = obs_start + timedelta(days=observation_days)
        pred_end = obs_end + timedelta(days=prediction_days)

        obs_df = df[(df["invoicedate"] >= obs_start) & (df["invoicedate"] < obs_end)]
        pred_df = purchases[(purchases["invoicedate"] >= obs_end) & (purchases["invoicedate"] < pred_end)]

        obs_customers = set(obs_df.loc[~obs_df["is_return"], "customer_id"].unique())
        pred_customers = set(pred_df["customer_id"].unique())

        churn_labels = {cust_id: (0 if cust_id in pred_customers else 1) for cust_id in obs_customers}

        windows.append({
            "obs_start": obs_start,
            "obs_end": obs_end,
            "pred_end": pred_end,
            "obs_data": obs_df,
            "churn_labels": churn_labels
        })

        window_start += timedelta(days=slide_days)

    return windows
