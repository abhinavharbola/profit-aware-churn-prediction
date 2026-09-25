import pandas as pd
from config import RAW_DATA_PATH


def load_raw_data():
    df_0910 = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2009-2010", engine="openpyxl")
    df_1011 = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2010-2011", engine="openpyxl")
    df = pd.concat([df_0910, df_1011], ignore_index=True)
    return df


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)

    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    df["is_cancellation"] = df["invoice"].astype(str).str.startswith("C")
    cancellations = df[df["is_cancellation"]].copy()
    transactions = df[~df["is_cancellation"]].copy()

    transactions = transactions[transactions["quantity"] > 0]

    cancellations["matched_invoice"] = cancellations["invoice"].astype(str).str.replace("^C", "", regex=True)
    cancellations["quantity"] = cancellations["quantity"].abs()

    cancellation_matches = cancellations.merge(
        transactions[["invoice", "customer_id", "stockcode"]].drop_duplicates(),
        left_on=["matched_invoice", "customer_id", "stockcode"],
        right_on=["invoice", "customer_id", "stockcode"],
        how="inner",
        suffixes=("_cancel", "_trans")
    )

    if not cancellation_matches.empty:
        matched_quantities = cancellation_matches.groupby(
            ["invoice_trans", "stockcode"]
        )["quantity"].sum()

        transactions = transactions.set_index(["invoice", "stockcode"])
        affected_mask = transactions.index.isin(matched_quantities.index)
        affected = transactions[affected_mask].copy()
        unaffected = transactions[~affected_mask]

        affected["_cancel_remaining"] = 0.0
        affected.loc[matched_quantities.index, "_cancel_remaining"] = matched_quantities.astype(float)

        def _net_group(group):
            remaining = group["_cancel_remaining"].iloc[0]
            if remaining <= 0:
                return group
            qty = group["quantity"].to_numpy(dtype=float)
            for i in range(len(qty)):
                take = min(remaining, qty[i])
                qty[i] -= take
                remaining -= take
                if remaining <= 0:
                    break
            group = group.copy()
            group["quantity"] = qty
            return group

        affected = affected.groupby(
            level=["invoice", "stockcode"], group_keys=False
        ).apply(_net_group)
        affected = affected.drop(columns="_cancel_remaining")

        transactions = pd.concat([unaffected, affected])
        transactions = transactions[transactions["quantity"] > 0].reset_index()

    df_clean = transactions.copy()

    df_clean["revenue"] = df_clean["quantity"] * df_clean["price"]

    df_clean = df_clean[df_clean["price"] > 0]

    df_clean = df_clean.sort_values(["customer_id", "invoicedate"]).reset_index(drop=True)

    return df_clean


def run_cleaning():
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    return df_clean
