import numpy as np
import pandas as pd
from config import RAW_DATA_PATH


def combine_sheets(df_first, df_second):
    first_invoices = set(df_first["Invoice"].astype(str))
    keep = ~df_second["Invoice"].astype(str).isin(first_invoices)
    return pd.concat([df_first, df_second[keep]], ignore_index=True)


def load_raw_data():
    df_0910 = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2009-2010", engine="openpyxl")
    df_1011 = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2010-2011", engine="openpyxl")
    return combine_sheets(df_0910, df_1011)


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)

    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    df["invoice"] = df["invoice"].astype(str)
    df["stockcode"] = df["stockcode"].astype(str)

    df["is_return"] = df["invoice"].str.startswith("C")
    df = df[df["price"] > 0].copy()

    df = df[df["is_return"] | (df["quantity"] > 0)].copy()
    df["quantity"] = np.where(df["is_return"], -df["quantity"].abs(), df["quantity"])
    df = df[df["quantity"] != 0].copy()

    df["revenue"] = df["quantity"] * df["price"]

    return df.sort_values(["customer_id", "invoicedate"]).reset_index(drop=True)


def run_cleaning():
    return clean_data(load_raw_data())
