import pandas as pd
from src.data.cleaner import clean_data


def _base_rows():
    return {
        "Invoice": ["536365", "536365", "536370", "C536365"],
        "StockCode": ["A", "B", "A", "A"],
        "Customer ID": [1, 1, 2, 1],
        "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-01-01", "2010-01-02", "2010-01-03"]),
        "Quantity": [10, 5, 8, 3],
        "Price": [2.0, 3.0, 1.5, 2.0],
    }


def test_cancellation_only_reduces_the_matched_stockcode():
    df = pd.DataFrame(_base_rows())
    result = clean_data(df)

    row_a = result[(result["invoice"] == "536365") & (result["stockcode"] == "A")].iloc[0]
    row_b = result[(result["invoice"] == "536365") & (result["stockcode"] == "B")].iloc[0]

    assert row_a["quantity"] == 7
    assert row_b["quantity"] == 5


def test_cancellation_does_not_affect_unrelated_invoice():
    df = pd.DataFrame(_base_rows())
    result = clean_data(df)

    row_other = result[(result["invoice"] == "536370")].iloc[0]
    assert row_other["quantity"] == 8


def test_fully_cancelled_line_is_dropped():
    df = pd.DataFrame({
        "Invoice": ["536365", "536365", "C536365"],
        "StockCode": ["A", "B", "A"],
        "Customer ID": [1, 1, 1],
        "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-01-01", "2010-01-03"]),
        "Quantity": [10, 5, 10],
        "Price": [2.0, 3.0, 2.0],
    })
    result = clean_data(df)

    assert len(result) == 1
    assert result.iloc[0]["stockcode"] == "B"


def test_no_crash_when_cancellation_matches_exist():
    df = pd.DataFrame(_base_rows())
    result = clean_data(df)
    assert len(result) > 0


def test_duplicate_line_items_are_netted_not_double_cancelled():
    df = pd.DataFrame({
        "Invoice": ["536365", "536365", "C536365"],
        "StockCode": ["A", "A", "A"],
        "Customer ID": [1, 1, 1],
        "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-01-01", "2010-01-03"]),
        "Quantity": [5, 5, 3],
        "Price": [2.0, 2.0, 2.0],
    })
    result = clean_data(df)
    assert result["quantity"].sum() == 7


def test_two_line_cancellation_against_same_product_sums_correctly():
    df = pd.DataFrame({
        "Invoice": ["536365", "C536365", "C536365"],
        "StockCode": ["A", "A", "A"],
        "Customer ID": [1, 1, 1],
        "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-01-03", "2010-01-03"]),
        "Quantity": [10, 3, 2],
        "Price": [2.0, 2.0, 2.0],
    })
    result = clean_data(df)
    assert result["quantity"].sum() == 5


def test_revenue_computed_after_netting():
    df = pd.DataFrame(_base_rows())
    result = clean_data(df)

    row_a = result[(result["invoice"] == "536365") & (result["stockcode"] == "A")].iloc[0]
    assert row_a["revenue"] == 7 * 2.0
