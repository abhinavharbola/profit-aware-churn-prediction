import pandas as pd
from src.data.cleaner import clean_data, combine_sheets


def _raw(invoices, stockcodes, customers, dates, quantities, prices):
    return pd.DataFrame({
        "Invoice": invoices,
        "StockCode": stockcodes,
        "Customer ID": customers,
        "InvoiceDate": pd.to_datetime(dates),
        "Quantity": quantities,
        "Price": prices,
    })


def _base_rows():
    return _raw(
        ["536365", "536365", "536370", "C536999"],
        ["A", "B", "A", "A"],
        [1, 1, 2, 1],
        ["2010-01-01", "2010-01-01", "2010-01-02", "2010-01-03"],
        [10, 5, 8, -3],
        [2.0, 3.0, 1.5, 2.0],
    )


def test_credit_note_is_kept_as_negative_revenue_at_its_own_date():
    result = clean_data(_base_rows())

    returns = result[result["is_return"]]
    assert len(returns) == 1
    row = returns.iloc[0]
    assert row["revenue"] == -3 * 2.0
    assert row["quantity"] == -3
    assert row["invoicedate"] == pd.Timestamp("2010-01-03")


def test_purchase_rows_are_not_modified_by_credit_notes():
    result = clean_data(_base_rows())

    purchases = result[~result["is_return"]]
    row_a = purchases[(purchases["invoice"] == "536365") & (purchases["stockcode"] == "A")].iloc[0]
    assert row_a["quantity"] == 10
    assert row_a["revenue"] == 20.0
    assert len(purchases) == 3


def test_credit_note_with_unrelated_number_is_still_recorded():
    df = _raw(
        ["489434", "C489449"], ["A", "A"], [1, 1],
        ["2009-12-01", "2009-12-02"], [10, -4], [2.0, 2.0],
    )
    result = clean_data(df)
    assert result["revenue"].sum() == 10 * 2.0 - 4 * 2.0


def test_positive_quantity_on_credit_note_is_treated_as_return():
    df = _raw(["C1"], ["A"], [1], ["2010-01-01"], [4], [2.0])
    result = clean_data(df)
    assert result.iloc[0]["revenue"] == -8.0


def test_negative_quantity_purchase_rows_are_dropped():
    df = _raw(["1", "2"], ["A", "A"], [1, 1], ["2010-01-01", "2010-01-02"], [-5, 5], [2.0, 2.0])
    result = clean_data(df)
    assert len(result) == 1
    assert result.iloc[0]["invoice"] == "2"


def test_non_positive_price_rows_are_dropped():
    df = _raw(["1", "2", "3"], ["A", "A", "A"], [1, 1, 1],
              ["2010-01-01", "2010-01-02", "2010-01-03"], [1, 1, 1], [0.0, -5.0, 2.0])
    result = clean_data(df)
    assert list(result["invoice"]) == ["3"]


def test_rows_without_customer_id_are_dropped():
    df = _raw(["1", "2"], ["A", "A"], [1.0, None], ["2010-01-01", "2010-01-02"], [1, 1], [2.0, 2.0])
    result = clean_data(df)
    assert list(result["customer_id"]) == [1]


def test_integer_invoice_numbers_are_normalised_to_strings():
    df = _raw([536365, "C536999"], ["A", 22423], [1, 1], ["2010-01-01", "2010-01-02"], [1, -1], [2.0, 2.0])
    result = clean_data(df)
    assert result["invoice"].map(type).eq(str).all()
    assert result["stockcode"].map(type).eq(str).all()
    assert result["is_return"].tolist() == [False, True]


def test_combine_sheets_drops_second_sheet_invoices_already_in_first():
    first = _raw([1, 2], ["A", "A"], [1, 1], ["2010-12-01", "2010-12-05"], [1, 1], [2.0, 2.0])
    second = _raw([2, 3], ["A", "A"], [1, 1], ["2010-12-05", "2010-12-10"], [1, 1], [2.0, 2.0])
    combined = combine_sheets(first, second)
    assert sorted(combined["Invoice"].astype(str)) == ["1", "2", "3"]


def test_combine_sheets_matches_mixed_type_invoice_numbers():
    first = _raw([2], ["A"], [1], ["2010-12-05"], [1], [2.0])
    second = _raw(["2"], ["A"], [1], ["2010-12-05"], [1], [2.0])
    assert len(combine_sheets(first, second)) == 1


def test_revenue_is_signed_quantity_times_price():
    result = clean_data(_base_rows())
    assert (result["revenue"] == result["quantity"] * result["price"]).all()
