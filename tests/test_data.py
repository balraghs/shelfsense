"""Unit tests: data generation and validation (the 'data quality gate')."""
import pandas as pd

from shelfsense.data import CATALOGUE, generate_sales, kpis, validate_sales


def test_generator_shape_and_columns():
    df = generate_sales(days=100, end_date="2026-10-31")
    assert len(df) == 100 * len(CATALOGUE)
    assert {"date", "sku", "units_sold", "revenue"} <= set(df.columns)
    assert (df["units_sold"] >= 0).all()


def test_generator_is_reproducible():
    a = generate_sales(days=50, end_date="2026-10-31", seed=3)
    b = generate_sales(days=50, end_date="2026-10-31", seed=3)
    assert a.equals(b)


def test_clean_data_passes_validation(sales):
    assert validate_sales(sales) == []


def test_validation_catches_missing_column(sales):
    assert "Missing required columns" in validate_sales(sales.drop(columns=["promo"]))[0]


def test_validation_catches_negative_units(sales):
    bad = sales.copy()
    bad.loc[0, "units_sold"] = -5
    assert any("Negative" in p for p in validate_sales(bad))


def test_validation_catches_duplicates(sales):
    bad = pd.concat([sales, sales.iloc[[0]]])
    assert any("Duplicate" in p for p in validate_sales(bad))


def test_validation_requires_enough_history(sales):
    short = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=30)]
    assert any("history" in p for p in validate_sales(short))


def test_kpis_are_positive(sales):
    k = kpis(sales)
    assert k["revenue"] > 0 and k["units"] > 0
