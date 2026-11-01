"""Synthetic data generation, loading and validation for ShelfSense."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "sku", "product_name", "category", "units_sold", "unit_price", "promo"]

# A small CPG catalogue: (sku, name, category, base daily units, unit price in INR)
CATALOGUE = [
    ("BEV-001", "Lemon Fizz 500ml", "Beverages", 420, 40),
    ("BEV-002", "Mango Nectar 1L", "Beverages", 260, 95),
    ("BEV-003", "Cold Coffee 200ml", "Beverages", 180, 45),
    ("SNK-001", "Masala Chips 50g", "Snacks", 520, 20),
    ("SNK-002", "Salted Peanuts 200g", "Snacks", 210, 60),
    ("SNK-003", "Choco Cookies 120g", "Snacks", 300, 35),
    ("DRY-001", "Toned Milk 500ml", "Dairy", 650, 28),
    ("DRY-002", "Greek Yogurt 400g", "Dairy", 140, 85),
    ("PCR-001", "Neem Soap 100g", "Personal Care", 230, 38),
    ("PCR-002", "Herbal Shampoo 180ml", "Personal Care", 120, 145),
]

# Rough festive window (Diwali season) and summer window, used for seasonality.
FESTIVE_MONTHS = {10, 11}
SUMMER_MONTHS = {4, 5, 6}


def generate_sales(days: int = 730, end_date: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Generate daily synthetic sales for the catalogue.

    Signals baked in: weekly pattern, summer lift for beverages, festive lift for
    snacks and personal care, slow upward trend, random promotions and noise.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")
    dow_effect = np.array([0.92, 0.90, 0.95, 1.00, 1.08, 1.22, 1.15])  # Mon..Sun

    frames = []
    for sku, name, category, base, price in CATALOGUE:
        t = np.arange(days)
        trend = 1 + 0.15 * t / days
        dow = dow_effect[dates.dayofweek]
        month = dates.month.values
        season = np.ones(days)
        if category == "Beverages":
            season = np.where(np.isin(month, list(SUMMER_MONTHS)), 1.35, 1.0)
        elif category in ("Snacks", "Personal Care"):
            season = np.where(np.isin(month, list(FESTIVE_MONTHS)), 1.30, 1.0)
        promo = (rng.random(days) < 0.08).astype(int)
        promo_lift = np.where(promo == 1, 1.25, 1.0)
        noise = rng.normal(1.0, 0.08, days)
        units = np.maximum(0, base * trend * dow * season * promo_lift * noise).round().astype(int)
        unit_price = np.where(promo == 1, round(price * 0.9, 2), price)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku": sku,
                    "product_name": name,
                    "category": category,
                    "units_sold": units,
                    "unit_price": unit_price.astype(float),
                    "promo": promo,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["revenue"] = df["units_sold"] * df["unit_price"]
    return df


def validate_sales(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable data-quality problems. Empty list means the data passed."""
    problems: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"]
    if df.empty:
        return ["The file has no rows."]
    if df[REQUIRED_COLUMNS].isna().any().any():
        cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isna().any()].tolist()
        problems.append(f"Null values found in: {', '.join(cols)}")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        problems.append("Some dates could not be parsed.")
    if (pd.to_numeric(df["units_sold"], errors="coerce") < 0).any():
        problems.append("Negative units_sold values found.")
    if (pd.to_numeric(df["unit_price"], errors="coerce") <= 0).any():
        problems.append("Non-positive unit_price values found.")
    if not set(pd.unique(df["promo"])).issubset({0, 1}):
        problems.append("promo must be 0 or 1.")
    if df.duplicated(subset=["date", "sku"]).any():
        problems.append("Duplicate (date, sku) rows found.")
    if not dates.isna().any():
        span = (dates.max() - dates.min()).days + 1
        if span < 60:
            problems.append(f"Only {span} days of history; at least 60 are needed to train.")
    return problems


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and add revenue if absent."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["units_sold"] = pd.to_numeric(out["units_sold"]).astype(float)
    out["unit_price"] = pd.to_numeric(out["unit_price"]).astype(float)
    out["promo"] = out["promo"].astype(int)
    if "revenue" not in out.columns:
        out["revenue"] = out["units_sold"] * out["unit_price"]
    return out.sort_values(["sku", "date"]).reset_index(drop=True)


def kpis(df: pd.DataFrame, window_days: int = 28) -> dict:
    """Headline KPIs for the latest window versus the window before it."""
    end = df["date"].max()
    cur = df[df["date"] > end - pd.Timedelta(days=window_days)]
    prev = df[(df["date"] <= end - pd.Timedelta(days=window_days)) & (df["date"] > end - pd.Timedelta(days=2 * window_days))]

    def pct(a, b):
        return (a - b) / b * 100 if b else 0.0

    rev, prev_rev = cur["revenue"].sum(), prev["revenue"].sum()
    units, prev_units = cur["units_sold"].sum(), prev["units_sold"].sum()
    top = cur.groupby("product_name")["revenue"].sum().idxmax() if not cur.empty else "-"
    return {
        "revenue": rev,
        "revenue_change_pct": pct(rev, prev_rev),
        "units": units,
        "units_change_pct": pct(units, prev_units),
        "avg_price": rev / units if units else 0.0,
        "top_product": top,
        "promo_share_pct": cur["promo"].mean() * 100 if not cur.empty else 0.0,
    }
