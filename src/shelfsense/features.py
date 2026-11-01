"""Feature engineering. All lag features are at least 7 days old, so one model can
forecast every day of the next week directly from known history (no recursion)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 7
LAGS = [7, 14, 21, 28]
FEATURES = [
    "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_28",
    "dow", "month", "day_of_year", "is_weekend", "is_festive",
    "promo", "unit_price", "sku_code",
]
TARGET = "units_sold"


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_festive"] = df["month"].isin([10, 11]).astype(int)
    return df


def build_features(df: pd.DataFrame, sku_codes: dict[str, int] | None = None) -> pd.DataFrame:
    """Add lag, rolling and calendar features. Rows without full history get NaN lags."""
    out = df.sort_values(["sku", "date"]).copy()
    g = out.groupby("sku")[TARGET]
    for lag in LAGS:
        out[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON)
    out["roll_mean_7"] = shifted.groupby(out["sku"]).transform(lambda s: s.rolling(7).mean())
    out["roll_mean_28"] = shifted.groupby(out["sku"]).transform(lambda s: s.rolling(28).mean())
    out["roll_std_28"] = shifted.groupby(out["sku"]).transform(lambda s: s.rolling(28).std())
    out = _calendar(out)
    codes = sku_codes or {s: i for i, s in enumerate(sorted(out["sku"].unique()))}
    out["sku_code"] = out["sku"].map(codes).fillna(-1).astype(int)
    return out


def future_frame(history: pd.DataFrame, horizon: int = HORIZON, promo_plan: dict[str, int] | None = None) -> pd.DataFrame:
    """Rows for the next `horizon` days per SKU, with unknown target and planned promo/price."""
    promo_plan = promo_plan or {}
    last = history["date"].max()
    rows = []
    latest = history.sort_values("date").groupby("sku").tail(1).set_index("sku")
    for sku, r in latest.iterrows():
        list_price = history[(history["sku"] == sku) & (history["promo"] == 0)]["unit_price"].median()
        if np.isnan(list_price):
            list_price = r["unit_price"]
        promo = int(promo_plan.get(sku, 0))
        for d in range(1, horizon + 1):
            rows.append({
                "date": last + pd.Timedelta(days=d), "sku": sku,
                "product_name": r["product_name"], "category": r["category"],
                TARGET: np.nan, "unit_price": round(list_price * (0.9 if promo else 1.0), 2),
                "promo": promo,
            })
    return pd.DataFrame(rows)
