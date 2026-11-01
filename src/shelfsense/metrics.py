"""Forecast accuracy metrics a business audience can read."""
from __future__ import annotations

import numpy as np


def wape(actual, pred) -> float:
    """Weighted absolute percentage error: total absolute error / total actual units."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(a - p).sum() / max(np.abs(a).sum(), 1e-9) * 100)


def mape(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mask = a > 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else float("nan")


def rmse(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def bias(actual, pred) -> float:
    """Positive means the forecast over-predicts (risk of excess stock)."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float((p.sum() - a.sum()) / max(a.sum(), 1e-9) * 100)


def all_metrics(actual, pred) -> dict:
    return {"WAPE %": wape(actual, pred), "MAPE %": mape(actual, pred),
            "RMSE units": rmse(actual, pred), "Bias %": bias(actual, pred)}
