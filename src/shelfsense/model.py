"""Training, evaluation, quality gate and 7-day forecasting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, HORIZON, TARGET, build_features, future_frame
from .metrics import all_metrics

# Release gate: the model must beat the naive baseline AND stay under this error.
MAX_WAPE = 20.0


@dataclass
class TrainResult:
    model: HistGradientBoostingRegressor
    sku_codes: dict
    metrics: dict
    baseline_metrics: dict
    holdout: pd.DataFrame  # date, sku, actual, predicted, baseline
    feature_importance: pd.DataFrame
    params: dict = field(default_factory=dict)

    @property
    def passes_gate(self) -> bool:
        return self.metrics["WAPE %"] < self.baseline_metrics["WAPE %"] and self.metrics["WAPE %"] < MAX_WAPE


def _new_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=7, **params)


def train_and_evaluate(df: pd.DataFrame, holdout_days: int = 28, params: dict | None = None) -> TrainResult:
    """Train on all but the last `holdout_days`, score on the holdout, then refit on everything."""
    params = params or {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31}
    codes = {s: i for i, s in enumerate(sorted(df["sku"].unique()))}
    feats = build_features(df, codes).dropna(subset=FEATURES + [TARGET])
    cutoff = feats["date"].max() - pd.Timedelta(days=holdout_days)
    train, test = feats[feats["date"] <= cutoff], feats[feats["date"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("Not enough history to create a train/holdout split.")

    model = _new_model(params).fit(train[FEATURES], train[TARGET])
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    baseline = test["lag_7"].values  # seasonal naive: same weekday last week

    holdout = test[["date", "sku", "product_name", "category"]].copy()
    holdout["actual"], holdout["predicted"], holdout["baseline"] = test[TARGET].values, pred, baseline

    importance = _importance(model, test)
    final = _new_model(params).fit(feats[FEATURES], feats[TARGET])
    return TrainResult(final, codes, all_metrics(test[TARGET], pred), all_metrics(test[TARGET], baseline),
                       holdout, importance, params)


def _importance(model, test: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance on a sample: how much error rises when a feature is shuffled."""
    rng = np.random.default_rng(0)
    sample = test.sample(min(len(test), 400), random_state=0)
    base_err = np.abs(model.predict(sample[FEATURES]) - sample[TARGET]).mean()
    rows = []
    for f in FEATURES:
        shuffled = sample[FEATURES].copy()
        shuffled[f] = rng.permutation(shuffled[f].values)
        err = np.abs(model.predict(shuffled) - sample[TARGET]).mean()
        rows.append({"feature": f, "importance": max(err - base_err, 0.0)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def forecast_next_days(result: TrainResult, history: pd.DataFrame, horizon: int = HORIZON,
                       promo_plan: dict | None = None) -> pd.DataFrame:
    """Predict units for the next `horizon` days for every SKU."""
    if horizon > HORIZON:
        raise ValueError(f"Model supports at most {HORIZON} days ahead.")
    fut = future_frame(history, horizon, promo_plan)
    combined = pd.concat([history, fut], ignore_index=True)
    feats = build_features(combined, result.sku_codes)
    feats = feats[feats["date"] > history["date"].max()]
    feats["forecast_units"] = np.clip(result.model.predict(feats[FEATURES]), 0, None).round()
    # Simple interval from holdout error spread per SKU
    err = (result.holdout["actual"] - result.holdout["predicted"]).groupby(result.holdout["sku"]).std()
    feats["lower"] = (feats["forecast_units"] - 1.28 * feats["sku"].map(err).fillna(0)).clip(lower=0).round()
    feats["upper"] = (feats["forecast_units"] + 1.28 * feats["sku"].map(err).fillna(0)).round()
    feats["forecast_revenue"] = feats["forecast_units"] * feats["unit_price"]
    return feats[["date", "sku", "product_name", "category", "promo", "unit_price",
                  "forecast_units", "lower", "upper", "forecast_revenue"]].reset_index(drop=True)


def log_to_mlflow(result: TrainResult, run_name: str = "shelfsense-hgb") -> str | None:
    """Log params, metrics and the model to MLflow if it is installed. Returns run id or None."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(result.params)
        mlflow.log_metrics({k.replace(" %", "_pct").replace(" ", "_"): v for k, v in result.metrics.items()})
        mlflow.log_metrics({"baseline_" + k.replace(" %", "_pct").replace(" ", "_"): v
                            for k, v in result.baseline_metrics.items()})
        mlflow.log_metric("passes_gate", int(result.passes_gate))
        mlflow.sklearn.log_model(result.model, "model")
        return run.info.run_id
