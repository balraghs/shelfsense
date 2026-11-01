"""Unit tests: features must never leak information from the forecast window."""
import pandas as pd

from shelfsense.features import FEATURES, HORIZON, build_features, future_frame


def test_lag_7_equals_value_a_week_ago(sales):
    f = build_features(sales)
    one = f[f["sku"] == f["sku"].iloc[0]].reset_index(drop=True)
    assert one.loc[20, "lag_7"] == one.loc[13, "units_sold"]


def test_no_feature_uses_data_newer_than_horizon(sales):
    """Changing the last 7 days of sales must not change any feature on those days."""
    f1 = build_features(sales)
    changed = sales.copy()
    last = changed["date"] > changed["date"].max() - pd.Timedelta(days=HORIZON)
    changed.loc[last, "units_sold"] *= 10
    f2 = build_features(changed)
    cols = [c for c in FEATURES if c.startswith(("lag", "roll"))]
    pd.testing.assert_frame_equal(f1.loc[last.values, cols].reset_index(drop=True),
                                  f2.loc[last.values, cols].reset_index(drop=True))


def test_future_frame_has_horizon_rows_per_sku(sales):
    fut = future_frame(sales)
    assert len(fut) == HORIZON * sales["sku"].nunique()
    assert fut["date"].min() == sales["date"].max() + pd.Timedelta(days=1)
