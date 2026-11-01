"""Model tests, including the release quality gate that CI enforces."""
import pytest

from shelfsense.metrics import bias, wape
from shelfsense.model import MAX_WAPE, forecast_next_days, train_and_evaluate


@pytest.fixture(scope="module")
def result(sales):
    return train_and_evaluate(sales, params={"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 31})


def test_metrics_basic():
    assert wape([10, 10], [10, 10]) == 0
    assert bias([10, 10], [12, 12]) == pytest.approx(20)


def test_quality_gate_model_beats_naive_baseline(result):
    """Release gate: a model that cannot beat 'same day last week' must not ship."""
    assert result.metrics["WAPE %"] < result.baseline_metrics["WAPE %"]


def test_quality_gate_error_under_threshold(result):
    assert result.metrics["WAPE %"] < MAX_WAPE


def test_forecast_is_seven_days_non_negative(result, sales):
    fc = forecast_next_days(result, sales)
    assert fc["date"].nunique() == 7
    assert (fc["forecast_units"] >= 0).all()
    assert (fc["lower"] <= fc["forecast_units"]).all() and (fc["forecast_units"] <= fc["upper"]).all()


def test_promotion_raises_forecast(result, sales):
    sku = sales["sku"].iloc[0]
    base = forecast_next_days(result, sales)
    promo = forecast_next_days(result, sales, promo_plan={sku: 1})
    assert promo[promo.sku == sku]["forecast_units"].sum() > base[base.sku == sku]["forecast_units"].sum()


def test_horizon_beyond_seven_days_is_rejected(result, sales):
    with pytest.raises(ValueError):
        forecast_next_days(result, sales, horizon=14)
