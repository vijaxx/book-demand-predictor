import numpy as np
import pandas as pd
import pytest

from recommender import DemandForecaster
from recommender.demand import FEATURES, build_features, mape


@pytest.fixture(scope="module")
def sales():
    return pd.read_csv("data/sales.csv")


@pytest.fixture(scope="module")
def model(sales):
    return DemandForecaster(sales)


def test_feature_matrix_has_no_missing_values(sales):
    featured = build_features(sales)
    assert not featured[FEATURES].isna().any().any()


def test_lag_features_do_not_leak_the_target(sales):
    """lag_1 for a row must equal units_sold of that book's previous month."""
    featured = build_features(sales)
    row = featured.iloc[100]
    history = sales[sales["book_id"] == row["book_id"]].sort_values("month")
    prior = history[history["month"] < row["month"]]
    assert row["lag_1"] == prior["units_sold"].iloc[-1]


def test_seasonality_encoding_is_on_unit_circle(sales):
    featured = build_features(sales)
    radius = featured["month_sin"] ** 2 + featured["month_cos"] ** 2
    assert np.allclose(radius, 1.0)


def test_model_beats_the_naive_baseline(model):
    assert model.metrics["mae"] < model.metrics["naive_mae"]


def test_model_explains_most_of_the_variance(model):
    assert model.metrics["r2"] > 0.5


def test_holdout_is_temporal_not_random(model, sales):
    """Training and test sets must not overlap in time."""
    months = sorted(sales["month"].unique())
    assert model.metrics["holdout_from"] == months[-6]


def test_forecast_returns_requested_horizon(model):
    assert len(model.forecast(1, horizon=6)) == 6


def test_forecast_months_are_consecutive_and_future(model, sales):
    last_month = sorted(sales["month"].unique())[-1]
    out = model.forecast(1, horizon=4)
    assert out[0]["month"] > last_month
    labels = [p["month"] for p in out]
    assert labels == sorted(labels)
    assert len(set(labels)) == 4


def test_forecast_is_never_negative(model):
    for book_id in (1, 50, 150, 299):
        assert all(p["predicted_units"] >= 0 for p in model.forecast(book_id, horizon=6))


def test_unknown_book_id_raises(model):
    with pytest.raises(KeyError):
        model.forecast(999_999)


def test_coefficients_cover_every_feature(model):
    assert [c["feature"] for c in model.coefficients()] == FEATURES


def test_mape_ignores_zero_demand_months():
    y_true = np.array([0, 100])
    y_pred = np.array([50, 110])
    assert mape(y_true, y_pred) == pytest.approx(10.0)
