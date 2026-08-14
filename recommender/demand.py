"""Demand forecasting for monthly book sales.

Framing
-------
This is a time-series problem reframed as supervised regression. For each
(book, month) observation the model sees:

  * the previous three months of units sold (lag_1, lag_2, lag_3)
  * a three-month rolling mean, which smooths single-month spikes
  * sin/cos of the month number, which encodes seasonality as a continuous
    cycle so that December and January are adjacent rather than 11 apart
  * the book's own historical mean, which stands in for title popularity

Ridge regression is used rather than plain least squares because the lag
features are strongly collinear -- last month, two months ago and the rolling
mean all carry overlapping information. L2 regularisation keeps the fitted
coefficients stable instead of letting them blow up in opposite directions.

Validation is a *temporal* split: the final HOLDOUT_MONTHS months are held out
for testing and never seen during training. A random split would leak future
information into the training set and produce a dishonestly good score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

HOLDOUT_MONTHS = 6
FEATURES = ["lag_1", "lag_2", "lag_3", "roll_3", "month_sin", "month_cos", "book_mean"]


def build_features(sales: pd.DataFrame) -> pd.DataFrame:
    """Turn the long (book_id, month, units_sold) table into a feature matrix."""
    df = sales.sort_values(["book_id", "month"]).copy()

    grouped = df.groupby("book_id")["units_sold"]
    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_3"] = grouped.shift(3)

    # Rolling mean of the three preceding months. shift(1) first so the current
    # month is never part of its own feature -- that would be target leakage.
    df["roll_3"] = grouped.shift(1).rolling(window=3, min_periods=3).mean()

    month_num = df["month"].str.slice(5, 7).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    # Expanding mean over prior months only, for the same leakage reason.
    df["book_mean"] = grouped.transform(
        lambda s: s.shift(1).expanding().mean()
    )

    return df.dropna(subset=FEATURES).reset_index(drop=True)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring zero-demand months."""
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class DemandForecaster:
    """Fits a Ridge model on lag features and forecasts future monthly demand."""

    def __init__(self, sales: pd.DataFrame) -> None:
        self.sales = sales.copy()
        self.months = sorted(sales["month"].unique())
        self.model = Ridge(alpha=1.0)
        self.metrics: dict[str, float] = {}
        self._fit()

    def _fit(self) -> None:
        featured = build_features(self.sales)

        split_at = self.months[-HOLDOUT_MONTHS]
        train = featured[featured["month"] < split_at]
        test = featured[featured["month"] >= split_at]

        x_train, y_train = train[FEATURES].to_numpy(), train["units_sold"].to_numpy()
        x_test, y_test = test[FEATURES].to_numpy(), test["units_sold"].to_numpy()

        self.model.fit(x_train, y_train)
        predicted = self.model.predict(x_test)

        # Baseline: "next month equals last month". A model that cannot beat
        # this is not earning its complexity, so it is reported alongside.
        naive = test["lag_1"].to_numpy()

        self.metrics = {
            "mae": round(float(mean_absolute_error(y_test, predicted)), 2),
            "mape": round(mape(y_test, predicted), 2),
            "r2": round(float(r2_score(y_test, predicted)), 4),
            "naive_mae": round(float(mean_absolute_error(y_test, naive)), 2),
            "naive_mape": round(mape(y_test, naive), 2),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "holdout_from": split_at,
        }
        self.metrics["improvement_pct"] = round(
            (self.metrics["naive_mae"] - self.metrics["mae"])
            / self.metrics["naive_mae"] * 100, 1
        )

    @staticmethod
    def _next_month(label: str) -> str:
        year, month = int(label[:4]), int(label[5:7])
        month += 1
        if month > 12:
            month, year = 1, year + 1
        return f"{year:04d}-{month:02d}"

    def forecast(self, book_id: int, horizon: int = 6) -> list[dict]:
        """Recursively forecast the next `horizon` months for one book.

        Each prediction is fed back in as the next step's lag_1, which is the
        standard recursive strategy. Errors compound across the horizon, so
        later months carry genuinely more uncertainty than earlier ones.
        """
        history = (
            self.sales[self.sales["book_id"] == book_id]
            .sort_values("month")
        )
        if history.empty:
            raise KeyError(f"unknown book_id: {book_id}")

        units = history["units_sold"].astype(float).tolist()
        month_label = history["month"].iloc[-1]
        running_mean = float(np.mean(units))

        out = []
        for _ in range(horizon):
            month_label = self._next_month(month_label)
            month_num = int(month_label[5:7])

            row = np.array([[
                units[-1],
                units[-2],
                units[-3],
                float(np.mean(units[-3:])),
                np.sin(2 * np.pi * month_num / 12),
                np.cos(2 * np.pi * month_num / 12),
                running_mean,
            ]])

            predicted = max(0.0, float(self.model.predict(row)[0]))
            out.append({"month": month_label, "predicted_units": round(predicted, 1)})

            units.append(predicted)
            running_mean = float(np.mean(units))

        return out

    def coefficients(self) -> list[dict]:
        """Fitted weight per feature -- what the model actually learned."""
        return [
            {"feature": name, "weight": round(float(w), 4)}
            for name, w in zip(FEATURES, self.model.coef_)
        ]
