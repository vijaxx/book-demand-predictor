"""Flask application: book catalogue, recommendations, and a demand dashboard.

Run:
    python scripts/generate_data.py    # once, to create data/*.csv
    python app.py                      # http://127.0.0.1:5000
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

from recommender import ContentBasedRecommender, DemandForecaster

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    books_path = DATA_DIR / "books.csv"
    sales_path = DATA_DIR / "sales.csv"
    if not books_path.exists() or not sales_path.exists():
        raise SystemExit(
            "data/books.csv or data/sales.csv is missing.\n"
            "Generate it first:  python scripts/generate_data.py"
        )
    return pd.read_csv(books_path), pd.read_csv(sales_path)


# Models are built once at import time. TF-IDF fitting and the Ridge fit both
# take well under a second at this dataset size, and holding them in memory
# keeps every request a lookup rather than a retrain.
BOOKS, SALES = load_data()
RECOMMENDER = ContentBasedRecommender(BOOKS)
FORECASTER = DemandForecaster(SALES)

# Total units per book, reused by several views.
TOTALS = SALES.groupby("book_id")["units_sold"].sum()


def book_record(book_id: int) -> dict:
    row = BOOKS[BOOKS["book_id"] == book_id]
    if row.empty:
        abort(404, description=f"No book with id {book_id}")
    rec = row.iloc[0].to_dict()
    rec["book_id"] = int(rec["book_id"])
    rec["total_units"] = int(TOTALS.get(book_id, 0))
    return rec


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()

    if query:
        # Over-fetch before filtering by genre, since search() has no genre
        # awareness of its own -- narrowing after the fact can otherwise leave
        # fewer than 24 results for a niche query, which is fine, but fetching
        # too few candidates would silently drop genuinely relevant matches.
        results = RECOMMENDER.search(query, top_n=100)
        if genre:
            results = [r for r in results if r["genre"] == genre]
        results = results[:24]
        for item in results:
            item["total_units"] = int(TOTALS.get(item["book_id"], 0))
    else:
        subset = BOOKS if not genre else BOOKS[BOOKS["genre"] == genre]
        subset = subset.copy()
        subset["total_units"] = subset["book_id"].map(TOTALS).fillna(0).astype(int)
        subset = subset.sort_values("total_units", ascending=False).head(24)
        results = subset.to_dict("records")
        for item in results:
            item["book_id"] = int(item["book_id"])

    return render_template(
        "index.html",
        books=results,
        query=query,
        genre=genre,
        genres=sorted(BOOKS["genre"].unique()),
    )


@app.route("/book/<int:book_id>")
def book_detail(book_id: int):
    book = book_record(book_id)
    history = (
        SALES[SALES["book_id"] == book_id]
        .sort_values("month")[["month", "units_sold"]]
        .to_dict("records")
    )
    # A book can exist in the catalogue before it has any sales rows, or with
    # only one or two months of them (e.g. a newly added title) -- forecast()
    # needs three months of lag to work with, so skip it rather than letting
    # the KeyError/ValueError turn into a 500.
    forecast = FORECASTER.forecast(book_id, horizon=6) if len(history) >= 3 else []
    return render_template(
        "book.html",
        book=book,
        similar=RECOMMENDER.similar_to(book_id, top_n=6),
        history=history,
        forecast=forecast,
    )


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        metrics=FORECASTER.metrics,
        coefficients=FORECASTER.coefficients(),
        catalogue_size=len(BOOKS),
        vocabulary_size=RECOMMENDER.vocabulary_size,
        total_units=int(SALES["units_sold"].sum()),
        months_tracked=SALES["month"].nunique(),
    )


# ---------------------------------------------------------------- JSON API --

@app.route("/api/genre-sales")
def api_genre_sales():
    """Total units per genre -- drives the dashboard bar chart."""
    merged = SALES.merge(BOOKS[["book_id", "genre"]], on="book_id")
    totals = (
        merged.groupby("genre")["units_sold"].sum()
        .sort_values(ascending=False)
    )
    return jsonify({
        "labels": totals.index.tolist(),
        "values": [int(v) for v in totals.to_numpy()],
    })


@app.route("/api/monthly-trend")
def api_monthly_trend():
    """Catalogue-wide units per month -- drives the dashboard line chart."""
    totals = SALES.groupby("month")["units_sold"].sum().sort_index()
    return jsonify({
        "labels": totals.index.tolist(),
        "values": [int(v) for v in totals.to_numpy()],
    })


@app.route("/api/forecast/<int:book_id>")
def api_forecast(book_id: int):
    """Actual history plus the forecast horizon for one book."""
    history = SALES[SALES["book_id"] == book_id].sort_values("month")
    if history.empty:
        abort(404, description=f"No book with id {book_id}")

    # Same guard as book_detail(): under 3 months of history isn't enough
    # for forecast() to build its lag features.
    predicted = FORECASTER.forecast(book_id, horizon=6) if len(history) >= 3 else []
    return jsonify({
        "book_id": book_id,
        "history": {
            "labels": history["month"].tolist(),
            "values": [int(v) for v in history["units_sold"].to_numpy()],
        },
        "forecast": {
            "labels": [p["month"] for p in predicted],
            "values": [p["predicted_units"] for p in predicted],
        },
    })


@app.route("/api/recommend/<int:book_id>")
def api_recommend(book_id: int):
    try:
        return jsonify(RECOMMENDER.similar_to(book_id, top_n=6))
    except KeyError:
        abort(404, description=f"No book with id {book_id}")


@app.errorhandler(404)
def not_found(err):
    return render_template("404.html", message=getattr(err, "description", "")), 404


if __name__ == "__main__":
    app.run(debug=True)
