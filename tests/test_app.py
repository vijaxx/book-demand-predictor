import re

import pandas as pd
import pytest

import app as app_module


@pytest.fixture(scope="module")
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_returns_200(client):
    assert client.get("/").status_code == 200


def test_book_detail_returns_200_for_valid_id(client):
    assert client.get("/book/1").status_code == 200


def test_book_detail_404_for_unknown_id(client):
    assert client.get("/book/999999").status_code == 404


def test_dashboard_returns_200(client):
    assert client.get("/dashboard").status_code == 200


def test_search_respects_genre_filter(client):
    """A search query combined with a genre filter must only return that genre.

    Regression guard: the genre <select> and the search box live in the same
    form, so a user filtering by genre while searching expects both to apply.
    Previously the genre parameter was read but silently dropped whenever a
    search query was present. "dragon heart kingdom wedding" is a real query
    that returns a genuine mix of Fantasy and Romance when unfiltered -- it's
    not a query that would pass this test by accident.
    """
    query = "dragon heart kingdom wedding"

    unfiltered = client.get(f"/?q={query}").get_data(as_text=True)
    ids = [int(i) for i in re.findall(r'/book/(\d+)"', unfiltered)]
    genres = set(app_module.BOOKS.set_index("book_id").loc[ids, "genre"])
    assert len(genres) > 1, "test query should span multiple genres unfiltered"

    filtered = client.get(f"/?q={query}&genre=Fantasy").get_data(as_text=True)
    ids = [int(i) for i in re.findall(r'/book/(\d+)"', filtered)]
    assert ids, "genre-filtered search returned no results"
    genres = set(app_module.BOOKS.set_index("book_id").loc[ids, "genre"])
    assert genres == {"Fantasy"}


def test_search_without_genre_is_unfiltered(client):
    resp = client.get("/?q=magic+kingdom+dragon")
    assert resp.status_code == 200


def test_api_recommend_404_for_unknown_book(client):
    assert client.get("/api/recommend/999999").status_code == 404


def test_api_forecast_404_for_unknown_book(client):
    assert client.get("/api/forecast/999999").status_code == 404


def test_book_detail_handles_book_with_no_sales_history(client):
    """A book can be in the catalogue before it has any sales rows.

    Regression guard: book_detail() used to call FORECASTER.forecast()
    unconditionally, which raises KeyError (-> 500) for a book with zero sales
    history. api_forecast() already guarded against this; the page route did
    not.
    """
    book_id = 1
    original_sales = app_module.SALES
    original_forecaster_sales = app_module.FORECASTER.sales
    trimmed = original_sales[original_sales["book_id"] != book_id]
    app_module.SALES = trimmed
    # forecast() reads FORECASTER's own copy of the sales table, not the
    # module-level SALES -- both have to reflect "no history" for this to
    # actually exercise the crash the fix guards against.
    app_module.FORECASTER.sales = trimmed
    try:
        resp = client.get(f"/book/{book_id}")
        assert resp.status_code == 200
    finally:
        app_module.SALES = original_sales
        app_module.FORECASTER.sales = original_forecaster_sales


def test_book_detail_handles_book_with_partial_sales_history(client):
    """A book can have 1-2 months of sales before forecast() has three
    months of lag to work with.

    Regression guard: forecast() raised a bare IndexError (-> 500) for a
    book with fewer than 3 months of history; book_detail() only guarded
    against zero months.
    """
    book_id = 1
    original_sales = app_module.SALES
    original_forecaster_sales = app_module.FORECASTER.sales
    rest = original_sales[original_sales["book_id"] != book_id]
    partial = (
        original_sales[original_sales["book_id"] == book_id]
        .sort_values("month")
        .head(2)
    )
    trimmed = pd.concat([rest, partial])
    app_module.SALES = trimmed
    app_module.FORECASTER.sales = trimmed
    try:
        resp = client.get(f"/book/{book_id}")
        assert resp.status_code == 200
    finally:
        app_module.SALES = original_sales
        app_module.FORECASTER.sales = original_forecaster_sales


def test_api_forecast_handles_partial_sales_history(client):
    """Same guard, exercised through the JSON endpoint the book page's
    chart actually calls."""
    book_id = 1
    original_sales = app_module.SALES
    original_forecaster_sales = app_module.FORECASTER.sales
    rest = original_sales[original_sales["book_id"] != book_id]
    partial = (
        original_sales[original_sales["book_id"] == book_id]
        .sort_values("month")
        .head(2)
    )
    trimmed = pd.concat([rest, partial])
    app_module.SALES = trimmed
    app_module.FORECASTER.sales = trimmed
    try:
        resp = client.get(f"/api/forecast/{book_id}")
        assert resp.status_code == 200
        assert resp.get_json()["forecast"]["labels"] == []
    finally:
        app_module.SALES = original_sales
        app_module.FORECASTER.sales = original_forecaster_sales
