import pandas as pd
import pytest

from recommender import ContentBasedRecommender


@pytest.fixture(scope="module")
def books():
    return pd.read_csv("data/books.csv")


@pytest.fixture(scope="module")
def rec(books):
    return ContentBasedRecommender(books)


def test_returns_requested_number_of_results(rec):
    assert len(rec.similar_to(1, top_n=5)) == 5


def test_never_recommends_the_book_itself(rec):
    assert all(r["book_id"] != 1 for r in rec.similar_to(1, top_n=10))


def test_results_are_sorted_by_descending_score(rec):
    scores = [r["score"] for r in rec.similar_to(42, top_n=8)]
    assert scores == sorted(scores, reverse=True)


def test_scores_stay_within_cosine_bounds(rec):
    for r in rec.similar_to(7, top_n=10):
        assert 0.0 <= r["score"] <= 1.0


def test_recommendations_favour_the_same_genre(rec, books):
    """Content overlap should surface same-genre titles at the top.

    Not asserted as 100% -- an author or keyword collision across genres is
    legitimate behaviour, not a bug -- but the majority should match.
    """
    seed_genre = books.loc[books["book_id"] == 1, "genre"].iloc[0]
    got = rec.similar_to(1, top_n=5)
    same = sum(1 for r in got if r["genre"] == seed_genre)
    assert same >= 3


def test_unknown_book_id_raises(rec):
    with pytest.raises(KeyError):
        rec.similar_to(999_999)


def test_search_matches_expected_genre(rec):
    """A horror-vocabulary query should return horror titles."""
    results = rec.search("haunted ritual crypt possession", top_n=5)
    assert results
    assert sum(1 for r in results if r["genre"] == "Horror") >= 3


def test_empty_search_returns_nothing(rec):
    assert rec.search("   ") == []


def test_search_scores_are_sorted(rec):
    scores = [r["score"] for r in rec.search("dragon magic kingdom", top_n=6)]
    assert scores == sorted(scores, reverse=True)
