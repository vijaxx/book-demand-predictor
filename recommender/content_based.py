"""Content-based book recommendation using TF-IDF and cosine similarity.

Approach
--------
Every book is flattened into a single "content document" built from its genre,
author and description keywords. Those documents are vectorised with TF-IDF,
and similarity between two books is the cosine of the angle between their
vectors.

Cosine similarity is the right choice here (over raw dot product) because it is
magnitude-invariant: a book with a longer description should not be considered
similar to everything simply because its vector is larger. Only the *direction*
of the vector -- the mix of terms -- matters.

The full 300x300 similarity matrix is computed once at load time and cached.
At that size it is trivial (~0.7 MB as float64), and it turns every subsequent
recommendation lookup into an array slice instead of a recomputation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Recommends books similar to a given book, by content overlap."""

    def __init__(self, books: pd.DataFrame) -> None:
        self.books = books.reset_index(drop=True)
        # Map book_id -> positional index, so callers can use real ids.
        self._index_of = {
            int(bid): pos for pos, bid in enumerate(self.books["book_id"])
        }

        corpus = self._build_corpus(self.books)

        # min_df=2 drops terms appearing in a single book: those carry no
        # similarity information but inflate the vocabulary.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.similarity = cosine_similarity(self.matrix)

    @staticmethod
    def _build_corpus(books: pd.DataFrame) -> list[str]:
        """Flatten each book into one text document.

        Genre is repeated three times as a cheap, explicit way of weighting it
        above description keywords -- two books in the same genre should start
        out closer than two books that merely share one adjective.
        """
        return [
            " ".join([
                str(row.genre) + " ",
                (str(row.genre) + " ") * 2,
                str(row.author),
                str(row.description),
                str(row.title),
            ])
            for row in books.itertuples(index=False)
        ]

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def similar_to(self, book_id: int, top_n: int = 5) -> list[dict]:
        """Return the top_n books most similar to book_id, excluding itself."""
        if book_id not in self._index_of:
            raise KeyError(f"unknown book_id: {book_id}")

        pos = self._index_of[book_id]
        scores = self.similarity[pos].copy()
        scores[pos] = -1.0  # never recommend the book itself

        # argsort ascending, take the tail, reverse -> descending top_n.
        best = scores.argsort()[-top_n:][::-1]

        out = []
        for other in best:
            row = self.books.iloc[int(other)]
            out.append({
                "book_id": int(row["book_id"]),
                "title": row["title"],
                "author": row["author"],
                "genre": row["genre"],
                "rating": float(row["rating"]),
                "price": float(row["price"]),
                "score": round(float(scores[other]), 4),
            })
        return out

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        """Free-text search: rank books by cosine similarity to the query.

        Reuses the fitted vectorizer, so the query lands in exactly the same
        vector space as the catalogue.
        """
        if not query.strip():
            return []

        vec = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = scores.argsort()[-top_n:][::-1]

        out = []
        for pos in best:
            if scores[pos] <= 0:
                continue
            row = self.books.iloc[int(pos)]
            out.append({
                "book_id": int(row["book_id"]),
                "title": row["title"],
                "author": row["author"],
                "genre": row["genre"],
                "rating": float(row["rating"]),
                "price": float(row["price"]),
                "score": round(float(scores[pos]), 4),
            })
        return out
