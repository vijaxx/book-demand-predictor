"""Generate the synthetic book catalogue and sales history used by the app.

The dataset is synthetic on purpose: it lets anyone clone the repo and get a
working demo with no external data download and no licensing questions. It is
seeded, so every run produces byte-identical CSVs and the numbers quoted in the
README stay reproducible.

Usage:
    python scripts/generate_data.py
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

SEED = 20260814
N_BOOKS = 300
N_MONTHS = 36
START_YEAR = 2023
START_MONTH = 1

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Each genre carries its own vocabulary, base demand level, and seasonal peak
# month. The recommender has to rediscover the vocabulary overlap on its own --
# it never sees the genre label as a feature by itself.
GENRES = {
    "Fantasy": {
        "base": 120,
        "peak_month": 12,
        "words": ["dragon", "magic", "kingdom", "sword", "prophecy", "wizard",
                  "quest", "realm", "elf", "curse", "throne", "spell"],
    },
    "Science Fiction": {
        "base": 95,
        "peak_month": 7,
        "words": ["starship", "colony", "android", "orbit", "quantum", "alien",
                  "terraform", "galaxy", "cyborg", "dystopia", "wormhole"],
    },
    "Mystery": {
        "base": 110,
        "peak_month": 10,
        "words": ["detective", "murder", "clue", "suspect", "alibi", "inspector",
                  "witness", "evidence", "forensic", "vanished", "confession"],
    },
    "Romance": {
        "base": 140,
        "peak_month": 2,
        "words": ["heart", "wedding", "letter", "summer", "reunion", "vow",
                  "longing", "affair", "promise", "kiss", "second-chance"],
    },
    "Thriller": {
        "base": 105,
        "peak_month": 8,
        "words": ["conspiracy", "agent", "hostage", "countdown", "assassin",
                  "surveillance", "manhunt", "betrayal", "sniper", "cipher"],
    },
    "Historical": {
        "base": 70,
        "peak_month": 11,
        "words": ["empire", "revolution", "war", "dynasty", "siege", "merchant",
                  "colonial", "regiment", "monarch", "plague", "voyage"],
    },
    "Self-Help": {
        "base": 130,
        "peak_month": 1,
        "words": ["habit", "focus", "discipline", "mindset", "productivity",
                  "clarity", "resilience", "purpose", "routine", "growth"],
    },
    "Horror": {
        "base": 60,
        "peak_month": 10,
        "words": ["haunted", "asylum", "ritual", "shadow", "possession", "crypt",
                  "nightmare", "coven", "descent", "whisper"],
    },
}

TITLE_PATTERNS = [
    "The {a} of {b}", "{a} and {b}", "A {a} of {b}", "The Last {a}",
    "{a} Rising", "Beneath the {a}", "The {a} Protocol", "Song of {a}",
    "{a} Season", "The {b} Question",
]

FIRST_NAMES = ["Aditi", "Marcus", "Elena", "Rohan", "Yuki", "Clara", "Idris",
               "Priya", "Tomas", "Naomi", "Hassan", "Greta", "Kenji", "Lucia",
               "Femi", "Anders", "Meera", "Oscar", "Ingrid", "Rafael"]
LAST_NAMES = ["Varma", "Holloway", "Serrano", "Nakamura", "Okonkwo", "Bergstrom",
              "Duarte", "Whitfield", "Kaur", "Lindqvist", "Abadi", "Moreau",
              "Castellanos", "Fitzgerald", "Ramaswamy", "Novak"]


def month_sequence(n: int) -> list[str]:
    """Return n consecutive YYYY-MM labels starting at START_YEAR/START_MONTH."""
    out = []
    year, month = START_YEAR, START_MONTH
    for _ in range(n):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def make_description(rng: random.Random, words: list[str]) -> str:
    picked = rng.sample(words, k=min(6, len(words)))
    return " ".join(picked)


def make_title(rng: random.Random, words: list[str]) -> str:
    pattern = rng.choice(TITLE_PATTERNS)
    a, b = rng.sample(words, k=2)
    return pattern.format(a=a.capitalize(), b=b.capitalize())


def build_books(rng: random.Random) -> list[dict]:
    genre_names = list(GENRES)
    books = []
    for book_id in range(1, N_BOOKS + 1):
        genre = genre_names[book_id % len(genre_names)]
        meta = GENRES[genre]
        books.append({
            "book_id": book_id,
            "title": make_title(rng, meta["words"]),
            "author": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "genre": genre,
            "description": make_description(rng, meta["words"]),
            "price": round(rng.uniform(149, 899), 2),
            "published_year": rng.randint(2005, 2024),
            "rating": round(rng.uniform(3.0, 4.9), 1),
        })
    return books


def build_sales(rng: random.Random, books: list[dict]) -> list[dict]:
    """Monthly units per book = trend x seasonality x popularity, plus noise.

    The three components are deliberately separable so the forecasting model has
    real structure to find rather than pure noise.
    """
    months = month_sequence(N_MONTHS)
    rows = []
    for book in books:
        meta = GENRES[book["genre"]]
        # Per-title popularity multiplier, lognormal-ish: a few big sellers,
        # a long tail of modest ones.
        popularity = math.exp(rng.gauss(0, 0.45))
        # Some titles are growing, some decaying.
        trend_per_month = rng.uniform(-0.010, 0.018)

        for idx, month_label in enumerate(months):
            month_num = int(month_label.split("-")[1])
            # Seasonality: cosine peaking at the genre's peak month.
            offset = (month_num - meta["peak_month"]) * (2 * math.pi / 12)
            seasonal = 1.0 + 0.35 * math.cos(offset)
            trend = 1.0 + trend_per_month * idx
            expected = meta["base"] * popularity * seasonal * trend
            noise = rng.gauss(1.0, 0.12)
            units = max(0, int(round(expected * noise)))
            rows.append({
                "book_id": book["book_id"],
                "month": month_label,
                "units_sold": units,
            })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    books = build_books(rng)
    sales = build_sales(rng, books)

    write_csv(DATA_DIR / "books.csv", books,
              ["book_id", "title", "author", "genre", "description",
               "price", "published_year", "rating"])
    write_csv(DATA_DIR / "sales.csv", sales, ["book_id", "month", "units_sold"])

    total_units = sum(r["units_sold"] for r in sales)
    print(f"books.csv  -> {len(books)} rows")
    print(f"sales.csv  -> {len(sales)} rows "
          f"({N_BOOKS} books x {N_MONTHS} months, {total_units:,} units total)")


if __name__ == "__main__":
    main()
