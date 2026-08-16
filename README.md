# Book Demand Predictor & Recommendation System

A Flask application that does two things over a book catalogue: **recommends similar titles** using content-based filtering, and **forecasts monthly demand** using a regression model validated against a naive baseline. Results are surfaced in an admin dashboard built with Chart.js.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/flask-3.x-lightgrey)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)

---

## What it does

**Recommendation.** Each book is flattened into a content document (genre, author, description keywords, title), vectorised with TF-IDF, and compared by cosine similarity. Cosine is used rather than a raw dot product because it is magnitude-invariant — a book with a longer description shouldn't look similar to everything simply because its vector is larger. Only the mix of terms matters, not its length.

The same fitted vectoriser powers free-text search, so a query lands in exactly the same vector space as the catalogue.

**Demand forecasting.** Monthly sales are reframed as a supervised regression problem. For each `(book, month)` the model sees the previous three months, a three-month rolling mean, sin/cos of the month number, and the book's own historical average.

Forecasting runs recursively — each predicted month becomes the next month's lag feature — so error compounds across the horizon, and later months are genuinely less certain than earlier ones.

---

## Results

Measured on a **temporal** holdout: the final 6 months (1,800 observations) are never seen during training. A random train/test split would leak future information and produce a dishonestly good score.

| Metric | Ridge model | Naive baseline |
|---|---:|---:|
| Mean absolute error (units) | **20.61** | 24.00 |
| Mean absolute percentage error | **15.67%** | 16.95% |
| R² on held-out data | **0.8827** | — |

The naive baseline is *"next month equals last month"* — a genuinely hard benchmark for monthly sales data. The model beats it by **14.1% on MAE**. That margin is deliberately reported rather than hidden: most of the signal in this data is autocorrelation, and a model that can't beat a one-line baseline isn't earning its complexity.

Trained on 8,100 observations, tested on 1,800.

### What the model learned

The fitted coefficients are interpretable, which is a large part of why Ridge was chosen here:

| Feature | Weight |
|---|---:|
| `lag_1` | 0.7515 |
| `month_sin` | −5.4045 |
| `month_cos` | 2.6251 |
| `book_mean` | 0.4071 |
| `roll_3` | 0.1428 |
| `lag_2` | 0.1164 |
| `lag_3` | −0.4394 |

Last month dominates, seasonality carries real weight, and the negative `lag_3` term acts as a mean-reversion correction against the positive `lag_1`.

**Why Ridge and not plain least squares:** the lag features are strongly collinear — last month, two months ago, and the rolling mean all encode overlapping information. Unregularised regression responds by inflating coefficients in opposing directions. L2 keeps them stable and interpretable.

---

## Screens

- **Catalogue** (`/`) — best-sellers, genre filter, and TF-IDF free-text search
- **Book detail** (`/book/<id>`) — 36 months of actual demand plus a 6-month forecast on one chart, and six similar titles with their similarity scores
- **Admin dashboard** (`/dashboard`) — catalogue KPIs, monthly trend, sales by genre, and the model's held-out performance against the baseline

---

## The data

The catalogue is **synthetic and seeded** (`SEED = 20260814`), so cloning the repo gives a working demo with no external download, no licensing questions, and byte-identical CSVs — every number in this README reproduces exactly.

`scripts/generate_data.py` builds 300 books across 8 genres and 36 months of sales (10,800 rows, ~1.39M units). Demand is composed of three separable components — a per-title popularity multiplier, a per-title linear trend, and genre-specific seasonality peaking in a different month per genre — plus Gaussian noise. They're separable on purpose: the forecasting model has real structure to find rather than pure noise, and the seasonality visible in the dashboard's trend chart is the signal the generator put there.

Genre is never given to the recommender as a standalone feature. That same-genre books cluster together is something it rediscovers from vocabulary overlap.

---

## Running it

```bash
git clone https://github.com/vijaxx/book-demand-predictor.git
cd book-demand-predictor

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py     # writes data/books.csv and data/sales.csv
python app.py                       # http://127.0.0.1:5000
```

Run the tests:

```bash
pip install pytest && python -m pytest tests/ -q
```

---

## JSON API

| Endpoint | Returns |
|---|---|
| `GET /api/forecast/<book_id>` | 36 months of history plus a 6-month forecast |
| `GET /api/recommend/<book_id>` | Six most similar books with scores |
| `GET /api/genre-sales` | Total units per genre |
| `GET /api/monthly-trend` | Catalogue-wide units per month |

---

## Tests

21 tests covering both components. The ones that matter most are the leakage guards:

- `test_lag_features_do_not_leak_the_target` — a row's `lag_1` must equal that book's actual previous-month sales
- `test_holdout_is_temporal_not_random` — the split boundary must be the 6th-from-last month
- `test_model_beats_the_naive_baseline` — fails if the model stops earning its complexity
- `test_never_recommends_the_book_itself`, `test_scores_stay_within_cosine_bounds`, `test_forecast_is_never_negative`

---

## Stack

Python · Flask · pandas · scikit-learn (TF-IDF, Ridge, cosine similarity) · Chart.js · pytest

---

## Scope and honest limitations

- The dataset is **synthetic**, not real sales data. The modelling approach, validation methodology, and metrics are real; the underlying numbers are generated.
- Data is served from CSV via pandas, held in memory. At 300 books that's the right call; a real catalogue would want a database and incremental retraining.
- The model is retrained at process start, not on a schedule, and there is no model persistence layer.
- Recommendations are purely content-based. With real user data, collaborative filtering would likely outperform this — content-based filtering's advantage is that it works from day one with no interaction history (no cold-start problem).

---

**Kondani Vijay Vardhan** · [GitHub](https://github.com/vijaxx) · [LinkedIn](https://www.linkedin.com/in/kondani-vijay-vardhan-b2729035a/)

---

## License

MIT.
