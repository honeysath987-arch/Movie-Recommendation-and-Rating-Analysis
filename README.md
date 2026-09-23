# 🎬 Movie Recommendation and Rating Analysis Using Machine Learning

A full-stack Python project that recommends similar movies from a plot
description (content-based filtering with TF-IDF + cosine similarity) and
visualizes rating/popularity trends across the dataset.

## Dataset

`data/movies.csv` — 10,000 movies with the following columns:

| Column | Description |
|---|---|
| `id` | Unique movie identifier |
| `title` | Movie title |
| `overview` | Plot summary text |
| `release_date` | Release date |
| `popularity` | Popularity score |
| `vote_average` | Average user rating (out of 10) |
| `vote_count` | Number of votes |

Source: [Kaggle](https://www.kaggle.com/) TMDB-style movie metadata export.
Swap in your own CSV with the same columns to reuse the pipeline.

## How it works

1. **Cleaning** (`train.py`): handles missing overviews, parses release
   dates, deduplicates by movie `id`.
2. **Feature extraction**: a `TfidfVectorizer` (unigrams + bigrams, English
   stop words removed) turns each movie's `overview` into a sparse vector.
3. **Recommendation** (`recommendation.py`): given a movie title, cosine
   similarity is computed on demand between that movie's TF-IDF vector and
   every other movie, and the top-N most similar titles are returned. This
   avoids storing a dense N×N similarity matrix, so the saved model stays
   small and the approach scales to much larger catalogs.
4. **App** (`app.py`): a Streamlit UI with three tabs — get recommendations,
   explore rating/popularity charts, and browse top-rated/most-popular lists.

## Project structure

```
Movie-Recommendation-and-Rating-Analysis/
│
├── data/
│   └── movies.csv                     # Dataset
├── models/
│   └── movie_similarity.pkl           # Trained TF-IDF vectorizer + matrix + cleaned data
├── notebooks/
│   └── MovieRecommendation.ipynb      # EDA, visualizations, model walkthrough
├── train.py                           # Cleans data, builds & saves the model
├── recommendation.py                  # Recommendation + top-rated/popular functions
├── app.py                             # Streamlit web app
├── requirements.txt
├── README.md
└── ProjectReport.docx
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train the model** (run once, or whenever `data/movies.csv` changes):

```bash
python train.py
```

This reads `data/movies.csv`, cleans it, fits the TF-IDF model, and saves
everything to `models/movie_similarity.pkl`.

**2. Get recommendations from the command line:**

```bash
python recommendation.py "The Godfather"
```

**3. Launch the web app:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

**4. Explore the analysis notebook:**

```bash
jupyter notebook notebooks/MovieRecommendation.ipynb
```

## Tech stack

- **pandas / numpy** — data loading and cleaning
- **scikit-learn** — `TfidfVectorizer`, `cosine_similarity`
- **Streamlit** — interactive web app
- **Plotly / Matplotlib / Seaborn** — charts (app + notebook)

## Notes

- Titles are not unique in the raw dataset (remakes, shared names); the
  lookup index keys on the *most popular* movie for a given title, with a
  fuzzy-match fallback (`difflib`) for typos or near-matches.
- `min_df=2` and `max_features=25000` in the TF-IDF vectorizer keep the
  saved model compact (~7–8 MB) while still capturing meaningful terms.

## Possible extensions

- Hybrid scoring that blends content similarity with `vote_average` /
  `popularity`.
- Incorporate genre, cast, or director metadata if available.
- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) for
  more semantic matching.
