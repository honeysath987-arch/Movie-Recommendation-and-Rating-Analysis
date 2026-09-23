"""
train.py
--------
Loads data/movies.csv, cleans it, builds a content-based recommendation
model (TF-IDF over movie overviews), and saves everything the app needs
to models/movie_similarity.pkl.

Run:
    python train.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path("data/movies.csv")
MODEL_PATH = Path("models/movie_similarity.pkl")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle movies CSV."""
    df = pd.read_csv(path)

    # Drop the unnamed index column Kaggle exports include, if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing text, types, and duplicates."""
    df = df.copy()

    # Overview is the text we recommend on -> fill missing with empty string
    df["overview"] = df["overview"].fillna("")

    # Parse release_date, pull out release_year for analysis/filtering
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # Make sure numeric columns are numeric
    for col in ["popularity", "vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that lost their id/title entirely (shouldn't normally happen)
    df = df.dropna(subset=["id", "title"])

    # Keep the first occurrence of each movie id (id is the real unique key;
    # title alone is NOT unique in this dataset)
    df = df.drop_duplicates(subset="id", keep="first")

    # Clean row index 0..n-1 -- this row position is what the similarity
    # matrix / TF-IDF matrix rows line up with
    df = df.reset_index(drop=True)

    return df


def build_tfidf(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer over the movie overviews."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(df["overview"])
    return vectorizer, tfidf_matrix


def build_title_index(df: pd.DataFrame) -> dict:
    """
    Map a lowercased title -> row position(s).
    Titles can repeat (remakes, foreign titles, etc.), so each value is a
    list of row positions, most-popular first.
    """
    index_map: dict[str, list[int]] = {}
    order = df.sort_values("popularity", ascending=False).index

    for pos in order:
        title_key = str(df.loc[pos, "title"]).strip().lower()
        index_map.setdefault(title_key, []).append(pos)

    return index_map


def main():
    print("Loading data...")
    raw_df = load_data()
    print(f"  raw shape: {raw_df.shape}")

    print("Cleaning data...")
    df = clean_data(raw_df)
    print(f"  cleaned shape: {df.shape}")

    print("Building TF-IDF matrix over movie overviews...")
    vectorizer, tfidf_matrix = build_tfidf(df)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("Building title lookup index...")
    title_index = build_title_index(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model artifacts to {MODEL_PATH} ...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "df": df,
                "title_index": title_index,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Done.")
    print(f"  Movies: {len(df)}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
