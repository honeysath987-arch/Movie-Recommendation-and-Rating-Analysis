"""
recommendation.py
------------------
Reusable functions for loading the trained model and generating
movie recommendations / rating-analysis helpers. Imported by app.py
and usable standalone from the command line:

    python recommendation.py "The Godfather"
"""

import pickle
import sys
import difflib
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = Path("models/movie_similarity.pkl")


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the pickled vectorizer, TF-IDF matrix, dataframe, and title index."""
    with open(path, "rb") as f:
        return pickle.load(f)


def find_movie_index(title: str, model: dict) -> int | None:
    """
    Resolve a user-typed title to a row position in model['df'].
    Tries an exact (case-insensitive) match first, then falls back to
    fuzzy matching against all known titles.
    """
    title_index = model["title_index"]
    key = title.strip().lower()

    if key in title_index:
        return title_index[key][0]  # most popular movie with that title

    # Fuzzy fallback
    close = difflib.get_close_matches(key, title_index.keys(), n=1, cutoff=0.6)
    if close:
        return title_index[close[0]][0]

    return None


def get_recommendations(title: str, model: dict, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top_n movies most similar (by overview content) to `title`.
    Raises ValueError if the title can't be resolved to a known movie.
    """
    idx = find_movie_index(title, model)
    if idx is None:
        raise ValueError(f"Could not find a movie matching '{title}'.")

    df = model["df"]
    tfidf_matrix = model["tfidf_matrix"]

    # Cosine similarity between this one movie and every other movie.
    # Done on demand (rather than storing a full dense N x N matrix) so the
    # saved model stays small and this scales to large catalogs.
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_positions = sims.argsort()[::-1]
    similar_positions = [p for p in similar_positions if p != idx][:top_n]

    result = df.iloc[similar_positions].copy()
    result["similarity"] = sims[similar_positions]

    cols = ["title", "release_year", "vote_average", "popularity", "similarity", "overview"]
    return result[cols].reset_index(drop=True)


def get_top_rated(model: dict, top_n: int = 10, min_votes: int = 1000) -> pd.DataFrame:
    """Highest vote_average movies with at least min_votes votes (avoids obscure outliers)."""
    df = model["df"]
    filtered = df[df["vote_count"] >= min_votes]
    top = filtered.sort_values("vote_average", ascending=False).head(top_n)
    return top[["title", "release_year", "vote_average", "vote_count", "popularity"]].reset_index(drop=True)


def get_most_popular(model: dict, top_n: int = 10) -> pd.DataFrame:
    """Highest popularity-score movies."""
    df = model["df"]
    top = df.sort_values("popularity", ascending=False).head(top_n)
    return top[["title", "release_year", "vote_average", "popularity"]].reset_index(drop=True)


def _cli():
    if len(sys.argv) < 2:
        print('Usage: python recommendation.py "Movie Title"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    model = load_model()

    try:
        recs = get_recommendations(query, model, top_n=10)
    except ValueError as e:
        print(e)
        sys.exit(1)

    idx = find_movie_index(query, model)
    matched_title = model["df"].loc[idx, "title"]
    print(f"Because you watched: {matched_title}\n")
    for i, row in recs.iterrows():
        print(f"{i + 1}. {row['title']} ({row['release_year']}) "
              f"- rating {row['vote_average']:.1f}, similarity {row['similarity']:.3f}")


if __name__ == "__main__":
    _cli()
