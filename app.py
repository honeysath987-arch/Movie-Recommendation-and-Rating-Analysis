"""
app.py
------
Streamlit web app for the Movie Recommendation and Rating Analysis project.

Run:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from recommendation import find_movie_index, get_most_popular, get_recommendations, get_top_rated

MODEL_PATH = Path("models/movie_similarity.pkl")

st.set_page_config(page_title="Movie Recommendation & Rating Analysis", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("🎬 Movie Recommendation & Rating Analysis")

if model is None:
    st.error(
        "No trained model found at `models/movie_similarity.pkl`. "
        "Run `python train.py` first, then restart the app."
    )
    st.stop()

df = model["df"]

tab_recommend, tab_analysis, tab_browse = st.tabs(
    ["🔎 Get Recommendations", "📊 Rating Analysis", "🏆 Top Movies"]
)

# ---------------------------------------------------------------- #
# Tab 1: Recommendations
# ---------------------------------------------------------------- #
with tab_recommend:
    st.subheader("Find movies similar to one you like")

    all_titles = sorted(df["title"].unique().tolist())
    col1, col2 = st.columns([3, 1])

    with col1:
        movie_title = st.selectbox(
            "Pick a movie (or type to search):",
            options=all_titles,
            index=all_titles.index("The Godfather") if "The Godfather" in all_titles else 0,
        )
    with col2:
        top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    if st.button("Recommend", type="primary"):
        try:
            recs = get_recommendations(movie_title, model, top_n=top_n)
        except ValueError as e:
            st.warning(str(e))
        else:
            idx = find_movie_index(movie_title, model)
            matched = df.loc[idx]

            st.markdown(f"### Because you watched *{matched['title']}* ({int(matched['release_year']) if pd.notna(matched['release_year']) else '—'})")
            st.caption(matched["overview"])
            st.divider()

            for _, row in recs.iterrows():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    c1.markdown(f"**{row['title']}**  \n{row['overview'][:220]}{'...' if len(row['overview']) > 220 else ''}")
                    year = int(row["release_year"]) if pd.notna(row["release_year"]) else "—"
                    c2.metric("Rating", f"{row['vote_average']:.1f}")
                    c3.metric("Match", f"{row['similarity'] * 100:.0f}%")

# ---------------------------------------------------------------- #
# Tab 2: Rating Analysis
# ---------------------------------------------------------------- #
with tab_analysis:
    st.subheader("Explore the dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total movies", f"{len(df):,}")
    c2.metric("Avg. rating", f"{df['vote_average'].mean():.2f}")
    c3.metric("Avg. popularity", f"{df['popularity'].mean():.1f}")
    c4.metric("Year range", f"{int(df['release_year'].min())}–{int(df['release_year'].max())}")

    left, right = st.columns(2)

    with left:
        fig1 = px.histogram(
            df, x="vote_average", nbins=30,
            title="Distribution of Vote Average",
            labels={"vote_average": "Vote Average"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        fig2 = px.scatter(
            df, x="popularity", y="vote_average",
            hover_name="title", opacity=0.4,
            title="Popularity vs. Vote Average",
            labels={"popularity": "Popularity", "vote_average": "Vote Average"},
        )
        fig2.update_xaxes(type="log")
        st.plotly_chart(fig2, use_container_width=True)

    movies_per_year = (
        df.dropna(subset=["release_year"])
        .groupby("release_year")
        .size()
        .reset_index(name="count")
    )
    movies_per_year = movies_per_year[movies_per_year["release_year"] >= 1960]
    fig3 = px.bar(
        movies_per_year, x="release_year", y="count",
        title="Movies per Release Year (1960+)",
        labels={"release_year": "Release Year", "count": "Number of Movies"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------- #
# Tab 3: Top Movies
# ---------------------------------------------------------------- #
with tab_browse:
    st.subheader("Browse the best of the dataset")

    left, right = st.columns(2)

    with left:
        st.markdown("#### ⭐ Top Rated (1,000+ votes)")
        st.dataframe(get_top_rated(model, top_n=15), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### 🔥 Most Popular")
        st.dataframe(get_most_popular(model, top_n=15), use_container_width=True, hide_index=True)
