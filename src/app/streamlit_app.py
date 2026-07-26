"""
streamlit_app.py — RecoIQ Streamlit Dashboard.

Screens:
  1. Home          — project overview + dataset stats
  2. For You       — personalised recommendations by user ID
  3. Similar Movies — item-item similarity
  4. New User      — cold start demo
  5. Model Compare — leaderboard table + charts
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

from src.data.loader import load_movies, load_all
from src.data.splitter import load_splits
from src.data.interaction_matrix import InteractionMatrix
from src.models.popularity import PopularityRecommender
from src.models.svd_model import SVDRecommender
from src.models.als_model import ALSRecommender
from src.models.ncf_model import NCFRecommender


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecoIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load everything once (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_data():
    data      = load_all("data/raw")
    train, val, test = load_splits("data/processed")
    im        = InteractionMatrix.load("data/processed/interaction_matrix.pkl")
    seen_lookup = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    return data, train, test, im, seen_lookup


@st.cache_resource
def load_models():
    return {
        "Popularity": PopularityRecommender.load("models/popularity.pkl"),
        "SVD":        SVDRecommender.load("models/svd.pkl"),
        "ALS":        ALSRecommender.load("models/als.pkl"),
        "NCF":        NCFRecommender.load("models/ncf.pkl"),
    }


data, train, test, im, seen_lookup = load_data()
models = load_models()
movies = data["movies"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/film-reel.png", width=60)
st.sidebar.title("RecoIQ 🎬")
st.sidebar.markdown("*Personalized Movie Recommendations*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎯 For You", "🎬 Similar Movies",
     "👤 New User", "📊 Model Comparison"],
)

st.sidebar.divider()
st.sidebar.caption("MovieLens 1M · 6,040 users · 3,706 movies")


# ── Helper ────────────────────────────────────────────────────────────────────
def enrich(recs_df: pd.DataFrame, score_col: str = "predicted_score") -> pd.DataFrame:
    return recs_df.merge(
        movies[["movie_id", "title", "genres", "year"]],
        on="movie_id", how="left"
    )


# ── Page 1: Home ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🎬 RecoIQ — Personalized Movie Recommendation Engine")
    st.markdown("""
    A production-grade recommender system built on **MovieLens 1M** comparing
    5 algorithms from simple baselines to Neural Collaborative Filtering.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings",  "1,000,209")
    col2.metric("Users",          "6,040")
    col3.metric("Movies",         "3,706")
    col4.metric("Matrix Density", "4.47%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Models Built")
        st.markdown("""
        | Model | Type | Key Idea |
        |---|---|---|
        | Popularity | Baseline | Most-rated globally |
        | User-CF | Memory-based | Similar users |
        | Item-CF | Memory-based | Similar items |
        | SVD | Matrix Factorisation | Latent factors |
        | ALS | Matrix Factorisation | Implicit feedback |
        | NCF | Neural | Embeddings + MLP |
        """)

    with col_b:
        st.subheader("Ratings Distribution")
        rating_counts = train["rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(showlegend=False, height=250, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Most Rated Movies")
    top_movies = (
        train.groupby("movie_id")
        .agg(n_ratings=("rating","count"), avg_rating=("rating","mean"))
        .reset_index()
        .merge(movies[["movie_id","title","genres"]], on="movie_id")
        .sort_values("n_ratings", ascending=False)
        .head(10)
    )
    st.dataframe(
        top_movies[["title","genres","n_ratings","avg_rating"]]
        .rename(columns={"n_ratings":"# Ratings","avg_rating":"Avg Rating",
                         "title":"Title","genres":"Genres"})
        .reset_index(drop=True),
        use_container_width=True,
    )


# ── Page 2: For You ───────────────────────────────────────────────────────────
elif page == "🎯 For You":
    st.title("🎯 Personalised Recommendations")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=6040,
                                  value=1, step=1)
    with col2:
        model_name = st.selectbox("Model", list(models.keys()))
    with col3:
        k = st.slider("Top-K", 5, 20, 10)

    if st.button("🎬 Get Recommendations", type="primary"):
        model = models[model_name]
        seen  = seen_lookup.get(user_id, set())

        with st.spinner("Generating recommendations..."):
            recs = model.recommend(user_id=user_id, k=k, seen_movie_ids=seen)

        if recs.empty:
            st.error(f"No recommendations for user {user_id}")
        else:
            recs = enrich(recs)
            st.success(f"Top {k} recommendations for User {user_id} via {model_name}")

            for _, row in recs.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([1, 5, 2])
                    c1.markdown(f"### #{int(row['rank'])}")
                    c2.markdown(f"**{row['title']}**  \n*{row['genres']}*")
                    score = row.get("predicted_score", row.get("score", 0))
                    c3.metric("Score", f"{score:.3f}")

        # Show what user has rated
        st.divider()
        st.subheader(f"User {user_id}'s Rating History (sample)")
        user_hist = train[train["user_id"] == user_id].merge(
            movies[["movie_id","title","genres"]], on="movie_id"
        ).sort_values("rating", ascending=False).head(10)

        if not user_hist.empty:
            st.dataframe(
                user_hist[["title","genres","rating"]]
                .rename(columns={"title":"Title","genres":"Genres","rating":"Your Rating"})
                .reset_index(drop=True),
                use_container_width=True,
            )


# ── Page 3: Similar Movies ────────────────────────────────────────────────────
elif page == "🎬 Similar Movies":
    st.title("🎬 Similar Movies")
    st.markdown("Find movies similar to one you love — powered by ALS item factors.")

    movie_titles = movies["title"].sort_values().tolist()
    selected_title = st.selectbox("Pick a movie", movie_titles, index=movie_titles.index("Toy Story (1995)"))
    k = st.slider("Number of similar movies", 5, 20, 10)

    selected_movie = movies[movies["title"] == selected_title].iloc[0]
    movie_id       = int(selected_movie["movie_id"])

    if st.button("🔍 Find Similar", type="primary"):
        als = models["ALS"]
        similar = als.similar_items(movie_id=movie_id, k=k)

        if similar.empty:
            st.warning("No similar movies found. Movie may not be in the training set.")
        else:
            similar = similar.merge(movies[["movie_id","title","genres"]], on="movie_id")
            st.success(f"Movies similar to **{selected_title}**")

            st.dataframe(
                similar[["title","genres","score"]]
                .rename(columns={"title":"Title","genres":"Genres","score":"Similarity"})
                .reset_index(drop=True),
                use_container_width=True,
            )

            fig = px.bar(
                similar.head(10),
                x="score", y="title",
                orientation="h",
                labels={"score":"Similarity Score","title":""},
                color="score",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=400, showlegend=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


# ── Page 4: New User ──────────────────────────────────────────────────────────
elif page == "👤 New User":
    st.title("👤 New User — Cold Start Demo")
    st.markdown("""
    Rate a few movies to get personalised recommendations instantly.
    This simulates what happens when a brand new user joins the platform.
    """)

    seed_movies = [
        "Toy Story (1995)", "GoodFellas (1990)", "Pulp Fiction (1994)",
        "The Silence of the Lambs (1991)", "Forrest Gump (1994)",
        "The Matrix (1999)", "Schindler's List (1993)",
        "Titanic (1997)", "The Lion King (1994)", "Jurassic Park (1993)",
    ]

    # Filter to movies that actually exist in our dataset
    available = movies[movies["title"].isin(seed_movies)]["title"].tolist()

    st.subheader("Rate these movies (0 = haven't seen)")
    user_ratings = {}
    cols = st.columns(2)
    for i, title in enumerate(available):
        with cols[i % 2]:
            rating = st.slider(title, 0.0, 5.0, 0.0, 0.5, key=f"rating_{i}")
            if rating > 0:
                mid = int(movies[movies["title"] == title]["movie_id"].values[0])
                user_ratings[mid] = rating

    st.markdown(f"*You've rated {len(user_ratings)} movies*")

    if st.button("🎯 Get My Recommendations", type="primary"):
        if not user_ratings:
            st.warning("Please rate at least one movie!")
        else:
            pop = models["Popularity"]
            seen = set(user_ratings.keys())

            liked_genres = set()
            for mid, r in user_ratings.items():
                if r >= 4.0:
                    row = movies[movies["movie_id"] == mid]
                    if not row.empty:
                        liked_genres.update(row["genres"].values[0].split("|"))

            recs = pop.recommend(user_id=-1, k=20, seen_movie_ids=seen)
            recs = recs.merge(movies[["movie_id","genres","title"]], on="movie_id")

            if liked_genres:
                recs["boost"] = recs["genres"].apply(
                    lambda g: len(set(str(g).split("|")) & liked_genres) * 0.1
                )
                recs["score"] = recs["score"] + recs["boost"]
                recs = recs.sort_values("score", ascending=False)

            recs = recs.head(10).reset_index(drop=True)
            recs["rank"] = range(1, len(recs) + 1)

            if liked_genres:
                st.info(f"Detected genre preferences: {', '.join(sorted(liked_genres))}")

            st.success("Your personalised recommendations:")
            st.dataframe(
                recs[["rank","title","genres","score"]]
                .rename(columns={"rank":"#","title":"Title",
                                 "genres":"Genres","score":"Score"})
                .reset_index(drop=True),
                use_container_width=True,
            )


# ── Page 5: Model Comparison ──────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison Leaderboard")

    eval_path = Path("reports/evaluation_results.csv")
    if not eval_path.exists():
        st.warning("No evaluation results found. Run `python src/evaluation/evaluator.py` first.")
        st.stop()

    results = pd.read_csv(eval_path)
    results = results.sort_values("ndcg@10", ascending=False)

    st.subheader("Leaderboard — Top-10 Recommendations")
    metric_cols = ["hit_rate@10","precision@10","recall@10","ndcg@10","map@10","coverage@10"]
    display_cols = ["Model"] + [c for c in metric_cols if c in results.columns]

    styled = results[display_cols].reset_index(drop=True)
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # NDCG bar chart
    if "ndcg@10" in results.columns:
        st.subheader("NDCG@10 by Model")
        fig = px.bar(
            results, x="Model", y="ndcg@10",
            color="ndcg@10", color_continuous_scale="Viridis",
            labels={"ndcg@10": "NDCG@10"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # Recall vs Precision scatter
    with col1:
        if "recall@10" in results.columns and "precision@10" in results.columns:
            st.subheader("Recall vs Precision@10")
            fig2 = px.scatter(
                results, x="recall@10", y="precision@10",
                text="Model", color="Model",
                labels={"recall@10":"Recall@10","precision@10":"Precision@10"},
            )
            fig2.update_traces(textposition="top center", marker_size=12)
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Coverage vs NDCG
    with col2:
        if "coverage@10" in results.columns and "ndcg@10" in results.columns:
            st.subheader("Coverage vs NDCG@10")
            fig3 = px.scatter(
                results, x="coverage@10", y="ndcg@10",
                text="Model", color="Model",
                labels={"coverage@10":"Coverage@10","ndcg@10":"NDCG@10"},
            )
            fig3.update_traces(textposition="top center", marker_size=12)
            fig3.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Key Insights")
    st.markdown("""
    - **Popularity beats complex models on NDCG** — a well-known RecSys result.
      Popular movies are popular for a reason: most users haven't seen them and tend to like them.
    - **NCF has the best personalisation** — similar NDCG to Popularity but 7× better Coverage,
      meaning it recommends diverse items tailored to each user rather than the same 20 movies.
    - **ALS excels at Coverage** — designed for implicit feedback, it explores the full catalogue.
    - **SVD underperforms here** — needs more epochs and hyperparameter tuning to shine.
    - **User-CF has highest Coverage** — finds niche films via similar users,
      but struggles with accuracy on this sparse dataset.
    """)