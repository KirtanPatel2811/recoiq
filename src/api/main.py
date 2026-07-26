"""
main.py — FastAPI REST API for RecoIQ recommendations.

Endpoints:
  GET  /recommend/{user_id}          — top-K recommendations
  GET  /recommend/{user_id}/explain  — recs + explanation
  GET  /similar/{movie_id}           — similar movies
  GET  /model/compare                — leaderboard metrics
  POST /recommend/new_user           — cold start
  GET  /health                       — health check
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.loader import load_movies
from src.data.splitter import load_splits
from src.data.interaction_matrix import InteractionMatrix
from src.models.popularity import PopularityRecommender
from src.models.als_model import ALSRecommender
from src.models.ncf_model import NCFRecommender
from src.models.svd_model import SVDRecommender

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Global state (loaded once at startup) ────────────────────────────────────
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data at startup."""
    logger.info("Loading models and data...")

    STATE["movies"]  = load_movies("data/raw")
    _, _, STATE["test"] = load_splits("data/processed")
    STATE["im"]      = InteractionMatrix.load("data/processed/interaction_matrix.pkl")

    STATE["models"] = {
        "popularity": PopularityRecommender.load("models/popularity.pkl"),
        "svd":        SVDRecommender.load("models/svd.pkl"),
        "als":        ALSRecommender.load("models/als.pkl"),
        "ncf":        NCFRecommender.load("models/ncf.pkl"),
    }

    # Load evaluation results if available
    eval_path = Path("reports/evaluation_results.json")
    if eval_path.exists():
        with open(eval_path) as f:
            STATE["eval_results"] = json.load(f)
    else:
        STATE["eval_results"] = {}

    # Build train seen-items lookup
    train, _, _ = load_splits("data/processed")
    STATE["seen_lookup"] = (
        train.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    logger.info("All models loaded. API ready!")
    yield
    STATE.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RecoIQ API",
    description="Personalized Movie Recommendation Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class MovieRec(BaseModel):
    movie_id:    int
    title:       str
    genres:      str
    score:       float
    rank:        int

class RecommendResponse(BaseModel):
    user_id:  int
    model:    str
    k:        int
    recommendations: list[MovieRec]

class NewUserRequest(BaseModel):
    ratings: dict[int, float]   # {movie_id: rating}
    k: int = 10
    model: str = "ncf"

class SimilarMoviesResponse(BaseModel):
    movie_id:    int
    title:       str
    similar:     list[MovieRec]


# ── Helpers ───────────────────────────────────────────────────────────────────
def enrich_recs(recs_df: pd.DataFrame, score_col: str = "predicted_score") -> list[MovieRec]:
    """Join recommendations with movie metadata."""
    movies = STATE["movies"]
    merged = recs_df.merge(
        movies[["movie_id", "title", "genres"]],
        on="movie_id", how="left"
    )
    result = []
    for _, row in merged.iterrows():
        result.append(MovieRec(
            movie_id = int(row["movie_id"]),
            title    = str(row.get("title", "Unknown")),
            genres   = str(row.get("genres", "")),
            score    = float(row.get(score_col, 0.0)),
            rank     = int(row.get("rank", 0)),
        ))
    return result


def get_model(model_name: str):
    models = STATE["models"]
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. "
                   f"Available: {list(models.keys())}"
        )
    return models[model_name]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(STATE.get("models", {}).keys()),
        "n_users":       STATE["im"].n_users if "im" in STATE else 0,
        "n_items":       STATE["im"].n_items if "im" in STATE else 0,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: int,
    k:       int = Query(default=10, ge=1, le=50),
    model:   str = Query(default="ncf"),
):
    """
    Get top-K personalised recommendations for a user.

    - **user_id**: MovieLens user ID (1-6040)
    - **k**: number of recommendations (1-50)
    - **model**: one of popularity, svd, als, ncf
    """
    rec_model = get_model(model)
    seen      = STATE["seen_lookup"].get(user_id, set())

    recs = rec_model.recommend(user_id=user_id, k=k, seen_movie_ids=seen)
    if recs.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations for user {user_id}. "
                   f"User may not exist or is a cold-start user."
        )

    return RecommendResponse(
        user_id         = user_id,
        model           = model,
        k               = k,
        recommendations = enrich_recs(recs),
    )


@app.get("/recommend/{user_id}/explain")
def recommend_explain(
    user_id: int,
    k:       int = Query(default=10, ge=1, le=50),
):
    """
    Get recommendations with explanations for WHY each movie was recommended.
    Uses NCF as the primary model.
    """
    rec_model = get_model("ncf")
    seen      = STATE["seen_lookup"].get(user_id, set())
    movies    = STATE["movies"]

    recs = rec_model.recommend(user_id=user_id, k=k, seen_movie_ids=seen)
    if recs.empty:
        raise HTTPException(status_code=404,
                            detail=f"No recommendations for user {user_id}")

    # Get user's top-rated movies as explanation seed
    im        = STATE["im"]
    top_rated = []
    if user_id in im.user_encoder:
        user_idx = im.user_encoder[user_id]
        row      = im.matrix.getrow(user_idx)
        rated    = {im.item_decoder[c]: float(v)
                    for c, v in zip(row.indices, row.data)}
        # Top 3 highest rated
        top_rated = sorted(rated.items(), key=lambda x: x[1], reverse=True)[:3]
        top_rated = [
            {"movie_id": mid, "rating": r,
             "title": movies[movies["movie_id"]==mid]["title"].values[0]
             if mid in movies["movie_id"].values else "Unknown"}
            for mid, r in top_rated
        ]

    enriched = enrich_recs(recs)

    return {
        "user_id":         user_id,
        "model":           "ncf",
        "because_you_liked": top_rated,
        "recommendations": [r.dict() for r in enriched],
        "explanation":     (
            f"Recommendations generated using Neural Collaborative Filtering. "
            f"The model learned your taste profile from your rating history "
            f"and found movies with similar latent features."
        ),
    }


@app.get("/similar/{movie_id}", response_model=SimilarMoviesResponse)
def similar_movies(
    movie_id: int,
    k:        int = Query(default=10, ge=1, le=50),
):
    """
    Find K movies most similar to a given movie.
    Uses ALS item factors for similarity.
    """
    als_model = get_model("als")
    movies    = STATE["movies"]

    movie_info = movies[movies["movie_id"] == movie_id]
    if movie_info.empty:
        raise HTTPException(status_code=404,
                            detail=f"Movie {movie_id} not found")

    similar = als_model.similar_items(movie_id=movie_id, k=k)
    if similar.empty:
        raise HTTPException(status_code=404,
                            detail=f"No similar movies found for {movie_id}")

    similar["rank"] = range(1, len(similar) + 1)
    similar = similar.rename(columns={"score": "predicted_score"})

    return SimilarMoviesResponse(
        movie_id = movie_id,
        title    = str(movie_info["title"].values[0]),
        similar  = enrich_recs(similar),
    )


@app.get("/model/compare")
def model_compare():
    """Return evaluation metrics for all models side by side."""
    eval_results = STATE.get("eval_results", {})
    if not eval_results:
        return {"message": "No evaluation results found. Run evaluator.py first."}
    return {
        "metrics":     eval_results,
        "description": "All metrics computed on MovieLens 1M test set (temporal split)",
        "k":           10,
    }


@app.post("/recommend/new_user")
def recommend_new_user(request: NewUserRequest):
    """
    Cold-start recommendation for a new user.
    Send their ratings for a few movies, get personalised recommendations.
    Falls back to popularity if too few ratings provided.
    """
    movies  = STATE["movies"]
    k       = request.k
    ratings = request.ratings   # {movie_id: rating}

    if len(ratings) < 3:
        # Too few ratings — fall back to popularity
        pop_model = get_model("popularity")
        seen      = set(ratings.keys())
        recs      = pop_model.recommend(user_id=-1, k=k, seen_movie_ids=seen)
        enriched  = enrich_recs(recs, score_col="score")
        return {
            "strategy":      "popularity_fallback",
            "reason":        f"Only {len(ratings)} ratings provided. "
                             f"Need at least 3 for personalised recommendations.",
            "recommendations": [r.dict() for r in enriched],
        }

    # With enough ratings, use SVD to find similar users
    # (True cold-start would require retraining — this is a practical approximation)
    pop_model = get_model("popularity")
    seen      = set(ratings.keys())
    recs      = pop_model.recommend(user_id=-1, k=k * 2, seen_movie_ids=seen)

    # Re-rank by genre overlap with highly-rated input movies
    liked_movies = [mid for mid, r in ratings.items() if r >= 4.0]
    liked_genres = set()
    for mid in liked_movies:
        movie_row = movies[movies["movie_id"] == mid]
        if not movie_row.empty:
            genres = movie_row["genres"].values[0].split("|")
            liked_genres.update(genres)

    # Boost movies matching liked genres
    recs = recs.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
    recs["genre_boost"] = recs["genres"].apply(
        lambda g: len(set(str(g).split("|")) & liked_genres) * 0.1
        if pd.notna(g) else 0
    )
    recs["final_score"] = recs["score"] + recs["genre_boost"]
    recs = recs.sort_values("final_score", ascending=False).head(k)
    recs["rank"] = range(1, len(recs) + 1)
    recs["predicted_score"] = recs["final_score"]

    enriched = enrich_recs(recs)
    return {
        "strategy":      "genre_boosted_popularity",
        "liked_genres":  list(liked_genres),
        "n_ratings":     len(ratings),
        "recommendations": [r.dict() for r in enriched],
    }


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=2)):
    """Search movies by title."""
    movies  = STATE["movies"]
    results = movies[movies["title"].str.contains(q, case=False, na=False)]
    return {
        "query":   q,
        "results": results[["movie_id", "title", "genres", "year"]]
                   .head(20)
                   .to_dict(orient="records")
    }