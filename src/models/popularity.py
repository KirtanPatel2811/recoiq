"""
popularity.py — Non-personalised popularity baseline.

WHY this matters: A good popularity baseline is embarrassingly hard to beat.
Netflix found that simple popularity outperforms complex models for new users.
Always include it — if your fancy model doesn't beat this, something is wrong.

Strategy: recommend the top-N globally most-rated movies to everyone.
Optionally filter by genre for light personalisation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class PopularityRecommender:
    """
    Recommends the globally most popular (most-rated) movies.

    This is the non-personalised baseline. Every user gets the same list,
    minus movies they've already seen.
    """

    def __init__(self, min_ratings: int = 50):
        """
        Parameters
        ----------
        min_ratings : minimum number of ratings a movie must have to be recommended.
                      Filters out obscure movies with 1-2 ratings that skew scores.
        """
        self.min_ratings  = min_ratings
        self.popularity_df = None   # DataFrame: movie_id, rating_count, avg_rating, score
        self.is_fitted     = False

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """
        Compute popularity scores from training ratings.

        Score = rating_count (pure popularity).
        We intentionally do NOT use avg_rating alone — a movie with 3 ratings
        of 5.0 is not more popular than one with 50,000 ratings of 4.5.
        """
        stats = (
            train_df
            .groupby("movie_id")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )

        # Filter low-count movies
        stats = stats[stats["rating_count"] >= self.min_ratings].copy()

        # Score: primarily by count, tiebreak by avg rating
        # Normalise both to [0,1] then combine 0.7/0.3
        stats["count_norm"]  = stats["rating_count"] / stats["rating_count"].max()
        stats["rating_norm"] = stats["avg_rating"]   / 5.0
        stats["score"]       = 0.7 * stats["count_norm"] + 0.3 * stats["rating_norm"]

        self.popularity_df = stats.sort_values("score", ascending=False).reset_index(drop=True)
        self.is_fitted = True

        logger.info(f"PopularityRecommender fitted: {len(self.popularity_df)} eligible movies")
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        seen_movie_ids: set | None = None,
    ) -> pd.DataFrame:
        """
        Return top-K recommendations for a user.

        Parameters
        ----------
        user_id       : used only to filter seen items (model is non-personalised)
        k             : number of recommendations
        seen_movie_ids: set of movie_ids the user has already rated — these are excluded

        Returns
        -------
        DataFrame with columns: [movie_id, score, rank]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        df = self.popularity_df.copy()

        if seen_movie_ids:
            df = df[~df["movie_id"].isin(seen_movie_ids)]

        top_k = df.head(k).copy()
        top_k["rank"] = range(1, len(top_k) + 1)

        return top_k[["movie_id", "score", "rank"]]

    def recommend_batch(
        self,
        user_ids: list[int],
        k: int = 10,
        train_df: pd.DataFrame | None = None,
    ) -> dict[int, pd.DataFrame]:
        """
        Recommend for multiple users at once.
        If train_df is provided, automatically filters seen movies per user.
        """
        # Build seen-movies lookup
        seen = {}
        if train_df is not None:
            seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

        return {
            uid: self.recommend(uid, k=k, seen_movie_ids=seen.get(uid))
            for uid in user_ids
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"PopularityRecommender saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PopularityRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_movies
    from src.data.splitter import load_splits

    train, val, test = load_splits("data/processed")
    movies = load_movies("data/raw")

    model = PopularityRecommender(min_ratings=50)
    model.fit(train)

    # Demo: recommend top 10 for user 1
    seen = set(train[train["user_id"] == 1]["movie_id"])
    recs = model.recommend(user_id=1, k=10, seen_movie_ids=seen)

    # Join with movie titles for readability
    recs = recs.merge(movies[["movie_id", "title"]], on="movie_id")
    print("\nTop 10 globally popular movies (unseen by user 1):")
    print(recs[["rank", "title", "score"]].to_string(index=False))

    model.save("models/popularity.pkl")