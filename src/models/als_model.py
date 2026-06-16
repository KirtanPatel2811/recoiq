"""
als_model.py — ALS (Alternating Least Squares) via the implicit library.

HOW IT WORKS:
ALS is matrix factorisation for IMPLICIT feedback (watched/not watched).

EXPLICIT vs IMPLICIT feedback — critical interview concept:
  Explicit: user gives a rating (1-5 stars) → we know they liked/disliked it
  Implicit: user watched a movie → we know they engaged, but not how much they felt

Real-world data is mostly IMPLICIT:
  - Netflix: watch history (did they finish? rewatch?)
  - Spotify: play counts, skips
  - Amazon: clicks, purchases (not star ratings)

ALS treats the problem differently from SVD:
  - SVD minimises (predicted_rating - actual_rating)²
  - ALS minimises a WEIGHTED loss: high confidence for rated items,
    low confidence for unrated (not "disliked", just "unobserved")

Confidence formula: c_ui = 1 + alpha * r_ui
  Where alpha=40 means "if user rated movie once, we're 40x more confident
  they like it than an unrated movie"

WHY ALS beats SGD for implicit:
  - ALS alternates: fix P, solve for Q exactly → fix Q, solve for P exactly
  - Each step is a closed-form least squares problem (fast, parallelisable)
  - The implicit library uses C++ with OpenMP under the hood → very fast
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
import logging
import pickle

import implicit

logger = logging.getLogger(__name__)


class ALSRecommender:
    """
    ALS Matrix Factorisation for implicit feedback using the implicit library.

    Treats ALL ratings as implicit positive signals (confidence weighted by rating).
    This is the industry approach: even a 1-star rating means the user engaged.
    """

    def __init__(
        self,
        factors: int          = 100,
        iterations: int       = 20,
        regularization: float = 0.01,
        alpha: float          = 40.0,
        random_state: int     = 42,
    ):
        """
        Parameters
        ----------
        factors        : latent dimension size (same as SVD's n_factors)
        iterations     : ALS alternating steps (converges faster than SGD epochs)
        regularization : L2 penalty (prevents overfitting)
        alpha          : confidence scaling factor.
                         c_ui = 1 + alpha * r_ui
                         Higher alpha = more weight on observed interactions.
        random_state   : reproducibility
        """
        self.factors        = factors
        self.iterations     = iterations
        self.regularization = regularization
        self.alpha          = alpha
        self.random_state   = random_state

        self.model         = None
        self.user_item_csr = None
        self.user_encoder  = {}
        self.item_encoder  = {}
        self.user_decoder  = {}
        self.item_decoder  = {}
        self.is_fitted     = False

    def fit(self, train_df: pd.DataFrame, interaction_matrix) -> "ALSRecommender":
        self.user_encoder = interaction_matrix.user_encoder
        self.item_encoder = interaction_matrix.item_encoder
        self.user_decoder = interaction_matrix.user_decoder
        self.item_decoder = interaction_matrix.item_decoder

        logger.info(f"Building confidence matrix: alpha={self.alpha}")

        df = train_df.copy()
        df["user_idx"] = df["user_id"].map(self.user_encoder)
        df["item_idx"] = df["movie_id"].map(self.item_encoder)

        before = len(df)
        df = df.dropna(subset=["user_idx", "item_idx"])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Skipped {dropped} rows outside interaction matrix "
                        f"(cold-filtered items — expected)")

        df["user_idx"] = df["user_idx"].astype(np.int32)
        df["item_idx"] = df["item_idx"].astype(np.int32)

        rows       = df["user_idx"].values
        cols       = df["item_idx"].values
        confidence = (1.0 + self.alpha * df["rating"].values).astype(np.float32)

        n_users = interaction_matrix.n_users
        n_items = interaction_matrix.n_items

        # Build user-item CSR matrix
        self.user_item_csr = csr_matrix(
            (confidence, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(f"Training ALS: factors={self.factors}, "
                    f"iterations={self.iterations}, reg={self.regularization}")

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
        )

        # implicit expects item-user (items as rows) — pass .T as CSR explicitly
        item_user_csr = self.user_item_csr.T.tocsr()
        self.model.fit(item_user_csr)

        self.is_fitted = True
        logger.info("ALS training complete")
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        seen_movie_ids: set = None,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not in training set (cold start)")
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        user_idx = self.user_encoder[user_id]

        filter_items = None
        if seen_movie_ids:
            filter_arr = [
                self.item_encoder[m] for m in seen_movie_ids
                if m in self.item_encoder
            ]
            if filter_arr:
                filter_items = np.array(filter_arr, dtype=np.int32)

        item_idxs, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_csr[user_idx],
            N=k,
            filter_already_liked_items=True,
            filter_items=filter_items,
        )

        # Only decode indices that exist in item_decoder (safety check)
        results = []
        for idx, score in zip(item_idxs, scores):
            idx_int = int(idx)
            if idx_int in self.item_decoder:
                results.append((self.item_decoder[idx_int], float(score)))

        result = pd.DataFrame(results, columns=["movie_id", "predicted_score"])
        result["rank"] = range(1, len(result) + 1)
        return result

    def similar_items(self, movie_id: int, k: int = 10) -> pd.DataFrame:
        """
        Find K most similar movies using learned ALS item factors.
        Used in the Streamlit 'Similar Movies' screen later.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before similar_items()")

        if movie_id not in self.item_encoder:
            logger.warning(f"Movie {movie_id} not in training set")
            return pd.DataFrame(columns=["movie_id", "score"])

        item_idx = self.item_encoder[movie_id]
        similar_idxs, scores = self.model.similar_items(item_idx, N=k + 10)

        results = []
        for idx, score in zip(similar_idxs, scores):
            idx_int = int(idx)
            if idx_int == item_idx:
                continue
            if idx_int in self.item_decoder:
                results.append((self.item_decoder[idx_int], float(score)))
            if len(results) == k:
                break

        return pd.DataFrame(results, columns=["movie_id", "score"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"ALSRecommender saved to {path}")

    @classmethod
    def load(cls, path) -> "ALSRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_movies
    from src.data.splitter import load_splits
    from src.data.interaction_matrix import InteractionMatrix

    train, val, test = load_splits("data/processed")
    movies = load_movies("data/raw")
    im     = InteractionMatrix.load("data/processed/interaction_matrix.pkl")

    model = ALSRecommender(factors=100, iterations=20, alpha=40.0)
    model.fit(train, im)

    seen = set(train[train["user_id"] == 1]["movie_id"])
    recs = model.recommend(user_id=1, k=10, seen_movie_ids=seen)
    recs = recs.merge(movies[["movie_id", "title"]], on="movie_id")

    print("\nALS top 10 for user 1:")
    print(recs[["rank", "title", "predicted_score"]].to_string(index=False))

    # Bonus: similar movies to The Matrix (movie_id=2571)
    similar = model.similar_items(movie_id=2571, k=5)
    similar = similar.merge(movies[["movie_id", "title"]], on="movie_id")
    print("\nMovies similar to The Matrix (ALS):")
    print(similar[["title", "score"]].to_string(index=False))

    model.save("models/als.pkl")