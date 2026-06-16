"""
svd_model.py — SVD Matrix Factorisation via the Surprise library.

HOW IT WORKS:
The user-item matrix R (6040 × 3260) is decomposed into:
    R ≈ P × Q^T
where:
    P = user latent factor matrix  (n_users × n_factors)
    Q = item latent factor matrix  (n_items × n_factors)

Each user gets a vector of n_factors numbers (their "taste profile").
Each item gets a vector of n_factors numbers (its "feature profile").
Predicted rating = dot product of user vector and item vector.

WHY this beats CF:
- CF uses raw ratings directly → sparse, noisy
- SVD learns DENSE low-rank representations → generalises better
- Latent factors capture hidden structure: genre preference, director style, etc.
- Works for users with few ratings (CF needs many overlapping items)

SURPRISE LIBRARY:
Surprise implements SVD++ and baseline SVD cleanly.
It handles train/test formatting, cross-validation, and prediction internally.
We wrap it to match our project's interface.

INTERVIEW INSIGHT:
"SVD" here is actually Simon Funk's SGD-based matrix factorisation
(not true SVD decomposition). True SVD is O(mn²) — too slow.
Funk SVD uses stochastic gradient descent to minimise:
    min Σ (r_ui - μ - b_u - b_i - p_u · q_i)²  +  λ(||p||² + ||q||²)
where b_u, b_i are user/item bias terms.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

logger = logging.getLogger(__name__)


class SVDRecommender:
    """
    Matrix Factorisation recommender using Surprise's SVD implementation.

    Wraps Surprise SVD to match the project's recommend() interface.
    """

    def __init__(
        self,
        n_factors: int  = 100,
        n_epochs: int   = 20,
        lr_all: float   = 0.005,
        reg_all: float  = 0.02,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_factors    : number of latent dimensions (50-200 typical)
        n_epochs     : SGD training epochs (more = better fit, slower)
        lr_all       : learning rate for SGD
        reg_all      : L2 regularisation to prevent overfitting
        random_state : reproducibility seed
        """
        self.n_factors    = n_factors
        self.n_epochs     = n_epochs
        self.lr_all       = lr_all
        self.reg_all      = reg_all
        self.random_state = random_state

        self.model        = None
        self.trainset     = None
        self.all_movie_ids = []
        self.is_fitted    = False

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        """
        Train SVD on the training ratings.

        Parameters
        ----------
        train_df : DataFrame with columns [user_id, movie_id, rating]
        """
        logger.info(f"Training SVD: n_factors={self.n_factors}, "
                    f"n_epochs={self.n_epochs}, reg={self.reg_all}")

        # Surprise needs a specific Dataset format
        reader = Reader(rating_scale=(1, 5))
        data   = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]],
            reader,
        )

        # Build full trainset from ALL training data (no val split inside Surprise)
        self.trainset = data.build_full_trainset()

        # Store all movie IDs seen in training (for candidate generation)
        self.all_movie_ids = train_df["movie_id"].unique().tolist()

        # Initialise and train
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
            verbose=True,
        )
        self.model.fit(self.trainset)

        self.is_fitted = True
        logger.info("SVD training complete")
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a single (user, movie) rating."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_rating()")
        pred = self.model.predict(str(user_id), str(movie_id))
        return float(np.clip(pred.est, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        seen_movie_ids: set = None,
    ) -> pd.DataFrame:
        """
        Return top-K recommendations for a user.

        Strategy: score ALL unseen movies using learned latent factors,
        return top-K by predicted rating.

        Returns
        -------
        DataFrame with columns: [movie_id, predicted_score, rank]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        candidates = [
            mid for mid in self.all_movie_ids
            if seen_movie_ids is None or mid not in seen_movie_ids
        ]

        scores = [
            (mid, self.predict_rating(user_id, mid))
            for mid in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]

        result = pd.DataFrame(top_k, columns=["movie_id", "predicted_score"])
        result["rank"] = range(1, len(result) + 1)
        return result

    def cross_validate(self, train_df: pd.DataFrame, cv: int = 3) -> dict:
        """
        Run k-fold cross-validation and return RMSE and MAE.
        Useful for hyperparameter tuning without touching the test set.
        """
        reader = Reader(rating_scale=(1, 5))
        data   = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]],
            reader,
        )
        model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        results = cross_validate(model, data, measures=["RMSE", "MAE"],
                                 cv=cv, verbose=True)
        return {
            "rmse_mean": results["test_rmse"].mean(),
            "rmse_std":  results["test_rmse"].std(),
            "mae_mean":  results["test_mae"].mean(),
        }

    def get_user_factors(self, user_id: int) -> np.ndarray:
        """Return the learned latent factor vector for a user."""
        inner_uid = self.trainset.to_inner_uid(str(user_id))
        return self.model.pu[inner_uid]

    def get_item_factors(self, movie_id: int) -> np.ndarray:
        """Return the learned latent factor vector for a movie."""
        inner_iid = self.trainset.to_inner_iid(str(movie_id))
        return self.model.qi[inner_iid]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SVDRecommender saved to {path}")

    @classmethod
    def load(cls, path) -> "SVDRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_movies
    from src.data.splitter import load_splits

    train, val, test = load_splits("data/processed")
    movies = load_movies("data/raw")

    model = SVDRecommender(n_factors=100, n_epochs=20)
    model.fit(train)

    seen = set(train[train["user_id"] == 1]["movie_id"])
    recs = model.recommend(user_id=1, k=10, seen_movie_ids=seen)
    recs = recs.merge(movies[["movie_id", "title"]], on="movie_id")

    print("\nSVD top 10 for user 1:")
    print(recs[["rank", "title", "predicted_score"]].to_string(index=False))

    model.save("models/svd.pkl")
    print("\nSVD model saved successfully!")