"""
user_cf.py — User-based Collaborative Filtering.

HOW IT WORKS:
1. Build a user-item rating matrix (users as rows, movies as cols)
2. For a target user, compute cosine similarity to ALL other users
3. Find the top-K most similar users (neighbours)
4. Aggregate the neighbours' ratings → predict scores for unseen movies
5. Return top-N movies by predicted score

WHY cosine similarity (not Pearson)?
Cosine works well for sparse matrices and is fast with scipy.
Pearson is better theoretically (handles rating scale bias) but slower.
We'll use mean-centered cosine (a middle ground) here.

SCALABILITY NOTE: For 6040 users, computing all pairwise similarities is
feasible (6040×6040 = 36M pairs). For 162K users (MovieLens 25M), you'd
need Approximate Nearest Neighbours (Faiss, Annoy) — we'll note this.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class UserCFRecommender:
    """
    Memory-based User-Collaborative Filtering using cosine similarity.

    This is a "memory-based" method — it stores the entire training matrix
    and computes similarities at recommendation time (no training phase).
    """

    def __init__(self, n_neighbours: int = 50, min_common_items: int = 5):
        """
        Parameters
        ----------
        n_neighbours     : number of similar users to aggregate from
        min_common_items : minimum items in common to consider a user a valid neighbour.
                           Prevents spurious similarities from 1-2 shared ratings.
        """
        self.n_neighbours      = n_neighbours
        self.min_common_items  = min_common_items

        self.user_item_matrix  = None
        self.user_encoder      = {}
        self.item_encoder      = {}
        self.user_decoder      = {}
        self.item_decoder      = {}
        self.n_users           = 0
        self.n_items           = 0
        self.user_means        = None
        self.is_fitted         = False

    def fit(self, train_df: pd.DataFrame, interaction_matrix) -> "UserCFRecommender":
        """
        Store the training matrix and compute user mean ratings.

        Parameters
        ----------
        train_df           : training ratings DataFrame
        interaction_matrix : fitted InteractionMatrix object (for encoders)
        """
        self.user_encoder = interaction_matrix.user_encoder
        self.item_encoder = interaction_matrix.item_encoder
        self.user_decoder = interaction_matrix.user_decoder
        self.item_decoder = interaction_matrix.item_decoder
        self.n_users      = interaction_matrix.n_users
        self.n_items      = interaction_matrix.n_items

        logger.info("Building user-item matrix from training data...")

        matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u = self.user_encoder.get(row.user_id)
            i = self.item_encoder.get(row.movie_id)
            if u is not None and i is not None:
                matrix[u, i] = row.rating

        # Mean-center each user's ratings
        # WHY: user A rates everything 4-5, user B rates 1-3. Without centering,
        # they look dissimilar even if they agree on relative preferences.
        counts = (matrix != 0).sum(axis=1)
        sums   = matrix.sum(axis=1)
        self.user_means = np.where(counts > 0, sums / counts, 0.0)

        self.user_item_matrix = matrix.copy()
        for u in range(self.n_users):
            nonzero_mask = matrix[u] != 0
            self.user_item_matrix[u, nonzero_mask] -= self.user_means[u]

        self.is_fitted = True
        logger.info(f"UserCF fitted: {self.n_users} users × {self.n_items} items")
        return self

    def _get_neighbours(self, user_idx: int) -> tuple:
        """
        Compute cosine similarity between target user and all others.
        Returns (neighbour_indices, similarity_scores) sorted by similarity.
        """
        target = self.user_item_matrix[user_idx].reshape(1, -1)
        sims   = cosine_similarity(target, self.user_item_matrix)[0]

        # Exclude self
        sims[user_idx] = -1

        # Filter: only neighbours with min_common_items overlap
        target_nonzero = set(np.nonzero(self.user_item_matrix[user_idx])[0])
        for other_idx in range(self.n_users):
            other_nonzero = set(np.nonzero(self.user_item_matrix[other_idx])[0])
            if len(target_nonzero & other_nonzero) < self.min_common_items:
                sims[other_idx] = -1

        top_indices = np.argsort(sims)[::-1][:self.n_neighbours]
        top_sims    = sims[top_indices]

        mask = top_sims > 0
        return top_indices[mask], top_sims[mask]

    def predict_score(self, user_idx: int, item_idx: int,
                      neighbour_idxs: np.ndarray,
                      neighbour_sims: np.ndarray) -> float:
        """
        Predict rating for (user, item) using weighted average of neighbours.

        Formula: pred(u,i) = mean(u) + Σ sim(u,v) * (r(v,i) - mean(v))
                                        ─────────────────────────────────
                                               Σ |sim(u,v)|
        """
        numerator   = 0.0
        denominator = 0.0

        for n_idx, sim in zip(neighbour_idxs, neighbour_sims):
            r_vi = self.user_item_matrix[n_idx, item_idx]
            if r_vi != 0:
                numerator   += sim * r_vi
                denominator += abs(sim)

        if denominator == 0:
            return 0.0

        raw = float(self.user_means[user_idx] + numerator / denominator)
        # Clip to valid rating range [1, 5]
        return float(np.clip(raw, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        seen_movie_ids: set = None,
    ) -> pd.DataFrame:
        """
        Return top-K recommendations for a user.

        Returns
        -------
        DataFrame with columns: [movie_id, predicted_score, rank]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not in training set (cold start)")
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        user_idx = self.user_encoder[user_id]
        neighbour_idxs, neighbour_sims = self._get_neighbours(user_idx)

        if len(neighbour_idxs) == 0:
            logger.warning(f"No valid neighbours for user {user_id}")
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        # Candidate items: union of all items rated by neighbours
        candidate_items = set()
        for n_idx in neighbour_idxs:
            rated = set(np.nonzero(self.user_item_matrix[n_idx])[0])
            candidate_items.update(rated)

        # Exclude already-seen items
        if seen_movie_ids:
            seen_idxs = {self.item_encoder[m] for m in seen_movie_ids
                         if m in self.item_encoder}
            candidate_items -= seen_idxs

        # Exclude items already rated in the matrix
        user_rated = set(np.nonzero(self.user_item_matrix[user_idx])[0])
        candidate_items -= user_rated

        # Score each candidate
        scores = []
        for item_idx in candidate_items:
            score = self.predict_score(user_idx, item_idx,
                                       neighbour_idxs, neighbour_sims)
            if score > 0:
                scores.append((self.item_decoder[item_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]

        result = pd.DataFrame(top_k, columns=["movie_id", "predicted_score"])
        result["rank"] = range(1, len(result) + 1)
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"UserCF saved to {path}")

    @classmethod
    def load(cls, path) -> "UserCFRecommender":
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

    model = UserCFRecommender(n_neighbours=50, min_common_items=5)
    model.fit(train, im)

    seen = set(train[train["user_id"] == 1]["movie_id"])
    recs = model.recommend(user_id=1, k=10, seen_movie_ids=seen)
    recs = recs.merge(movies[["movie_id", "title"]], on="movie_id")

    print("\nUser-CF top 10 for user 1:")
    print(recs[["rank", "title", "predicted_score"]].to_string(index=False))

    model.save("models/user_cf.pkl")