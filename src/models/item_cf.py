"""
item_cf.py — Item-based Collaborative Filtering.

HOW IT WORKS (different from User-CF):
1. Compute item-item similarity matrix (movies similar to each other)
2. For a target user, look at what they've already rated
3. For each unseen movie, score it by: Σ sim(unseen, seen) * rating(seen)
4. Return top-N by aggregated score

WHY item-CF is often BETTER than user-CF in practice:
- Item similarities are more stable over time (movie genres don't change)
- User preferences drift; item relationships don't
- Easier to explain: "because you liked Inception, we recommend Interstellar"
- Amazon famously switched from user-CF to item-CF in 2003 for exactly this reason

KEY optimisation: We don't compute ALL item-item similarities upfront.
For production, you'd precompute and cache top-K similar items per movie.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class ItemCFRecommender:
    """
    Memory-based Item-Collaborative Filtering using cosine similarity.

    Item vectors are the COLUMNS of the user-item matrix (i.e., each item
    is represented by the vector of ratings it received from all users).
    """

    def __init__(self, n_similar_items: int = 20):
        """
        Parameters
        ----------
        n_similar_items : for each item the user rated, how many similar items
                          to consider as candidates.
        """
        self.n_similar_items  = n_similar_items
        self.item_matrix      = None
        self.user_encoder     = {}
        self.item_encoder     = {}
        self.user_decoder     = {}
        self.item_decoder     = {}
        self.n_users          = 0
        self.n_items          = 0
        self.is_fitted        = False

    def fit(self, train_df: pd.DataFrame, interaction_matrix) -> "ItemCFRecommender":
        """
        Build the item-user matrix (transpose of user-item matrix).

        WHY transpose? For item-CF, we want item vectors.
        Item vector = all users who rated this movie and what they gave it.
        Cosine similarity between two item vectors = how similarly users rate them.
        """
        self.user_encoder = interaction_matrix.user_encoder
        self.item_encoder = interaction_matrix.item_encoder
        self.user_decoder = interaction_matrix.user_decoder
        self.item_decoder = interaction_matrix.item_decoder
        self.n_users      = interaction_matrix.n_users
        self.n_items      = interaction_matrix.n_items

        logger.info("Building item-user matrix from training data...")

        user_item = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u = self.user_encoder.get(row.user_id)
            i = self.item_encoder.get(row.movie_id)
            if u is not None and i is not None:
                user_item[u, i] = row.rating

        # Transpose: rows = items, cols = users
        self.item_matrix = user_item.T

        self.is_fitted = True
        logger.info(f"ItemCF fitted: {self.n_items} items × {self.n_users} users")
        return self

    def _get_similar_items(self, item_idx: int) -> tuple:
        """
        Compute cosine similarity between one item and all others.
        Returns (similar_item_indices, similarity_scores).
        """
        target = self.item_matrix[item_idx].reshape(1, -1)
        sims   = cosine_similarity(target, self.item_matrix)[0]
        sims[item_idx] = -1

        top_idxs = np.argsort(sims)[::-1][:self.n_similar_items]
        top_sims  = sims[top_idxs]

        mask = top_sims > 0
        return top_idxs[mask], top_sims[mask]

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        seen_movie_ids: set = None,
    ) -> pd.DataFrame:
        """
        Return top-K item-CF recommendations.

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

        # Get this user's rated items from the matrix
        user_row      = self.item_matrix[:, user_idx]
        rated_idxs    = np.nonzero(user_row)[0]
        rated_ratings = user_row[rated_idxs]

        if len(rated_idxs) == 0:
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        # Excluded set = already rated by user
        excluded = set(rated_idxs)
        if seen_movie_ids:
            excluded.update(
                self.item_encoder[m] for m in seen_movie_ids
                if m in self.item_encoder
            )

        # Score candidates: Σ sim(candidate, seen_item) * rating(seen_item)
        candidate_scores   = {}
        candidate_sims_sum = {}

        for item_idx, rating in zip(rated_idxs, rated_ratings):
            similar_idxs, similar_sims = self._get_similar_items(item_idx)
            for sim_idx, sim in zip(similar_idxs, similar_sims):
                if sim_idx in excluded:
                    continue
                candidate_scores[sim_idx]   = candidate_scores.get(sim_idx, 0.0)   + sim * rating
                candidate_sims_sum[sim_idx] = candidate_sims_sum.get(sim_idx, 0.0) + abs(sim)

        if not candidate_scores:
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        # Normalise by total similarity weight and clip to [1, 5]
        scores = [
            (
                self.item_decoder[idx],
                float(np.clip(candidate_scores[idx] / candidate_sims_sum[idx], 1.0, 5.0))
            )
            for idx in candidate_scores
        ]
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
        logger.info(f"ItemCF saved to {path}")

    @classmethod
    def load(cls, path) -> "ItemCFRecommender":
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

    model = ItemCFRecommender(n_similar_items=20)
    model.fit(train, im)

    seen = set(train[train["user_id"] == 1]["movie_id"])
    recs = model.recommend(user_id=1, k=10, seen_movie_ids=seen)
    recs = recs.merge(movies[["movie_id", "title"]], on="movie_id")

    print("\nItem-CF top 10 for user 1:")
    print(recs[["rank", "title", "predicted_score"]].to_string(index=False))

    model.save("models/item_cf.pkl")