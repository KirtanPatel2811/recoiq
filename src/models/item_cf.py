"""
item_cf.py — Item-based Collaborative Filtering (vectorised, precomputed).

KEY OPTIMISATION vs naive version:
Naive:  For each user → for each rated item → compute cosine with all items
        = O(users × avg_ratings × n_items²) — extremely slow at inference

Fast:   Precompute the full item-item similarity matrix ONCE at fit() time
        = O(n_items²) once, then O(avg_ratings × k) per user at inference
        3260 × 3260 matrix fits in ~40MB RAM easily.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class ItemCFRecommender:

    def __init__(self, n_similar_items: int = 20):
        self.n_similar_items   = n_similar_items
        self.item_matrix       = None   # (n_items × n_users)
        self.similarity_matrix = None   # (n_items × n_items) — PRECOMPUTED
        self.top_similar_items = None   # (n_items × n_similar_items) indices
        self.top_similar_sims  = None   # (n_items × n_similar_items) scores
        self.user_encoder      = {}
        self.item_encoder      = {}
        self.user_decoder      = {}
        self.item_decoder      = {}
        self.n_users           = 0
        self.n_items           = 0
        self.is_fitted         = False

    def fit(self, train_df: pd.DataFrame, interaction_matrix) -> "ItemCFRecommender":
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

        self.item_matrix = user_item.T  # (n_items × n_users)

        # PRECOMPUTE full item-item cosine similarity matrix
        # 3260 × 3260 = ~42MB — fits in RAM easily
        logger.info(f"Precomputing {self.n_items}×{self.n_items} "
                    f"item similarity matrix...")
        sim = cosine_similarity(self.item_matrix)  # (n_items × n_items)
        np.fill_diagonal(sim, -1)                  # exclude self-similarity

        # For each item, store indices and scores of top-N similar items
        logger.info("Caching top similar items per item...")
        self.top_similar_items = np.argsort(sim, axis=1)[:, ::-1][:, :self.n_similar_items]
        self.top_similar_sims  = np.take_along_axis(sim, self.top_similar_items, axis=1)

        self.is_fitted = True
        logger.info(f"ItemCF fitted: {self.n_items} items × {self.n_users} users | "
                    f"top-{self.n_similar_items} similarities cached")
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

        user_idx      = self.user_encoder[user_id]
        user_row      = self.item_matrix[:, user_idx]      # (n_items,)
        rated_idxs    = np.nonzero(user_row)[0]
        rated_ratings = user_row[rated_idxs]

        if len(rated_idxs) == 0:
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        excluded = set(rated_idxs.tolist())
        if seen_movie_ids:
            excluded.update(
                self.item_encoder[m] for m in seen_movie_ids
                if m in self.item_encoder
            )

        # Score candidates using precomputed similarities — very fast now
        candidate_scores   = {}
        candidate_sims_sum = {}

        for item_idx, rating in zip(rated_idxs, rated_ratings):
            sim_idxs  = self.top_similar_items[item_idx]  # (n_similar_items,)
            sim_vals  = self.top_similar_sims[item_idx]   # (n_similar_items,)

            for sim_idx, sim in zip(sim_idxs, sim_vals):
                sim_idx = int(sim_idx)
                if sim_idx in excluded or sim <= 0:
                    continue
                candidate_scores[sim_idx]   = candidate_scores.get(sim_idx, 0.0)   + sim * rating
                candidate_sims_sum[sim_idx] = candidate_sims_sum.get(sim_idx, 0.0) + abs(sim)

        if not candidate_scores:
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        scores = [
            (
                self.item_decoder[idx],
                float(np.clip(candidate_scores[idx] / candidate_sims_sum[idx], 1.0, 5.0))
            )
            for idx in candidate_scores
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        result = pd.DataFrame(scores[:k], columns=["movie_id", "predicted_score"])
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
    print("\nItem-CF model saved!")