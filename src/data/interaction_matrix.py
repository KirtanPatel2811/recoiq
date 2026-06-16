"""
interaction_matrix.py — Build a sparse user-item interaction matrix.

WHY sparse? The full 1M matrix would be (6040 users × 3706 movies) = 22M cells.
But only 1M are filled (4.5% density) — storing the 95.5% zeros wastes RAM.
scipy CSR (Compressed Sparse Row) stores ONLY non-zero values → ~10x smaller.

WHY two encoders? Raw IDs are non-contiguous (movie IDs skip numbers).
We map them to 0-based contiguous indices for the matrix and all models.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class InteractionMatrix:
    """
    Wraps a scipy CSR sparse matrix with user/item index encoders.

    Attributes
    ----------
    matrix          : csr_matrix  (n_users × n_items) — explicit ratings
    implicit_matrix : csr_matrix  (n_users × n_items) — binary (1 = interacted)
    user_encoder    : dict  {original_user_id → row_index}
    item_encoder    : dict  {original_movie_id → col_index}
    user_decoder    : dict  {row_index → original_user_id}
    item_decoder    : dict  {col_index → original_movie_id}
    """

    def __init__(self):
        self.matrix          = None
        self.implicit_matrix = None
        self.user_encoder    = {}
        self.item_encoder    = {}
        self.user_decoder    = {}
        self.item_decoder    = {}
        self.n_users         = 0
        self.n_items         = 0

    def fit(self, ratings: pd.DataFrame,
            min_user_ratings: int = 5,
            min_item_ratings: int = 10) -> "InteractionMatrix":
        """
        Build sparse matrix from a ratings DataFrame.

        Parameters
        ----------
        ratings           : must have columns [user_id, movie_id, rating]
        min_user_ratings  : drop users with fewer than this many ratings (cold filter)
        min_item_ratings  : drop items with fewer than this many ratings (cold filter)
        """
        df = ratings.copy()

        # ── Filter cold users and items ───────────────────────────────────────
        before = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["movie_id"].value_counts()

        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index

        df = df[df["user_id"].isin(valid_users) & df["movie_id"].isin(valid_items)]
        logger.info(f"Cold filtering: {before:,} → {len(df):,} ratings | "
                    f"{df['user_id'].nunique():,} users | {df['movie_id'].nunique():,} items")

        # ── Build contiguous index encoders ───────────────────────────────────
        unique_users  = sorted(df["user_id"].unique())
        unique_items  = sorted(df["movie_id"].unique())

        self.user_encoder = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_encoder = {iid: idx for idx, iid in enumerate(unique_items)}
        self.user_decoder = {idx: uid for uid, idx in self.user_encoder.items()}
        self.item_decoder = {idx: iid for iid, idx in self.item_encoder.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

        # ── Map IDs to matrix indices ─────────────────────────────────────────
        rows = df["user_id"].map(self.user_encoder).values
        cols = df["movie_id"].map(self.item_encoder).values
        vals = df["rating"].values.astype(np.float32)

        # ── Build explicit rating matrix ──────────────────────────────────────
        self.matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        # ── Build implicit (binary) matrix ────────────────────────────────────
        self.implicit_matrix = csr_matrix(
            (np.ones(len(vals), dtype=np.float32), (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        density = self.matrix.nnz / (self.n_users * self.n_items) * 100
        logger.info(f"Matrix built: {self.n_users} users × {self.n_items} items | "
                    f"{self.matrix.nnz:,} interactions | density={density:.2f}%")
        return self

    def get_user_ratings(self, user_id: int) -> dict[int, float]:
        """Return {movie_id: rating} dict for a given original user_id."""
        if user_id not in self.user_encoder:
            return {}
        row_idx = self.user_encoder[user_id]
        row = self.matrix.getrow(row_idx)
        col_idxs = row.indices
        ratings  = row.data
        return {self.item_decoder[c]: float(r) for c, r in zip(col_idxs, ratings)}

    def save(self, path: str | Path):
        """Pickle the whole InteractionMatrix object."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"InteractionMatrix saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "InteractionMatrix":
        """Load a pickled InteractionMatrix."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"InteractionMatrix loaded from {path} | "
                    f"{obj.n_users} users × {obj.n_items} items")
        return obj


if __name__ == "__main__":
    from loader import load_ratings
    ratings = load_ratings("data/raw")

    im = InteractionMatrix()
    im.fit(ratings, min_user_ratings=5, min_item_ratings=10)
    im.save("data/processed/interaction_matrix.pkl")

    # Spot check
    sample_user = list(im.user_encoder.keys())[0]
    print(f"\nSample ratings for user {sample_user}:")
    print(im.get_user_ratings(sample_user))