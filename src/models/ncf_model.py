"""
ncf_model.py — Neural Collaborative Filtering (PyTorch).

ARCHITECTURE:
    User ID ──► Embedding(n_users, 64) ──►
                                            Concat(128) ──► Linear(128→64) ──► ReLU
    Item ID ──► Embedding(n_items, 64) ──►              ──► Linear(64→32)  ──► ReLU
                                                         ──► Linear(32→1)  ──► Sigmoid × 5

WHY embeddings instead of one-hot vectors?
One-hot: user 1 = [1,0,0,...,0] (6040-dim sparse)
Embedding: user 1 = [0.23, -0.41, 0.87, ...] (64-dim dense, LEARNED)
The model learns that "user 42 and user 117 have similar taste"
by pushing their embedding vectors close together during training.

WHY MLP on top of embeddings?
Dot product (like SVD) only captures LINEAR interactions.
MLP captures NON-LINEAR interactions — "user likes sci-fi AND
dislikes romance UNLESS it's directed by Nolan" type patterns.

LOSS FUNCTION — BPR (Bayesian Personalised Ranking):
Instead of predicting exact ratings, BPR optimises RANKING:
  For each user u, positive item i (rated), negative item j (not rated):
  Loss = -log(sigmoid(score(u,i) - score(u,j)))
This directly optimises "rank positive items above negative items"
which is what we actually care about in RecSys.

NEGATIVE SAMPLING:
We don't have explicit "disliked" data. For each positive (user, item)
pair, we randomly sample K items the user hasn't rated as negatives.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import pickle
import time

logger = logging.getLogger(__name__)

# ── Detect device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"NCF will train on: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class BPRDataset(Dataset):
    """
    Dataset for BPR training.

    For each positive (user, item) interaction, samples n_neg random
    negative items the user hasn't rated.

    Returns (user_idx, pos_item_idx, neg_item_idx) triplets.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        user_encoder: dict,
        item_encoder: dict,
        n_items: int,
        n_neg: int = 4,
    ):
        self.n_items      = n_items
        self.n_neg        = n_neg
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

        # Build user → set of rated item indices (for negative sampling)
        self.user_pos_items: dict[int, set] = {}

        valid_rows = []
        for row in train_df.itertuples(index=False):
            u = user_encoder.get(row.user_id)
            i = item_encoder.get(row.movie_id)
            if u is not None and i is not None:
                if u not in self.user_pos_items:
                    self.user_pos_items[u] = set()
                self.user_pos_items[u].add(i)
                valid_rows.append((u, i))

        self.interactions = valid_rows
        logger.info(f"BPRDataset: {len(self.interactions):,} interactions, "
                    f"n_neg={n_neg}")

    def __len__(self):
        return len(self.interactions) * self.n_neg

    def __getitem__(self, idx):
        # Map back to base interaction
        base_idx = idx // self.n_neg
        user_idx, pos_item_idx = self.interactions[base_idx]

        # Sample a negative item the user hasn't rated
        neg_item_idx = np.random.randint(0, self.n_items)
        while neg_item_idx in self.user_pos_items[user_idx]:
            neg_item_idx = np.random.randint(0, self.n_items)

        return (
            torch.tensor(user_idx,      dtype=torch.long),
            torch.tensor(pos_item_idx,  dtype=torch.long),
            torch.tensor(neg_item_idx,  dtype=torch.long),
        )


# ── Model architecture ────────────────────────────────────────────────────────

class NCFNet(nn.Module):
    """
    Neural Collaborative Filtering network.

    User and item embeddings are concatenated and passed through an MLP.
    Output is a scalar score (higher = more relevant).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: list  = None,
        dropout: float     = 0.2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        input_dim = embedding_dim * 2   # concat of user + item embeddings
        layers    = []
        prev_dim  = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Weight initialisation — Xavier uniform for stable training
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        user_ids : (batch_size,) tensor of user indices
        item_ids : (batch_size,) tensor of item indices

        Returns
        -------
        scores : (batch_size,) tensor of relevance scores
        """
        user_emb = self.user_embedding(user_ids)   # (B, embedding_dim)
        item_emb = self.item_embedding(item_ids)   # (B, embedding_dim)
        x        = torch.cat([user_emb, item_emb], dim=-1)   # (B, 2*embedding_dim)
        score    = self.mlp(x).squeeze(-1)         # (B,)
        return score

    def get_user_embeddings(self) -> np.ndarray:
        """Return all user embedding vectors as numpy array."""
        return self.user_embedding.weight.detach().cpu().numpy()

    def get_item_embeddings(self) -> np.ndarray:
        """Return all item embedding vectors as numpy array."""
        return self.item_embedding.weight.detach().cpu().numpy()


# ── BPR Loss ─────────────────────────────────────────────────────────────────

class BPRLoss(nn.Module):
    """
    Bayesian Personalised Ranking loss.

    Loss = -mean(log(sigmoid(pos_score - neg_score)))

    Intuition: push the score of the positive item ABOVE the negative item.
    We don't care about exact values, only relative order.
    """

    def forward(self, pos_scores: torch.Tensor,
                neg_scores: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))


# ── Trainer ───────────────────────────────────────────────────────────────────

class NCFRecommender:
    """
    End-to-end Neural CF recommender.
    Wraps NCFNet with training loop, evaluation, and recommend() interface.
    """

    def __init__(
        self,
        embedding_dim: int  = 64,
        hidden_dims: list   = None,
        dropout: float      = 0.2,
        n_neg: int          = 4,
        batch_size: int     = 2048,
        n_epochs: int       = 20,
        lr: float           = 1e-3,
        weight_decay: float = 1e-5,
        random_state: int   = 42,
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.embedding_dim  = embedding_dim
        self.hidden_dims    = hidden_dims
        self.dropout        = dropout
        self.n_neg          = n_neg
        self.batch_size     = batch_size
        self.n_epochs       = n_epochs
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.random_state   = random_state

        self.model          = None
        self.user_encoder   = {}
        self.item_encoder   = {}
        self.user_decoder   = {}
        self.item_decoder   = {}
        self.n_users        = 0
        self.n_items        = 0
        self.all_item_idxs  = None
        self.train_losses   = []
        self.is_fitted      = False

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def fit(self, train_df: pd.DataFrame, interaction_matrix) -> "NCFRecommender":
        """
        Train NCF on training ratings.

        Parameters
        ----------
        train_df           : training ratings DataFrame
        interaction_matrix : fitted InteractionMatrix (for encoders)
        """
        self.user_encoder = interaction_matrix.user_encoder
        self.item_encoder = interaction_matrix.item_encoder
        self.user_decoder = interaction_matrix.user_decoder
        self.item_decoder = interaction_matrix.item_decoder
        self.n_users      = interaction_matrix.n_users
        self.n_items      = interaction_matrix.n_items
        self.all_item_idxs = np.arange(self.n_items)

        logger.info(f"NCF training on {DEVICE} | "
                    f"embedding_dim={self.embedding_dim} | "
                    f"hidden={self.hidden_dims} | "
                    f"epochs={self.n_epochs} | batch={self.batch_size}")

        # Dataset and loader
        dataset = BPRDataset(
            train_df, self.user_encoder, self.item_encoder,
            self.n_items, n_neg=self.n_neg,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,       # Windows: keep at 0
            pin_memory=(DEVICE.type == "cuda"),
        )

        # Model, loss, optimiser
        self.model = NCFNet(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(DEVICE)

        criterion = BPRLoss()
        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10, gamma=0.5
        )

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss  = 0.0
            n_batches   = 0
            t_start     = time.time()

            for user_ids, pos_ids, neg_ids in loader:
                user_ids = user_ids.to(DEVICE)
                pos_ids  = pos_ids.to(DEVICE)
                neg_ids  = neg_ids.to(DEVICE)

                optimiser.zero_grad()

                pos_scores = self.model(user_ids, pos_ids)
                neg_scores = self.model(user_ids, neg_ids)

                loss = criterion(pos_scores, neg_scores)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches
            elapsed  = time.time() - t_start
            self.train_losses.append(avg_loss)

            logger.info(f"Epoch {epoch+1:>2}/{self.n_epochs} | "
                        f"BPR Loss: {avg_loss:.4f} | "
                        f"Time: {elapsed:.1f}s")

        self.model.eval()
        self.is_fitted = True
        logger.info("NCF training complete!")
        return self

    def _score_all_items(self, user_idx: int) -> np.ndarray:
        """Score all items for a user using the trained model."""
        user_tensor = torch.tensor(
            [user_idx] * self.n_items, dtype=torch.long, device=DEVICE
        )
        item_tensor = torch.tensor(
            self.all_item_idxs, dtype=torch.long, device=DEVICE
        )

        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()

        return scores

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        seen_movie_ids: set = None,
    ) -> pd.DataFrame:
        """
        Return top-K NCF recommendations for a user.

        Returns
        -------
        DataFrame with columns: [movie_id, predicted_score, rank]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not in training set")
            return pd.DataFrame(columns=["movie_id", "predicted_score", "rank"])

        user_idx = self.user_encoder[user_id]
        scores   = self._score_all_items(user_idx)

        # Mask seen items
        if seen_movie_ids:
            for mid in seen_movie_ids:
                if mid in self.item_encoder:
                    scores[self.item_encoder[mid]] = -np.inf

        top_idxs   = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_idxs]
        top_movies = [self.item_decoder[int(i)] for i in top_idxs]

        result = pd.DataFrame({
            "movie_id":        top_movies,
            "predicted_score": top_scores.tolist(),
        })
        result["rank"] = range(1, len(result) + 1)
        return result

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Return the learned embedding vector for a user."""
        if user_id not in self.user_encoder:
            raise ValueError(f"User {user_id} not in training set")
        idx = self.user_encoder[user_id]
        return self.model.get_user_embeddings()[idx]

    def get_item_embedding(self, movie_id: int) -> np.ndarray:
        """Return the learned embedding vector for a movie."""
        if movie_id not in self.item_encoder:
            raise ValueError(f"Movie {movie_id} not in training set")
        idx = self.item_encoder[movie_id]
        return self.model.get_item_embeddings()[idx]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save model weights separately for portability
        torch.save(self.model.state_dict(),
                   path.parent / (path.stem + "_weights.pt"))
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"NCFRecommender saved to {path}")

    @classmethod
    def load(cls, path) -> "NCFRecommender":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"NCFRecommender loaded | device: {DEVICE}")
        return obj


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_movies
    from src.data.splitter import load_splits
    from src.data.interaction_matrix import InteractionMatrix

    # Verify GPU
    print(f"\n{'='*50}")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*50}\n")

    train, val, test = load_splits("data/processed")
    movies = load_movies("data/raw")
    im     = InteractionMatrix.load("data/processed/interaction_matrix.pkl")

    model = NCFRecommender(
        embedding_dim = 64,
        hidden_dims   = [128, 64, 32],
        dropout       = 0.2,
        n_neg         = 4,
        batch_size    = 2048,
        n_epochs      = 20,
        lr            = 1e-3,
    )
    model.fit(train, im)

    # Recommend for user 1
    seen = set(train[train["user_id"] == 1]["movie_id"])
    recs = model.recommend(user_id=1, k=10, seen_movie_ids=seen)
    recs = recs.merge(movies[["movie_id", "title"]], on="movie_id")

    print("\nNCF top 10 for user 1:")
    print(recs[["rank", "title", "predicted_score"]].to_string(index=False))

    # Show embedding shape
    ue = model.get_user_embedding(1)
    print(f"\nUser 1 embedding shape : {ue.shape}")
    print(f"First 8 dimensions     : {ue[:8].round(3)}")

    model.save("models/ncf.pkl")
    print("\nNCF model saved!")