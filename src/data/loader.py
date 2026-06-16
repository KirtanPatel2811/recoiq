"""
loader.py — Load and join MovieLens 1M tables into clean DataFrames.

Design decisions:
- MovieLens 1M uses '::' separator with no header → handle explicitly
- Timestamps are Unix epoch → convert to datetime for temporal split later
- Keep raw user/item IDs as integers (Surprise and implicit both expect this)
- Return a single merged DataFrame AND individual tables for flexibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Column schemas ────────────────────────────────────────────────────────────
RATINGS_COLS  = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLS   = ["movie_id", "title", "genres"]
USERS_COLS    = ["user_id", "gender", "age", "occupation", "zip_code"]

RATINGS_DTYPES = {"user_id": np.int32, "movie_id": np.int32,
                  "rating": np.float32, "timestamp": np.int64}
MOVIES_DTYPES  = {"movie_id": np.int32}
USERS_DTYPES   = {"user_id": np.int32}


def load_ratings(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load ratings.dat → DataFrame with columns:
        user_id, movie_id, rating, timestamp, datetime
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / "ratings.dat"
    logger.info(f"Loading ratings from {path}")

    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=RATINGS_COLS,
        dtype=RATINGS_DTYPES,
        engine="python",   # needed for multi-char separator
    )

    # Convert Unix timestamp → datetime (UTC)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    logger.info(f"Ratings loaded: {len(df):,} rows | "
                f"{df['user_id'].nunique():,} users | "
                f"{df['movie_id'].nunique():,} movies")
    return df


def load_movies(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load movies.dat → DataFrame with columns:
        movie_id, title, genres (as list), year (extracted from title)
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / "movies.dat"
    logger.info(f"Loading movies from {path}")

    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=MOVIES_COLS,
        dtype=MOVIES_DTYPES,
        engine="python",
        encoding="latin-1",   # some titles have accented chars
    )

    # Split pipe-separated genres into a list: "Action|Comedy" → ["Action","Comedy"]
    df["genres_list"] = df["genres"].str.split("|")

    # Extract year from title: "Toy Story (1995)" → 1995
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int32")

    logger.info(f"Movies loaded: {len(df):,} movies | "
                f"{df['genres_list'].explode().nunique()} unique genres")
    return df


def load_users(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load users.dat → DataFrame with columns:
        user_id, gender, age, occupation, zip_code
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / "users.dat"
    logger.info(f"Loading users from {path}")

    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=USERS_COLS,
        dtype=USERS_DTYPES,
        engine="python",
    )

    logger.info(f"Users loaded: {len(df):,} users")
    return df


def load_all(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all three tables and return a dict.

    Returns
    -------
    {
        "ratings": DataFrame,
        "movies":  DataFrame,
        "users":   DataFrame,
        "merged":  ratings LEFT JOIN movies LEFT JOIN users
    }
    """
    raw_dir = Path(raw_dir)

    ratings = load_ratings(raw_dir)
    movies  = load_movies(raw_dir)
    users   = load_users(raw_dir)

    merged = (
        ratings
        .merge(movies[["movie_id", "title", "genres_list", "year"]], on="movie_id", how="left")
        .merge(users[["user_id", "gender", "age"]], on="user_id", how="left")
    )

    logger.info(f"Merged table shape: {merged.shape}")
    return {"ratings": ratings, "movies": movies, "users": users, "merged": merged}


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    data = load_all(raw_dir)
    print("\n── Ratings sample ──")
    print(data["ratings"].head(3))
    print("\n── Movies sample ──")
    print(data["movies"].head(3))
    print("\n── Merged sample ──")
    print(data["merged"].head(3))