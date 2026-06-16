"""
splitter.py — Temporal train/validation/test split for RecSys.

WHY temporal split (not random)?
Random split leaks the future into training. If a user rated 'Inception' in 2010
and we randomly put it in train, we're training on information that didn't exist
when the earlier ratings were made. In production, a model always predicts FUTURE
preferences from PAST behaviour. Temporal split mirrors this exactly.

Split strategy:
- Sort all ratings by timestamp
- Train:  first 70% of each user's ratings (chronologically)
- Val:    next 15%
- Test:   last 15%
This is per-user temporal split — ensures every user appears in all three sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def temporal_split(
    ratings: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    min_ratings_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal train/val/test split.

    Parameters
    ----------
    ratings               : DataFrame with [user_id, movie_id, rating, timestamp]
    train_frac            : fraction of each user's ratings for training
    val_frac              : fraction for validation (test gets the rest)
    min_ratings_per_user  : users with fewer ratings are excluded entirely

    Returns
    -------
    (train_df, val_df, test_df) — non-overlapping, chronologically ordered
    """
    assert train_frac + val_frac < 1.0, "train + val fracs must be < 1.0"

    # ── Filter users with too few ratings ─────────────────────────────────────
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    df = ratings[ratings["user_id"].isin(valid_users)].copy()
    logger.info(f"Users after min-rating filter: {df['user_id'].nunique():,}")

    # ── Sort by user, then timestamp ──────────────────────────────────────────
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── Assign split label per user ───────────────────────────────────────────
    def assign_split(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        train_end = int(np.floor(n * train_frac))
        val_end   = int(np.floor(n * (train_frac + val_frac)))

        labels = pd.Series(["train"] * n, index=group.index)
        labels.iloc[train_end:val_end] = "val"
        labels.iloc[val_end:]          = "test"
        return labels

    df["split"] = df.groupby("user_id", group_keys=False).apply(assign_split)

    train = df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)
    val   = df[df["split"] == "val"  ].drop(columns="split").reset_index(drop=True)
    test  = df[df["split"] == "test" ].drop(columns="split").reset_index(drop=True)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    _log_split_stats(train, val, test)
    _verify_no_leakage(train, val, test)

    return train, val, test


def _log_split_stats(train, val, test):
    total = len(train) + len(val) + len(test)
    logger.info(
        f"Split complete:\n"
        f"  Train : {len(train):>8,} ratings ({len(train)/total*100:.1f}%) | "
        f"{train['user_id'].nunique():,} users\n"
        f"  Val   : {len(val):>8,} ratings ({len(val)/total*100:.1f}%) | "
        f"{val['user_id'].nunique():,} users\n"
        f"  Test  : {len(test):>8,} ratings ({len(test)/total*100:.1f}%) | "
        f"{test['user_id'].nunique():,} users"
    )


def _verify_no_leakage(train, val, test):
    """
    Verify temporal integrity: every rating in val/test is AFTER all train ratings
    for that user.
    """
    train_max_ts = train.groupby("user_id")["timestamp"].max()

    for split_name, split_df in [("val", val), ("test", test)]:
        split_min_ts = split_df.groupby("user_id")["timestamp"].min()
        common_users = train_max_ts.index.intersection(split_min_ts.index)

        violations = (split_min_ts[common_users] < train_max_ts[common_users]).sum()
        if violations > 0:
            logger.warning(f"Temporal leakage in {split_name}: {violations} users "
                           f"have {split_name} ratings BEFORE their latest train rating!")
        else:
            logger.info(f"✓ No temporal leakage in {split_name} split")


def save_splits(train, val, test, processed_dir: str | Path):
    """Save all three splits as parquet files (fast + compressed)."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(  processed_dir / "val.parquet",   index=False)
    test.to_parquet( processed_dir / "test.parquet",  index=False)
    logger.info(f"Splits saved to {processed_dir}")


def load_splits(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved splits from parquet."""
    processed_dir = Path(processed_dir)
    train = pd.read_parquet(processed_dir / "train.parquet")
    val   = pd.read_parquet(processed_dir / "val.parquet")
    test  = pd.read_parquet(processed_dir / "test.parquet")
    return train, val, test


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_ratings

    ratings = load_ratings("data/raw")
    train, val, test = temporal_split(ratings)
    save_splits(train, val, test, "data/processed")

    print("\n── Train sample (first 3) ──")
    print(train.head(3))
    print("\n── Test sample (first 3) ──")
    print(test.head(3))