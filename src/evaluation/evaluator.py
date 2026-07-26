"""
evaluator.py — Run all models on the test set and produce a comparison table.

DESIGN:
- Load all trained models
- For a sample of test users, generate top-K recommendations
- Compute all metrics
- Print a leaderboard table
- Save results to reports/evaluation_results.csv

WHY sample users (not all 6040)?
- User-CF and Item-CF are slow at inference (~2s per user)
- 500 users × 5 models = good statistical estimate
- SVD and NCF can run on all users quickly
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json

from src.data.loader import load_movies
from src.data.splitter import load_splits
from src.data.interaction_matrix import InteractionMatrix
from src.models.popularity import PopularityRecommender
from src.models.user_cf import UserCFRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.svd_model import SVDRecommender
from src.models.als_model import ALSRecommender
from src.models.ncf_model import NCFRecommender
from src.evaluation.metrics import evaluate_model, coverage_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
K                   = 10       # recommendation cutoff
RELEVANCE_THRESHOLD = 4.0      # rating >= 4.0 = "liked"
N_EVAL_USERS        = 500      # users to evaluate CF models on (slow inference)
RANDOM_SEED         = 42


def load_all_models() -> dict:
    """Load all trained model artifacts."""
    logger.info("Loading all models...")
    models = {}

    models["Popularity"] = PopularityRecommender.load("models/popularity.pkl")
    logger.info("  ✓ Popularity loaded")

    models["User-CF"] = UserCFRecommender.load("models/user_cf.pkl")
    logger.info("  ✓ User-CF loaded")

    models["Item-CF"] = ItemCFRecommender.load("models/item_cf.pkl")
    logger.info("  ✓ Item-CF loaded")

    models["SVD"] = SVDRecommender.load("models/svd.pkl")
    logger.info("  ✓ SVD loaded")

    models["ALS"] = ALSRecommender.load("models/als.pkl")
    logger.info("  ✓ ALS loaded")

    models["NCF"] = NCFRecommender.load("models/ncf.pkl")
    logger.info("  ✓ NCF loaded")

    return models


def get_recommendations(
    model_name: str,
    model,
    eval_users: list,
    train_df: pd.DataFrame,
    k: int,
) -> dict:
    """
    Generate top-K recommendations for all eval users.
    Returns {user_id: [movie_id, ...]} dict.
    """
    # Build seen-movies lookup once
    seen_lookup = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    recommendations = {}
    t_start = time.time()

    for i, user_id in enumerate(eval_users):
        seen = seen_lookup.get(user_id, set())

        try:
            recs = model.recommend(user_id=user_id, k=k, seen_movie_ids=seen)
            recommendations[user_id] = recs["movie_id"].tolist()
        except Exception as e:
            logger.warning(f"  {model_name} failed for user {user_id}: {e}")
            recommendations[user_id] = []

        # Progress log every 100 users
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            logger.info(f"  {model_name}: {i+1}/{len(eval_users)} users "
                        f"({rate:.1f} users/sec)")

    elapsed = time.time() - t_start
    logger.info(f"  {model_name} done: {len(eval_users)} users in {elapsed:.1f}s")
    return recommendations


def run_evaluation(
    n_eval_users: int = N_EVAL_USERS,
    k: int = K,
    slow_models: bool = True,
) -> pd.DataFrame:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    n_eval_users : number of users to evaluate (all models)
    k            : recommendation cutoff
    slow_models  : if False, skip User-CF and Item-CF (for quick testing)

    Returns
    -------
    DataFrame with one row per model and columns for each metric
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train, val, test = load_splits("data/processed")
    im     = InteractionMatrix.load("data/processed/interaction_matrix.pkl")
    movies = load_movies("data/raw")

    # ── Select eval users ─────────────────────────────────────────────────────
    # Users must appear in both train AND test
    train_users = set(train["user_id"].unique())
    test_users  = set(test["user_id"].unique())
    common_users = list(train_users & test_users)

    np.random.seed(RANDOM_SEED)
    eval_users = np.random.choice(
        common_users,
        size=min(n_eval_users, len(common_users)),
        replace=False,
    ).tolist()

    logger.info(f"Evaluating {len(eval_users)} users | K={k} | "
                f"relevance_threshold={RELEVANCE_THRESHOLD}")

    # ── Load models ───────────────────────────────────────────────────────────
    models = load_all_models()

    if not slow_models:
        logger.info("Skipping User-CF and Item-CF (slow_models=False)")
        models.pop("User-CF", None)
        models.pop("Item-CF", None)

    # ── Evaluate each model ───────────────────────────────────────────────────
    results = []
    total_items = im.n_items

    for model_name, model in models.items():
        logger.info(f"\n── Evaluating {model_name} ──")

        recs = get_recommendations(model_name, model, eval_users, train, k)

        metrics = evaluate_model(
            recommendations=recs,
            test_df=test[test["user_id"].isin(eval_users)],
            k=k,
            relevance_threshold=RELEVANCE_THRESHOLD,
        )

        coverage = coverage_at_k(recs, total_items, k)
        metrics[f"coverage@{k}"] = coverage

        row = {"Model": model_name, **metrics}
        results.append(row)

        logger.info(
            f"  Hit Rate@{k}  : {metrics[f'hit_rate@{k}']:.4f}\n"
            f"  Precision@{k} : {metrics[f'precision@{k}']:.4f}\n"
            f"  Recall@{k}    : {metrics[f'recall@{k}']:.4f}\n"
            f"  NDCG@{k}      : {metrics[f'ndcg@{k}']:.4f}\n"
            f"  MAP@{k}       : {metrics[f'map@{k}']:.4f}\n"
            f"  Coverage@{k}  : {coverage:.4f}"
        )

    # ── Build results table ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    metric_cols = [
        f"hit_rate@{k}", f"precision@{k}", f"recall@{k}",
        f"ndcg@{k}", f"map@{k}", f"coverage@{k}",
    ]

    # Sort by NDCG (best metric for ranking quality)
    results_df = results_df.sort_values(f"ndcg@{k}", ascending=False)

    return results_df, metric_cols


def print_leaderboard(results_df: pd.DataFrame, metric_cols: list, k: int = K):
    """Print a formatted leaderboard table."""
    print("\n" + "="*75)
    print(f"  RECOIQ MODEL LEADERBOARD — Top-{k} Recommendations")
    print("="*75)

    # Header
    header = f"{'Model':<12}"
    for col in metric_cols:
        label = col.split("@")[0].replace("_", " ").title()[:9]
        header += f"  {label:>9}"
    print(header)
    print("-"*75)

    # Rows
    for _, row in results_df.iterrows():
        line = f"{row['Model']:<12}"
        for col in metric_cols:
            val = row.get(col, 0.0)
            line += f"  {val:>9.4f}"
        print(line)

    print("="*75)
    best_model = results_df.iloc[0]["Model"]
    best_ndcg  = results_df.iloc[0][f"ndcg@{k}"]
    print(f"\n🏆 Best model by NDCG@{k}: {best_model} ({best_ndcg:.4f})")
    print()


def save_results(results_df: pd.DataFrame, k: int = K):
    """Save evaluation results to reports/."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    csv_path = reports_dir / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Also save as JSON for the API/dashboard later
    json_path = reports_dir / "evaluation_results.json"
    results_dict = results_df.set_index("Model").to_dict(orient="index")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users",    type=int,  default=500,
                        help="Number of users to evaluate")
    parser.add_argument("--k",          type=int,  default=10,
                        help="Recommendation cutoff")
    parser.add_argument("--fast",       action="store_true",
                        help="Skip slow CF models (User-CF, Item-CF)")
    args = parser.parse_args()

    results_df, metric_cols = run_evaluation(
        n_eval_users=args.n_users,
        k=args.k,
        slow_models=not args.fast,
    )

    print_leaderboard(results_df, metric_cols, k=args.k)
    save_results(results_df, k=args.k)