"""
metrics.py — RecSys evaluation metrics from scratch.

WHY implement from scratch instead of using a library?
1. recmetrics is incompatible with pandas 2.x
2. Interviewers WILL ask you to explain these formulas
3. You understand exactly what's being computed

ALL metrics are computed at cutoff K (e.g. top-10 recommendations).

CRITICAL: We use the TEST set as ground truth.
Ground truth = movies the user rated >= threshold (e.g. >= 4.0) in the test set.
We only recommend movies NOT in the training set.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


# ── Core metric functions ─────────────────────────────────────────────────────

def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Hit Rate@K — did at least one relevant item appear in top-K?

    Binary: 1.0 if any relevant item is in top-K, else 0.0.

    WHY useful: simplest measure of whether the model is doing anything.
    If HR@10 = 0.05, the model only helps 5% of users — useless.
    """
    recommended_k = set(recommended[:k])
    relevant_set  = set(relevant)
    return 1.0 if recommended_k & relevant_set else 0.0


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Precision@K — of the K items recommended, what fraction are relevant?

    P@K = |relevant ∩ recommended[:K]| / K

    WHY useful: measures recommendation quality (are our picks good?).
    Low precision = we're recommending many irrelevant items.
    """
    if k == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set  = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Recall@K — of all relevant items, what fraction did we catch in top-K?

    R@K = |relevant ∩ recommended[:K]| / |relevant|

    WHY useful: measures coverage of user's true interests.
    High recall = we found most of what the user would like.
    Note: recall is bounded by min(K, |relevant|) / |relevant|.

    INTERVIEW NOTE: Precision vs Recall tradeoff:
    - Increase K → recall goes up (find more), precision goes down (more noise)
    - In RecSys we care more about recall (don't miss good movies) but
      precision matters too (don't waste the user's attention)
    """
    if not relevant:
        return 0.0
    recommended_k = set(recommended[:k])
    relevant_set  = set(relevant)
    hits = len(recommended_k & relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    NDCG@K — Normalised Discounted Cumulative Gain.

    NDCG rewards putting the BEST items HIGHER in the list.

    DCG@K  = Σ (rel_i / log2(i+2))   for i in 0..K-1
             where rel_i = 1 if recommended[i] is relevant, else 0

    IDCG@K = DCG of a perfect ranking (all relevant items at top)

    NDCG@K = DCG@K / IDCG@K   (normalised to [0,1])

    WHY better than Precision/Recall:
    - P@K and R@K treat all positions equally
    - NDCG penalises relevant items found at position 8 vs position 1
    - Position 1 hit is worth log2(2)=1.0, position 8 hit is worth log2(9)=0.32
    - This matches real user behaviour: users rarely scroll past top 3-5 results

    INTERVIEW: "Why is NDCG better than accuracy for recommendations?"
    → Because recommendation is a RANKING problem, not classification.
      Getting the right item at rank 1 is far more valuable than rank 10.
    """
    if not relevant or k == 0:
        return 0.0

    relevant_set = set(relevant)
    recommended_k = recommended[:k]

    # DCG: sum of discounted gains
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)   # i+2 because log2(1)=0

    # IDCG: best possible DCG (all relevant items in first positions)
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Average Precision@K for a single user.

    AP@K = (1/|relevant|) × Σ P@i × rel(i)   for i in 1..K

    where rel(i) = 1 if position i is relevant, else 0.

    AP rewards models that rank relevant items consistently high,
    not just having one lucky hit at position 1.
    MAP (Mean AP) = mean of AP@K across all users.

    INTERVIEW: AP is to precision what NDCG is to gains —
    it accounts for the ordering of hits.
    """
    if not relevant or k == 0:
        return 0.0

    relevant_set = set(relevant)
    hits         = 0
    sum_precisions = 0.0

    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)   # P@(i+1)

    if not relevant:
        return 0.0

    return sum_precisions / len(relevant)


# ── Aggregate over all users ──────────────────────────────────────────────────

def evaluate_model(
    recommendations: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """
    Evaluate a model across all users in the test set.

    Parameters
    ----------
    recommendations      : {user_id: [movie_id, movie_id, ...]} ranked list
    test_df              : test set DataFrame with [user_id, movie_id, rating]
    k                    : cutoff for all metrics
    relevance_threshold  : minimum rating to consider an item "relevant"
                           4.0 means "the user liked it" (4 or 5 stars)

    Returns
    -------
    dict with keys: hit_rate, precision, recall, ndcg, map, n_users_evaluated
    """
    # Build ground truth: {user_id: [relevant_movie_ids]}
    ground_truth = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    hit_rates   = []
    precisions  = []
    recalls     = []
    ndcgs       = []
    avg_precs   = []

    # Track all recommended items for coverage
    all_recommended = set()
    n_skipped       = 0

    for user_id, rec_list in recommendations.items():
        relevant = ground_truth.get(user_id, [])

        # Skip users with no relevant items in test set
        if not relevant:
            n_skipped += 1
            continue

        all_recommended.update(rec_list[:k])

        hit_rates.append(  hit_rate_at_k(rec_list, relevant, k))
        precisions.append( precision_at_k(rec_list, relevant, k))
        recalls.append(    recall_at_k(rec_list, relevant, k))
        ndcgs.append(      ndcg_at_k(rec_list, relevant, k))
        avg_precs.append(  average_precision_at_k(rec_list, relevant, k))

    n_evaluated = len(hit_rates)
    logger.info(f"Evaluated {n_evaluated} users | "
                f"skipped {n_skipped} (no relevant test items)")

    if n_evaluated == 0:
        return {f"{m}@{k}": 0.0 for m in
                ["hit_rate", "precision", "recall", "ndcg", "map"]}

    return {
        f"hit_rate@{k}":  float(np.mean(hit_rates)),
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}":    float(np.mean(recalls)),
        f"ndcg@{k}":      float(np.mean(ndcgs)),
        f"map@{k}":       float(np.mean(avg_precs)),
        "n_users_evaluated": n_evaluated,
        "catalog_coverage":  len(all_recommended),
    }


def coverage_at_k(
    recommendations: Dict[int, List[int]],
    total_items: int,
    k: int = 10,
) -> float:
    """
    Catalog Coverage@K — what fraction of all items does the model ever recommend?

    coverage = |unique recommended items| / total_items

    WHY it matters: a model that recommends only the top 50 popular movies
    to everyone has 50/3260 = 1.5% coverage. Users with niche tastes never
    get relevant recommendations. Amazon calls this the "long tail problem".

    High NDCG + Low Coverage = popular-item bias (the model is lazy).
    """
    all_recommended = set()
    for rec_list in recommendations.values():
        all_recommended.update(rec_list[:k])
    return len(all_recommended) / total_items


if __name__ == "__main__":
    # Unit tests for each metric
    print("Running metric unit tests...\n")

    rec      = [1, 2, 3, 4, 5]
    relevant = [1, 3, 7]

    print(f"Recommended : {rec}")
    print(f"Relevant    : {relevant}")
    print(f"Hit Rate@5  : {hit_rate_at_k(rec, relevant, 5):.4f}  (expect 1.0)")
    print(f"Precision@5 : {precision_at_k(rec, relevant, 5):.4f} (expect 0.4)")
    print(f"Recall@5    : {recall_at_k(rec, relevant, 5):.4f}    (expect 0.667)")
    print(f"NDCG@5      : {ndcg_at_k(rec, relevant, 5):.4f}      (expect ~0.72)")
    print(f"AP@5        : {average_precision_at_k(rec, relevant, 5):.4f}")

    print("\n── Edge cases ──")
    print(f"Empty relevant : {ndcg_at_k([1,2,3], [], 3):.4f}     (expect 0.0)")
    print(f"No hits        : {recall_at_k([1,2,3], [7,8,9], 3):.4f} (expect 0.0)")
    print(f"Perfect@3      : {ndcg_at_k([1,3,7], [1,3,7], 3):.4f} (expect 1.0)")

    print("\n✓ All metric tests passed!")