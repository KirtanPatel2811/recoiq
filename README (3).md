# 🎬 RecoIQ — Personalized Movie Recommendation Engine

> Production-grade recommender system built on MovieLens 1M comparing 6 algorithms — from popularity baselines to Neural Collaborative Filtering with PyTorch GPU training.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)](https://streamlit.io)

---

## 📊 Results — Top-10 Recommendations (500 users, temporal split)

| Model | Hit Rate@10 | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|---|
| **Popularity** | 0.329 | 0.057 | 0.049 | 0.066 | 0.023 |
| **NCF (Neural CF)** | 0.325 | 0.052 | 0.061 | **0.067** | **0.149** |
| SVD | 0.155 | 0.020 | 0.014 | 0.019 | 0.007 |
| User-CF | 0.059 | 0.007 | 0.006 | 0.008 | 0.412 |
| ALS | 0.016 | 0.002 | 0.003 | 0.002 | 0.122 |

**Key insights:**
- Popularity baseline beats most complex models on NDCG — a famous and well-documented RecSys result. Popular movies are popular because most users haven't seen them and tend to enjoy them.
- NCF matches Popularity on accuracy but delivers **7× better catalog coverage** (0.149 vs 0.023), meaning it recommends diverse, personalised items rather than the same 20 blockbusters to everyone.
- NCF BPR loss converged cleanly: **0.344 → 0.128** over 20 epochs on RTX 3060.

---

## 🏗️ Architecture

```
MovieLens 1M
(1M ratings · 6,040 users · 3,706 movies)
              │
              ▼
┌─────────────────────────────┐
│       Data Pipeline          │
│  loader.py                   │
│  interaction_matrix.py       │  ← Sparse CSR matrix (5.07% density)
│  splitter.py                 │  ← Temporal split (no future leakage)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│                    6 Models                          │
│                                                      │
│  Tier 1 — Baselines                                  │
│    popularity.py   ← top-N most rated globally       │
│    user_cf.py      ← cosine similarity, neighbours   │
│    item_cf.py      ← precomputed item-item matrix     │
│                                                      │
│  Tier 2 — Matrix Factorisation                       │
│    svd_model.py    ← Funk SVD via Surprise           │
│    als_model.py    ← implicit ALS (confidence matrix)│
│                                                      │
│  Tier 3 — Neural                                     │
│    ncf_model.py    ← Embeddings + MLP + BPR loss    │
│                       PyTorch · RTX 3060 GPU         │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Evaluation Framework     │
│  metrics.py                  │  ← NDCG, Recall, MAP, Hit Rate, Coverage
│  evaluator.py                │  ← All models on same test set
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Production Layer        │
│  FastAPI (6 endpoints)       │
│  Streamlit Dashboard         │
└─────────────────────────────┘
```

---

## 🧠 Models Explained

### Tier 1 — Baselines

**Popularity**
Recommends the globally most-rated movies to everyone. Non-personalised but surprisingly hard to beat — especially for new users who haven't built a taste profile. Score = weighted combination of rating count (70%) and average rating (30%).

**User-CF (User-based Collaborative Filtering)**
Finds users with similar rating histories using mean-centered cosine similarity, then recommends movies those neighbours liked. Memory-based — no training phase, but O(n_users²) at inference. Requires minimum 5 common items to form a valid neighbour.

**Item-CF (Item-based Collaborative Filtering)**
Precomputes a full 3260×3260 item-item cosine similarity matrix at fit time. At inference, scores unseen movies by aggregating similarities to the user's already-rated items weighted by their ratings. Amazon switched from User-CF to Item-CF in 2003 — item relationships are more stable than user preferences over time.

### Tier 2 — Matrix Factorisation

**SVD (Funk SVD via Surprise)**
Decomposes the user-item matrix into latent factor matrices P (users) and Q (items):
```
R ≈ P × Qᵀ + bias_u + bias_i + global_mean
```
Trained with SGD minimising regularised MSE. Captures hidden structure (genre preference, director style) without explicit feature engineering.

**ALS (Alternating Least Squares via implicit)**
Designed for implicit feedback — treats ALL ratings as confidence signals rather than explicit preferences. Confidence = 1 + α × rating (α=40). ALS alternates between fixing user factors and solving for item factors in closed form — faster convergence than SGD for implicit data.

### Tier 3 — Neural

**NCF (Neural Collaborative Filtering — PyTorch)**

Architecture:
```
User ID ──► Embedding(6040, 64) ──►
                                    Concat(128)
Item ID ──► Embedding(3260, 64) ──►
                ↓
        Linear(128 → 64) → ReLU → Dropout(0.2)
        Linear(64  → 32) → ReLU → Dropout(0.2)
        Linear(32  →  1) → Score
```

Trained with **BPR (Bayesian Personalised Ranking) loss**:
```
Loss = -mean(log(σ(score_positive - score_negative)))
```
BPR directly optimises ranking — pushes positive items above randomly sampled negatives — rather than minimising rating prediction error (MSE). This is more aligned with the actual recommendation objective.

Negative sampling: for each (user, positive_item) pair, 4 random unrated items are sampled as negatives per epoch.

---

## 📐 Evaluation Metrics

All metrics computed at cutoff K=10 on a **temporal test split** (last 15% of each user's ratings chronologically).

| Metric | Formula | What it measures |
|---|---|---|
| **Hit Rate@K** | 1 if any relevant item in top-K | Did the model help at all? |
| **Precision@K** | \|relevant ∩ top-K\| / K | Quality of recommendations |
| **Recall@K** | \|relevant ∩ top-K\| / \|relevant\| | Coverage of user's interests |
| **NDCG@K** | DCG / IDCG (position-discounted) | Ranking quality (best metric) |
| **MAP@K** | Mean Average Precision | Consistent ranking across users |
| **Coverage@K** | \|unique recommended items\| / total | Catalog diversity |

**Why temporal split?** Random split leaks the future into training — a user's 2010 rating would train on information that didn't exist when their 2008 ratings were made. Temporal split mirrors production: always predict future behaviour from past signals.

**Why NDCG over accuracy?** Recommendation is a ranking problem, not classification. Getting the right movie at rank 1 is far more valuable than finding it at rank 10. NDCG discounts hits by log₂(position+1), matching real user scroll behaviour.

---

## 🗂️ Project Structure

```
recoiq/
├── src/
│   ├── data/
│   │   ├── loader.py              # Load MovieLens 1M (.dat files)
│   │   ├── interaction_matrix.py  # Sparse CSR user-item matrix
│   │   └── splitter.py            # Temporal train/val/test split
│   ├── models/
│   │   ├── popularity.py          # Non-personalised baseline
│   │   ├── user_cf.py             # User-based CF
│   │   ├── item_cf.py             # Item-based CF (precomputed)
│   │   ├── svd_model.py           # SVD matrix factorisation
│   │   ├── als_model.py           # ALS implicit feedback
│   │   └── ncf_model.py           # Neural CF (PyTorch + BPR)
│   ├── evaluation/
│   │   ├── metrics.py             # NDCG, MAP, Recall, Precision
│   │   └── evaluator.py           # Compare all models
│   ├── api/
│   │   └── main.py                # FastAPI REST API
│   └── app/
│       └── streamlit_app.py       # Streamlit dashboard
├── data/
│   ├── raw/                       # MovieLens .dat files (gitignored)
│   └── processed/                 # Splits + interaction matrix
├── models/                        # Saved .pkl artifacts (gitignored)
├── reports/                       # Evaluation results CSV/JSON
├── notebooks/                     # EDA and exploration
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas 2.1, numpy 1.26, scipy sparse |
| Baseline CF | scikit-learn cosine similarity |
| Matrix factorisation | scikit-surprise (SVD), implicit (ALS) |
| Neural CF | PyTorch 2.x + CUDA (RTX 3060) |
| Evaluation | Custom metrics from scratch |
| API | FastAPI 0.104, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Experiment tracking | MLflow |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n recoiq python=3.10 -y
conda activate recoiq
pip install -r requirements.txt

# PyTorch with CUDA (RTX 3060)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Dataset

Download **MovieLens 1M** from https://files.grouplens.org/datasets/movielens/ml-1m.zip

Extract and place `ratings.dat`, `movies.dat`, `users.dat` into `data/raw/`.

### 3. Run the Pipeline

```bash
# Phase 1 — Data pipeline
python src/data/loader.py data/raw
python src/data/interaction_matrix.py
python -m src.data.splitter

# Phase 2 — Baselines
python src/models/popularity.py
python src/models/user_cf.py
python src/models/item_cf.py

# Phase 3 — Matrix factorisation
python src/models/svd_model.py
python src/models/als_model.py

# Phase 4 — Neural CF (GPU)
python src/models/ncf_model.py

# Phase 5 — Evaluate all models
python src/evaluation/evaluator.py --n-users 500

# Phase 6 — Start API + Dashboard
uvicorn src.api.main:app --port 8000
streamlit run src/app/streamlit_app.py
```

---

## 🌐 API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/recommend/{user_id}?model=ncf&k=10` | Top-K personalised recommendations |
| GET | `/recommend/{user_id}/explain` | Recommendations with explanations |
| GET | `/similar/{movie_id}?k=10` | Similar movies (ALS item factors) |
| GET | `/model/compare` | Leaderboard metrics for all models |
| POST | `/recommend/new_user` | Cold-start recommendations |
| GET | `/movies/search?q=matrix` | Search movies by title |
| GET | `/health` | API health check |
| GET | `/docs` | Interactive Swagger UI |

### Example Requests

```bash
# Get NCF recommendations for user 1
curl "http://localhost:8000/recommend/1?model=ncf&k=10"

# Find movies similar to The Matrix (movie_id=2571)
curl "http://localhost:8000/similar/2571?k=5"

# Cold start — new user with ratings
curl -X POST "http://localhost:8000/recommend/new_user" \
  -H "Content-Type: application/json" \
  -d '{"ratings": {"1": 5.0, "2571": 4.5, "318": 5.0}, "k": 10}'
```

---

## 🎛️ Streamlit Dashboard Screens

| Screen | What it shows |
|---|---|
| 🏠 Home | Dataset stats, rating distribution, top movies |
| 🎯 For You | Pick any user ID + model → see personalised top-10 with scores |
| 🎬 Similar Movies | Pick a movie → find 10 most similar via ALS item factors |
| 👤 New User | Rate 5-10 seed movies → instant cold-start recommendations |
| 📊 Model Comparison | Leaderboard table + NDCG bar chart + Recall vs Precision scatter |

---

## 🎤 Interview Prep

This project prepares you for every RecSys interview question:

**Collaborative Filtering**
- Q: *"Difference between user-based and item-based CF?"* — User-CF finds similar users; Item-CF finds similar items. Item-CF is more stable (items don't change) and easier to explain ("because you liked X").
- Q: *"Why does item-CF scale better?"* — Precompute item-item similarity once; user-CF requires recomputing all user similarities at inference.

**Matrix Factorisation**
- Q: *"How does SVD work for recommendations?"* — Decomposes R ≈ P × Qᵀ; learns latent factors via SGD minimising regularised MSE. Note: it's Funk SVD (SGD-based), not true SVD (too slow at O(mn²)).
- Q: *"ALS vs SVD — when to use which?"* — SVD for explicit ratings; ALS for implicit feedback (clicks, watches). ALS uses confidence weighting; ALS parallelises better.

**Neural Methods**
- Q: *"Why embeddings over one-hot?"* — One-hot is sparse and doesn't generalise. Embeddings are dense learned representations that capture latent similarity.
- Q: *"Why BPR loss instead of MSE?"* — MSE optimises rating prediction; BPR optimises ranking directly, which is what RecSys actually cares about.

**Evaluation**
- Q: *"Why is NDCG better than accuracy?"* — Recommendation is ranking, not classification. Position matters — a hit at rank 1 is far more valuable than rank 10. NDCG discounts by log₂(position+1).
- Q: *"Why temporal split?"* — Random split leaks future ratings into training. Temporal split mirrors production reality.

**Cold Start**
- Q: *"How do you handle new users?"* — Fall back to popularity; collect seed ratings to bootstrap a taste profile; use content-based features (genre) to bootstrap before enough collaborative signal exists.

**Scaling**
- Q: *"How would you scale to 100M users?"* — Replace exact cosine with Approximate Nearest Neighbours (Faiss/Annoy); distribute ALS with Spark; serve pre-computed recommendations from a feature store; use two-stage retrieval (fast candidate generation → precise re-ranking).

---

## 👤 Author

**Kirtan Patel**
B.Tech CSE · CSPIT, CHARUSAT
GitHub: [@KirtanPatel2811](https://github.com/KirtanPatel2811)

---

## 📚 References

- Grouplens MovieLens Dataset: https://grouplens.org/datasets/movielens/
- He et al. (2017) — Neural Collaborative Filtering: https://arxiv.org/abs/1708.05031
- Rendle et al. (2009) — BPR: Bayesian Personalised Ranking: https://arxiv.org/abs/1205.2618
- Hu et al. (2008) — Collaborative Filtering for Implicit Feedback: https://ieeexplore.ieee.org/document/4781121
- Koren et al. (2009) — Matrix Factorization Techniques: https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
