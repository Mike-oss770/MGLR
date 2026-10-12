#!/usr/bin/env python3
"""Minimal, self-contained implementation of MGLR."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd


EPS = 1e-12

CONFIG = {
    "latent_dim_bank": (6, 10, 14, 18, 24),
    "gamma_multiplier_bank": (0.35, 0.60, 1.00, 1.70, 3.00),
    "rff_features": 64,
    "rff_replicates": 3,
    "trim_each_side": 3,
    "svd_n_iter": 2,
    "pairwise_distance_sample": 300,
    "local_k": 10,
    "global_weight": 0.75,
    "local_weight": 0.25,
    "redundancy_eta": 0.02,
    "redundancy_candidate_pool": 300,
}


def _center_unit_columns(Xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(Xz, dtype=np.float32).copy()
    X -= X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0)
    valid = norms > 1e-10
    X[:, valid] /= norms[valid]
    X[:, ~valid] = 0.0
    return X, valid


def _rank_percentile_desc(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ids = np.where(valid)[0]
    order = ids[np.argsort(-score[ids], kind="stable")]
    ranks = np.ones(len(score), dtype=np.float64)
    ranks[order] = (np.arange(len(order)) + 1) / (len(order) + 1)
    return ranks


def _positive_median(values: np.ndarray) -> float:
    positive = values[values > EPS]
    return float(np.median(positive)) if len(positive) else 1.0


def _clipped_dims(dims: tuple[int, ...], n: int, d: int) -> tuple[int, ...]:
    upper = max(2, min(int(n) - 1, int(d) - 1))
    return tuple(max(2, min(int(q), upper)) for q in dims)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= EPS:
        return 0.0
    value = float(np.dot(a, b) / denominator)
    return value if np.isfinite(value) else 0.0


def _local_rank(
    X: np.ndarray, Z: np.ndarray, k: int, valid: np.ndarray
) -> np.ndarray:
    n = Z.shape[0]
    if n < 2:
        return np.ones(X.shape[1], dtype=np.float64)
    kk = min(int(k), n - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1, algorithm="auto", n_jobs=1)
    nn.fit(np.asarray(Z, dtype=np.float32))
    distances, neighbors = nn.kneighbors(
        np.asarray(Z, dtype=np.float32), return_distance=True
    )
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]
    ii = np.repeat(np.arange(n), kk)
    jj = neighbors.reshape(-1)
    dd = distances.reshape(-1)
    sigma = _positive_median(dd)
    weights = np.exp(-(dd * dd) / (2.0 * sigma * sigma + EPS)).astype(np.float32)
    degree = np.bincount(ii, weights=weights, minlength=n).astype(np.float32)
    diffs = np.asarray(X[ii] - X[jj], dtype=np.float32)
    numerator = np.sum((diffs * diffs) * weights[:, None], axis=0)
    denominator = (
        np.sum((np.asarray(X, dtype=np.float32) ** 2) * degree[:, None], axis=0)
        + EPS
    )
    smoothness = numerator / denominator
    smoothness[~valid] = np.inf
    return _rank_percentile_desc(-smoothness, valid)


def _compute_view_evidence(
    Xtr: np.ndarray,
    seed: int,
    *,
    dims: tuple[int, ...],
    gammas: tuple[float, ...],
    n_rff: int,
    rff_reps: int,
    svd_n_iter: int,
    pairwise_sample: int,
    local_k: int,
) -> dict:
    X, valid = _center_unit_columns(Xtr)
    dims = _clipped_dims(tuple(dims), X.shape[0], X.shape[1])
    maxd = max(2, min(max(dims), min(X.shape) - 1))
    U, singular_values, _ = randomized_svd(
        X,
        n_components=maxd,
        n_iter=int(svd_n_iter),
        random_state=int(seed),
    )
    Xc = X - X.mean(axis=0, keepdims=True)
    xn = np.sum(Xc * Xc, axis=0)[:, None]
    global_ranks = []
    local_ranks = []
    reliabilities = []
    metadata = []

    for dim in dims:
        dd = min(int(dim), U.shape[1])
        Z = U[:, :dd] * singular_values[:dd][None, :]
        local_ranks.append(_local_rank(X, Z, local_k, valid))

        rng0 = np.random.default_rng(int(seed) + int(dim) * 101)
        ids = rng0.choice(len(Z), size=min(int(pairwise_sample), len(Z)), replace=False)
        sampled = Z[ids]
        sq = np.sum(sampled * sampled, axis=1, keepdims=True)
        distances2 = np.maximum(sq + sq.T - 2 * sampled @ sampled.T, 0)
        triangle = distances2[np.triu_indices(len(ids), 1)]
        median_distance2 = _positive_median(triangle)

        for gamma_multiplier in gammas:
            replicate_scores = []
            replicate_ranks = []
            gamma = float(gamma_multiplier) / (median_distance2 + EPS)
            for rep in range(int(rff_reps)):
                rng = np.random.default_rng(
                    int(seed)
                    + int(dim) * 1009
                    + int(float(gamma_multiplier) * 1000)
                    + 10007 * int(rep)
                )
                W = rng.normal(
                    scale=np.sqrt(2 * gamma),
                    size=(Z.shape[1], int(n_rff)),
                )
                b = rng.uniform(0.0, 2.0 * np.pi, int(n_rff))
                Phi = np.sqrt(2.0 / float(n_rff)) * np.cos(Z @ W + b)
                Phi -= Phi.mean(axis=0, keepdims=True)
                cross = Xc.T @ Phi
                denominator = xn * np.sum(Phi * Phi, axis=0)[None, :] + EPS
                score = np.sum((cross * cross) / denominator, axis=1)
                score[~valid] = -np.inf
                replicate_scores.append(score)
                replicate_ranks.append(_rank_percentile_desc(score, valid))

            mean_score = np.mean(replicate_scores, axis=0)
            mean_score[~valid] = -np.inf
            global_ranks.append(_rank_percentile_desc(mean_score, valid))

            correlations = []
            for a in range(len(replicate_ranks)):
                for b in range(a + 1, len(replicate_ranks)):
                    correlations.append(
                        _safe_corr(replicate_ranks[a][valid], replicate_ranks[b][valid])
                    )
            reliabilities.append(float(np.mean(correlations)) if correlations else 1.0)
            metadata.append({"dim": int(dim), "gamma_multiplier": float(gamma_multiplier)})

    return {
        "global_ranks": np.asarray(global_ranks, dtype=np.float64),
        "local_ranks": np.asarray(local_ranks, dtype=np.float64),
        "reliabilities": np.asarray(reliabilities, dtype=np.float64),
        "metadata": metadata,
        "valid": valid,
        "X_normalized": X,
    }


def _robust_aggregate(view_ranks: np.ndarray, trim: int) -> np.ndarray:
    ranks = np.asarray(view_ranks, dtype=np.float64)
    order = np.argsort(ranks, axis=0, kind="stable")
    sorted_ranks = np.take_along_axis(ranks, order, axis=0)
    kept = sorted_ranks[int(trim) : ranks.shape[0] - int(trim)]
    return np.mean(kept, axis=0)


def _rank_from_score(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).copy()
    score[~valid] = np.inf
    return np.argsort(score, kind="stable").astype(np.int64)


def _diversified_ordering(
    valid: np.ndarray,
    X: np.ndarray,
    relevance_score: np.ndarray,
    eta: float,
    candidate_pool: int,
) -> np.ndarray:
    d = X.shape[1]
    base_order = _rank_from_score(relevance_score, valid)
    relevance = 1.0 - np.asarray(relevance_score, dtype=np.float64)
    relevance[~valid] = -np.inf
    pool_size = min(int(candidate_pool), int(np.sum(valid)), d)
    candidates = base_order[:pool_size]
    candidate_X = np.asarray(X[:, candidates], dtype=np.float32)
    abs_corr = np.abs(np.asarray(candidate_X.T @ candidate_X, dtype=np.float64))
    np.fill_diagonal(abs_corr, 1.0)

    remaining = list(range(pool_size))
    selected_local: list[int] = []
    while remaining:
        if not selected_local:
            scores = relevance[candidates[remaining]]
        else:
            selected = np.asarray(selected_local, dtype=np.int64)
            redundancy = np.max(abs_corr[np.ix_(remaining, selected)], axis=1)
            scores = relevance[candidates[remaining]] - float(eta) * redundancy
        best_position = int(np.argmax(scores))
        selected_local.append(remaining.pop(best_position))

    diversified = candidates[np.asarray(selected_local, dtype=np.int64)].tolist()
    selected_set = set(diversified)
    tail = [int(j) for j in base_order if int(j) not in selected_set]
    return np.asarray(diversified + tail, dtype=np.int64)


def mglr_rank(X_train: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return one label-free feature ordering (best to worst)."""
    details = _compute_view_evidence(
        X_train,
        int(seed),
        dims=CONFIG["latent_dim_bank"],
        gammas=CONFIG["gamma_multiplier_bank"],
        n_rff=CONFIG["rff_features"],
        rff_reps=CONFIG["rff_replicates"],
        svd_n_iter=CONFIG["svd_n_iter"],
        pairwise_sample=CONFIG["pairwise_distance_sample"],
        local_k=CONFIG["local_k"],
    )
    global_ranks = details["global_ranks"]
    local_ranks = np.repeat(details["local_ranks"], len(CONFIG["gamma_multiplier_bank"]), axis=0)
    fused = (
        CONFIG["global_weight"] * global_ranks
        + CONFIG["local_weight"] * local_ranks
    )
    relevance = _robust_aggregate(fused, CONFIG["trim_each_side"])
    relevance[~details["valid"]] = np.inf
    return _diversified_ordering(
        details["valid"],
        details["X_normalized"],
        relevance,
        CONFIG["redundancy_eta"],
        CONFIG["redundancy_candidate_pool"],
    )


__all__ = ["CONFIG", "mglr_rank"]
