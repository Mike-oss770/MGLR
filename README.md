# Reusable Unsupervised Feature Ranking with Multi-Resolution Global–Local Structural Evidence

**MGLR: Multi-Resolution Global–Local Evidence Ranking**

> MGLR integrates globally associated and locally smooth feature evidence across multiple structural resolutions into a single reusable ranking, followed by bounded redundancy control over its leading candidates.

## Use

Requires Python 3.9+ with NumPy, SciPy, and scikit-learn.

```bash
python -m pip install numpy scipy scikit-learn
```

```python
from mglr import mglr_rank

ranking = mglr_rank(X_train, seed=2026)  # best-to-worst, zero-based feature indices
selected_20 = ranking[:20]
```

`X_train` must be a finite dense numeric array shaped `(samples, features)`. Fit the ranking on training data only and reuse its prefixes for feature budgets. The selector does not accept labels.

## Method

The implementation centers each feature and scales nonconstant columns to unit L2 norm. Randomized SVD produces latent representations at dimensions `[6, 10, 14, 18, 24]`, clipped to the available dimensions. For each dimension, five RBF scales use multipliers `[0.35, 0.6, 1.0, 1.7, 3.0]` relative to the median positive pairwise squared distance (estimated from at most 300 rows). Each of the resulting 25 views estimates feature-to-geometry dependence with 3 deterministic random Fourier draws of dimension 64; higher dependence receives a better percentile rank.

For each latent dimension, a `k=10` nearest-neighbor graph gives a local feature-smoothness ranking; lower normalized graph smoothness is better. The same local ranking is reused across that dimension's five RBF scales. Each view combines ranks as `0.75 * global + 0.25 * local`. Per feature, the 25 fused ranks are sorted, three values are trimmed from each end, and the remaining 19 are averaged.

Finally, the leading 300 candidates are ordered greedily by `relevance - 0.02 * maximum_absolute_correlation_to_selected`. Since absolute correlation is in `[0,1]`, the per-step redundancy penalty is bounded by `0.02`. Remaining features retain their relevance order. This produces one reusable ranking for all budgets.

The fixed settings are in `CONFIG` in [`mglr.py`](mglr.py). Randomness is controlled by the `seed` argument. This minimal repository contains only the method and this README; it includes no datasets, baselines, or evaluation runner.
