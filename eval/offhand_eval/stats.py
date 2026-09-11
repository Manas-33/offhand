"""Small statistics helpers (standard library only).

Decoding is greedy, so repeated runs are identical and seed-based variance is
meaningless. We get confidence intervals by bootstrapping over the eval items
instead: resample the per-item outcomes with replacement and read percentiles.
"""

from __future__ import annotations

import random


def bootstrap_ci(outcomes, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and a (1 - alpha) percentile bootstrap CI for 0/1 outcomes.

    Returns ``(mean, lo, hi)``, or ``(None, None, None)`` for an empty list.
    """
    vals = [1.0 if o else 0.0 for o in outcomes]
    n = len(vals)
    if n == 0:
        return (None, None, None)
    mean = sum(vals) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    lo = means[max(0, int((alpha / 2) * n_boot))]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (mean, lo, hi)
