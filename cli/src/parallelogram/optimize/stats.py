"""Uncertainty and the rule for when a comparison supports a conclusion.

Two different kinds of noise are kept apart, because they answer different
questions and are easy to confuse:

* variation across runs: train the same arm with another seed and the number
  moves. Measured as the spread of per-seed means.
* evaluation uncertainty: the development or test set is a finite sample.
  Measured by bootstrapping examples.

A difference between two arms is called supported only when it clears both,
and only with at least three seeds per arm. Everything else is "insufficient
evidence", which is the default outcome and not a failure.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

MIN_SEEDS = 3
SUPPORTED, CONTRADICTED, INSUFFICIENT = "supported", "contradicted", "insufficient_evidence"

# Two-sided 95% Student t critical values by degrees of freedom.
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042)


def t95(df: float) -> float:
    if df < 1:
        return math.inf
    if df < 31:
        return _T95[int(df) - 1]
    # Beyond the table, round toward the wider interval: never below the true value.
    return 2.042 if df < 40 else 2.021 if df < 60 else 2.000 if df < 120 else 1.980


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def across_seeds(values: Sequence[float]) -> dict:
    """Spread of one arm's per-seed results (variation across runs)."""
    return {"n": len(values), "mean": mean(values), "std": sample_std(values),
            "min": min(values), "max": max(values), "values": list(values)}


def bootstrap_ci(values: Sequence[float], samples: int, seed: int = 0) -> list[float]:
    """Percentile 95% interval of the mean over resampled examples."""
    n = len(values)
    rng = random.Random(f"bootstrap:{seed}")
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def compare(candidate: Sequence[float], baseline: Sequence[float], *, higher_is_better: bool,
            candidate_examples: Sequence[float] | None = None,
            baseline_examples: Sequence[float] | None = None, samples: int = 1000) -> dict:
    """Is the candidate arm better than the baseline arm?

    `candidate` and `baseline` are per-seed means. The optional per-example
    values (averaged over seeds, aligned by example) give a paired bootstrap
    interval for the same difference. Differences are oriented so that
    positive always means "candidate better".
    """
    sign = 1.0 if higher_is_better else -1.0
    difference = sign * (mean(candidate) - mean(baseline))
    out: dict = {"difference": difference, "seeds": [len(candidate), len(baseline)],
                 "across_seed_ci95": None, "evaluation_ci95": None}
    if candidate_examples is not None and baseline_examples is not None and candidate_examples:
        paired = [sign * (c - b) for c, b in zip(candidate_examples, baseline_examples)]
        out["evaluation_ci95"] = bootstrap_ci(paired, samples)
    if min(len(candidate), len(baseline)) < MIN_SEEDS:
        out.update(verdict=INSUFFICIENT,
                   reason=f"fewer than {MIN_SEEDS} seeds per arm; run-to-run variation cannot be estimated")
        return out
    var_c, var_b = sample_std(candidate) ** 2 / len(candidate), sample_std(baseline) ** 2 / len(baseline)
    se = math.sqrt(var_c + var_b)
    if se <= 1e-9 * max(1.0, abs(mean(candidate)), abs(mean(baseline))):  # zero up to rounding
        out.update(verdict=INSUFFICIENT, reason="no variation across seeds was measured, so noise is unknown")
        return out
    # Welch-Satterthwaite degrees of freedom.
    df = (var_c + var_b) ** 2 / sum(v ** 2 / (n - 1) for v, n in ((var_c, len(candidate)), (var_b, len(baseline))) if v > 0)
    half = t95(df) * se
    out["across_seed_ci95"] = [difference - half, difference + half]
    low_seed, high_seed = out["across_seed_ci95"]
    eval_ci = out["evaluation_ci95"]
    if low_seed > 0 and (eval_ci is None or eval_ci[0] > 0):
        out.update(verdict=SUPPORTED, reason="the difference exceeds both run-to-run variation and "
                                             "evaluation uncertainty at the 95% level")
    elif high_seed < 0 and (eval_ci is None or eval_ci[1] < 0):
        out.update(verdict=CONTRADICTED, reason="the baseline is better beyond both run-to-run variation "
                                                "and evaluation uncertainty at the 95% level")
    else:
        out.update(verdict=INSUFFICIENT, reason="the difference is within run-to-run variation or "
                                                "evaluation uncertainty")
    return out
