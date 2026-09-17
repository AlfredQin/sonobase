"""Statistical utilities for the analysis suite.

Wilcoxon signed-rank, Cohen's kappa, bootstrap CI, BH-FDR correction, and
Bland-Altman summary statistics. All functions are pure (no I/O), accept
NumPy arrays, and return plain Python floats / dataclasses.

The statistical methodology, including the BH-FDR convention for the primary
endpoint, is described in the Methods of the SonoBase paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Wilcoxon paired signed-rank test
# ---------------------------------------------------------------------------


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test on per-sample errors.

    Returns ``(W_statistic, p_value)``. Both arrays must be 1-D and the same
    length (paired observations). For ``n < 10`` the result may be
    inaccurate (mode='exact' is used when available; SciPy emits a warning
    that callers should propagate to the report).
    """
    from scipy.stats import wilcoxon

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D arrays, got {x.ndim}-D")

    # If all paired differences are zero, wilcoxon raises. Guard explicitly.
    if np.allclose(x, y):
        return 0.0, 1.0

    res = wilcoxon(x, y, alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# Cohen's kappa for clinical reclassification
# ---------------------------------------------------------------------------


def cohens_kappa(gt_class: np.ndarray, pred_class: np.ndarray) -> float:
    """Cohen's kappa between two integer-valued classifications (paired).

    Used at the EF 40% / 35% thresholds: classify each patient as ≤threshold
    or not, for both GT EF and model-predicted EF; kappa = agreement
    corrected for chance.

    Interpretation key:
      κ < 0.20 = poor; 0.21–0.40 = fair; 0.41–0.60 = moderate;
      0.61–0.80 = substantial; 0.81–1.00 = almost perfect.
    """
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(np.asarray(gt_class).astype(int),
                                   np.asarray(pred_class).astype(int)))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for the mean
# ---------------------------------------------------------------------------


def bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean of a 1-D array.

    Returns ``(mean, lower, upper)``. Uses NumPy's per-call random
    generator, deterministic given ``seed``. Sample-with-replacement
    n_bootstrap times; report the mean's percentile range.
    """
    arr = np.asarray(values).ravel()
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return float(arr.mean()), lo, hi


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------


def bh_fdr_correct(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[bool], List[float]]:
    """Apply Benjamini-Hochberg FDR correction to a family of raw p-values.

    Wraps ``statsmodels.stats.multitest.multipletests(method='fdr_bh')``.
    Returns:
        ``(rejected, q_values)`` — both lists in the same order as the input.
        ``rejected[i]`` is True iff ``q_values[i] < alpha``.

    BH-FDR is applied as a SINGLE FAMILY across all secondary
    tests in a study. The pre-specified primary endpoint is NOT included in
    this family.
    """
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return [], []

    rejected, q_values, _, _ = multipletests(
        np.asarray(p_values, dtype=float), alpha=alpha, method="fdr_bh"
    )
    return [bool(r) for r in rejected], [float(q) for q in q_values]


# ---------------------------------------------------------------------------
# Bland-Altman summary statistics
# ---------------------------------------------------------------------------


@dataclass
class BlandAltmanResult:
    """Numeric outputs of a Bland-Altman analysis.

    All values in the same units as the input arrays (mm for HC/AC, % for
    EF, mL for prostate volume).
    """

    bias: float          # mean(pred - gt); positive = model overestimates
    std_diff: float      # sample std of (pred - gt)
    loa_lower: float     # bias - 1.96 * std
    loa_upper: float     # bias + 1.96 * std
    n: int

    def to_dict(self) -> dict:
        return {
            "bias": self.bias,
            "std_diff": self.std_diff,
            "loa_lower": self.loa_lower,
            "loa_upper": self.loa_upper,
            "n": self.n,
        }


def bland_altman_stats(gt: np.ndarray, pred: np.ndarray) -> BlandAltmanResult:
    """Compute Bland-Altman bias and 95% limits of agreement.

    The differences are computed as ``pred - gt`` so that a positive bias
    indicates the model overestimates the clinical quantity.
    """
    gt = np.asarray(gt, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")

    diff = pred - gt
    bias = float(diff.mean())
    std = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return BlandAltmanResult(
        bias=bias,
        std_diff=std,
        loa_lower=bias - 1.96 * std,
        loa_upper=bias + 1.96 * std,
        n=int(diff.size),
    )


# ---------------------------------------------------------------------------
# Convenience aggregators (used by per-analysis report writers)
# ---------------------------------------------------------------------------


def percent_within(values: np.ndarray, threshold: float) -> float:
    """Fraction of values strictly below ``threshold``, expressed as a percent."""
    arr = np.asarray(values).ravel()
    if arr.size == 0:
        return float("nan")
    return float((arr < threshold).mean() * 100.0)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient. NaN-safe (drops paired NaNs)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])
