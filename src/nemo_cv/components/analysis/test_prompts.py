"""Tests for the randomised prompt samplers (prompt-noise study).

The point of this file is one claim: our numpy `uniform` point sampler draws
from the same distribution as SAM2's `sample_random_points_from_errors`, which
we chose *not* to call (it allocates a [B,1,H,W,2] tensor on the GPU and its
RNG consumption scales with image size, so seeding it is neither cheap nor
device-independent). Distributional equivalence is asserted here rather than
argued in a docstring.

Run:  uv run pytest nemo_cv/components/analysis/test_prompts.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nemo_cv.components.analysis.prompts import initial_point_from_gt, jitter_box
from nemo_cv.components.models.sam2.modeling.sam2_utils import (
    sample_random_points_from_errors,
)


def _mask(h=32, w=32, shape="blob") -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if shape == "blob":
        m[8:20, 6:26] = 1
    elif shape == "crescent":
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        m[(r < 14) & (r > 9) & (yy < h / 2)] = 1
    elif shape == "single":
        m[5, 7] = 1
    return m


# --------------------------------------------------------------------------
# uniform sampler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["blob", "crescent", "single"])
def test_uniform_point_is_always_inside_the_mask(shape):
    m = _mask(shape=shape)
    rng = np.random.default_rng(0)
    for _ in range(500):
        (x, y), label = initial_point_from_gt(m, method="uniform", rng=rng)
        assert m[int(y), int(x)] == 1, "sampled a background pixel"
        assert label == 1, "initial click against an empty prediction must be positive"


def test_uniform_matches_sam2_sampler_in_distribution():
    """Chi-square GOF: our draws vs SAM2's, over the same foreground.

    SAM2 masks a uniform noise tensor to the error region and takes an argmax,
    which is a uniform draw from that region; ours indexes the foreground list
    directly. Same distribution, so neither should be distinguishable from the
    other at n=6000.
    """
    m = _mask(shape="crescent")
    n_fg = int(m.sum())
    assert n_fg > 40, "test mask is too small to be informative"

    N = 6000
    rng = np.random.default_rng(12345)
    ours = np.zeros(n_fg, dtype=int)
    fg_index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(*np.nonzero(m)))}
    for _ in range(N):
        (x, y), _ = initial_point_from_gt(m, method="uniform", rng=rng)
        ours[fg_index[(int(y), int(x))]] += 1

    torch.manual_seed(12345)
    gt_t = torch.from_numpy(m.astype(bool))[None, None]
    empty = torch.zeros_like(gt_t)
    theirs = np.zeros(n_fg, dtype=int)
    for _ in range(N):
        pts, lbl = sample_random_points_from_errors(gt_t, empty)
        x, y = pts.reshape(-1).tolist()
        assert int(lbl.item()) == 1
        theirs[fg_index[(int(y), int(x))]] += 1

    # Both should be uniform over the foreground; compare each to the flat
    # expectation rather than to each other, so neither sampler is the oracle.
    expected = N / n_fg
    for name, obs in (("ours", ours), ("sam2", theirs)):
        chi2 = float(((obs - expected) ** 2 / expected).sum())
        # 99.9th percentile of chi2 with df = n_fg - 1, generously bounded.
        bound = (n_fg - 1) + 5.0 * np.sqrt(2 * (n_fg - 1))
        assert chi2 < bound, f"{name} is not uniform over the foreground: chi2={chi2:.1f}"


def test_uniform_requires_an_rng():
    with pytest.raises(ValueError, match="requires a seeded"):
        initial_point_from_gt(_mask(), method="uniform", rng=None)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown point_sampling"):
        initial_point_from_gt(_mask(), method="centre", rng=None)


# --------------------------------------------------------------------------
# seeding: the property the 3-seed study depends on
# --------------------------------------------------------------------------


def test_same_seed_reproduces_and_different_seeds_diverge():
    m = _mask(shape="blob")
    a = [tuple(initial_point_from_gt(m, "uniform", np.random.default_rng(42))[0])
         for _ in range(20)]
    b = [tuple(initial_point_from_gt(m, "uniform", np.random.default_rng(42))[0])
         for _ in range(20)]
    c = [tuple(initial_point_from_gt(m, "uniform", np.random.default_rng(2026))[0])
         for _ in range(20)]
    assert a == b, "same seed must reproduce"
    assert a != c, "different seeds must produce different prompts"


def test_center_is_deterministic_and_ignores_the_rng():
    m = _mask(shape="blob")
    p0, _ = initial_point_from_gt(m)
    p1, _ = initial_point_from_gt(m, method="center", rng=np.random.default_rng(7))
    p2, _ = initial_point_from_gt(m, method="center", rng=np.random.default_rng(9))
    assert np.array_equal(p0, p1) and np.array_equal(p1, p2)


def test_zero_jitter_does_not_consume_the_stream():
    """The property that lets one rng serve both prompt types.

    If `jitter_box(pct=0)` drew from the stream, a box run and a point run at
    the same seed would desynchronise, and turning jitter off would shift every
    subsequent point.
    """
    box = np.array([4.0, 4.0, 20.0, 20.0], dtype=np.float32)
    rng = np.random.default_rng(3)
    before = rng.integers(0, 1_000_000)
    rng = np.random.default_rng(3)
    jitter_box(box, 0.0, (32, 32), rng)
    after = rng.integers(0, 1_000_000)
    assert before == after
