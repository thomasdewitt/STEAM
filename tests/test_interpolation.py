"""The class-to-class resample must preserve horizontal periodicity.

torch's interpolate clamps at the array edge, so before 2026-08-05 every
class hop replicated the outer half source cell and the delivered field was
not periodic at all -- an end-to-end mean ramp of 0.2 to 3 times the field's
own standard deviation, worst where the coarsest class grid is only a few
cells across. align_corners=False did not fix this; it is a separate defect
from the corner-alignment ramp.
"""

import numpy as np
import pytest

from steam.utils import zoom_trilinear, zoom_bilinear, VOLUME_GUARD_CELLS


def periodic_linear_reference(coarse, n_out):
    """Exact periodic linear interpolant at target cell centres."""
    n_in = coarse.size
    xs = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
    padded = np.r_[coarse[-1], coarse, coarse[0]]
    return np.interp(xs, np.arange(-1, n_in + 1), padded)


@pytest.mark.parametrize("n_in,n_out", [(8, 16), (8, 12), (4, 6), (2, 3),
                                        (8, 24), (6, 18)])
def test_resample_matches_periodic_interpolant(n_in, n_out):
    """Integer and fractional ratios alike, to float32 roundoff."""
    coarse = np.cos(2 * np.pi * np.arange(n_in) / n_in)
    field = (coarse[:, None, None] * np.ones((1, 4, 4))).astype(np.float32)
    got = np.asarray(zoom_trilinear(field, (n_out, 4, 4)))[:, 0, 0]
    assert np.allclose(got, periodic_linear_reference(coarse, n_out),
                       atol=1e-5)


def test_resample_does_not_clamp_the_edge():
    """The clamped path repeated the end sample; the periodic one must not."""
    coarse = np.cos(2 * np.pi * np.arange(8) / 8)
    field = (coarse[:, None, None] * np.ones((1, 4, 4))).astype(np.float32)
    got = np.asarray(zoom_trilinear(field, (32, 4, 4)))[:, 0, 0]
    assert got[0] != got[1], "first two target cells are flat -- edge clamped"
    assert got[-1] != got[-2], "last two target cells are flat -- edge clamped"
    # A periodic field's seam gap is an ordinary gap, not a discontinuity.
    seam = abs(got[-1] - got[0])
    assert seam < 2 * np.median(np.abs(np.diff(got)))


def test_z_axis_is_not_wrapped():
    """z is zero-padded in the cascade, so it must keep the clamped edge."""
    field = np.zeros((4, 4, 4), dtype=np.float32)
    field[:, :, 0] = 1.0                      # a bottom-only feature
    out = np.asarray(zoom_trilinear(field, (8, 8, 8)))
    assert out[0, 0, -1] == pytest.approx(0.0, abs=1e-6), \
        "bottom level leaked to the top -- z was wrapped"


def test_bilinear_is_periodic_on_both_axes():
    coarse = np.cos(2 * np.pi * np.arange(8) / 8)
    field = (coarse[:, None] * np.ones((1, 8))).astype(np.float32)
    got = np.asarray(zoom_bilinear(field, (16, 16)))[:, 0]
    ref = periodic_linear_reference(coarse, 16)
    assert np.allclose(got, ref, atol=1e-5)


def test_bilinear_honours_non_periodic_axes():
    """A partial nest's p_bottom does not wrap; it must clamp as before."""
    field = np.zeros((4, 4), dtype=np.float32)
    field[0, :] = 1.0
    out = np.asarray(zoom_bilinear(field, (8, 8), periodic=(False, False)))
    assert out[-1, 0] == pytest.approx(0.0, abs=1e-6)


def test_downsampling_is_refused():
    field = np.zeros((8, 4, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="downsamples"):
        zoom_trilinear(field, (4, 4, 4))


def test_expensive_pad_is_refused_on_large_grids():
    """A ratio with no common factor needs a pad as big as the array.

    Cheap at test sizes, so the guard only fires above VOLUME_GUARD_CELLS --
    which is the point: correctness is always available, the refusal is
    about not paying for it silently on a production grid.
    """
    n = int(VOLUME_GUARD_CELLS ** (1 / 3)) + 1
    field = np.zeros((n, n, n), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="larger working array"):
        zoom_trilinear(field, (2 * n + 1, 2 * n + 1, n))
