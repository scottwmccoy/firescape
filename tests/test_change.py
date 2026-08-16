"""Known-answer tests for the change-statistics layer (all offline)."""
import numpy as np
import pytest

from firescape import change as ch


def test_indices_known_answers():
    nir, red = np.ma.array([0.30]), np.ma.array([0.10])
    assert ch.ndvi(nir, red)[0] == pytest.approx(0.5)
    # MSAVI2 closed form: (2N+1 - sqrt((2N+1)^2 - 8(N-R)))/2
    s = 2 * 0.30 + 1
    want = (s - np.sqrt(s * s - 8 * 0.20)) / 2
    assert ch.msavi2(nir, red)[0] == pytest.approx(want)
    assert ch.brightness(np.ma.array([0.1]), np.ma.array([0.2]),
                         np.ma.array([0.3]), np.ma.array([0.4]))[0] \
        == pytest.approx(0.25)
    assert ch.redness(np.ma.array([0.3]), np.ma.array([0.2]))[0] \
        == pytest.approx(1.5)


def test_rdndvi_floor_prevents_blowup():
    pre = np.ma.array([0.25, 0.01])
    post = np.ma.array([0.10, -0.02])
    out = ch.rdndvi(pre, post)
    # sagebrush case: (0.10-0.25)/sqrt(0.35)*100
    assert out[0] == pytest.approx(100 * -0.15 / np.sqrt(0.35), rel=1e-3)
    # near-zero-NDVI case: denominator clamps at the floor, stays finite
    assert out[1] == pytest.approx(100 * -0.03 / np.sqrt(ch.RDNDVI_FLOOR),
                                   rel=1e-3)
    assert np.isfinite(out).all()


def test_epoch_stats_median_mad_count():
    stack = np.ma.masked_invalid(np.array([
        [[1.0, 10.0]], [[2.0, 20.0]], [[3.0, np.nan]],
    ]))
    med, mad, count = ch.epoch_stats(stack)
    assert med[0, 0] == 2.0 and med[0, 1] == 15.0
    assert mad[0, 0] == 1.0 and mad[0, 1] == 5.0
    assert count[0, 0] == 3 and count[0, 1] == 2
    with pytest.raises(ValueError, match="stack"):
        ch.epoch_stats(np.ma.array([1.0, 2.0]))


def test_robust_z_scales_and_floors():
    rng = np.random.default_rng(7)
    pre = np.ma.array(rng.normal(0.2, 0.01, (6, 20, 20)))
    pre_med, pre_mad, _ = ch.epoch_stats(pre)
    post = pre_med.copy()
    post[5, 5] -= 0.05                       # a real drop
    z = ch.robust_z(pre_med, post, pre_mad)
    assert z[5, 5] < -3                      # detected
    assert abs(np.ma.median(z)) < 0.5        # background centred
    # dead-stable pre pixels cannot make infinite z
    pre_mad0 = np.ma.array(np.zeros((20, 20)))
    z0 = ch.robust_z(pre_med, post, pre_mad0)
    assert np.isfinite(np.ma.asarray(z0)).all()


def test_scene_offset_recovers_known_shift():
    pytest.importorskip("skimage")
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(120, 120))
    img = np.roll(ref, (2, -3), axis=(0, 1))
    dy, dx = ch.scene_offset(ref, img)
    assert dy == pytest.approx(2, abs=0.2)
    assert dx == pytest.approx(-3, abs=0.2)
