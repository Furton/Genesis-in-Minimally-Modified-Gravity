"""Normalization conventions: powers of b and M_Pl, Wronskian, tensor factor."""
import numpy as np
import pytest

from vcdm_genesis import spectra as sp


def test_scalar_b_scaling_at_fixed_kappa():
    """P_R = S_R/(b^2 M_Pl^2): doubling b at fixed kappa divides P_R by 4."""
    S = 3.7e19
    p1 = sp.to_physical(S, b=1.0)
    p2 = sp.to_physical(S, b=2.0)
    assert p2 / p1 == pytest.approx(0.25, rel=1e-15)
    assert sp.from_physical(p2, b=2.0) == pytest.approx(S, rel=1e-15)


def test_M_Pl_scaling_independent_of_b():
    S = 2.0
    assert sp.to_physical(S, b=3.0, M_Pl=1.0) / sp.to_physical(S, b=3.0, M_Pl=5.0) == pytest.approx(25.0, rel=1e-15)


def test_u_hat_rescaling():
    b = 4.0
    u = np.array([1 + 1j, 0.5j])
    u_hat = sp.u_hat_from_u(u, b)
    np.testing.assert_allclose(u_hat, u / 2.0)
    np.testing.assert_allclose(sp.u_from_u_hat(u_hat, b), u)
    # an unscaled physical u overestimates S_R by exactly the factor b
    z = 0.3
    assert sp.scalar_power_dimensionless(1e-7, u, z) / sp.scalar_power_dimensionless(1e-7, u_hat, z) == pytest.approx(b, rel=1e-14)


def test_vacuum_wronskian_and_derivative_sign():
    for kappa in [1e-9, 1e-4, 3.0]:
        for x in [0.0, 1.0, 1e5]:
            u, du = sp.vacuum_plane_wave(kappa, x)
            assert abs(u) ** 2 == pytest.approx(1.0 / (2.0 * kappa), rel=1e-14)
            assert du == pytest.approx(1j * kappa * u, rel=1e-14)
            assert sp.wronskian(u, du) == pytest.approx(-1j, rel=1e-13)
    # the opposite-sign derivative (e^{-i kappa x}) gives +i, so the sign test is meaningful
    u = np.exp(-1j * 0.7) / np.sqrt(2.0)
    assert sp.wronskian(u, -1j * u) == pytest.approx(+1j)


def test_tensor_normalization_two_routes_agree():
    """u = a M_Pl h/2, two equal polarizations: S_h/(b^2 M_Pl^2) == k^3/pi^2 |h|^2."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        b = float(rng.uniform(0.2, 5.0)); M = float(rng.uniform(0.5, 2.0)); a = float(rng.uniform(1.0, 80.0))
        kappa = float(10 ** rng.uniform(-9, -4)); k = kappa / b
        h = complex(rng.normal(), rng.normal()) * 1e-3
        u = sp.tensor_mode_from_h(h, a, M)           # physical canonical mode
        u_hat = sp.u_hat_from_u(u, b)
        S_h = sp.tensor_power_dimensionless(kappa, u_hat, a)
        P_h = sp.to_physical(S_h, b, M)
        assert P_h == pytest.approx(sp.tensor_power_from_h_physical(k, h), rel=1e-13)
        assert sp.h_from_tensor_mode(u, a, M) == pytest.approx(h, rel=1e-14)


def test_invalid_scales_raise():
    with pytest.raises(ValueError):
        sp.to_physical(1.0, b=0.0)
    with pytest.raises(ValueError):
        sp.u_hat_from_u(1.0, b=-1.0)
