"""Finite-difference derivative layer: exact weights, analytic tests, noise budget."""
import numpy as np
import pytest

from vcdm_genesis import derivatives as dv


def test_fd_weights_classical_stencils():
    h = 0.02
    np.testing.assert_allclose(dv.fd_weights(2, [-2, -1, 0, 1, 2], h), np.array([-1, 16, -30, 16, -1]) / (12 * h * h), rtol=1e-12)
    np.testing.assert_allclose(dv.fd_weights(1, [-2, -1, 0, 1, 2], h), np.array([1, -8, 0, 8, -1]) / (12 * h), rtol=1e-12)
    np.testing.assert_allclose(dv.fd_weights(2, dv.stencil_offsets(7), h), np.array([2, -27, 270, -490, 270, -27, 2]) / (180 * h * h), rtol=1e-11)
    np.testing.assert_allclose(dv.fd_weights(1, dv.stencil_offsets(7), h), np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * h), rtol=1e-11)


def test_noise_budget_matches_documented_bound():
    nb = dv.noise_budget(delta=1.5e-11, h=0.02, stencil=5)
    assert nb["second_derivative_bound"] == pytest.approx(16.0 / 3.0 * 1.5e-11 / 0.02 ** 2, rel=1e-12)
    assert nb["second_derivative_bound"] == pytest.approx(2e-7, rel=1e-12)
    nb7 = dv.noise_budget(delta=1.0, h=1.0, stencil=7)
    assert nb7["sum_abs_w2_times_h2"] == pytest.approx(1088 / 180, rel=1e-12)


def test_constant_and_polynomial_log_power_are_exact():
    l = np.linspace(-20.7, -9.2, 80)
    for stencil in (5, 7):
        dL, d2L, valid = dv.derivatives_uniform(l, np.full_like(l, 3.7), stencil)
        assert np.allclose(dL[valid], 0.0, atol=1e-12) and np.allclose(d2L[valid], 0.0, atol=1e-12)
        L = 0.4 - 0.13 * l + 0.007 * l ** 2
        dL, d2L, valid = dv.derivatives_uniform(l, L, stencil)
        np.testing.assert_allclose(dL[valid], -0.13 + 0.014 * l[valid], rtol=0, atol=1e-11)
        np.testing.assert_allclose(d2L[valid], 0.014, rtol=0, atol=1e-10)
    # degree-6 polynomial: exact for the 7-point stencil, not for the 5-point one
    c = np.array([0.3, -0.2, 0.05, -0.01, 0.002, -3e-4, 4e-5])
    P = np.polynomial.Polynomial(c)
    x = np.linspace(-1.5, 1.5, 61)
    dL, d2L, valid = dv.derivatives_uniform(x, P(x), 7)
    np.testing.assert_allclose(d2L[valid], P.deriv(2)(x[valid]), rtol=0, atol=1e-9)
    dL5, d2L5, valid5 = dv.derivatives_uniform(x, P(x), 5)
    err5 = np.max(np.abs(d2L5[valid5] - P.deriv(2)(x[valid5])))
    err7 = np.max(np.abs(d2L[valid] - P.deriv(2)(x[valid])))
    assert err5 > 1e-10 and err5 > 100 * err7      # h^4/90 P^(6) ~ 2e-9 for h = 0.05


def test_truncation_error_scales_as_h6():
    f = lambda x: np.exp(np.sin(1.3 * x))
    d2 = lambda x: (1.3 ** 2 * np.cos(1.3 * x) ** 2 - 1.3 ** 2 * np.sin(1.3 * x)) * f(x)
    errs = []
    for n in (41, 81):
        x = np.linspace(-1.0, 1.0, n)
        _, d2L, valid = dv.derivatives_uniform(x, f(x), 7)
        errs.append(np.max(np.abs(d2L[valid] - d2(x[valid]))))
    assert 40 < errs[0] / errs[1] < 90     # ~2^6 = 64


def test_derivative_columns_with_pads_on_polynomial():
    kappa = np.logspace(-9, -4, 80)
    l = np.log(kappa)
    h = np.diff(l).mean()
    L = lambda t: 1.0 + 0.2 * t - 0.03 * t ** 2 + 0.001 * t ** 3
    pad_low = [L(l[0] - j * h) for j in (3, 2, 1)]
    pad_high = [L(l[-1] + j * h) for j in (1, 2, 3)]
    ns, al = dv.derivative_columns(kappa, L(l), pad_low, pad_high, stencil=7)
    np.testing.assert_allclose(ns, 1.0 + 0.2 - 0.06 * l + 0.003 * l ** 2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(al, -0.06 + 0.006 * l, rtol=0, atol=1e-9)
    with pytest.raises(ValueError):
        dv.derivative_columns(kappa, L(l), pad_low[:2], pad_high, stencil=7)


def test_non_uniform_grid_rejected():
    with pytest.raises(ValueError):
        dv.derivatives_uniform(np.array([0.0, 1.0, 2.5, 3.0, 4.0, 5.0, 6.0]), np.zeros(7), 5)
