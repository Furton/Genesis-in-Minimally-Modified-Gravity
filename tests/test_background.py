"""Independent checks of the exact background and quadratic-action coefficients.

The reference implementation below is written directly from the manuscript
definitions in ``mpmath`` at 40 digits and differentiated numerically; it does
not import the closed-form derivative code under test.
"""
import math

import mpmath as mp
import numpy as np
import pytest

from vcdm_genesis import background as bg

mp.mp.dps = 40

# (kappa, alpha0, d): regression tuples, scan corners, the R6 selector tuple,
# and the manuscript c_R^2 counterexample parameters.
CASES = [
    (1e-9, 1e-30, 0.1),
    (1e-4, 1e-21, 0.4),
    (1e-9, 1e-21, 0.1),
    (1e-4, 1e-30, 0.4),
    (1e-7, 1e-23, 0.3),
    (1e-4, 1e-23, 0.35),
    (2.2616759492228647e-06, 1e-30, 0.15316455696202533),
    (1e-4, 1e-5, 1.0),
]
XS = [0.0, 1e-7, 1e-3, 0.5, 1.0, 37.0, 1e3, 1e8, 1e12]


def mp_block(kappa, alpha0, d, x):
    """Manuscript definitions (with the regular product E_eta) in mpmath."""
    kappa, alpha0, d, x = (mp.mpf(v) for v in (kappa, alpha0, d, x))
    T = 1 + x
    a = mp.exp((1 + d) / d * T ** (-d))
    Ht = (1 + d) * T ** (-1 - d)
    al = alpha0 * T ** (2 * d)
    eps = 1 - T ** d
    beta = -2 * d / (1 + d) * T ** d
    Ee = d / (1 + d) * T ** (2 * d)
    A1 = 2 * al * (12 + al - 2 * eps)
    A2 = 3 * al ** 2 * (6 + al - 2 * eps)
    B1 = 2 * (al * (-4 * beta + 2 * eps + al) + 4 * eps * (6 + beta - 2 * eps) + 4 * Ee)
    B2 = al * (-(al - 2 * eps) ** 2 + 36 * eps + 6 * al * (-1 - beta + eps) + 12 * eps * (1 - eps) + 12 * Ee)
    z2 = 2 * a ** 2 * al * (2 * kappa ** 2 + 3 * al * Ht ** 2) / (4 * kappa ** 2 + al * Ht ** 2 * (6 + al - 2 * eps))
    cR2 = (8 * kappa ** 4 + B1 * Ht ** 2 * kappa ** 2 + B2 * Ht ** 4) / (8 * kappa ** 4 + A1 * Ht ** 2 * kappa ** 2 + A2 * Ht ** 4)
    return dict(a=a, Ht=Ht, al=al, eps=eps, beta=beta, Ee=Ee, A1=A1, A2=A2, B1=B1, B2=B2, z2=z2, cR2=cR2)


def mp_lnz2(kappa, alpha0, d):
    return lambda x: mp.log(mp_block(kappa, alpha0, d, x)["z2"])


def mp_lna(d):
    return lambda x: (1 + mp.mpf(d)) / mp.mpf(d) * (1 + x) ** (-mp.mpf(d))


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa,alpha0,d", CASES)
def test_coefficients_match_manuscript_definitions(kappa, alpha0, d):
    for x in XS:
        ref = mp_block(kappa, alpha0, d, x)
        assert rel(bg.scale_factor(d, x), ref["a"]) < 1e-13
        assert rel(bg.hubble_tilde(d, x), ref["Ht"]) < 1e-13
        assert rel(bg.alpha(alpha0, d, x), ref["al"]) < 1e-13
        assert abs(float(bg.epsilon(d, x)) - float(ref["eps"])) < 1e-13 * max(1.0, abs(float(ref["eps"])))
        assert rel(bg.beta_alpha(d, x), ref["beta"]) < 1e-13
        assert rel(bg.E_eta(d, x), ref["Ee"]) < 1e-13
        assert rel(bg.A1(alpha0, d, x), ref["A1"]) < 1e-12
        assert rel(bg.A2(alpha0, d, x), ref["A2"]) < 1e-12
        assert rel(bg.B1(alpha0, d, x), ref["B1"]) < 1e-12
        assert rel(bg.B2(alpha0, d, x), ref["B2"]) < 1e-12
        assert rel(bg.z_squared(kappa, alpha0, d, x), ref["z2"]) < 1e-12
        assert rel(bg.cR_squared(kappa, alpha0, d, x), ref["cR2"]) < 1e-11


def test_endpoint_is_regular_and_uses_E_eta():
    """At x=0, epsilon=0 and eta diverges; every coefficient must stay finite."""
    for kappa, alpha0, d in CASES:
        vals = bg.scalar_coefficients_scalar(kappa, alpha0, d, 0.0)
        assert all(math.isfinite(v) for v in vals)
        assert bg.E_eta(d, 0.0) == pytest.approx(d / (1.0 + d), rel=1e-15)
        beta0 = -2.0 * d / (1.0 + d)
        Ee0 = d / (1.0 + d)
        # explicit endpoint limits of B1, B2 with epsilon = 0
        assert bg.B1(alpha0, d, 0.0) == pytest.approx(2.0 * (alpha0 * (-4.0 * beta0 + alpha0) + 4.0 * Ee0), rel=1e-13)
        assert bg.B2(alpha0, d, 0.0) == pytest.approx(alpha0 * (-alpha0 ** 2 + 6.0 * alpha0 * (-1.0 - beta0) + 12.0 * Ee0), rel=1e-13)
        assert bg.z_squared(kappa, alpha0, d, 0.0) > 0.0
        assert math.isfinite(bg.omega_squared_scalar(kappa, alpha0, d, 0.0))
    # the naive singular product 0 * inf would give nan; we never produce nan
    assert not np.isnan(bg.B1(1e-23, 0.3, np.array([0.0, 1e-300, 1e-7])).any())


def test_matches_singular_eta_form_away_from_endpoint():
    """For x > 0 the manuscript's eta-based forms of B1, B2 agree with the regular E_eta forms."""
    for kappa, alpha0, d in CASES[:6]:
        for x in [1e-3, 0.5, 1.0, 1e3]:
            T = 1.0 + x
            al = alpha0 * T ** (2 * d)
            eps = 1.0 - T ** d
            eta = -d * T ** (2 * d) / ((1.0 + d) * (T ** d - 1.0))  # eta = E_eta / epsilon, singular at x = 0
            beta = -(2.0 * d / (1.0 + d)) * T ** d
            B1_nb = 2.0 * (al * (-4.0 * beta + 2.0 * eps + al) + 4.0 * eps * (6.0 + beta - 2.0 * eps + eta))
            B2_nb = al * (-(al - 2.0 * eps) ** 2 + 36.0 * eps + 6.0 * (al * (-1.0 - beta + eps) + 2.0 * eps * (1.0 - eps + eta)))
            assert bg.B1(alpha0, d, x) == pytest.approx(B1_nb, rel=1e-10)
            assert bg.B2(alpha0, d, x) == pytest.approx(B2_nb, rel=1e-10)


@pytest.mark.parametrize("kappa,alpha0,d", CASES)
def test_scalar_pump_matches_high_precision_differentiation(kappa, alpha0, d):
    f = mp_lnz2(kappa, alpha0, d)
    for x in XS:
        L1_ref = mp.diff(f, mp.mpf(x), 1)
        L2_ref = mp.diff(f, mp.mpf(x), 2)
        pump_ref = L2_ref / 2 + L1_ref ** 2 / 4
        L1, L2 = bg.log_z2_derivatives(kappa, alpha0, d, x)
        pump = bg.z_xx_over_z(kappa, alpha0, d, x)
        scale = abs(float(L2_ref)) / 2 + float(L1_ref) ** 2 / 4  # guards zeros of the pump
        assert abs(float(L1) - float(L1_ref)) <= 1e-10 * abs(float(L1_ref))
        assert abs(float(L2) - float(L2_ref)) <= 1e-9 * max(abs(float(L2_ref)), scale)
        assert abs(float(pump) - float(pump_ref)) <= 1e-9 * max(abs(float(pump_ref)), scale)


@pytest.mark.parametrize("d", [0.1, 0.3, 0.5, 0.7, 1.0])
def test_tensor_pump_matches_high_precision_differentiation(d):
    f = mp_lna(d)
    for x in XS:
        ref = mp.diff(f, mp.mpf(x), 2) + mp.diff(f, mp.mpf(x), 1) ** 2
        assert rel(bg.a_xx_over_a(d, x), ref) < 1e-10
        assert rel(bg.tensor_omega_squared_scalar(1e-7, d, x), 1e-14 - ref) < 1e-9


def test_cR2_counterexample_from_manuscript():
    """d=1, alpha0=1e-5, kappa=1e-4, x=1e3: kappa/Htilde=50.10005 and c_R^2=-65.9361815185 (S2)."""
    ratio = 1e-4 / bg.hubble_tilde(1.0, 1e3)
    assert ratio == pytest.approx(50.10005, rel=2e-7)
    assert bg.cR_squared(1e-4, 1e-5, 1.0, 1e3) == pytest.approx(-65.9361815185, rel=1e-9)


def test_fixed_time_uv_limit():
    """At fixed x, c_R^2 -> 1 and z^2 -> a^2 alpha as kappa -> infinity."""
    for alpha0, d, x in [(1e-23, 0.3, 1.0), (1e-5, 1.0, 1e3), (1e-21, 0.4, 0.0)]:
        for kappa in [1.0, 1e2, 1e4]:
            c = bg.cR_squared(kappa, alpha0, d, x)
            z2 = bg.z_squared(kappa, alpha0, d, x)
            a2al = bg.scale_factor(d, x) ** 2 * bg.alpha(alpha0, d, x)
            Ht2 = bg.hubble_tilde(d, x) ** 2
            bound = 10.0 * Ht2 / kappa ** 2 * max(abs(bg.B1(alpha0, d, x)), abs(bg.A1(alpha0, d, x)), 1.0)
            assert abs(c - 1.0) <= bound
            # exact: z^2/(a^2 alpha) - 1 = alpha Ht^2 (2 eps - alpha) / D  ->  O(alpha Ht^2 |2 eps - alpha| / kappa^2)
            al = float(bg.alpha(alpha0, d, x)); ep = float(bg.epsilon(d, x))
            assert abs(z2 / a2al - 1.0) <= 10.0 * Ht2 / kappa ** 2 * al * (abs(2.0 * ep - al) + 1.0)
        # this is the fixed-time limit, not the large-x limit of the notebook's approximations
        assert bg.cR_squared(1e4, alpha0, d, x) == pytest.approx(1.0, abs=1e-6)


def test_small_alpha0_frequency_expansion():
    """omega^2 = kappa^2 - [2 + 3(1+d)^2 T^-d (1 - T^-d)]/T^2 + O(alpha0) (manuscript)."""
    alpha0 = 1e-30
    for d in [0.1, 0.25, 0.4]:
        for kappa in [1e-9, 1e-6, 1e-4]:
            for x in [0.0, 1e-7, 0.3, 1.0, 10.0, 1e3]:
                T = 1.0 + x
                expected = kappa ** 2 - (2.0 + 3.0 * (1.0 + d) ** 2 * T ** (-d) * (1.0 - T ** (-d))) / T ** 2
                got = bg.omega_squared_scalar(kappa, alpha0, d, x)
                # the neglected terms are O(alpha0 Ht^2 / kappa^2) (from B2 Ht^4/kappa^2, A1, and N/D),
                # i.e. ~1e-12 at kappa=1e-9, x=0: allow that remainder explicitly.
                Ht = float(bg.hubble_tilde(d, x))
                slack = 200.0 * alpha0 * Ht ** 2 * (1.0 + Ht ** 2) / kappa ** 2
                assert abs(got - expected) <= 1e-13 * max(abs(expected), 1.0) + slack


def test_scalar_and_array_paths_agree():
    for kappa, alpha0, d in CASES:
        xs = np.array(XS)
        arr = bg.omega_squared(kappa, alpha0, d, xs)
        sc = np.array([bg.omega_squared_scalar(kappa, alpha0, d, float(x)) for x in xs])
        np.testing.assert_allclose(arr, sc, rtol=1e-14, atol=0.0)
        z_arr = np.sqrt(bg.z_squared(kappa, alpha0, d, xs))
        z_sc = np.array([bg.z_scalar(kappa, alpha0, d, float(x)) for x in xs])
        np.testing.assert_allclose(z_arr, z_sc, rtol=1e-14, atol=0.0)


def test_z_squared_positive_on_scan_domain():
    xs = np.geomspace(1e-7, 1e12, 400)
    xs = np.concatenate([[0.0], xs])
    for kappa, alpha0, d in CASES:
        z2 = bg.z_squared(kappa, alpha0, d, xs)
        assert np.all(np.isfinite(z2)) and np.all(z2 > 0.0)


def test_cR2_is_not_clamped():
    """Negative finite-wavenumber c_R^2 must be reported, never replaced by zero."""
    assert bg.cR_squared(1e-4, 1e-5, 1.0, 1e3) < 0.0
    assert bg.omega_squared_scalar(1e-9, 1e-30, 0.3, 0.0) < 0.0  # endpoint is tachyonic for small kappa


@pytest.mark.parametrize("bad", [
    dict(d=0.0), dict(d=-0.3), dict(d=float("nan")), dict(alpha0=0.0), dict(alpha0=-1e-23),
    dict(kappa=0.0), dict(kappa=float("inf")),
])
def test_invalid_parameters_raise(bad):
    args = dict(alpha0=1e-23, d=0.3, kappa=1e-7)
    args.update(bad)
    with pytest.raises(ValueError):
        bg.check_parameters(**args)
    with pytest.raises(ValueError):
        bg.z_squared(args["kappa"], args["alpha0"], args["d"], 1.0)


def test_negative_or_nonfinite_x_rejected():
    with pytest.raises(ValueError):
        bg.z_squared(1e-7, 1e-23, 0.3, -1e-3)
    with pytest.raises(ValueError):
        bg.a_xx_over_a(0.3, float("nan"))
