"""Vacuum selection, initial data, mode evolution: regression cases and convergence.

Reference values (prior independent computations, see the validation record):
    R1  d=0.3, alpha0=1e-23, kappa=1e-7, x_f=0            S_R ~ 6.91139484127e19
    R2  R1 at both endpoints                               S_R(1e-7)/S_R(0) ~ 0.9999999773106846, d ln|R|/dx(0) ~ -0.1134465522
    R5  d=0.5, kappa=1e-7 (tensor)                         S_h ~ 2.0196314518e-15
    R6  alpha0=1e-30, d=0.15316455696202533, kappa=2.2616759492228647e-6: pump-zero spike (selector test)
    R7  alpha0=1e-30, d=0.13417721518987344, kappa=1.2263306841775643e-7, x_f=1e-7: converged S_R ~ 1.48036818276e24
Acceptance targets: <= 1e-6 relative change under each refinement and <= 1e-6 between formulations.
"""
import math

import mpmath as mp
import numpy as np
import pytest

from vcdm_genesis import background as bg
from vcdm_genesis import modes as md
from vcdm_genesis.spectra import wronskian

CFG = md.SolverConfig()
R1 = (1e-7, 1e-23, 0.3)
R6 = (2.2616759492228647e-06, 1e-30, 0.15316455696202533)
R7 = (1.2263306841775643e-07, 1e-30, 0.13417721518987344)
TARGET = 1e-6


@pytest.fixture(scope="module")
def r1_zero():
    return md.evolve_scalar_mode(*R1, x_final=0.0, cfg=CFG)


def test_R1_R2_regression(r1_zero):
    r0 = r1_zero
    assert r0.S_R == pytest.approx(6.91139484127e19, rel=5e-9)
    r7 = md.evolve_scalar_mode(*R1, x_ini=r0.x_ini, x_final=1e-7, cfg=CFG)
    assert r7.S_R / r0.S_R == pytest.approx(0.9999999773106846, abs=1e-12)
    L1 = bg.log_z2_derivatives(*R1, 0.0)[0]
    dlnR = (r0.ux_final / r0.u_final).real - 0.5 * L1
    assert dlnR == pytest.approx(-0.1134465522, abs=5e-10)
    assert r0.selection["ratio_at_returned"] >= CFG.pump_ratio_min
    assert r0.success


def test_R7_regression():
    r = md.evolve_scalar_mode(*R7, x_final=1e-7, cfg=CFG)
    assert r.S_R == pytest.approx(1.48036818276e24, rel=5e-9)


def test_R5_tensor_regression():
    r = md.evolve_tensor_mode(1e-7, 0.5, x_final=0.0, cfg=CFG)
    assert r.S_h == pytest.approx(2.0196314518e-15, rel=1e-8)      # measured 1.5e-10 with the tensor threshold 1e7
    deeper = md.evolve_tensor_mode(1e-7, 0.5, x_ini=2.0 * r.x_ini, x_final=0.0, cfg=CFG)
    assert abs(deeper.S_h / r.S_h - 1.0) <= 1e-8
    assert r.selection["ratio_at_returned"] >= CFG.tensor_pump_ratio_min
    assert r.a_final == pytest.approx(math.exp(3.0))


# --------------------------------------------------------------------------- #
# selector
# --------------------------------------------------------------------------- #
def test_R6_pump_zero_spike_is_rejected():
    kappa, alpha0, d = R6
    # a single-time test passes at x=1.365114e6 (ratio 137962.98999) although 2x fails (ratio 2446.93459)
    x_hist = 1.365114e6
    c = md.vacuum_checks(kappa, alpha0, d, x_hist, CFG)
    assert c.ratio > 1e5, "the spike exists in the exact coefficients too"
    c2 = md.vacuum_checks(kappa, alpha0, d, 2.0 * x_hist, CFG)
    assert c2.ratio < 1e4 and not c2.passed
    xs = np.geomspace(1e6, 2e6, 2000)
    pump = bg.z_xx_over_z(kappa, alpha0, d, xs)
    assert np.any(np.sign(pump[:-1]) != np.sign(pump[1:])), "pump changes sign near the spike"
    ok, _ = md._bracket_passes(lambda x: md.vacuum_checks(kappa, alpha0, d, x, CFG), 2.0 * x_hist, CFG)
    assert not ok
    x_ret, diag = md.select_initial_time(kappa, alpha0, d, CFG)
    assert x_ret > 2.0 * x_hist
    assert md.vacuum_checks(kappa, alpha0, d, x_ret, CFG).passed
    assert md.vacuum_checks(kappa, alpha0, d, 2.0 * x_ret, CFG).passed
    assert diag["ratio_at_returned"] >= CFG.pump_ratio_min and not diag["capped"]


@pytest.mark.parametrize("kappa,alpha0,d", [R1, R6, R7, (1e-9, 1e-30, 0.1), (1e-4, 1e-21, 0.4), (1e-4, 1e-30, 0.1), (1e-9, 1e-21, 0.4)])
def test_returned_time_passes_all_checks_after_multiplier(kappa, alpha0, d):
    x_ret, diag = md.select_initial_time(kappa, alpha0, d, CFG)
    for f in CFG.bracket:
        c = md.vacuum_checks(kappa, alpha0, d, x_ret * f, CFG)
        assert c.passed, (f, c)
    assert x_ret == pytest.approx(min(diag["x_first_pass"] * CFG.multiplier, CFG.x_search_max))
    assert x_ret < CFG.x_search_max


def test_selection_fails_explicitly_instead_of_returning_the_cap():
    cfg = md.SolverConfig(x_search_max=1e3, n_search=200)
    with pytest.raises(md.SelectionError):
        md.select_initial_time(1e-9, 1e-30, 0.1, cfg)     # needs x ~ 1e11
    cfg2 = md.SolverConfig(pump_ratio_min=1e40)
    with pytest.raises(md.SelectionError):
        md.select_initial_time(*R1, cfg2)


def test_intentional_invalid_starts_fail():
    kappa, alpha0, d = R1
    assert bg.omega_squared_scalar(kappa, alpha0, d, 0.5) < 0.0
    with pytest.raises(ValueError):
        md.adiabatic_initial_data(kappa, alpha0, d, 0.5)      # omega^2 < 0: no |omega^2| trick
    with pytest.raises(ValueError):
        md.evolve_scalar_mode(kappa, alpha0, d, x_ini=0.5, x_final=0.0, cfg=CFG)
    with pytest.raises(ValueError):
        md.evolve_scalar_mode(kappa, alpha0, d, x_ini=1e-8, x_final=1e-7, cfg=CFG)   # x_ini < x_final
    with pytest.raises(ValueError):
        md.omega_derivative(kappa, alpha0, d, 0.0)


# --------------------------------------------------------------------------- #
# initial data
# --------------------------------------------------------------------------- #
def mp_omega(kappa, alpha0, d):
    from tests.test_background import mp_block  # manuscript definitions in mpmath
    kappa_m = mp.mpf(kappa)

    def lnz2(x):
        return mp.log(mp_block(kappa, alpha0, d, x)["z2"])

    def omega(x):
        blk = mp_block(kappa, alpha0, d, x)
        pump = mp.diff(lnz2, x, 2) / 2 + mp.diff(lnz2, x, 1) ** 2 / 4
        return mp.sqrt(blk["cR2"] * kappa_m ** 2 - pump)
    return omega


@pytest.mark.parametrize("kappa,alpha0,d,x", [(1e-7, 1e-23, 0.3, 3e9), (1e-4, 1e-21, 0.4, 5e6), (2.26e-6, 1e-30, 0.153, 7e7)])
def test_omega_derivative_estimate_matches_mpmath(kappa, alpha0, d, x):
    mp.mp.dps = 40
    om = mp_omega(kappa, alpha0, d)
    ref = mp.diff(om, mp.mpf(x), 1)
    got = md.omega_derivative(kappa, alpha0, d, x)
    assert abs(got / float(ref) - 1.0) < 1e-6
    assert float(om(mp.mpf(x))) == pytest.approx(md.omega_of_x(kappa, alpha0, d, x), rel=1e-11)


def test_adiabatic_initial_data_normalization_and_sign():
    for kappa, alpha0, d in [R1, R6, R7]:
        x, _ = md.select_initial_time(kappa, alpha0, d, CFG)
        u, ux, Om, Om_x = md.adiabatic_initial_data(kappa, alpha0, d, x)
        assert abs(u) ** 2 == pytest.approx(1.0 / (2.0 * Om), rel=1e-14)
        assert wronskian(u, ux) == pytest.approx(-1j, abs=1e-12)
        assert (ux / u).imag > 0.0                       # e^{+i kappa x} convention
        assert (ux / u).real == pytest.approx(-0.5 * Om_x / Om, rel=1e-12)
        assert Om == pytest.approx(kappa, rel=CFG.cR2_tol + 10 * CFG.adiabatic_tol)


# --------------------------------------------------------------------------- #
# convergence and independent formulation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa,alpha0,d,x_final", [R1 + (0.0,), R7 + (1e-7,), (1e-4, 1e-21, 0.4, 1e-7), (1e-9, 1e-30, 0.1, 1e-7)])
def test_refinements_and_independent_formulation(kappa, alpha0, d, x_final):
    res = md.convergence_study(kappa, alpha0, d, x_final=x_final, cfg=CFG, x_final_alt=None)
    assert res["rel_change_x_ini"] <= TARGET
    assert res["rel_change_rtol"] <= TARGET
    assert res["rel_diff_independent"] <= TARGET
    assert res["S_R"] > 0.0 and math.isfinite(res["S_R"])


def test_frozen_corner_and_interior_modes_finite_and_converged():
    pts = [(1e-9, 1e-21, 0.1), (1e-4, 1e-30, 0.4), (3.2e-7, 1e-25, 0.25), (1e-6, 8.2e-30, 0.25)]
    for kappa, alpha0, d in pts:
        r = md.evolve_scalar_mode(kappa, alpha0, d, x_final=1e-7, cfg=CFG)
        assert math.isfinite(r.S_R) and r.S_R > 0.0
        r2 = md.evolve_scalar_mode(kappa, alpha0, d, x_ini=2.0 * r.x_ini, x_final=1e-7, cfg=CFG)
        assert abs(r2.S_R / r.S_R - 1.0) <= TARGET
        assert abs(r.wronskian_ini + 1j) < 1e-12


def test_endpoint_variation_is_reported_not_hidden(r1_zero):
    r7 = md.evolve_scalar_mode(*R1, x_ini=r1_zero.x_ini, x_final=1e-7, cfg=CFG)
    assert 0.0 < 1.0 - r7.S_R / r1_zero.S_R < 1e-7   # finite endpoint difference, documented not suppressed


def test_record_serializes():
    r = md.evolve_scalar_mode(*R7, x_final=1e-7, cfg=CFG)
    rec = r.to_record()
    import json
    json.dumps(rec)
    assert rec["formulation"].startswith("canonical")
