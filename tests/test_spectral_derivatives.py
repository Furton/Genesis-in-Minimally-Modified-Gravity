"""Spectral index and running from independently evolved modes: R3, R4, step-size and stencil checks.

References (prior independent computation, S2):
    R3  d=0.35, alpha0=1e-23, kappa=1e-4:  n_s ~ 0.90964681576,  alpha_s ~ -0.03030813031
    R4  d=0.10, alpha0=1e-23, kappa=1e-4:  n_s ~ 0.48958130079
Acceptance target: 2e-6 absolute on n_s and alpha_s.  These tests are heavier
(7 modes per stencil evaluation) and are the derivative-specific gate.
"""
import numpy as np
import pytest

from vcdm_genesis import derivatives as dv

ALPHA0 = 1e-23
TARGET = 2e-6
H_TABLE = float(np.log(1e-4 / 1e-9) / 79)     # step of the 80-point scan grid in ln kappa (0.1457)


@pytest.mark.parametrize("d,ns_ref,as_ref", [(0.35, 0.90964681576, -0.03030813031), (0.10, 0.48958130079, None)])
def test_R3_R4_with_two_step_sizes(d, ns_ref, as_ref):
    res = {}
    for h in (0.05, H_TABLE):
        r = dv.derivatives_from_modes(1e-4, ALPHA0, d, h=h, stencil=7)
        res[h] = r
        assert abs(r.n_s - ns_ref) < TARGET, (h, r.n_s)
        if as_ref is not None:
            assert abs(r.alpha_s - as_ref) < TARGET, (h, r.alpha_s)
    # two resolutions agree with each other (truncation + noise both below target)
    assert abs(res[0.05].n_s - res[H_TABLE].n_s) < TARGET
    assert abs(res[0.05].alpha_s - res[H_TABLE].alpha_s) < TARGET


def test_stencil_order_and_noise_budget_consistent():
    """Five- and seven-point stencils on the table step agree; the noise bound is below the target."""
    r7 = dv.derivatives_from_modes(1e-4, ALPHA0, 0.35, h=H_TABLE, stencil=7)
    r5 = dv.derivatives_from_modes(1e-4, ALPHA0, 0.35, h=H_TABLE, stencil=5)
    assert abs(r7.n_s - r5.n_s) < TARGET
    assert abs(r7.alpha_s - r5.alpha_s) < TARGET
    # measured per-point log-power uncertainty of the solver is ~1e-9 (convergence study); budget at the table step
    nb = dv.noise_budget(delta=2e-9, h=H_TABLE, stencil=7)
    assert nb["second_derivative_bound"] < TARGET / 2
    assert nb["first_derivative_bound"] < TARGET / 10
    # the notebook example: h=0.02 with delta=1.5e-11 gives 2e-7 for the five-point running (documented budget)
    assert dv.noise_budget(1.5e-11, 0.02, 5)["second_derivative_bound"] == pytest.approx(2e-7, rel=1e-9)


def test_interior_and_display_edge_points_stable():
    for d, kappa in [(0.30, 1e-9), (0.40, 1e-4), (0.35, 3e-7)]:
        a = dv.derivatives_from_modes(kappa, ALPHA0, d, h=H_TABLE, stencil=7)
        b = dv.derivatives_from_modes(kappa, ALPHA0, d, h=0.5 * H_TABLE, stencil=7)
        assert abs(a.n_s - b.n_s) < TARGET and abs(a.alpha_s - b.alpha_s) < TARGET
        assert 0.4 < a.n_s < 1.0
