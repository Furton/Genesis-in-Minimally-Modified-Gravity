"""The compact regression fixture must be reproduced by the released solver.

``tests/fixtures/regression_cases.json`` stores prior independent reference
values and the values this release's solver produced when the fixture was
generated.  Reproducing the stored values guards against silent solver drift;
the reference comparisons in ``test_modes.py`` guard scientific agreement.
"""
import json
from pathlib import Path

import pytest

from vcdm_genesis import modes as md

FIXTURE = Path(__file__).parent / "fixtures" / "regression_cases.json"


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_config_matches_default_solver(fixture):
    assert fixture["solver_config"] == md.SolverConfig().to_dict()


@pytest.mark.parametrize("name", ["R1", "R6", "R7", "corner_a0_d0_k0", "corner_a79_d79_k79", "interior_mid"])
def test_scalar_cases_reproduce(fixture, name):
    c = fixture["cases"][name]
    r = md.evolve_scalar_mode(c["kappa"], c["alpha0"], c["d"], x_final=c["x_final"], cfg=md.SolverConfig())
    assert r.S_R == pytest.approx(c["measured"]["S_R"], rel=1e-9)
    assert r.x_ini == pytest.approx(c["measured"]["x_ini"], rel=1e-12)
    if "S_R" in c["reference"]:
        assert r.S_R == pytest.approx(c["reference"]["S_R"], rel=5e-9)
    # the convergence metrics are recomputed live (not only read back from the fixture)
    deeper = md.evolve_scalar_mode(c["kappa"], c["alpha0"], c["d"], x_ini=2.0 * r.x_ini, x_final=c["x_final"], cfg=md.SolverConfig())
    indep = md.evolve_curvature_mode(c["kappa"], c["alpha0"], c["d"], x_ini=r.x_ini, x_final=c["x_final"], cfg=md.SolverConfig())
    rel_x2 = abs(deeper.S_R / r.S_R - 1.0)
    rel_indep = abs(indep.S_R / r.S_R - 1.0)
    assert rel_x2 <= 1e-6 and rel_indep <= 1e-6
    assert rel_x2 == pytest.approx(c["measured"]["rel_change_x_ini_x2"], abs=1e-9)
    assert rel_indep == pytest.approx(c["measured"]["rel_diff_independent_R_formulation"], abs=1e-9)


def test_tensor_case_reproduces(fixture):
    c = fixture["cases"]["R5"]
    r = md.evolve_tensor_mode(c["kappa"], c["d"], x_final=c["x_final"], cfg=md.SolverConfig())
    assert r.S_h == pytest.approx(c["measured"]["S_h"], rel=1e-9)
    assert r.S_h == pytest.approx(c["reference"]["S_h"], rel=1e-8)
