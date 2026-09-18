"""Fit template: cross-representation checks, coefficient record, grouped split, LP."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vcdm_genesis import fit

COEFFS = fit.load_coefficients()
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "genesis_scan_table.csv"


def test_coefficient_record_is_the_released_set():
    for k, v in fit.RELEASED_COEFFICIENTS.items():
        assert COEFFS[k] == v
    assert COEFFS["status"] == "released"


def test_template_matches_wolfram_notebook_saved_values():
    """Cross-representation check against the Wolfram evaluation of the same template.

    The four rows were evaluated with WolframScript 1.9.0 from the definitions of 'Analytic fit.nb'
    (refit coefficients of 2026-09-18); the Python template agrees with them to 3e-14 (docs/VALIDATION.md).
    """
    cases = [
        (2.3285662984981965e-29, 0.14177215189873418, 2.196707090793235e-7, 1.1239216240945546e+23),
        (2.6938893095901644e-22, 0.24810126582278486, 1.64130719537513e-7, 1.0593155471464253e+18),
        (9.162739011886693e-28, 0.1, 1.548365256585496e-9, 4.869083992113025e+19),
        (2.5413430367026287e-23, 0.14556962025316456, 3.401304938279253e-7, 1.2915390115154e+17),
    ]
    for a, d, k, ps in cases:
        assert fit.S_R_fit(a, d, k, COEFFS) == pytest.approx(ps, rel=1e-11)


def test_template_matches_manuscript_table_at_exact_grid_points():
    """The paper's Table 1 (data/fit_sample_table.csv) lists rounded parameters; the fit values
    (8 significant digits) are reproduced at the exact grid points."""
    A = np.logspace(-30, -21, 80); D = np.linspace(0.1, 0.4, 80); K = np.logspace(-9, -4, 80)
    table = [(7.69265e-22, 0.160759, 2.19671e-7, 1.4046197e16), (1.09138e-24, 0.183544, 3.02700e-6, 2.2404214e19),
             (4.55227e-22, 0.251899, 5.91769e-8, 7.0641840e17), (2.26168e-25, 0.164557, 6.27290e-6, 2.9927273e19)]
    for a, d, k, ps in table:
        ae = A[np.argmin(np.abs(A / a - 1))]; de = D[np.argmin(np.abs(D - d))]; ke = K[np.argmin(np.abs(K / k - 1))]
        assert fit.S_R_fit(ae, de, ke, COEFFS) == pytest.approx(ps, rel=2e-7)


def test_fit_derivatives_are_smooth_and_step_independent():
    k = np.logspace(-9, -4, 25)
    n1, a1 = fit.fit_derivatives(1e-23, 0.35, k, COEFFS, h=1e-2)
    n2, a2 = fit.fit_derivatives(1e-23, 0.35, k, COEFFS, h=2e-2)
    np.testing.assert_allclose(n1, n2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-9)
    assert np.all(n1 < 1.0) and np.all(np.diff(n1) < 0)       # red tilt increasing with kappa in this range
    # alpha0 factors out: derivatives independent of alpha0
    n3, a3 = fit.fit_derivatives(1e-27, 0.35, k, COEFFS)
    np.testing.assert_array_equal(n1, n3)
    np.testing.assert_array_equal(a1, a3)


def test_template_residuals_against_independent_mpmath_implementation():
    """Per-row residuals of the released table against an mpmath transcription of the manuscript template.

    Independent of vcdm_genesis.fit (own arithmetic at 30 digits, mp.loggamma instead of scipy.gammaln);
    validates the residual computation that the aggregate statistics below are built from.
    """
    import mpmath as mp
    mp.mp.dps = 30
    assert CSV.exists(), "the released scan table must be present"
    rows = []
    with open(CSV, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh); next(rd)
        for n, row in enumerate(rd):
            if n % 9973 == 0:            # 52 rows spread over the table, plus R7's row
                rows.append(row)
            if n == 753:
                rows.append(row)
    c0, c1, q = (mp.mpf(COEFFS[k]) for k in ("c0", "c1", "q"))
    worst = 0.0
    for row in rows:
        a0, d, k, ps = (mp.mpf(v) for v in row[:4])
        A = 3 * (1 + d) ** 2; xi = k ** d; nu = mp.sqrt(mp.mpf(9) / 4 + A * xi * (1 - xi)); u = A * (1 - xi) / (nu + mp.mpf(3) / 2)
        J = 3 + mp.sqrt(A) * mp.atan(u / mp.sqrt(A)) + mp.mpf(3) / 2 * mp.log(A / (A + u ** 2)) - A * (3 + u) / (A + u ** 2)
        lnC = (2 * nu - 3) * mp.log(2) + 2 * (mp.loggamma(nu) - mp.loggamma(mp.mpf(3) / 2))
        lnS = -mp.log(4 * mp.pi ** 2 * a0) - 2 * (1 + d) / d + 2 * J / d + c0 + c1 * d + q * lnC
        rel_ref = float(mp.exp(lnS - mp.log(ps)) - 1)
        rel_prod = float(np.expm1(fit.log_S_R_fit(float(row[0]), float(row[1]), float(row[2]), COEFFS) - np.log(float(row[3]))))
        worst = max(worst, abs(rel_prod - rel_ref))
        assert abs(rel_prod - rel_ref) < 1e-12, row[:3]
        assert abs(rel_ref) < 0.0033
    assert worst < 1e-12
    assert len(rows) >= 50


def test_grouped_split_and_residuals_reproduce_notebook_statistics():
    assert CSV.exists(), "the released scan table must be present (mandatory workflow, never skipped)"
    alpha, d, kappa, ps = [], [], [], []
    with open(CSV, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh); next(rd)
        for row in rd:
            alpha.append(float(row[0])); d.append(float(row[1])); kappa.append(float(row[2])); ps.append(float(row[3]))
    alpha, d, kappa, ps = map(np.asarray, (alpha, d, kappa, ps))
    train = fit.split_by_d_kappa(d, kappa)
    assert train.sum() == 409600 and (~train).sum() == 102400
    # every (d,kappa) pair is entirely on one side
    pair = d * 1e6 + np.log10(kappa)
    for side in (train, ~train):
        assert np.isin(pair[side], pair[~side]).sum() == 0
    st = fit.evaluate_fit(alpha[train], d[train], kappa[train], ps[train], COEFFS)
    sv = fit.evaluate_fit(alpha[~train], d[~train], kappa[~train], ps[~train], COEFFS)
    sa = fit.evaluate_fit(alpha, d, kappa, ps, COEFFS)
    # the validation numbers stored in the coefficient record
    rv = COEFFS["validation"]
    assert st["max_abs_rel_percent"] == pytest.approx(rv["training"]["max_abs_rel_percent"], abs=1e-9)
    assert sv["max_abs_rel_percent"] == pytest.approx(rv["verification"]["max_abs_rel_percent"], abs=1e-9)
    assert st["mean_abs_rel_percent"] == pytest.approx(rv["training"]["mean_abs_rel_percent"], abs=1e-9)
    assert sv["mean_abs_rel_percent"] == pytest.approx(rv["verification"]["mean_abs_rel_percent"], abs=1e-9)
    assert sa["mean_abs_rel_percent"] == pytest.approx(rv["full_grid"]["mean_abs_rel_percent"], abs=1e-9)
    assert sa["max_abs_rel_percent"] == pytest.approx(rv["full_grid"]["max_abs_rel_percent"], abs=1e-9)
    # the paper's claim: mean about 0.11 percent, maximum below 0.28 percent (fraction 0.0028)
    assert sa["mean_abs_rel"] < 0.0012 and sa["max_abs_rel"] < 0.0028
    # the README table (4 significant digits)
    assert (round(st["mean_abs_rel_percent"], 4), round(st["max_abs_rel_percent"], 4)) == (0.1145, 0.2781)
    assert (round(sv["mean_abs_rel_percent"], 4), round(sv["max_abs_rel_percent"], 4)) == (0.1141, 0.2782)
    # the released coefficients are the LP optimum of the released training set rounded to 12 significant
    # digits; the record stores the full-precision solution (lp_solution)
    coeffs, t = fit.minimax_fit(alpha[train], d[train], kappa[train], ps[train])
    lp = COEFFS["lp_solution"]
    for k in ("c0", "c1", "q"):
        assert coeffs[k] == pytest.approx(lp[k], abs=1e-7)
        assert coeffs[k] == pytest.approx(fit.RELEASED_COEFFICIENTS[k], abs=1e-7)
    assert 100 * np.expm1(t) == pytest.approx(lp["max_abs_rel_percent_training"], abs=2e-5)
    # the superseded previous coefficients are kept in the record for reference only
    prev = COEFFS["previous_coefficients"]
    assert prev["adopted"] is False
    sp = fit.evaluate_fit(alpha, d, kappa, ps, prev)
    assert sp["max_abs_rel_percent"] == pytest.approx(prev["full_grid"]["max_abs_rel_percent"], abs=1e-9)
    assert sp["mean_abs_rel_percent"] == pytest.approx(prev["full_grid"]["mean_abs_rel_percent"], abs=1e-9)


def test_wolfram_notebook_literals_match_record():
    """'Analytic fit.nb' defines a, b, q literally; they must equal the declared record."""
    import re
    nb = (ROOT / "Analytic fit.nb").read_text(encoding="utf-8")
    a = float(re.search(r'RowBox\[\{"a", "=", "([0-9.]+)"\}\]', nb).group(1))
    b = -float(re.search(r'RowBox\[\{"b", "=", \s*RowBox\[\{"-", "([0-9.]+)"\}\]\}\]', nb).group(1))
    q = float(re.search(r'RowBox\[\{"q", "=", "([0-9.]+)"\}\]', nb).group(1))
    assert (a, b, q) == (COEFFS["c0"], COEFFS["c1"], COEFFS["q"])
    assert "RandomSample" not in nb and "samplePoints" in nb


def test_fixed_sample_table_consistent_with_record_and_table():
    rows = list(csv.DictReader(open(ROOT / "data" / "fit_sample_table.csv", newline="", encoding="utf-8")))
    assert len(rows) == 4
    for r in rows:
        assert float(r["Ps_fit"]) == pytest.approx(fit.S_R_fit(float(r["alpha0"]), float(r["d"]), float(r["kappa"]), COEFFS), rel=1e-14)
        assert float(r["Delta_rel_percent"]) == pytest.approx(100 * abs(float(r["Ps_fit"]) / float(r["Ps_numerical"]) - 1), rel=1e-12)


def test_stored_split_manifest_matches_procedural_split():
    """data/fit_split_manifest.json freezes the verification groups produced by split_by_d_kappa (seed 1234)."""
    man = json.loads((ROOT / "data" / "fit_split_manifest.json").read_text(encoding="utf-8"))
    assert man["seed"] == 1234 and man["train_frac"] == 0.8 and man["n_groups"] == 6400
    assert man["n_verification_groups"] == 1280 and len(man["verification_groups_id_ik"]) == 1280
    A = np.logspace(-30, -21, 80); D = np.linspace(0.1, 0.4, 80); K = np.logspace(-9, -4, 80)
    # rebuild the (d, kappa) columns in table order and apply the procedural split
    d = np.tile(np.repeat(D, 80), 80); kappa = np.tile(K, 80 * 80)
    train = fit.split_by_d_kappa(d, kappa)
    idd = np.tile(np.repeat(np.arange(80), 80), 80); ik = np.tile(np.arange(80), 80 * 80)
    ver = sorted({(int(i), int(j)) for i, j in zip(idd[~train], ik[~train])})
    assert ver == [tuple(x) for x in man["verification_groups_id_ik"]]
    assert int(train.sum()) == man["n_training_rows"] and int((~train).sum()) == man["n_verification_rows"]
