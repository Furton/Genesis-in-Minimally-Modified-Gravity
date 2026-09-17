"""Independent validation of the reconstructed potential U_hat(y; d) and Figure 1 generator.

The reference values here are computed by direct quadrature of the canonical
matter equation in ``mpmath`` (never by calling the production gamma helper):

    dU_hat/dy = -C^{-p} e^{-y} y^{p-1} (1/d + y),   U_hat(y) = C^{-p} int_y^inf e^{-t} (t^{p-1}/d + t^p) dt,

which follows from U_chi = chi_1 (1 + d y)/(a^2 s^2), dy/dchi = -d y/chi_1,
a^2 = e^y, s = b (C/y)^{1/d} and U(y -> infinity) = 0.
"""
import csv
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from vcdm_genesis import potential as pot
from vcdm_genesis.figures import plot_u

mp.mp.dps = 30

THIRD = 1.0 / 3.0
ELEMENTARY_D = [1.0, 0.5, THIRD]
GENERIC_D = [0.3, 0.7]
TARGET_REL = 1e-11


def quad_U_hat(y, d):
    """Independent quadrature target (mpmath)."""
    d = mp.mpf(d)
    y = mp.mpf(y)
    p = 2 / d
    C = 2 * (1 + d) / d
    integrand = lambda t: mp.e ** (-t) * (t ** (p - 1) / d + t ** p)
    return C ** (-p) * mp.quad(integrand, [y, mp.inf])


def quad_dU_hat_dy(y, d):
    """Fundamental theorem: derivative of the quadrature = minus the integrand."""
    d = mp.mpf(d)
    y = mp.mpf(y)
    p = 2 / d
    C = 2 * (1 + d) / d
    return -C ** (-p) * mp.e ** (-y) * (y ** (p - 1) / d + y ** p)


def elementary(y, d):
    y = mp.mpf(y)
    if d == 1.0:
        return mp.e ** (-y) / 16 * (y ** 2 + 3 * y + 3)
    if d == 0.5:
        return mp.e ** (-y) / 1296 * (y ** 4 + 6 * y ** 3 + 18 * y ** 2 + 36 * y + 36)
    return mp.e ** (-y) / 262144 * (y ** 6 + 9 * y ** 5 + 45 * y ** 4 + 180 * y ** 3 + 540 * y ** 2 + 1080 * y + 1080)


def sample_points(d):
    C = pot.genesis_endpoint(d)
    return [0.01, 0.1, 1.0, C, 10.0, 20.0, 32.0, 0.5 * C, 2.0 * C]


@pytest.mark.parametrize("d", ELEMENTARY_D)
def test_elementary_cases(d):
    worst = 0.0
    for y in sample_points(d):
        got = pot.U_hat(y, d)
        ref = float(elementary(y, d))
        worst = max(worst, abs(got / ref - 1.0))
        assert abs(got / ref - 1.0) < TARGET_REL, (d, y, got, ref)
    assert worst < TARGET_REL


@pytest.mark.parametrize("d", ELEMENTARY_D + GENERIC_D)
def test_independent_quadrature(d):
    worst = 0.0
    for y in sample_points(d):
        got = pot.U_hat(y, d)
        ref = float(quad_U_hat(y, d))
        worst = max(worst, abs(got / ref - 1.0))
        assert abs(got / ref - 1.0) < TARGET_REL, (d, y, got, ref)
    assert worst < TARGET_REL


@pytest.mark.parametrize("d", ELEMENTARY_D + GENERIC_D)
def test_derivative_matches_quadrature_derivative(d):
    for y in [0.01, 0.1, 1.0, pot.genesis_endpoint(d), 20.0]:
        got = pot.dU_hat_dy(y, d)
        ref = float(quad_dU_hat_dy(y, d))
        assert abs(got / ref - 1.0) < TARGET_REL
        assert got < 0.0
    # and the derivative of the elementary polynomial where available
    if d in ELEMENTARY_D:
        for y in [0.3, 3.0]:
            ref = float(mp.diff(lambda t: elementary(t, d), mp.mpf(y)))
            assert abs(pot.dU_hat_dy(y, d) / ref - 1.0) < 1e-9


def test_plateaus_and_reference_values():
    exact = {1.0: 3.0 / 16.0, 0.5: 1.0 / 36.0, THIRD: 135.0 / 32768.0}
    ref001 = {1.0: 0.1874968750777096, 0.5: 0.02777777777393519, THIRD: 0.004119873046874998}
    for d in ELEMENTARY_D:
        assert pot.plateau(d) == pytest.approx(exact[d], rel=1e-13)
        assert pot.U_hat(0.0, d) == pytest.approx(exact[d], rel=1e-13)       # y = 0 handled analytically
        assert pot.U_hat(0.01, d) == pytest.approx(ref001[d], rel=1e-12)
    assert [pot.genesis_endpoint(d) for d in ELEMENTARY_D] == pytest.approx([4.0, 6.0, 8.0])
    # generic d: plateau formula 3 Gamma(2/d) / (d C^{2/d}) against mpmath
    for d in GENERIC_D:
        ref = 3 * mp.gamma(2 / mp.mpf(d)) / (mp.mpf(d) * (2 * (1 + mp.mpf(d)) / mp.mpf(d)) ** (2 / mp.mpf(d)))
        assert pot.plateau(d) == pytest.approx(float(ref), rel=1e-12)


def test_positive_decreasing_and_large_y_limit():
    y = np.geomspace(1e-3, 300.0, 600)
    for d in ELEMENTARY_D + GENERIC_D + [pot.D_MIN, pot.D_MAX]:
        U = pot.U_hat(y, d)
        assert np.all(np.isfinite(U)) and np.all(U > 0.0)
        assert np.all(pot.dU_hat_dy(y, d) < 0.0)
        # strict decrease only where a finite difference is resolvable in double precision:
        # the plateau (y << 2/d) is numerically flat and must not be tested that way
        slope = np.abs(pot.dU_hat_dy(y[:-1], d)) * np.diff(y) / U[:-1]
        resolvable = slope > 1e-10
        assert np.count_nonzero(resolvable) > 100
        assert np.all(np.diff(U)[resolvable] < 0.0)
        # large-y limit: many orders below the plateau, but still positive and representable
        assert 0.0 < U[-1] < 1e-50 * pot.plateau(d)
        assert pot.U_hat(300.0, d) > 0.0


def test_derivative_at_y_zero_is_finite_where_the_limit_is():
    """dU_hat/dy at y=0: -C^-p (1/d) y^(p-1) -> 0 for d<2, -1/(2 C) for d=2 (p=1), -inf for d>2."""
    assert pot.dU_hat_dy(0.0, 1.0) == 0.0 and pot.dU_hat_dy(0.0, 0.5) == 0.0
    assert pot.dU_hat_dy(0.0, 2.0) == pytest.approx(-1.0 / (2.0 * pot.genesis_endpoint(2.0)), rel=1e-15)
    assert pot.dU_hat_dy(0.0, 4.0) == -np.inf
    arr = pot.dU_hat_dy(np.array([0.0, 1e-3, 1.0]), 2.0)
    assert np.all(np.isfinite(arr)) and arr[0] < 0.0
    assert not np.isnan(pot.dU_hat_dy(np.array([0.0]), 2.0)).any()


def test_extremes_are_reported_not_clipped():
    with pytest.raises(ValueError):
        pot.U_hat(pot.Y_MAX + 1.0, 1.0)
    for bad_d in [0.0, -0.5, float("nan"), pot.D_MIN / 2, pot.D_MAX * 2]:
        with pytest.raises(ValueError):
            pot.U_hat(1.0, bad_d)
    with pytest.raises(ValueError):
        pot.U_hat(-1e-3, 1.0)
    with pytest.raises(ValueError):
        pot.U_hat(float("nan"), 1.0)
    with pytest.raises(ValueError):
        pot.U_hat(float("inf"), 1.0)


def test_labels_derived_from_values():
    assert pot.d_label(1.0) == "d = 1"
    assert pot.d_label(0.5) == "d = 1/2"
    assert pot.d_label(THIRD) == "d = 1/3"
    assert pot.d_label(0.3) == "d = 3/10"


# --------------------------------------------------------------------------- #
# generator: run headless into a temporary directory and check the written data
# --------------------------------------------------------------------------- #
def test_generator_end_to_end(tmp_path: Path):
    manifest = plot_u.main(["--outdir", str(tmp_path)])
    pdf = tmp_path / "Plot_U.pdf"
    csv_path = tmp_path / "Plot_U_curves.csv"
    man = tmp_path / "Plot_U_manifest.json"
    assert pdf.exists() and pdf.stat().st_size > 5000
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert json.loads(man.read_text()) == manifest
    assert manifest["curves_csv_sha256"] == plot_u.sha256_of(csv_path)
    assert manifest["pdf_sha256"] == plot_u.sha256_of(pdf)
    assert manifest["d_values"] == pytest.approx(list(plot_u.DEFAULT_D))
    assert manifest["y_range"] == [plot_u.DEFAULT_YMIN, plot_u.DEFAULT_YMAX]

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    assert set(rows[0].keys()) == {"d", "y", "U_hat", "in_genesis_segment"}
    by_d = {}
    for r in rows:
        by_d.setdefault(float(r["d"]), []).append((float(r["y"]), float(r["U_hat"]), int(r["in_genesis_segment"])))
    assert sorted(by_d) == sorted(plot_u.DEFAULT_D)
    for d, pts in by_d.items():
        ys = np.array([p[0] for p in pts]); Us = np.array([p[1] for p in pts]); flags = np.array([p[2] for p in pts])
        C = pot.genesis_endpoint(d)
        assert len(ys) >= plot_u.DEFAULT_N
        assert np.any(ys == C), "endpoint y=C must be present exactly"
        assert np.all(np.diff(ys) > 0)
        assert np.all((ys <= C) == flags.astype(bool))
        # written values against the independent elementary formulas at the exported coordinates
        ref = np.array([float(elementary(y, d)) for y in ys])
        assert np.max(np.abs(Us / ref - 1.0)) < TARGET_REL
        assert ys.min() == pytest.approx(plot_u.DEFAULT_YMIN) and ys.max() == pytest.approx(plot_u.DEFAULT_YMAX)


def test_generator_denser_sampling_agrees(tmp_path: Path):
    m1 = plot_u.main(["--outdir", str(tmp_path / "a")])
    m2 = plot_u.main(["--outdir", str(tmp_path / "b"), "--n-samples", "6400"])
    assert m2["samples_written_per_curve"][0] >= 4 * m1["samples_written_per_curve"][0] - 4
    r1 = {(r["d"], r["y"]): float(r["U_hat"]) for r in csv.DictReader(open(tmp_path / "a" / "Plot_U_curves.csv"))}
    r2 = {(r["d"], r["y"]): float(r["U_hat"]) for r in csv.DictReader(open(tmp_path / "b" / "Plot_U_curves.csv"))}
    common = set(r1) & set(r2)
    assert len(common) >= 3  # at least the exact endpoints and range limits coincide
    assert all(r1[k] == r2[k] for k in common)


def test_released_plot_u_manifest_is_consistent():
    """The released Figure 1 manifest hashes match the released files (LF-normalized text hashing)."""
    root = Path(__file__).resolve().parents[1]
    man = json.loads((root / "figures" / "Plot_U_manifest.json").read_text(encoding="utf-8"))
    assert man["pdf_sha256"] == plot_u.sha256_of(root / "figures" / "Plot_U.pdf")
    assert man["curves_csv_sha256"] == plot_u.sha256_of(root / "figures" / "Plot_U_curves.csv")
    assert man["d_values"] == pytest.approx(list(plot_u.DEFAULT_D)) and "LF-normalized" in man["hash_convention"]
    rows = list(csv.DictReader(open(root / "figures" / "Plot_U_curves.csv", newline="", encoding="utf-8")))
    assert len(rows) == 3 * 1601
    worst = 0.0
    for r in rows[::7]:
        worst = max(worst, abs(float(r["U_hat"]) / float(elementary(float(r["y"]), float(r["d"]))) - 1.0))
    assert worst < TARGET_REL
