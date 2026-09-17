"""Figure 5 generator: bounded regression (labels from configuration, curve data, manifest)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vcdm_genesis import fit
from vcdm_genesis.figures import spectral_index as si


def test_spectral_index_generator_bounded(tmp_path: Path):
    man = si.main(["--outdir", str(tmp_path), "--n-kappa", "17", "--processes", "3", "--d", "0.30", "--d", "0.35"])
    assert (tmp_path / "n_s.pdf").stat().st_size > 3000 and (tmp_path / "alpha_s.pdf").stat().st_size > 3000
    assert man["configuration"]["d_values"] == [0.30, 0.35] and man["configuration"]["n_kappa"] == 17
    assert man["fit_curves"]["coefficient_record_sha256"] == si.sha256_of(fit.COEFFICIENT_FILE)
    assert man["fit_curves"]["coefficients"] == {k: fit.load_coefficients()[k] for k in ("c0", "c1", "q")}
    assert "h=1e-2" in man["fit_curves"]["method"]
    rows = list(csv.DictReader(open(tmp_path / "spectral_index_curves.csv", newline="", encoding="utf-8")))
    ds = sorted({float(r["d"]) for r in rows})
    assert ds == [0.30, 0.35], "curve parameters come from the configuration"
    num = [r for r in rows if r["curve"] == "numerical" and float(r["d"]) == 0.35]
    assert len(num) == 17
    kappas = np.array([float(r["kappa"]) for r in num])
    assert kappas[0] == pytest.approx(1e-9) and kappas[-1] == pytest.approx(1e-4)
    ns = np.array([float(r["n_s"]) for r in num]); al = np.array([float(r["alpha_s"]) for r in num])
    assert np.all(np.isfinite(ns)) and np.all(np.isfinite(al)) and np.all(ns < 1.0) and np.all(al < 0.0)
    # numerical curve at kappa = 1e-4 vs the R3 reference: the 17-point step (h = 0.72 in ln kappa) is coarse,
    # so only truncation-limited agreement is expected here (the release step 0.1457 reaches 1e-7)
    assert ns[-1] == pytest.approx(0.90964681576, abs=1e-5) and al[-1] == pytest.approx(-0.03030813031, abs=1e-4)
    # fit curve derivatives equal the closed-form template derivatives
    fitrows = [r for r in rows if r["curve"] == "fit" and float(r["d"]) == 0.35]
    kf = np.array([float(r["kappa"]) for r in fitrows])
    nf, af = fit.fit_derivatives(1e-23, 0.35, kf, fit.load_coefficients())
    np.testing.assert_allclose([float(r["n_s"]) for r in fitrows], nf, rtol=0, atol=1e-12)
    np.testing.assert_allclose([float(r["alpha_s"]) for r in fitrows], af, rtol=0, atol=1e-12)
    assert json.loads((tmp_path / "spectral_index_manifest.json").read_text())["curves_csv_sha256"] == si.sha256_of(tmp_path / "spectral_index_curves.csv")
