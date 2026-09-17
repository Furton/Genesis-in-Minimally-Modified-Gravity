"""Derivative stage of the table pipeline on a small log-uniform grid (Ps preserved, columns validated)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vcdm_genesis import derivatives as dv
from vcdm_genesis import table as tb
from vcdm_genesis.table_derivatives import run_derivative_stage

KAPPAS = [repr(float(k)) for k in np.logspace(-7, -6, 9)]      # 9 log-uniform points, h = ln(10)/8


def _amplitudes(tmp_path: Path, extra=()):
    argv = ["amplitudes", "--outdir", str(tmp_path / "amp"), "--alpha0", "1e-24", "--d", "0.3", "--kappa", *KAPPAS,
            "--processes", "3", "--chunk-size", "5", *extra]
    tb.main(argv)
    tb.main(["assemble", "--outdir", str(tmp_path / "amp")])
    return tmp_path / "amp" / "amplitudes.csv"


def test_stage_preserves_Ps_and_matches_pointwise_stencils(tmp_path: Path):
    amp = _amplitudes(tmp_path)
    out = tmp_path / "table.csv"
    man = run_derivative_stage(amp, tmp_path / "work", out, processes=3)
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    src = list(csv.DictReader(open(amp, newline="", encoding="utf-8")))
    assert [r["Ps"] for r in rows] == [r["Ps"] for r in src]                 # verbatim
    assert list(rows[0].keys()) == ["alpha", "d", "kappa", "Ps", "ns", "alpha_s"]
    assert man["pad_modes"]["n"] == 6 and man["Ps_preserved_verbatim"]
    h = man["grid"]["h_ln_kappa"]
    assert h == pytest.approx(np.log(10) / 8)
    # interior/edge rows vs the direct stencil at the same h: stencil nodes differ from grid nodes at 1e-16,
    # so agreement is at the solver noise level, not bitwise
    for idx in (0, 4, 8):
        r = rows[idx]
        direct = dv.derivatives_from_modes(float(r["kappa"]), 1e-24, 0.3, h=h, stencil=7)
        assert float(r["ns"]) == pytest.approx(direct.n_s, abs=1e-8)
        assert float(r["alpha_s"]) == pytest.approx(direct.alpha_s, abs=1e-7)
    assert man["five_point_vs_seven_point"]["max_abs_diff_alpha_s"] < 1e-5
    assert (tmp_path / "work" / "derivatives_manifest.json").exists()


def test_stage_refuses_failed_or_incomplete_amplitudes(tmp_path: Path):
    amp = _amplitudes(tmp_path)
    rows = list(csv.DictReader(open(amp, newline="", encoding="utf-8")))
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(rows):
            if i == 3:
                r = {**r, "Ps": "", "status": "failed"}
            w.writerow(r)
    with pytest.raises(RuntimeError, match="failed/missing"):
        run_derivative_stage(bad, tmp_path / "w2", tmp_path / "t2.csv", processes=2)
    with open(tmp_path / "short.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for r in rows[:-1]:
            w.writerow({**r, "alpha": "2e-24"})   # second alpha block with one row missing
    with pytest.raises(ValueError, match="complete"):
        run_derivative_stage(tmp_path / "short.csv", tmp_path / "w3", tmp_path / "t3.csv", processes=2)


def test_stage_is_resumable_via_pad_checkpoint(tmp_path: Path):
    amp = _amplitudes(tmp_path)
    m1 = run_derivative_stage(amp, tmp_path / "work", tmp_path / "t1.csv", processes=3)
    m2 = run_derivative_stage(amp, tmp_path / "work", tmp_path / "t2.csv", processes=3)   # pads already present
    assert m1["pad_modes"]["fingerprint"] == m2["pad_modes"]["fingerprint"]
    assert (tmp_path / "t1.csv").read_bytes() == (tmp_path / "t2.csv").read_bytes()
    assert m2["seconds"] < m1["seconds"]
