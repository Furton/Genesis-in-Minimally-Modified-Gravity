"""Resumable amplitude-table generator: bounded grid, resume equivalence, rejection of bad state."""
import csv
import json
from pathlib import Path

import pytest

from vcdm_genesis import modes as md
from vcdm_genesis import table as tb

SMALL = ["--alpha0", "3e-27", "7e-24", "--d", "0.22", "0.33", "--kappa", "2e-8", "5e-7", "--processes", "2", "--chunk-size", "3"]


def _read(p: Path):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def test_small_grid_matches_direct_calls_and_is_ordered(tmp_path: Path):
    man = tb.main(["amplitudes", "--outdir", str(tmp_path)] + SMALL)
    assert man["status"] == "complete" and man["n_records"] == 8 and man["n_failed"] == 0
    aud = tb.main(["assemble", "--outdir", str(tmp_path)])
    assert aud["complete_and_clean"] and aud["all_finite_positive"]
    rows = _read(tmp_path / "amplitudes.csv")
    assert [r["alpha"] for r in rows] == ["3e-27"] * 4 + ["7e-24"] * 4          # alpha-major order
    assert [r["kappa"] for r in rows[:2]] == ["2e-08", "5e-07"]
    for r in rows:
        direct = md.evolve_scalar_mode(float(r["kappa"]), float(r["alpha"]), float(r["d"]), x_final=1e-7).S_R
        assert float(r["Ps"]) == direct
        assert r["status"] == "ok"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["x_final"] == 1e-7 and manifest["backend"].startswith("python")
    assert set(manifest["code_sha256"]) == {"background.py", "spectra.py", "modes.py", "table.py"}


def test_interrupted_then_resumed_equals_uninterrupted(tmp_path: Path):
    a = tmp_path / "a"; b = tmp_path / "b"
    tb.main(["amplitudes", "--outdir", str(a)] + SMALL)
    tb.main(["assemble", "--outdir", str(a)])
    tb.main(["amplitudes", "--outdir", str(b), "--stop-after-chunks", "1"] + SMALL)
    partial = tb.read_manifest(b)
    assert partial["status"] == "partial" and partial["n_records"] == 3
    tb.main(["amplitudes", "--outdir", str(b)] + SMALL)
    tb.main(["assemble", "--outdir", str(b)])
    assert (a / "amplitudes.csv").read_bytes() == (b / "amplitudes.csv").read_bytes()
    assert len(list((b / "chunks").glob("chunk_*.jsonl"))) == 3


def test_incompatible_resume_and_duplicate_keys_are_rejected(tmp_path: Path):
    tb.main(["amplitudes", "--outdir", str(tmp_path), "--stop-after-chunks", "1"] + SMALL)
    with pytest.raises(tb.IncompatibleResume):
        tb.main(["amplitudes", "--outdir", str(tmp_path), "--x-final", "0"] + SMALL)
    with pytest.raises(tb.IncompatibleResume):
        tb.main(["amplitudes", "--outdir", str(tmp_path), "--rtol", "1e-8"] + SMALL)
    # duplicate key injected into a second chunk file
    first = next((tmp_path / "chunks").glob("chunk_*.jsonl"))
    (tmp_path / "chunks" / "chunk_000099.jsonl").write_text(first.read_text().splitlines()[0] + "\n")
    with pytest.raises(RuntimeError, match="duplicate key"):
        tb.read_chunks(tmp_path)


def test_failed_modes_are_recorded_not_dropped(tmp_path: Path):
    # kappa <= 0 is rejected by the solver: the record must carry status=failed with the error text
    # (given through a points file: argparse before Python 3.13 does not accept "-1e-8" as a value)
    pts = tmp_path / "pts.csv"
    pts.write_text("alpha,d,kappa\n1e-25,0.3,-1e-8\n1e-25,0.3,1e-7\n")
    man = tb.main(["amplitudes", "--outdir", str(tmp_path), "--points", str(pts), "--processes", "2", "--chunk-size", "2"])
    assert man["n_failed"] == 1 and man["n_records"] == 2
    aud = tb.main(["assemble", "--outdir", str(tmp_path)])
    assert not aud["complete_and_clean"] and aud["n_failed"] == 1 and aud["failed_keys"] == [0]
    rows = _read(tmp_path / "amplitudes.csv")
    assert rows[0]["status"] == "failed" and rows[0]["Ps"] == ""
    assert rows[1]["status"] == "ok" and float(rows[1]["Ps"]) > 0


def test_points_file_and_axes_file(tmp_path: Path):
    pts = tmp_path / "pts.csv"
    pts.write_text("alpha,d,kappa\n1e-25,0.3,1e-7\n1e-25,0.3,2e-7\n")
    man = tb.main(["amplitudes", "--outdir", str(tmp_path / "p"), "--points", str(pts), "--processes", "2"])
    assert man["n_records"] == 2 and man["key_spec"]["kind"] == "points"
    axes = tmp_path / "axes.json"
    axes.write_text(json.dumps({"alpha": [1e-25], "d": [0.3], "kappa": [1e-7, 2e-7]}))
    man2 = tb.main(["amplitudes", "--outdir", str(tmp_path / "x"), "--axes", str(axes), "--processes", "2"])
    tb.main(["assemble", "--outdir", str(tmp_path / "p")]); tb.main(["assemble", "--outdir", str(tmp_path / "x")])
    assert [r["Ps"] for r in _read(tmp_path / "p" / "amplitudes.csv")] == [r["Ps"] for r in _read(tmp_path / "x" / "amplitudes.csv")]
    with pytest.raises(ValueError):
        tb.load_axes(_bad_axes(tmp_path))


def _bad_axes(tmp_path: Path) -> Path:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"alpha": [1e-25, 1e-26], "d": [0.3], "kappa": [1e-7]}))   # not increasing
    return p


def test_generated_axes_options(tmp_path: Path):
    man = tb.main(["amplitudes", "--outdir", str(tmp_path), "--alpha0", "1e-25", "--d-linspace", "0.2", "0.3", "2",
                   "--kappa-logspace", "1e-8", "1e-5", "7", "--processes", "2", "--chunk-size", "7"])
    assert man["n_records"] == 14 and man["key_spec"]["axes"]["d"] == [0.2, 0.3]
    k = man["key_spec"]["axes"]["kappa"]
    assert len(k) == 7 and k[0] == 1e-8 and k[-1] == 1e-5
    import numpy as np
    assert np.allclose(np.diff(np.log(k)), np.log(10) / 2, rtol=0, atol=1e-12)


def test_released_axes_file_is_exact_grid():
    axes = tb.load_axes(Path(__file__).resolve().parents[1] / "data" / "grid_axes.json")
    assert [len(axes[k]) for k in ("alpha", "d", "kappa")] == [80, 80, 80]
    keys = tb.keys_from_axes(axes)
    assert len(keys) == 512000 and keys[753][1:4] == (0, 9, 33)
    assert keys[753][4:] == (1e-30, 0.13417721518987344, 1.2263306841775643e-07)
