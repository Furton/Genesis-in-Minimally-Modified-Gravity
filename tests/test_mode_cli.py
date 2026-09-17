"""Bounded command-line entry points write JSON records into an explicit output directory."""
import json
from pathlib import Path

import pytest

from vcdm_genesis import mode_cli


def test_single_mode_command(tmp_path: Path):
    rec = mode_cli.main(["single", "--alpha0", "1e-23", "--d", "0.3", "--kappa", "1e-7", "--x-final", "0", "--outdir", str(tmp_path)])
    files = list(tmp_path.glob("mode_*.json"))
    assert len(files) == 1
    on_disk = json.loads(files[0].read_text())
    assert on_disk["S_R"] == rec["S_R"] == pytest.approx(6.91139484127e19, rel=5e-9)
    assert on_disk["x_final"] == 0.0 and on_disk["config"]["rtol"] == 1e-10
    assert "selection" in on_disk and on_disk["selection"]["ratio_at_returned"] >= 1e5


def test_single_mode_R_formulation_and_override(tmp_path: Path):
    rec = mode_cli.main(["single", "--alpha0", "1e-23", "--d", "0.3", "--kappa", "1e-7", "--x-final", "0",
                         "--x-ini", "3e9", "--formulation", "R", "--outdir", str(tmp_path)])
    assert rec["x_ini"] == 3e9 and rec["formulation"].startswith("curvature")
    assert rec["S_R"] == pytest.approx(6.91139484127e19, rel=5e-9)


def test_converge_command(tmp_path: Path):
    rec = mode_cli.main(["converge", "--alpha0", "1e-30", "--d", "0.13417721518987344", "--kappa", "1.2263306841775643e-07",
                         "--x-final", "1e-7", "--outdir", str(tmp_path)])
    assert rec["rel_change_x_ini"] <= 1e-6 and rec["rel_change_rtol"] <= 1e-6 and rec["rel_diff_independent"] <= 1e-6
    assert rec["S_R"] == pytest.approx(1.48036818276e24, rel=5e-9)
    assert list(tmp_path.glob("converge_*.json"))


def test_tensor_command(tmp_path: Path):
    rec = mode_cli.main(["tensor", "--d", "0.5", "--kappa", "1e-7", "--x-final", "0", "--outdir", str(tmp_path)])
    assert rec["S_h"] == pytest.approx(2.0196314518e-15, rel=1e-8)
    assert list(tmp_path.glob("tensor_*.json"))


def test_benchmark_command_small(tmp_path: Path):
    rep = mode_cli.main(["benchmark", "--outdir", str(tmp_path), "--processes", "2", "--repeat", "1", "--limit", "3"])
    assert rep["sample_size"] == 3 and rep["parallel"]["max_rel_mismatch_vs_serial"] < 1e-12
    assert rep["projection_512000"]["hours_measured_rate"] > 0
    assert (tmp_path / "benchmark_modes.json").exists()
    full = mode_cli.benchmark_sample()
    assert len(full) == 45 and set(mode_cli.REGRESSION) <= set(full)
