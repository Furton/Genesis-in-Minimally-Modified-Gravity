"""Reserved final-check sample: the released table against the independent formulation.

``tests/fixtures/final_check_sample.json`` holds 24 grid points (seed 20260918)
frozen before the table regeneration finished.  Each point is re-evolved with
the curvature-variable formulation (which never uses the pump ``z_xx/z``) and
compared with the released ``Ps``; the canonical solver must reproduce the
released value exactly (same code, same configuration).
"""
import csv
import json
from pathlib import Path

import pytest

from vcdm_genesis import modes as md

FIXTURE = Path(__file__).parent / "fixtures" / "final_check_sample.json"
TABLE = Path(__file__).resolve().parents[1] / "genesis_scan_table.csv"
TARGET = 1e-6


def _released_rows(indices):
    wanted = set(indices)
    rows = {}
    with open(TABLE, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for n, r in enumerate(rd):
            if n in wanted:
                rows[n] = r
                if len(rows) == len(wanted):
                    break
    return rows


def test_reserved_sample_against_released_table():
    assert TABLE.exists(), "the released scan table must be present"
    fix = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pts = fix["points"]
    assert len(pts) == 24 and fix["seed"] == 20260918
    idx = {p["ia"] * 6400 + p["id"] * 80 + p["ik"]: p for p in pts}
    rows = _released_rows(idx)
    worst_independent = 0.0
    for n, p in idx.items():
        r = rows[n]
        assert (float(r[0]), float(r[1]), float(r[2])) == (p["alpha"], p["d"], p["kappa"])
        Ps = float(r[3])
        indep = md.evolve_curvature_mode(p["kappa"], p["alpha"], p["d"], x_final=1e-7)
        canon = md.evolve_scalar_mode(p["kappa"], p["alpha"], p["d"], x_final=1e-7)
        assert canon.S_R == Ps, (n, canon.S_R, Ps)
        worst_independent = max(worst_independent, abs(indep.S_R / Ps - 1.0))
    assert worst_independent < TARGET
    assert worst_independent < 1e-8   # measured 4.3e-10 at release; guards silent drift
