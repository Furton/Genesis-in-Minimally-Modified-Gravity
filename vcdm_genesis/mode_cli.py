"""Bounded single-mode commands, convergence study and a fixed benchmark sample.

Examples (all outputs are JSON records written into ``--outdir``)::

    python -m vcdm_genesis.mode_cli single   --alpha0 1e-23 --d 0.3 --kappa 1e-7 --x-final 0 --outdir out
    python -m vcdm_genesis.mode_cli converge --alpha0 1e-23 --d 0.3 --kappa 1e-7 --x-final 1e-7 --outdir out
    python -m vcdm_genesis.mode_cli tensor   --d 0.5 --kappa 1e-7 --x-final 0 --outdir out
    python -m vcdm_genesis.mode_cli benchmark --outdir out --processes 8

The benchmark sample is fixed (regression cases, grid corners, typical
interior points, selector edge cases and a seeded stratified sample) so that
timings are comparable between runs and machines.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .modes import (SolverConfig, X_FINAL_PAPER, convergence_study, evolve_curvature_mode,
                    evolve_scalar_mode, evolve_tensor_mode)

# ----------------------------------------------------------------------------- #
# fixed benchmark sample
# ----------------------------------------------------------------------------- #
REGRESSION = {
    "R1": (1e-23, 0.3, 1e-7),
    "R3": (1e-23, 0.35, 1e-4),
    "R4": (1e-23, 0.1, 1e-4),
    "R6": (1e-30, 0.15316455696202533, 2.2616759492228647e-06),
    "R7": (1e-30, 0.13417721518987344, 1.2263306841775643e-07),
}
ALPHA_GRID = np.logspace(-30, -21, 80)
D_GRID = np.linspace(0.1, 0.4, 80)
KAPPA_GRID = np.logspace(-9, -4, 80)
CORNERS = {f"corner_{i}": (float(a), float(d), float(k)) for i, (a, d, k) in enumerate(
    (a, d, k) for a in (ALPHA_GRID[0], ALPHA_GRID[-1]) for d in (D_GRID[0], D_GRID[-1]) for k in (KAPPA_GRID[0], KAPPA_GRID[-1]))}
TYPICAL = {
    "mid": (float(ALPHA_GRID[40]), float(D_GRID[40]), float(KAPPA_GRID[40])),
    "fig5_d030_kmid": (1e-23, 0.30, 1e-6),
    "fig5_d040_kmax": (1e-23, 0.40, 1e-4),
    "small_d_small_k": (float(ALPHA_GRID[10]), float(D_GRID[2]), float(KAPPA_GRID[5])),
    "large_alpha_small_d": (1e-21, 0.1, 1e-9),
}
STRATIFIED_SEED = 20260917
STRATIFIED_N = 27


def stratified_sample(n: int = STRATIFIED_N, seed: int = STRATIFIED_SEED):
    """Frozen stratified sample of grid points: one per (alpha, d, kappa) octant bin, seeded."""
    rng = np.random.default_rng(seed)
    pts = {}
    bins = 3
    i = 0
    for ia in range(bins):
        for idd in range(bins):
            for ik in range(bins):
                if i >= n:
                    break
                a = int(rng.integers(ia * 80 // bins, (ia + 1) * 80 // bins))
                d = int(rng.integers(idd * 80 // bins, (idd + 1) * 80 // bins))
                k = int(rng.integers(ik * 80 // bins, (ik + 1) * 80 // bins))
                pts[f"strat_{i:02d}_a{a}_d{d}_k{k}"] = (float(ALPHA_GRID[a]), float(D_GRID[d]), float(KAPPA_GRID[k]))
                i += 1
    return pts


def benchmark_sample():
    s = {}
    s.update(REGRESSION)
    s.update(CORNERS)
    s.update(TYPICAL)
    s.update(stratified_sample())
    return s


# ----------------------------------------------------------------------------- #
# workers
# ----------------------------------------------------------------------------- #
def _run_one(item):
    name, (alpha0, d, kappa), x_final, cfg_dict = item
    cfg = SolverConfig(**{**cfg_dict, "bracket": tuple(cfg_dict["bracket"])})
    t0 = time.perf_counter()
    r = evolve_scalar_mode(kappa, alpha0, d, x_final=x_final, cfg=cfg)
    dt = time.perf_counter() - t0
    return {"name": name, "alpha0": alpha0, "d": d, "kappa": kappa, "x_final": x_final, "S_R": r.S_R,
            "x_ini": r.x_ini, "nfev": r.nfev, "n_steps": r.n_steps, "seconds": dt,
            "ratio_at_returned": r.selection.get("ratio_at_returned")}


def _noop(_):
    return None


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------- #
# subcommands
# ----------------------------------------------------------------------------- #
def cmd_single(a):
    cfg = SolverConfig(rtol=a.rtol)
    fn = evolve_curvature_mode if a.formulation == "R" else evolve_scalar_mode
    t0 = time.perf_counter()
    r = fn(a.kappa, a.alpha0, a.d, x_ini=a.x_ini, x_final=a.x_final, cfg=cfg)
    rec = r.to_record()
    rec["seconds"] = time.perf_counter() - t0
    out = a.outdir / f"mode_a{a.alpha0:.6e}_d{a.d:.10g}_k{a.kappa:.6e}_xf{a.x_final:g}_{a.formulation}.json"
    _write_json(out, rec)
    print(f"S_R = {r.S_R:.12e}  x_ini = {r.x_ini:.6e}  nfev = {r.nfev}  ({rec['seconds']:.2f} s)  -> {out}")
    return rec


def cmd_converge(a):
    cfg = SolverConfig(rtol=a.rtol)
    t0 = time.perf_counter()
    res = convergence_study(a.kappa, a.alpha0, a.d, x_final=a.x_final, cfg=cfg)
    rec = {k: (v.to_record() if hasattr(v, "to_record") else v) for k, v in res.items()}
    rec["seconds"] = time.perf_counter() - t0
    out = a.outdir / f"converge_a{a.alpha0:.6e}_d{a.d:.10g}_k{a.kappa:.6e}_xf{a.x_final:g}.json"
    _write_json(out, rec)
    print(f"S_R = {res['S_R']:.12e}  d(x_ini x2) = {res['rel_change_x_ini']:.3e}  d(rtol/10) = {res['rel_change_rtol']:.3e}  "
          f"independent = {res.get('rel_diff_independent', float('nan')):.3e}  -> {out}")
    return rec


def cmd_tensor(a):
    cfg = SolverConfig(rtol=a.rtol)
    t0 = time.perf_counter()
    r = evolve_tensor_mode(a.kappa, a.d, x_ini=a.x_ini, x_final=a.x_final, cfg=cfg)
    rec = {"kappa": r.kappa, "d": r.d, "x_ini": r.x_ini, "x_final": r.x_final, "S_h": r.S_h, "a_final": r.a_final,
           "u_final": [r.u_final.real, r.u_final.imag], "nfev": r.nfev, "n_steps": r.n_steps, "selection": r.selection,
           "config": r.config, "seconds": time.perf_counter() - t0}
    out = a.outdir / f"tensor_d{a.d:.10g}_k{a.kappa:.6e}_xf{a.x_final:g}.json"
    _write_json(out, rec)
    print(f"S_h = {r.S_h:.12e}  x_ini = {r.x_ini:.6e}  nfev = {r.nfev}  -> {out}")
    return rec


def cmd_benchmark(a):
    cfg = SolverConfig(rtol=a.rtol)
    sample = benchmark_sample()
    if a.limit:
        sample = dict(list(sample.items())[: a.limit])
    items = [(name, tup, a.x_final, cfg.to_dict()) for name, tup in sample.items()]
    # serial
    t0 = time.perf_counter()
    serial = [_run_one(it) for it in items]
    serial_wall = time.perf_counter() - t0
    secs = [r["seconds"] for r in serial]
    # parallel: measure pool start-up separately, then steady-state throughput on the sample repeated
    procs = a.processes or max(1, (os.cpu_count() or 2) - 2)
    rep_items = items * max(1, a.repeat)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=procs) as ex:
        list(ex.map(_noop, range(procs)))          # warm-up: interpreter start + imports
        startup_wall = time.perf_counter() - t0
        t1 = time.perf_counter()
        par = list(ex.map(_run_one, rep_items, chunksize=1))
        par_wall = time.perf_counter() - t1
    mismatch = max(abs(p["S_R"] / s["S_R"] - 1.0) for p, s in zip(par[: len(items)], serial))
    n_full = 80 ** 3
    mean_s, med_s = statistics.mean(secs), statistics.median(secs)
    p90 = float(np.percentile(secs, 90)); mx = max(secs)
    eff = (serial_wall * max(1, a.repeat)) / (procs * par_wall) if par_wall > 0 else float("nan")
    per_mode_parallel = par_wall / len(rep_items)
    proj_hours_meas = n_full * per_mode_parallel / 3600.0
    proj_hours_p90 = n_full * p90 / (procs * max(eff, 1e-6)) / 3600.0
    proj_hours_max = n_full * mx / (procs * max(eff, 1e-6)) / 3600.0
    report = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "processes": procs, "rtol": a.rtol, "x_final": a.x_final,
        "sample_size": len(items), "sample_names": [it[0] for it in items],
        "serial": {"wall_seconds": serial_wall, "mean_seconds": mean_s, "median_seconds": med_s,
                   "p90_seconds": p90, "max_seconds": mx, "max_name": serial[int(np.argmax(secs))]["name"],
                   "mean_nfev": statistics.mean(r["nfev"] for r in serial)},
        "parallel": {"wall_seconds": par_wall, "efficiency": eff, "seconds_per_mode_wall": per_mode_parallel,
                     "pool_startup_seconds": startup_wall, "modes_timed": len(rep_items), "repeat": a.repeat,
                     "max_rel_mismatch_vs_serial": mismatch},
        "projection_512000": {
            "method": "measured parallel wall per mode x 512000 (includes process startup, pickling and I/O of records); "
                      "upper-tail variants use p90/max serial cost divided by processes*efficiency",
            "hours_measured_rate": proj_hours_meas, "hours_p90_rate": proj_hours_p90, "hours_max_rate": proj_hours_max,
            "hours_serial_mean": n_full * mean_s / 3600.0,
        },
        "records": serial,
    }
    out = a.outdir / "benchmark_modes.json"
    _write_json(out, report)
    print(f"sample {len(items)} modes: serial mean {mean_s*1e3:.1f} ms, median {med_s*1e3:.1f} ms, p90 {p90*1e3:.1f} ms, "
          f"max {mx*1e3:.1f} ms ({report['serial']['max_name']}); parallel x{procs}: {len(rep_items)} modes in {par_wall:.1f} s wall "
          f"(startup {startup_wall:.1f} s), eff {eff:.2f}")
    print(f"512000-mode projection: {proj_hours_meas:.2f} h at measured rate, {proj_hours_p90:.2f} h at p90, {proj_hours_max:.2f} h at max; "
          f"serial {report['projection_512000']['hours_serial_mean']:.1f} h  -> {out}")
    return report


def build_parser():
    ap = argparse.ArgumentParser(description="Bounded mode calculations for the VCDM Genesis scan.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, tensor=False):
        if not tensor:
            p.add_argument("--alpha0", type=float, required=True)
        p.add_argument("--d", type=float, required=True)
        p.add_argument("--kappa", type=float, required=True)
        p.add_argument("--x-final", type=float, default=(0.0 if tensor else X_FINAL_PAPER))
        p.add_argument("--x-ini", type=float, default=None, help="override the selected initial time")
        p.add_argument("--rtol", type=float, default=SolverConfig.rtol)
        p.add_argument("--outdir", type=Path, required=True)

    p = sub.add_parser("single"); common(p)
    p.add_argument("--formulation", choices=["u", "R"], default="u")
    p.set_defaults(func=cmd_single)
    p = sub.add_parser("converge"); common(p); p.set_defaults(func=cmd_converge)
    p = sub.add_parser("tensor"); common(p, tensor=True); p.set_defaults(func=cmd_tensor)
    p = sub.add_parser("benchmark")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--processes", type=int, default=0, help="worker processes (default: cpu_count - 2)")
    p.add_argument("--rtol", type=float, default=SolverConfig.rtol)
    p.add_argument("--x-final", type=float, default=X_FINAL_PAPER)
    p.add_argument("--repeat", type=int, default=4, help="repeat the sample this many times in the parallel stage")
    p.add_argument("--limit", type=int, default=0, help="use only the first N sample points (tests)")
    p.set_defaults(func=cmd_benchmark)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    return a.func(a)


if __name__ == "__main__":
    main()
