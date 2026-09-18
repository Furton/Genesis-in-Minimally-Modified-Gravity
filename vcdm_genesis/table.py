"""Configurable, resumable amplitude-table generator.

Stages
------
``amplitudes``  evolve one converged scalar mode per parameter key and checkpoint
                the records atomically in chunk files (``<outdir>/chunks/*.jsonl``)
``assemble``    collect the chunks into ``amplitudes.csv`` (``alpha,d,kappa,Ps,status``)
                with a completeness/failure audit (``assembly_audit.json``)
``audit``       report completeness and failures without writing a table

The derivative stage and the final six-column assembly are in
:mod:`vcdm_genesis.derivatives` / the ``derivatives`` subcommand (:mod:`vcdm_genesis.table_derivatives`).

Examples::

    # bounded example: 2 x 2 x 9 grid outside the authoritative table (log-uniform kappa,
    # sufficient for the derivatives stage)
    python -m vcdm_genesis.table amplitudes --outdir out/small --alpha0 3e-27 7e-24 --d 0.22 0.33 \
        --kappa-logspace 1e-8 1e-5 9 --processes 4
    python -m vcdm_genesis.table assemble --outdir out/small
    python -m vcdm_genesis.table derivatives --amplitudes out/small/amplitudes.csv --workdir out/small/deriv \
        --output out/small/table.csv --processes 4

    # the released 80x80x80 axes (NOT run by default; ~2.5 h on 18 processes)
    python -m vcdm_genesis.table amplitudes --outdir out/full --axes data/grid_axes.json --processes 18

Records carry the parameter key, endpoint, solver settings hash, backend
identity, initial time, selection ratio, function evaluations, status and
error text.  Failed modes are recorded as ``status=failed`` and are never
replaced by a fit value or silently omitted.  Resuming into a directory whose
manifest was produced with different axes/settings is rejected; duplicate keys
in the checkpoint are rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .modes import SolverConfig, X_FINAL_PAPER, evolve_scalar_mode

SCHEMA_VERSION = 1
BACKEND = "python:scipy.integrate.solve_ivp/DOP853 exact-coefficient scalar mode solver (vcdm_genesis.modes)"


class IncompatibleResume(RuntimeError):
    """The output directory holds a checkpoint from a different configuration."""


# --------------------------------------------------------------------------- #
# keys and configuration fingerprint
# --------------------------------------------------------------------------- #
def load_axes(path: Path) -> dict:
    ax = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("alpha", "d", "kappa"):
        v = np.asarray(ax[k], dtype=float)
        if v.ndim != 1 or len(v) == 0 or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0):
            raise ValueError(f"axis {k} must be a strictly increasing finite list")
    return {"alpha": [float(v) for v in ax["alpha"]], "d": [float(v) for v in ax["d"]], "kappa": [float(v) for v in ax["kappa"]]}


def keys_from_axes(axes: dict):
    """Canonical order: alpha-major, then d, then kappa (the order of the released table)."""
    A, D, K = axes["alpha"], axes["d"], axes["kappa"]
    out = []
    i = 0
    for ia, a in enumerate(A):
        for idd, d in enumerate(D):
            for ik, k in enumerate(K):
                out.append((i, ia, idd, ik, a, d, k))
                i += 1
    return out


def keys_from_points(points):
    return [(i, -1, -1, -1, float(a), float(d), float(k)) for i, (a, d, k) in enumerate(points)]


def code_hashes() -> dict:
    here = Path(__file__).parent
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in ("background.py", "spectra.py", "modes.py", "table.py")}


def config_fingerprint(axes_or_points: dict, x_final: float, cfg: SolverConfig, chunk_size: int) -> str:
    payload = {"schema": SCHEMA_VERSION, "keys": axes_or_points, "x_final": x_final, "solver": cfg.to_dict(),
               "chunk_size": chunk_size, "code": code_hashes()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def _worker(args):
    idx, ia, idd, ik, alpha0, d, kappa, x_final, cfg_dict = args
    cfg = SolverConfig(**{**cfg_dict, "bracket": tuple(cfg_dict["bracket"])})
    t0 = time.perf_counter()
    rec = {"key": idx, "ia": ia, "id": idd, "ik": ik, "alpha": alpha0, "d": d, "kappa": kappa, "x_final": x_final}
    try:
        r = evolve_scalar_mode(kappa, alpha0, d, x_final=x_final, cfg=cfg)
        rec.update({"Ps": r.S_R, "x_ini": r.x_ini, "ratio_at_returned": r.selection.get("ratio_at_returned"),
                    "nfev": r.nfev, "status": "ok", "error": None})
    except Exception as exc:  # recorded, never hidden
        rec.update({"Ps": None, "x_ini": None, "ratio_at_returned": None, "nfev": None, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"})
    rec["seconds"] = time.perf_counter() - t0
    return rec


# --------------------------------------------------------------------------- #
# checkpoint I/O
# --------------------------------------------------------------------------- #
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_chunks(outdir: Path) -> dict:
    """Return ``{key: record}``; raises on duplicate keys."""
    recs = {}
    for f in sorted((outdir / "chunks").glob("chunk_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["key"] in recs:
                raise RuntimeError(f"duplicate key {r['key']} in checkpoint {f.name}")
            recs[r["key"]] = r
    return recs


def write_manifest(outdir: Path, manifest: dict) -> None:
    _atomic_write_text(outdir / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def read_manifest(outdir: Path) -> dict | None:
    p = outdir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------- #
# amplitude stage
# --------------------------------------------------------------------------- #
def run_amplitudes(outdir: Path, keys, key_spec: dict, x_final: float, cfg: SolverConfig, processes: int,
                   chunk_size: int = 500, limit: int = 0, stop_after_chunks: int = 0, quiet: bool = False) -> dict:
    outdir = Path(outdir)
    (outdir / "chunks").mkdir(parents=True, exist_ok=True)
    fp = config_fingerprint(key_spec, x_final, cfg, chunk_size)
    manifest = read_manifest(outdir)
    if manifest is None:
        manifest = {
            "schema_version": SCHEMA_VERSION, "fingerprint": fp, "backend": BACKEND,
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "platform": platform.platform(), "x_final": x_final, "solver_config": cfg.to_dict(),
            "key_spec": key_spec, "n_keys": len(keys), "chunk_size": chunk_size, "code_sha256": code_hashes(),
            "key_order": "alpha-major, d, kappa-minor (index 'key' is the position in that order)",
            "Ps_meaning": "S_R = b^2 M_Pl^2 P_R = kappa^3/(2 pi^2) |u_hat/z|^2 at x_final (dimensionless)",
            "status": "running",
        }
        write_manifest(outdir, manifest)
    elif manifest.get("fingerprint") != fp:
        raise IncompatibleResume(f"{outdir}/manifest.json was written for a different configuration "
                                 f"(fingerprint {manifest.get('fingerprint')} != {fp}); refusing to resume")
    done = read_chunks(outdir)
    todo = [k for k in keys if k[0] not in done]
    if limit:
        todo = todo[:limit]
    next_chunk = 1 + max([int(p.stem.split("_")[1]) for p in (outdir / "chunks").glob("chunk_*.jsonl")] or [-1])
    t_start = time.perf_counter()
    n_done_now = 0
    n_chunks = 0
    if not quiet:
        print(f"[table] {len(done)} records already present; {len(todo)} to compute; {processes} processes; chunk {chunk_size}")
    with ProcessPoolExecutor(max_workers=processes) as ex:
        for start in range(0, len(todo), chunk_size):
            batch = todo[start:start + chunk_size]
            args = [(k[0], k[1], k[2], k[3], k[4], k[5], k[6], x_final, cfg.to_dict()) for k in batch]
            recs = list(ex.map(_worker, args, chunksize=max(1, len(args) // (4 * processes) or 1)))
            text = "".join(json.dumps(r) + "\n" for r in recs)
            _atomic_write_text(outdir / "chunks" / f"chunk_{next_chunk:06d}.jsonl", text)
            next_chunk += 1
            n_chunks += 1
            n_done_now += len(recs)
            n_fail = sum(1 for r in recs if r["status"] != "ok")
            if not quiet:
                el = time.perf_counter() - t_start
                rate = n_done_now / el if el > 0 else float("nan")
                remaining = (len(todo) - n_done_now) / rate if rate > 0 else float("nan")
                print(f"[table] {len(done) + n_done_now}/{len(keys)} done  ({rate:.1f} modes/s, ~{remaining/3600:.2f} h left)"
                      f"{'  FAILURES: ' + str(n_fail) if n_fail else ''}", flush=True)
            if stop_after_chunks and n_chunks >= stop_after_chunks:
                break
    all_recs = read_chunks(outdir)
    complete = len(all_recs) == len(keys)
    manifest["status"] = "complete" if complete else "partial"
    manifest["n_records"] = len(all_recs)
    manifest["n_failed"] = sum(1 for r in all_recs.values() if r["status"] != "ok")
    manifest["last_run_seconds"] = time.perf_counter() - t_start
    write_manifest(outdir, manifest)
    return manifest


# --------------------------------------------------------------------------- #
# assembly and audit
# --------------------------------------------------------------------------- #
def audit_records(outdir: Path, n_keys: int) -> dict:
    recs = read_chunks(Path(outdir))
    missing = sorted(set(range(n_keys)) - set(recs))
    failed = sorted(k for k, r in recs.items() if r["status"] != "ok")
    ok = [r for r in recs.values() if r["status"] == "ok"]
    ps = np.array([r["Ps"] for r in ok], dtype=float) if ok else np.array([])
    secs = np.array([r["seconds"] for r in ok], dtype=float) if ok else np.array([])
    return {
        "n_keys": n_keys, "n_records": len(recs), "n_ok": len(ok), "n_failed": len(failed), "n_missing": len(missing),
        "failed_keys": failed[:1000], "missing_keys_head": missing[:1000],
        "all_finite_positive": bool(len(ps) and np.all(np.isfinite(ps)) and np.all(ps > 0.0)),
        "seconds_mean": float(secs.mean()) if len(secs) else None, "seconds_max": float(secs.max()) if len(secs) else None,
        "complete_and_clean": bool(not missing and not failed),
    }


def assemble(outdir: Path, output: Path | None = None) -> dict:
    outdir = Path(outdir)
    manifest = read_manifest(outdir)
    if manifest is None:
        raise FileNotFoundError(f"{outdir}/manifest.json not found")
    n_keys = manifest["n_keys"]
    recs = read_chunks(outdir)
    aud = audit_records(outdir, n_keys)
    output = Path(output) if output else outdir / "amplitudes.csv"
    rows = [recs[k] for k in sorted(recs)]
    tmp = output.with_name(output.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["alpha", "d", "kappa", "Ps", "status"])
        for r in rows:
            w.writerow([repr(float(r["alpha"])), repr(float(r["d"])), repr(float(r["kappa"])),
                        "" if r["Ps"] is None else repr(float(r["Ps"])), r["status"]])
    os.replace(tmp, output)
    aud["output"] = str(output)
    aud["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    aud["fingerprint"] = manifest["fingerprint"]
    _atomic_write_text(outdir / "assembly_audit.json", json.dumps(aud, indent=2) + "\n")
    return aud


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _key_spec_and_keys(a):
    if a.axes:
        axes = load_axes(a.axes)
        return {"kind": "axes", "axes": axes, "source": str(a.axes)}, keys_from_axes(axes)
    if a.points:
        pts = []
        with open(a.points, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pts.append((float(row["alpha"]), float(row["d"]), float(row["kappa"])))
        return {"kind": "points", "points": pts, "source": str(a.points)}, keys_from_points(pts)
    alpha = list(a.alpha0 or [])
    if a.alpha0_logspace:
        lo, hi, n = a.alpha0_logspace
        alpha += [float(v) for v in np.logspace(np.log10(lo), np.log10(hi), int(n))]
    dd = list(a.d or [])
    if a.d_linspace:
        lo, hi, n = a.d_linspace
        dd += [float(v) for v in np.linspace(lo, hi, int(n))]
    kappa = list(a.kappa or [])
    if a.kappa_logspace:
        lo, hi, n = a.kappa_logspace
        kappa += [float(v) for v in np.logspace(np.log10(lo), np.log10(hi), int(n))]
    if alpha and dd and kappa:
        axes = {"alpha": sorted(set(float(v) for v in alpha)), "d": sorted(set(float(v) for v in dd)),
                "kappa": sorted(set(float(v) for v in kappa))}
        return {"kind": "axes", "axes": axes, "source": "command line"}, keys_from_axes(axes)
    raise SystemExit("specify --axes FILE, --points FILE, or explicit/generated axes "
                     "(--alpha0/--alpha0-logspace, --d/--d-linspace, --kappa/--kappa-logspace)")


def cmd_amplitudes(a):
    key_spec, keys = _key_spec_and_keys(a)
    cfg = SolverConfig(rtol=a.rtol)
    procs = a.processes or max(1, (os.cpu_count() or 2) - 2)
    man = run_amplitudes(a.outdir, keys, key_spec, a.x_final, cfg, procs, chunk_size=a.chunk_size, limit=a.limit,
                         stop_after_chunks=a.stop_after_chunks)
    print(f"[table] status={man['status']} records={man['n_records']} failed={man['n_failed']} -> {a.outdir}")
    return man


def cmd_assemble(a):
    aud = assemble(a.outdir, a.output)
    print(f"[table] assembled {aud['n_ok']} ok / {aud['n_failed']} failed / {aud['n_missing']} missing of {aud['n_keys']} -> {aud['output']}")
    return aud


def cmd_derivatives(a):
    from .table_derivatives import run_derivative_stage
    procs = a.processes or max(1, (os.cpu_count() or 2) - 2)
    man = run_derivative_stage(a.amplitudes, a.workdir, a.output, procs, x_final=a.x_final, rtol=a.rtol,
                               stencil=a.stencil, chunk_size=a.chunk_size, delta=a.delta)
    print(f"[table] derivatives written -> {man['output']} (5- vs 7-point max |diff|: ns {man['five_point_vs_seven_point']['max_abs_diff_ns']:.2e}, "
          f"alpha_s {man['five_point_vs_seven_point']['max_abs_diff_alpha_s']:.2e})")
    return man


def cmd_audit(a):
    man = read_manifest(a.outdir)
    aud = audit_records(a.outdir, man["n_keys"])
    print(json.dumps({k: v for k, v in aud.items() if not k.endswith("_head") and k != "failed_keys"}, indent=2))
    return aud


def build_parser():
    ap = argparse.ArgumentParser(description="Resumable amplitude-table generator for the VCDM Genesis scan.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("amplitudes")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--axes", type=Path, help="JSON with alpha, d, kappa axis lists (e.g. data/grid_axes.json)")
    p.add_argument("--points", type=Path, help="CSV with columns alpha,d,kappa")
    p.add_argument("--alpha0", type=float, nargs="*"); p.add_argument("--d", type=float, nargs="*"); p.add_argument("--kappa", type=float, nargs="*")
    p.add_argument("--alpha0-logspace", type=float, nargs=3, metavar=("MIN", "MAX", "N"), help="add N log-spaced alpha0 values")
    p.add_argument("--d-linspace", type=float, nargs=3, metavar=("MIN", "MAX", "N"), help="add N linearly spaced d values")
    p.add_argument("--kappa-logspace", type=float, nargs=3, metavar=("MIN", "MAX", "N"),
                   help="add N log-spaced kappa values (the derivatives stage needs a log-uniform kappa axis with >= 7 points)")
    p.add_argument("--x-final", type=float, default=X_FINAL_PAPER)
    p.add_argument("--rtol", type=float, default=SolverConfig.rtol)
    p.add_argument("--processes", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--limit", type=int, default=0, help="compute at most N remaining keys (bounded runs)")
    p.add_argument("--stop-after-chunks", type=int, default=0, help="stop after N chunks (resume tests)")
    p.set_defaults(func=cmd_amplitudes)
    p = sub.add_parser("assemble")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_assemble)
    p = sub.add_parser("audit")
    p.add_argument("--outdir", type=Path, required=True)
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("derivatives", help="add ns/alpha_s columns to an amplitude table (Ps preserved verbatim)")
    p.add_argument("--amplitudes", type=Path, required=True, help="alpha,d,kappa,Ps[,status] table on a log-uniform kappa axis")
    p.add_argument("--workdir", type=Path, required=True, help="directory for the resumable pad modes and the manifest")
    p.add_argument("--output", type=Path, required=True, help="final alpha,d,kappa,Ps,ns,alpha_s table")
    p.add_argument("--x-final", type=float, default=X_FINAL_PAPER)
    p.add_argument("--rtol", type=float, default=SolverConfig.rtol)
    p.add_argument("--stencil", type=int, default=7)
    p.add_argument("--processes", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--delta", type=float, default=2e-9, help="assumed per-point |error| of ln Ps for the noise budget")
    p.set_defaults(func=cmd_derivatives)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
