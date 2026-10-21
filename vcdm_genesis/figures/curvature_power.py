"""Rebuild Figure 4: scalar-mode evolution and endpoint power spectra.

Run from the repository root::

    python -m vcdm_genesis.figures.curvature_power --outdir figures --processes 8

Both panels use the exact numerical layer and the common endpoint x_f=1e-7.
The longer initial-time search accommodates the infrared end kappa=1e-14.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from .. import background as bg
from ..modes import (
    SolverConfig, X_FINAL_PAPER, adiabatic_initial_data,
    evolve_scalar_mode, select_initial_time,
)

ALPHA0 = 1e-22
D_VALUES = (0.3, 0.5, 0.7)
EVOLUTION_D, EVOLUTION_KAPPA = 0.3, 1e-7
KAPPA_MIN, KAPPA_MAX = 1e-14, 1e-5
N_KAPPA, N_EVOLUTION = 100, 1500


def evolution_curve(cfg: SolverConfig):
    """Sample the canonical mode using the same vacuum and tolerances as modes.py."""
    kappa, d = EVOLUTION_KAPPA, EVOLUTION_D
    x_ini, selection = select_initial_time(kappa, ALPHA0, d, cfg)
    u0, ux0, omega0, _ = adiabatic_initial_data(kappa, ALPHA0, d, x_ini)
    y0 = [u0.real, u0.imag, ux0.real, ux0.imag]
    scale = abs(u0)
    atol = np.array([scale, scale, scale * omega0, scale * omega0]) * cfg.rtol * cfg.atol_factor
    x = np.geomspace(X_FINAL_PAPER, x_ini, N_EVOLUTION)

    def rhs(xx, y):
        omega2 = bg.omega_squared_scalar(kappa, ALPHA0, d, xx)
        return [y[2], y[3], -omega2 * y[0], -omega2 * y[1]]

    sol = solve_ivp(rhs, (x_ini, X_FINAL_PAPER), y0, method=cfg.method,
                    rtol=cfg.rtol, atol=atol, t_eval=x[::-1])
    if not sol.success:
        raise RuntimeError(f"Figure 4 evolution failed: {sol.message}")
    u = (sol.y[0] + 1j * sol.y[1])[::-1]
    amplitude = np.abs(u / np.sqrt(bg.z_squared(kappa, ALPHA0, d, x)))
    if not np.all(np.isfinite(amplitude) & (amplitude > 0)):
        raise RuntimeError("Figure 4 evolution contains invalid amplitudes")
    return {"x": x, "amplitude": amplitude, "x_ini": x_ini, "selection": selection}


def _mode(task):
    kappa, d, cfg = task
    return evolve_scalar_mode(kappa, ALPHA0, d, x_final=X_FINAL_PAPER, cfg=cfg)


def build_curves(processes: int = 1):
    """Evolve all 300 spectrum modes and the independently sampled left panel."""
    if processes < 1:
        raise ValueError("processes must be at least 1")
    cfg = SolverConfig(x_search_max=1e23)
    evolution = evolution_curve(cfg)
    kappas = np.geomspace(KAPPA_MIN, KAPPA_MAX, N_KAPPA)
    tasks = [(float(k), d, cfg) for d in D_VALUES for k in kappas]
    if processes == 1:
        modes = list(map(_mode, tasks))
    else:
        with ProcessPoolExecutor(max_workers=processes) as executor:
            modes = list(executor.map(_mode, tasks, chunksize=4))
    if not all(np.isfinite(m.S_R) and m.S_R > 0 for m in modes):
        raise RuntimeError("Figure 4 spectrum contains invalid powers")
    return evolution, modes, cfg


def render(evolution, modes, outdir: Path):
    """Write the vector panels and return the figures for notebook display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []

    def finish(fig, ax, basename, xlabel, ylabel, legend_loc):
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="both", which="both", labelsize=16)
        ax.tick_params(axis="both", which="major", length=6, width=0.8)
        ax.tick_params(axis="both", which="minor", length=3, width=0.6)
        ax.legend(loc=legend_loc, frameon=True, fontsize=14)
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.25)
        fig.tight_layout()
        tmp = outdir / f"{basename}.tmp.pdf"
        fig.savefig(tmp, bbox_inches="tight", format="pdf",
                    metadata={"Creator": "vcdm_genesis curvature_power", "CreationDate": None})
        tmp.replace(outdir / f"{basename}.pdf")
        figures.append(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    amplitude_label = r"$|M_{\rm Pl}\mathcal{R}_k/\sqrt{-\tau_B}|$"
    ax.loglog(evolution["x"], evolution["amplitude"], color="tab:blue", lw=1.8,
              label=amplitude_label)
    # Reference slopes only; their vertical offsets are arbitrary, as in the supplied notebook.
    x1 = np.geomspace(3, 10**7.8, 500)
    x2 = np.geomspace(10**6.3, 10**10.5, 500)
    ax.loglog(x1, x1**(-1 - EVOLUTION_D) * 10**21.4, color="black", lw=1.2,
              ls="--", label=r"$\sim x^{-1-d}$")
    ax.loglog(x2, x2**(-EVOLUTION_D) * 10**14.35, color="black", lw=1.2,
              ls="-.", label=r"$\sim x^{-d}$")
    finish(fig, ax, "Evolution_R", r"$x$", amplitude_label, "lower left")

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    for d, color in zip(D_VALUES, ("tab:blue", "tab:red", "tab:green")):
        selected = [m for m in modes if m.d == d]
        ax.loglog([m.kappa for m in selected], [m.S_R for m in selected],
                  color=color, lw=1.8, label=rf"$d={d:g}$")
    finish(fig, ax, "P_R", r"$\kappa$", r"$\mathcal{P}_{\mathcal{R}}\,\tau_B^2 M_{\rm Pl}^2$",
           "lower right")
    return figures


def _write_csv(path, header, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def _sha256(path):
    data = path.read_bytes()
    if path.suffix in (".csv", ".py"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def rebuild(outdir: Path, processes: int = 1):
    """Rebuild both panels, their curve tables and provenance; return manifest and figures."""
    outdir = Path(outdir)
    evolution, modes, cfg = build_curves(processes)
    outdir.mkdir(parents=True, exist_ok=True)
    evolution_path = outdir / "curvature_evolution_curves.csv"
    power_path = outdir / "curvature_power_curves.csv"
    _write_csv(evolution_path, ["x", "abs_u_hat_over_z"],
               zip(evolution["x"], evolution["amplitude"]))
    _write_csv(power_path, ["alpha0", "d", "kappa", "S_R", "x_ini", "x_final"],
               ((m.alpha0, m.d, m.kappa, m.S_R, m.x_ini, m.x_final) for m in modes))
    figures = render(evolution, modes, outdir)

    import matplotlib
    files = ["Evolution_R.pdf", "P_R.pdf", evolution_path.name, power_path.name]
    manifest = {
        "figure": "Figure 4 (fig:Evolution_Power_R)",
        "generator": "vcdm_genesis.figures.curvature_power",
        "configuration": {"alpha0": ALPHA0, "evolution_d": EVOLUTION_D,
                          "evolution_kappa": EVOLUTION_KAPPA, "d_values": list(D_VALUES),
                          "kappa_range": [KAPPA_MIN, KAPPA_MAX], "n_kappa_per_d": N_KAPPA,
                          "n_evolution": N_EVOLUTION, "x_final": X_FINAL_PAPER},
        "normalization": {"evolution": "abs(u_hat/z) = abs(M_Pl R_k/sqrt(-tau_B))",
                          "spectrum": "S_R = tau_B^2 M_Pl^2 P_R = kappa^3/(2 pi^2) abs(u_hat/z)^2"},
        "numerical_method": "DOP853 with exact coefficients, retested adiabatic vacuum, no coefficient interpolation",
        "solver": cfg.to_dict(),
        "evolution_initial_time": evolution["x_ini"],
        "evolution_vacuum_checks": evolution["selection"],
        "minimum_spectrum_bracket_pump_ratio": min(m.selection["bracket_min_ratio"] for m in modes),
        "slope_guides": "x^(-1-d) and x^(-d); arbitrary vertical offsets from the supplied notebook",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "files_sha256": {name: _sha256(outdir / name) for name in files},
        "source_sha256": _sha256(Path(__file__)),
        "hash_convention": "SHA-256 of LF-normalized CSV/source bytes and raw PDF bytes",
        "corrections": ["Left ordinate identifies u_hat/z with M_Pl R_k/sqrt(-tau_B)",
                        "Both panels use the paper endpoint x_f=1e-7",
                        "Uses the maintained solver instead of interpolated coefficients"],
    }
    manifest_path = outdir / "curvature_power_manifest.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"Rebuilt Figure 4 in {outdir} (301 successful mode evolutions).")
    return manifest, figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.processes < 1:
        parser.error("--processes must be at least 1")
    manifest, _ = rebuild(args.outdir, processes=args.processes)
    return manifest


if __name__ == "__main__":
    main()
