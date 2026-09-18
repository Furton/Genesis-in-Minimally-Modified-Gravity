"""Generate Figure 5: ``figures/n_s.pdf`` and ``figures/alpha_s.pdf``.

Declared configuration (manuscript caption and rendered page)::

    alpha0 = 1e-23,   d in (0.30, 0.35, 0.40),   1e-9 <= kappa <= 1e-4

Solid curves: spectral index and running from independently evolved modes,
differentiated by centred seven-point finite differences in ``ln kappa`` on a
log-uniform grid with three extra converged modes beyond each boundary
(``vcdm_genesis.derivatives``).  Dashed curves: derivatives of the fit template
with the single declared coefficient record (``vcdm_genesis.fit``).  Both sets
of curve parameters and the legends are derived from this configuration.
The Planck 2018 marginal bands are drawn as given in the manuscript
(``n_s = 0.9649 +- 0.0042`` without running; ``alpha_s = -0.0045 +- 0.0067``
with running); they are separate marginal benchmarks, not a joint region.

Headless regeneration::

    python -m vcdm_genesis.figures.spectral_index --outdir figures --processes 8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..derivatives import derivative_columns, noise_budget
from ..fit import COEFFICIENT_FILE, fit_derivatives, load_coefficients
from ..modes import SolverConfig, X_FINAL_PAPER, evolve_scalar_mode

ALPHA0 = 1e-23
D_VALUES = (0.30, 0.35, 0.40)
KAPPA_MIN, KAPPA_MAX = 1e-9, 1e-4
N_KAPPA = 41
STENCIL = 7
PLANCK_NS = (0.9649, 0.0042)
PLANCK_AS = (-0.0045, 0.0067)


def _mode(args):
    kappa, alpha0, d, x_final, rtol = args
    r = evolve_scalar_mode(kappa, alpha0, d, x_final=x_final, cfg=SolverConfig(rtol=rtol))
    return float(np.log(r.S_R))


def numerical_curves(ds, alpha0, n_kappa, processes, x_final=X_FINAL_PAPER, rtol=1e-10, stencil=STENCIL):
    pad = stencil // 2
    kappa = np.logspace(np.log10(KAPPA_MIN), np.log10(KAPPA_MAX), n_kappa)
    h = float(np.diff(np.log(kappa)).mean())
    ext = np.concatenate([kappa[0] * np.exp(-h * np.arange(pad, 0, -1)), kappa, kappa[-1] * np.exp(h * np.arange(1, pad + 1))])
    tasks = [(float(k), alpha0, float(d), x_final, rtol) for d in ds for k in ext]
    with ProcessPoolExecutor(max_workers=processes) as ex:
        Ls = list(ex.map(_mode, tasks, chunksize=4))
    out = {}
    n = len(ext)
    for i, d in enumerate(ds):
        L = np.array(Ls[i * n:(i + 1) * n])
        ns, al = derivative_columns(kappa, L[pad:-pad], L[:pad], L[-pad:], stencil)
        out[float(d)] = {"kappa": kappa, "L": L[pad:-pad], "n_s": ns, "alpha_s": al, "L_ext": L, "kappa_ext": ext}
    return out, h


def fit_curves(ds, alpha0, coeffs, n_dense=400):
    kd = np.logspace(np.log10(KAPPA_MIN), np.log10(KAPPA_MAX), n_dense)
    out = {}
    for d in ds:
        ns, al = fit_derivatives(alpha0, float(d), kd, coeffs)
        out[float(d)] = {"kappa": kd, "n_s": ns, "alpha_s": al}
    return out


def render(num, fitc, ds, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for key, fname, ylabel, band in (("n_s", "n_s.pdf", r"$n_s$", PLANCK_NS), ("alpha_s", "alpha_s.pdf", r"$\alpha_s$", PLANCK_AS)):
        fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=150)
        ax.axhspan(band[0] - band[1], band[0] + band[1], color="0.85", zorder=0,
                   label=rf"Planck $1\sigma$ ({band[0]:.4f} $\pm$ {band[1]:.4f})")
        for i, d in enumerate(ds):
            col = colors[i % len(colors)]
            ax.semilogx(num[d]["kappa"], num[d][key], "-", color=col, lw=1.7, label=f"$d = {d:.2f}$ numerical")
            ax.semilogx(fitc[d]["kappa"], fitc[d][key], "--", color=col, lw=1.3, label=f"$d = {d:.2f}$ fit")
        if key == "n_s":
            ax.axhline(1.0, color="0.5", lw=0.6, ls=":")
        else:
            ax.axhline(0.0, color="0.5", lw=0.6, ls=":")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(KAPPA_MIN, KAPPA_MAX)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(fontsize=7.5, loc="lower left", ncol=1)
        fig.tight_layout()
        tmp = outdir / (fname + ".tmp.pdf")
        fig.savefig(tmp, format="pdf", metadata={"Creator": "vcdm_genesis spectral_index", "CreationDate": None})
        plt.close(fig)
        tmp.replace(outdir / fname)


def write_curves(path: Path, num, fitc, ds):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["curve", "d", "kappa", "n_s", "alpha_s", "ln_S_R"])
        for d in ds:
            for k, ns, al, L in zip(num[d]["kappa"], num[d]["n_s"], num[d]["alpha_s"], num[d]["L"]):
                w.writerow(["numerical", repr(float(d)), repr(float(k)), repr(float(ns)), repr(float(al)), repr(float(L))])
            for k, ns, al in zip(fitc[d]["kappa"], fitc[d]["n_s"], fitc[d]["alpha_s"]):
                w.writerow(["fit", repr(float(d)), repr(float(k)), repr(float(ns)), repr(float(al)), ""])
    tmp.replace(path)


def sha256_of(p: Path) -> str:
    """SHA-256 of a file; text files (.csv, .json) are hashed with LF-normalized bytes so the
    value equals the committed Git blob regardless of the checkout's line-ending conversion."""
    b = Path(p).read_bytes()
    if Path(p).suffix.lower() in (".csv", ".json", ".txt", ".md"):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate Figure 5 (n_s.pdf, alpha_s.pdf).")
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    ap.add_argument("--n-kappa", type=int, default=N_KAPPA)
    ap.add_argument("--processes", type=int, default=0)
    ap.add_argument("--alpha0", type=float, default=ALPHA0)
    ap.add_argument("--d", type=float, action="append")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--delta", type=float, default=1e-9, help="assumed per-point |error| of ln S_R for the noise budget")
    a = ap.parse_args(argv)
    ds = tuple(float(x) for x in (a.d or D_VALUES))
    procs = a.processes or max(1, (os.cpu_count() or 2) - 2)
    a.outdir.mkdir(parents=True, exist_ok=True)
    coeffs = load_coefficients()
    num, h = numerical_curves(ds, a.alpha0, a.n_kappa, procs, rtol=a.rtol)
    fitc = fit_curves(ds, a.alpha0, coeffs)
    render(num, fitc, ds, a.outdir)
    curves = a.outdir / "spectral_index_curves.csv"
    write_curves(curves, num, fitc, ds)
    import matplotlib, scipy
    manifest = {
        "figure": "Figure 5 (fig:Spectral_Index_and_running): figures/n_s.pdf, figures/alpha_s.pdf",
        "configuration": {"alpha0": a.alpha0, "d_values": list(ds), "kappa_range": [KAPPA_MIN, KAPPA_MAX], "n_kappa": a.n_kappa,
                          "x_final": X_FINAL_PAPER, "rtol": a.rtol, "stencil": STENCIL, "h_ln_kappa": h, "pad_modes_per_side": STENCIL // 2},
        "numerical_method": "independently evolved modes (vcdm_genesis.modes) differentiated by centred 7-point finite differences in ln kappa; no smoothing",
        "noise_budget": noise_budget(a.delta, h, STENCIL),
        "hash_convention": "SHA-256 of LF-normalized bytes for text files (.csv/.json), raw bytes for PDF; equals the committed Git blob hashes",
        "fit_curves": {"coefficient_record": "data/fit_coefficients.json", "coefficient_record_sha256": sha256_of(COEFFICIENT_FILE),
                       "coefficients": {k: coeffs[k] for k in ("c0", "c1", "q")},
                       "method": "7-point finite differences of the closed-form template in ln kappa (vcdm_genesis.fit.fit_derivatives, default step h=1e-2)"},
        "planck_bands": {"n_s": PLANCK_NS, "alpha_s": PLANCK_AS, "note": "separate marginal benchmarks, drawn as in the manuscript"},
        "generator": "vcdm_genesis.figures.spectral_index",
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "curves_csv": curves.name, "curves_csv_sha256": sha256_of(curves),
        "n_s_pdf_sha256": sha256_of(a.outdir / "n_s.pdf"), "alpha_s_pdf_sha256": sha256_of(a.outdir / "alpha_s.pdf"),
    }
    (a.outdir / "spectral_index_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for d in ds:
        i = int(np.argmin(np.abs(num[d]["kappa"] / 1e-4 - 1)))
        print(f"d={d:.2f}: n_s(kappa=1e-4)={num[d]['n_s'][i]:.8f}  alpha_s={num[d]['alpha_s'][i]:.8f}")
    print(f"wrote {a.outdir / 'n_s.pdf'}, {a.outdir / 'alpha_s.pdf'}, {curves}")
    return manifest


if __name__ == "__main__":
    main()
