"""Generate Figure 1: ``figures/Plot_U.pdf``, the matter potential ``U_hat(y; d)``.

Headless, deterministic regeneration::

    python -m vcdm_genesis.figures.plot_u --outdir figures

Outputs (in ``--outdir``):

* ``Plot_U.pdf``            vector figure (paper asset, basename fixed)
* ``Plot_U_curves.csv``     columns ``d, y, U_hat, in_genesis_segment`` (every curve
                            contains its endpoint ``y = C = 2(1+d)/d`` exactly)
* ``Plot_U_manifest.json``  formula, normalization, d values, range, sample count,
                            generator version, library versions and SHA-256 of the
                            data and PDF

Presentation: log-log axes, horizontal ``y``, vertical ``tau_B^2 U/chi_1^2``.
The Genesis segment ``0 < y <= C`` is drawn solid; the potential *function*
beyond the endpoint (``y > C``) is drawn dashed and lighter, because no
post-Genesis trajectory is computed here.  Endpoints are marked.  No legend
is drawn (the caption of the paper describes the curves); ``--legend`` adds one.
The default range ``0.01 <= y <= 32`` with 1600 logarithmic samples is the
configuration of the paper's Figure 1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .. import __version__
from ..potential import D_MAX, D_MIN, U_hat, d_label, genesis_endpoint, plateau

DEFAULT_D = (1.0, 0.5, 1.0 / 3.0)
DEFAULT_YMIN = 0.01
DEFAULT_YMAX = 32.0
DEFAULT_N = 1600
FORMULA = "U_hat(y;d) = b^2 U/chi1^2 = C^(-p) [Gamma(p,y)/d + Gamma(p+1,y)], C=2(1+d)/d, p=2/d, Gamma = upper unregularized incomplete gamma; U(y->inf)=0"


def curve_samples(d: float, ymin: float, ymax: float, n: int) -> np.ndarray:
    """Logarithmic samples in ``[ymin, ymax]`` with the endpoint ``C`` inserted exactly."""
    y = np.geomspace(ymin, ymax, int(n))
    C = genesis_endpoint(d)
    if ymin <= C <= ymax:
        y = np.unique(np.concatenate([y, [C]]))
    return y


def build_curves(ds, ymin: float, ymax: float, n: int):
    curves = []
    for d in ds:
        if not (D_MIN <= d <= D_MAX):
            raise ValueError(f"d={d} outside supported range")
        y = curve_samples(d, ymin, ymax, n)
        U = U_hat(y, d)
        C = genesis_endpoint(d)
        curves.append({"d": float(d), "C": float(C), "y": y, "U": np.asarray(U), "in_genesis": y <= C,
                       "plateau": plateau(d), "label": d_label(d)})
    return curves


def write_curves_csv(path: Path, curves) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["d", "y", "U_hat", "in_genesis_segment"])
        for c in curves:
            for y, U, g in zip(c["y"], c["U"], c["in_genesis"]):
                w.writerow([repr(c["d"]), repr(float(y)), repr(float(U)), int(bool(g))])
    tmp.replace(path)


def render(curves, pdf_path: Path, legend: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        g = c["in_genesis"]
        ax.loglog(c["y"][g], c["U"][g], "-", color=col, lw=1.8, label=c["label"])
        if np.any(~g):
            # continue the function beyond the endpoint, visually distinct
            j = np.flatnonzero(~g)
            ax.loglog(np.concatenate([[c["C"]], c["y"][j]]), np.concatenate([[U_hat(c["C"], c["d"])], c["U"][j]]),
                      "--", color=col, lw=1.2, alpha=0.6)
        ax.plot([c["C"]], [U_hat(c["C"], c["d"])], "o", color=col, ms=5.5, zorder=5)
    if legend:
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color="0.35", ls="--", lw=1.2, alpha=0.8),
                    Line2D([], [], color="0.35", marker="o", ls="", ms=5.5)]
        labels += ["potential function for $y>2(1+d)/d$ (beyond Genesis)", "Genesis endpoint $y=2(1+d)/d$"]
        ax.legend(handles, labels, loc="lower left", fontsize=8.5, frameon=True)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$\tau_B^{2}\,U/\chi_1^{2}$")
    allU = np.concatenate([c["U"] for c in curves])
    ax.set_ylim(10 ** np.floor(np.log10(allU.min()) - 0.2), 10 ** np.ceil(np.log10(allU.max()) + 0.2))
    ax.set_xlim(min(c["y"].min() for c in curves), max(c["y"].max() for c in curves))
    ax.grid(True, which="major", lw=0.4, alpha=0.5)
    ax.grid(True, which="minor", lw=0.2, alpha=0.25)
    fig.tight_layout()
    tmp = pdf_path.with_suffix(".tmp.pdf")
    fig.savefig(tmp, format="pdf", metadata={"Creator": f"vcdm_genesis {__version__} plot_u", "CreationDate": None})
    plt.close(fig)
    tmp.replace(pdf_path)


def sha256_of(path: Path) -> str:
    """SHA-256 of a file; text files (.csv, .json) are hashed with LF-normalized bytes so the
    value equals the committed Git blob regardless of the checkout's line-ending conversion."""
    b = Path(path).read_bytes()
    if Path(path).suffix.lower() in (".csv", ".json", ".txt", ".md"):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def write_manifest(path: Path, curves, args, csv_path: Path, pdf_path: Path) -> dict:
    import matplotlib
    import scipy
    manifest = {
        "figure": "Figure 1 (fig:Matter_Potential), paper path figures/Plot_U.pdf",
        "formula": FORMULA,
        "normalization": "absolute; vertical axis tau_B^2 U/chi_1^2 = U_hat; no per-curve rescaling",
        "d_values": [c["d"] for c in curves],
        "labels": [c["label"] for c in curves],
        "genesis_endpoints_C": [c["C"] for c in curves],
        "plateaus_U_hat_0": [c["plateau"] for c in curves],
        "y_range": [args.ymin, args.ymax],
        "log_samples_per_curve": args.n_samples,
        "endpoint_inserted_exactly": True,
        "samples_written_per_curve": [int(len(c["y"])) for c in curves],
        "legend": bool(args.legend),
        "generator": "vcdm_genesis.figures.plot_u",
        "generator_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "hash_convention": "SHA-256 of LF-normalized bytes for text files (.csv/.json), raw bytes for PDF; equals the committed Git blob hashes",
        "curves_csv": csv_path.name, "curves_csv_sha256": sha256_of(csv_path),
        "pdf": pdf_path.name, "pdf_sha256": sha256_of(pdf_path),
        "note": "PDF bytes may differ across matplotlib/font versions; compare curve data and rendered content.",
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return manifest


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate Figure 1 (Plot_U.pdf) from the verified potential.")
    ap.add_argument("--outdir", type=Path, default=Path("figures"), help="output directory (created if missing)")
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N, help="logarithmic samples per curve (default 1600)")
    ap.add_argument("--ymin", type=float, default=DEFAULT_YMIN)
    ap.add_argument("--ymax", type=float, default=DEFAULT_YMAX)
    ap.add_argument("--d", type=float, action="append", help="shape parameter (repeatable); default 1, 1/2, 1/3")
    ap.add_argument("--legend", action="store_true", help="draw a legend (the paper's figure has none)")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    ds = tuple(args.d) if args.d else DEFAULT_D
    if args.n_samples < 16 or not (0.0 < args.ymin < args.ymax):
        raise SystemExit("invalid sampling arguments")
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    curves = build_curves(ds, args.ymin, args.ymax, args.n_samples)
    csv_path = outdir / "Plot_U_curves.csv"
    pdf_path = outdir / "Plot_U.pdf"
    man_path = outdir / "Plot_U_manifest.json"
    write_curves_csv(csv_path, curves)
    render(curves, pdf_path, legend=args.legend)
    manifest = write_manifest(man_path, curves, args, csv_path, pdf_path)
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes), {csv_path}, {man_path}")
    for c in curves:
        print(f"  {c['label']}: C={c['C']:.6g}  U_hat(0)={c['plateau']:.12g}  U_hat(C)={U_hat(c['C'], c['d']):.12g}  U_hat(ymax)={c['U'][-1]:.6g}")
    return manifest


if __name__ == "__main__":
    main()
