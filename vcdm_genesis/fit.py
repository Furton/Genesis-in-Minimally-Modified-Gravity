"""Phenomenological amplitude template (manuscript Appendix) and the declared coefficient record.

Template for the dimensionless power ``S_R = b^2 M_Pl^2 P_R`` (``tau_B = -1``, ``M_Pl = 1`` convention)::

    ln S_R^fit = -ln(4 pi^2 alpha0) - 2(1+d)/d + 2 J/d + c0 + c1 d + q ln C_{nu*}

    A = 3(1+d)^2,   xi = kappa^d,   nu* = sqrt(9/4 + A xi (1 - xi)),   u_g = A (1 - xi) / (nu* + 3/2)
    J = 3 + sqrt(A) arctan(u_g/sqrt(A)) + (3/2) ln[A/(A + u_g^2)] - A (3 + u_g)/(A + u_g^2)
    ln C_nu = (2 nu - 3) ln 2 + 2 [lnGamma(nu) - lnGamma(3/2)]

``u_g`` is an auxiliary fit variable, not a mode function.  The coefficients
``(c0, c1, q)`` live in one authoritative record, ``data/fit_coefficients.json``;
every maintained representation (this module, the notebooks, the figures)
must use that record.

The minimax fit minimizes the maximum absolute logarithmic residual on the
training groups (a linear programme).  Validation statistics are *table
relative*: they describe the template's deviation from the stored reference
spectra, not the integration error of those spectra.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from .derivatives import fd_weights, stencil_offsets

__all__ = ["COEFFICIENT_FILE", "RELEASED_COEFFICIENTS", "load_coefficients", "template_parts", "log_S_R_fit",
           "S_R_fit", "fit_derivatives", "split_by_d_kappa", "minimax_fit", "evaluate_fit"]

COEFFICIENT_FILE = Path(__file__).resolve().parent.parent / "data" / "fit_coefficients.json"
RELEASED_COEFFICIENTS = {"c0": 0.0139557668608, "c1": -0.275286738213, "q": 1.32186006083}


def load_coefficients(path: Path | None = None) -> dict:
    rec = json.loads(Path(path or COEFFICIENT_FILE).read_text(encoding="utf-8"))
    for k in ("c0", "c1", "q"):
        if k not in rec or not np.isfinite(rec[k]):
            raise ValueError(f"coefficient record lacks a finite '{k}'")
    return rec


def template_parts(alpha0, d, kappa):
    """Return ``(log_base, logC)`` with ``ln S_R^fit = log_base + c0 + c1 d + q logC``."""
    alpha0 = np.asarray(alpha0, dtype=float)
    d = np.asarray(d, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    A = 3.0 * (1.0 + d) ** 2
    xi = np.exp(d * np.log(kappa))
    nu = np.sqrt(2.25 + A * xi * (1.0 - xi))
    u = A * (1.0 - xi) / (nu + 1.5)
    rootA = np.sqrt(A)
    J = 3.0 + rootA * np.arctan(u / rootA) + 1.5 * np.log(A / (A + u ** 2)) - A * (3.0 + u) / (A + u ** 2)
    log_base = -np.log(4.0 * np.pi ** 2 * alpha0) - 2.0 * (1.0 + d) / d + 2.0 * J / d
    logC = (2.0 * nu - 3.0) * np.log(2.0) + 2.0 * (gammaln(nu) - gammaln(1.5))
    return log_base, logC


def log_S_R_fit(alpha0, d, kappa, coeffs: dict):
    log_base, logC = template_parts(alpha0, d, kappa)
    return log_base + coeffs["c0"] + coeffs["c1"] * np.asarray(d, dtype=float) + coeffs["q"] * logC


def S_R_fit(alpha0, d, kappa, coeffs: dict):
    return np.exp(log_S_R_fit(alpha0, d, kappa, coeffs))


def fit_derivatives(alpha0, d, kappa, coeffs: dict, h: float = 1e-2, stencil: int = 7):
    """``n_s^fit`` and ``alpha_s^fit`` from the closed-form template by finite differences in ``ln kappa``.

    ``alpha0`` enters the template only through the additive constant ``-ln(4 pi^2 alpha0)``,
    so the derivatives are evaluated with that constant removed (they are exactly
    independent of ``alpha0``).  The default step ``h = 1e-2`` keeps the ``O(h^6)``
    truncation error and the round-off ``~1e-16 |L| / h^2`` both far below 1e-9.
    These are derivatives of the *template*, never of independently evolved modes.
    """
    if not np.all(np.isfinite(np.asarray(alpha0, dtype=float))) or np.any(np.asarray(alpha0, dtype=float) <= 0):
        raise ValueError("alpha0 must be finite and positive")
    kappa = np.asarray(kappa, dtype=float)
    off = stencil_offsets(stencil)
    w1 = fd_weights(1, off, h)
    w2 = fd_weights(2, off, h)
    const = -np.log(4.0 * np.pi ** 2)            # log_S_R_fit with alpha0 = 1 still carries -ln(4 pi^2)
    Ls = np.stack([log_S_R_fit(1.0, d, kappa * np.exp(j * h), coeffs) - const for j in off], axis=0)
    Ls = Ls - Ls[stencil // 2]                  # remove the remaining constant before differencing
    dL = np.tensordot(w1, Ls, axes=1)
    d2L = np.tensordot(w2, Ls, axes=1)
    return 1.0 + dL, d2L


# --------------------------------------------------------------------------- #
# grouped split and minimax fit (as in Fitting.ipynb)
# --------------------------------------------------------------------------- #
def split_by_d_kappa(d, kappa, train_frac: float = 0.8, seed: int = 1234) -> np.ndarray:
    """Boolean training mask: complete ``(d, kappa)`` groups, all alpha0 values together.

    Reproduces the split of ``Fitting.ipynb``: group ids are assigned in order of
    first appearance (``pandas.factorize`` on the ``(d, kappa)`` pairs), shuffled
    with ``numpy.random.default_rng(seed)``, and the first ``int(train_frac * n_groups)``
    ids form the training set.
    """
    import pandas as pd
    pair_id = pd.factorize(pd.MultiIndex.from_arrays([np.asarray(d), np.asarray(kappa)]))[0]
    unique_ids = np.arange(pair_id.max() + 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    train_ids = unique_ids[: int(train_frac * len(unique_ids))]
    return np.isin(pair_id, train_ids)


def minimax_fit(alpha0, d, kappa, S_R):
    """Solve ``min t  s.t. |c0 + c1 d + q logC - y| <= t`` (LP, HiGHS); returns ``(coeffs, t)``."""
    from scipy import sparse
    from scipy.optimize import linprog
    d = np.asarray(d, dtype=float)
    log_base, logC = template_parts(alpha0, d, kappa)
    y = np.log(np.asarray(S_R, dtype=float)) - log_base
    n = len(y)
    X = sparse.csr_matrix(np.column_stack([np.ones(n), d, logC]))
    minus_t = sparse.csr_matrix(-np.ones((n, 1)))
    A_ub = sparse.vstack([sparse.hstack([X, minus_t]), sparse.hstack([-X, minus_t])], format="csr")
    b_ub = np.concatenate([y, -y])
    res = linprog(c=np.array([0.0, 0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (None, None), (0.0, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"minimax LP failed: {res.message}")
    return {"c0": float(res.x[0]), "c1": float(res.x[1]), "q": float(res.x[2])}, float(res.x[3])


def evaluate_fit(alpha0, d, kappa, S_R, coeffs: dict) -> dict:
    """Relative-error statistics of the template against reference powers (fractions and percent)."""
    log_true = np.log(np.asarray(S_R, dtype=float))
    log_pred = log_S_R_fit(alpha0, d, kappa, coeffs)
    rel = np.expm1(log_pred - log_true)
    a = np.abs(rel)
    i = int(np.argmax(a))
    return {
        "n": int(len(a)),
        "mean_abs_rel": float(a.mean()), "median_abs_rel": float(np.median(a)), "max_abs_rel": float(a[i]),
        "q90_abs_rel": float(np.quantile(a, 0.9)), "q99_abs_rel": float(np.quantile(a, 0.99)),
        "mean_abs_rel_percent": float(100 * a.mean()), "max_abs_rel_percent": float(100 * a[i]),
        "argmax": {"alpha0": float(np.asarray(alpha0, dtype=float).reshape(-1)[i] if np.ndim(alpha0) else alpha0),
                   "d": float(np.asarray(d, dtype=float).reshape(-1)[i] if np.ndim(d) else d),
                   "kappa": float(np.asarray(kappa, dtype=float).reshape(-1)[i] if np.ndim(kappa) else kappa),
                   "ratio_fit_over_ref": float(np.exp(log_pred[i] - log_true[i]))},
        "convention": "abs(S_fit/S_ref - 1); 'percent' fields are 100x the fraction",
    }
