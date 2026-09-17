"""Spectral index and running from converged log-power samples.

With ``L = ln S_R`` and ``l = ln kappa`` at fixed ``(alpha0, d)``::

    n_s = 1 + dL/dl,      alpha_s = d^2 L / dl^2.

Derivatives are taken by centred finite differences on a uniform ``l`` grid
(Fornberg weights; default seven-point stencil, truncation error ``O(h^6)``).
No smoothing is applied.  Values at the ends of a range need centred stencils,
so the callers supply extra converged modes beyond the displayed/tabulated
boundaries (``pad`` points on each side).

Noise propagation: with a per-point absolute error ``delta`` in ``L``, the
finite-difference error is bounded by ``delta * sum|w|``.  For the five-point
second derivative that bound is ``(16/3) delta / h^2``; for the seven-point
stencil it is ``(272/45) delta / h^2 ~ 6.04 delta / h^2``.  ``noise_budget``
returns these bounds so that the integration precision and the step size are
chosen together (an ODE tolerance is not by itself a measured ``delta``).

These finite-difference derivatives of *independently evolved* modes must
never be confused with derivatives of the fitting template
(:func:`vcdm_genesis.fit.fit_derivatives`).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from .modes import SolverConfig, X_FINAL_PAPER, evolve_scalar_mode

__all__ = ["fd_weights", "stencil_offsets", "noise_budget", "derivatives_uniform", "DerivativeResult",
           "derivatives_from_modes", "derivative_columns"]


def stencil_offsets(stencil: int):
    if stencil % 2 == 0 or stencil < 3:
        raise ValueError("stencil must be an odd integer >= 3")
    r = stencil // 2
    return list(range(-r, r + 1))


def fd_weights(m: int, offsets, h: float = 1.0) -> np.ndarray:
    """Finite-difference weights for the ``m``-th derivative at 0 from nodes ``offsets*h`` (Fornberg)."""
    offsets = np.asarray(offsets, dtype=float)
    n = len(offsets)
    if m >= n:
        raise ValueError("need more than m nodes")
    # solve Vandermonde system: sum_j w_j (x_j)^k / k! = delta_{k,m}
    V = np.vander(offsets, n, increasing=True).T / np.array([factorial(k) for k in range(n)])[:, None]
    rhs = np.zeros(n)
    rhs[m] = 1.0
    w = np.linalg.solve(V, rhs)
    return w / h ** m


def noise_budget(delta: float, h: float, stencil: int = 7) -> dict:
    """Conservative bounds ``delta * sum|w|`` on the first and second derivative errors."""
    off = stencil_offsets(stencil)
    w1 = fd_weights(1, off, h)
    w2 = fd_weights(2, off, h)
    return {"stencil": stencil, "h": h, "delta": delta,
            "first_derivative_bound": float(delta * np.abs(w1).sum()),
            "second_derivative_bound": float(delta * np.abs(w2).sum()),
            "sum_abs_w1_times_h": float(np.abs(w1).sum() * h),
            "sum_abs_w2_times_h2": float(np.abs(w2).sum() * h * h)}


def derivatives_uniform(l, L, stencil: int = 7):
    """Centred derivatives on a uniform grid.

    Returns ``(dL, d2L, valid)`` where ``valid`` marks the points with a full stencil;
    the others are ``nan``.
    """
    l = np.asarray(l, dtype=float)
    L = np.asarray(L, dtype=float)
    if l.ndim != 1 or L.shape != l.shape or len(l) < stencil:
        raise ValueError("l and L must be 1-D of equal length >= stencil")
    dl = np.diff(l)
    h = float(dl.mean())
    if not np.allclose(dl, h, rtol=1e-8, atol=0.0):
        raise ValueError("grid in l = ln kappa must be uniform")
    off = stencil_offsets(stencil)
    w1 = fd_weights(1, off, h)
    w2 = fd_weights(2, off, h)
    r = stencil // 2
    n = len(l)
    dL = np.full(n, np.nan)
    d2L = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i in range(r, n - r):
        seg = L[i - r:i + r + 1]
        dL[i] = float(np.dot(w1, seg))
        d2L[i] = float(np.dot(w2, seg))
        valid[i] = True
    return dL, d2L, valid


@dataclass
class DerivativeResult:
    kappa: float
    alpha0: float
    d: float
    h: float
    stencil: int
    n_s: float
    alpha_s: float
    L_centre: float
    kappas: list
    log_powers: list
    x_final: float


def derivatives_from_modes(kappa: float, alpha0: float, d: float, h: float, stencil: int = 7,
                           x_final: float = X_FINAL_PAPER, cfg: SolverConfig | None = None) -> DerivativeResult:
    """Evolve the stencil modes ``kappa * exp(j h)`` and differentiate ``ln S_R`` at ``kappa``."""
    cfg = cfg or SolverConfig()
    off = stencil_offsets(stencil)
    ks = [kappa * float(np.exp(j * h)) for j in off]
    Ls = [float(np.log(evolve_scalar_mode(k, alpha0, d, x_final=x_final, cfg=cfg).S_R)) for k in ks]
    w1 = fd_weights(1, off, h)
    w2 = fd_weights(2, off, h)
    return DerivativeResult(kappa=kappa, alpha0=alpha0, d=d, h=h, stencil=stencil,
                            n_s=1.0 + float(np.dot(w1, Ls)), alpha_s=float(np.dot(w2, Ls)),
                            L_centre=Ls[stencil // 2], kappas=ks, log_powers=Ls, x_final=x_final)


def derivative_columns(kappa_main, L_main, L_pad_low, L_pad_high, stencil: int = 7):
    """``n_s`` and ``alpha_s`` on a log-uniform main grid using ``pad`` extra log-powers on each side.

    ``L_pad_low`` are the log-powers at ``kappa_main[0] * exp(-j h)``, ``j = pad..1``
    (increasing kappa order), and ``L_pad_high`` at ``kappa_main[-1] * exp(+j h)``, ``j = 1..pad``.
    """
    kappa_main = np.asarray(kappa_main, dtype=float)
    r = stencil // 2
    L_pad_low = np.asarray(L_pad_low, dtype=float)
    L_pad_high = np.asarray(L_pad_high, dtype=float)
    if len(L_pad_low) != r or len(L_pad_high) != r:
        raise ValueError(f"need {r} pad values on each side for a {stencil}-point stencil")
    l = np.log(kappa_main)
    h = float(np.diff(l).mean())
    l_ext = np.concatenate([l[0] - h * np.arange(r, 0, -1), l, l[-1] + h * np.arange(1, r + 1)])
    L_ext = np.concatenate([L_pad_low, np.asarray(L_main, dtype=float), L_pad_high])
    dL, d2L, valid = derivatives_uniform(l_ext, L_ext, stencil)
    assert valid[r:-r].all()
    return 1.0 + dL[r:-r], d2L[r:-r]
