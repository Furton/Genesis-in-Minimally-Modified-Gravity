"""Reconstructed matter potential ``U`` of Eq. (2.16) (Figure 1).

Dimensionless form (``b = -tau_B``)::

    U_hat(y; d) = b^2 U / chi_1^2
                = C^{-p} [ Gamma(p, y)/d + Gamma(p+1, y) ],
    C = 2(1+d)/d,   p = 2/d,   Gamma(p, y) = int_y^inf t^{p-1} e^{-t} dt  (upper, unregularized)

with ``y = C * T^{-d}`` along the Genesis trajectory, so the Genesis segment is
``0 < y <= C`` and ``y -> 0`` is the deep past.  The additive constant is fixed
by ``U(y -> infinity) = 0``.

This is identical to the manuscript's generalized-exponential-integral form
through ``E_nu(y) = y^{nu-1} Gamma(1-nu, y)``.  SciPy's ``gammaincc`` is the
*regularized* upper function ``Q(p, y) = Gamma(p, y)/Gamma(p)``; the
unregularized value is restored in log space with ``gammaln``.  ``scipy.special.expn``
supports only nonnegative integer orders and is therefore not used.

Useful identities::

    U_hat = C^{-p} [ (3/d) Gamma(p, y) + y^p e^{-y} ]
    dU_hat/dy = -C^{-p} e^{-y} y^{p-1} (1/d + y) < 0          (y > 0)
    U_hat(0) = 3 Gamma(2/d) / (d C^{2/d}),   U_hat(y -> inf) -> 0

Elementary cases (tested)::

    d = 1   : e^{-y} (y^2 + 3y + 3) / 16
    d = 1/2 : e^{-y} (y^4 + 6y^3 + 18y^2 + 36y + 36) / 1296
    d = 1/3 : e^{-y} (y^6 + 9y^5 + 45y^4 + 180y^3 + 540y^2 + 1080y + 1080) / 262144

Supported domain (declared, tested): ``D_MIN <= d <= D_MAX`` and
``0 <= y <= Y_MAX``.  Outside it the function raises instead of returning an
underflow-induced artificial plateau or zero.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.special import gammaincc, gammaln

__all__ = [
    "D_MIN", "D_MAX", "Y_MAX", "genesis_endpoint", "plateau", "U_hat", "dU_hat_dy",
    "elementary_U_hat", "d_label",
]

D_MIN = 0.05     # p = 2/d <= 40: Gamma(p) and C^{-p} stay far from double overflow/underflow
D_MAX = 10.0     # p >= 0.2
Y_MAX = 300.0    # regularized Q(p, y) stays representable (e^{-300} ~ 5e-131)


def _validate(y, d):
    if not (np.isfinite(d) and D_MIN <= d <= D_MAX):
        raise ValueError(f"d={d!r} outside the supported range [{D_MIN}, {D_MAX}]")
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    if np.any(y < 0.0):
        raise ValueError("y must be >= 0")
    if np.any(y > Y_MAX):
        raise ValueError(f"y > {Y_MAX} is outside the validated range (would underflow)")
    return y


def genesis_endpoint(d: float) -> float:
    """``C = 2(1+d)/d``: the value of ``y`` at the Genesis endpoint ``x = 0``."""
    return 2.0 * (1.0 + d) / d


def plateau(d: float) -> float:
    """Analytic deep-past plateau ``U_hat(0) = 3 Gamma(2/d) / (d C^{2/d})`` (log-space evaluation)."""
    _validate(0.0, d)
    p = 2.0 / d
    return float(np.exp(np.log(3.0) - np.log(d) + gammaln(p) - p * np.log(genesis_endpoint(d))))


def U_hat(y, d: float):
    """``b^2 U / chi_1^2`` at ``y`` for shape parameter ``d`` (vectorized in ``y``)."""
    y = _validate(y, d)
    p = 2.0 / d
    C = genesis_endpoint(d)
    # U_hat = C^{-p} Gamma(p) [ Q(p,y)/d + p Q(p+1,y) ]      (Gamma(p+1) = p Gamma(p))
    bracket = gammaincc(p, y) / d + p * gammaincc(p + 1.0, y)
    if np.any(bracket <= 0.0):
        raise FloatingPointError("incomplete-gamma bracket underflowed; y outside the representable range")
    logU = -p * np.log(C) + gammaln(p) + np.log(bracket)
    out = np.exp(logU)
    if np.ndim(y) == 0:
        return float(out)
    return out


def dU_hat_dy(y, d: float):
    """``dU_hat/dy = -C^{-p} e^{-y} y^{p-1} (1/d + y)`` (strictly negative for ``y > 0``)."""
    y = _validate(y, d)
    p = 2.0 / d
    C = genesis_endpoint(d)
    # y^(p-1) evaluated as exp((p-1) ln y); at y = 0 this is 0 for p > 1, 1 for p == 1
    # (d = 2) and +inf for p < 1 (d > 2), which is the true limit of the derivative.
    with np.errstate(divide="ignore", invalid="ignore"):
        ypm1 = np.where(y > 0.0, np.exp((p - 1.0) * np.log(np.where(y > 0.0, y, 1.0))),
                        0.0 if p > 1.0 else (1.0 if p == 1.0 else np.inf))
    val = -np.exp(-p * np.log(C) - y) * ypm1 * (1.0 / d + y)
    if np.ndim(y) == 0:
        return float(val)
    return val


def elementary_U_hat(y, d: float):
    """Closed elementary polynomials for ``d`` in {1, 1/2, 1/3} (reference only)."""
    y = np.asarray(y, dtype=float)
    if d == 1.0:
        return np.exp(-y) / 16.0 * (y ** 2 + 3.0 * y + 3.0)
    if d == 0.5:
        return np.exp(-y) / 1296.0 * (y ** 4 + 6.0 * y ** 3 + 18.0 * y ** 2 + 36.0 * y + 36.0)
    if abs(d - 1.0 / 3.0) < 1e-15:
        return np.exp(-y) / 262144.0 * (y ** 6 + 9.0 * y ** 5 + 45.0 * y ** 4 + 180.0 * y ** 3
                                          + 540.0 * y ** 2 + 1080.0 * y + 1080.0)
    raise ValueError("elementary form available only for d in {1, 1/2, 1/3}")


def d_label(d: float) -> str:
    """Legend text derived from the value itself, e.g. ``d = 1/3``."""
    f = Fraction(d).limit_denominator(1000)
    if abs(float(f) - d) < 1e-12:
        return f"d = {f.numerator}" if f.denominator == 1 else f"d = {f.numerator}/{f.denominator}"
    return f"d = {d:g}"
