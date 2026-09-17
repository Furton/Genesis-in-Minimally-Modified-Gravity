"""Exact Genesis background and quadratic-action coefficients.

Conventions (paper Secs. 2-3)::

    b = -tau_B > 0,   x = tau / tau_B,   T = 1 + x,   kappa = b k > 0,
    d > 0,            alpha0 > 0.

Physical time runs from large positive ``x`` down to ``x = 0``, the exact
Genesis endpoint.  All derivatives are with respect to ``x`` (equivalently
``T``, because ``dT/dx = 1``).  The conformal-time derivative is
``d/dtau = -(1/b) d/dx``.

Background::

    a        = exp[(1+d)/d * T^-d]
    Htilde   = (1+d) T^(-1-d)                 (dimensionless conformal Hubble)
    alpha    = alpha0 T^(2d)
    epsilon  = 1 - T^d
    beta_a   = -2d/(1+d) T^d
    E_eta    = epsilon*eta = d/(1+d) T^(2d)   (regular product, finite at x=0)

The coefficients ``A1, A2, B1, B2, z^2, c_R^2`` follow the manuscript
(Eqs. Definition_zsq / definition_cR) written with the regular product
``E_eta`` instead of the divergent ``eta``.

The pump ``z_xx/z = (ln z^2)_xx / 2 + [(ln z^2)_x]^2 / 4`` is evaluated from
closed-form first and second derivatives of ``ln z^2``; these are verified
against high-precision numerical differentiation in ``tests/test_background.py``.

Every public function accepts floats or NumPy arrays for ``x``.  The
``*_scalar`` helpers use the ``math`` module and are the fast path for ODE
right-hand sides.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = [
    "T_of_x", "scale_factor", "log_scale_factor", "hubble_tilde", "alpha",
    "epsilon", "beta_alpha", "E_eta",
    "A1", "A2", "B1", "B2", "z_squared", "cR_squared",
    "log_z2_derivatives", "z_xx_over_z", "a_xx_over_a",
    "omega_squared", "tensor_omega_squared",
    "scalar_coefficients_scalar", "omega_squared_scalar", "tensor_omega_squared_scalar",
    "z_scalar", "check_parameters",
]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def check_parameters(alpha0: float | None = None, d: float | None = None,
                     kappa: float | None = None, x=None) -> None:
    """Raise ``ValueError`` for nonfinite or out-of-domain model parameters."""
    if d is not None:
        if not (np.isfinite(d) and d > 0.0):
            raise ValueError(f"d must be finite and > 0, got {d!r}")
    if alpha0 is not None:
        if not (np.isfinite(alpha0) and alpha0 > 0.0):
            raise ValueError(f"alpha0 must be finite and > 0, got {alpha0!r}")
    if kappa is not None:
        if not (np.isfinite(kappa) and kappa > 0.0):
            raise ValueError(f"kappa must be finite and > 0, got {kappa!r}")
    if x is not None:
        xa = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(xa)) or np.any(xa < 0.0):
            raise ValueError("x must be finite and >= 0 (Genesis segment)")


# --------------------------------------------------------------------------- #
# background
# --------------------------------------------------------------------------- #
def T_of_x(x):
    """``T = 1 + x``."""
    return 1.0 + np.asarray(x, dtype=float)


def log_scale_factor(d, x):
    """``ln a = (1+d)/d * T^-d``."""
    T = T_of_x(x)
    return (1.0 + d) / d * T ** (-d)


def scale_factor(d, x):
    """``a = exp[(1+d)/d * T^-d]``; ``a(0) = e^{(1+d)/d}``."""
    return np.exp(log_scale_factor(d, x))


def hubble_tilde(d, x):
    """Dimensionless conformal Hubble parameter ``Htilde = b*H_conf = (1+d) T^(-1-d)``."""
    T = T_of_x(x)
    return (1.0 + d) * T ** (-1.0 - d)


def alpha(alpha0, d, x):
    """``alpha = alpha0 T^(2d)``."""
    T = T_of_x(x)
    return alpha0 * T ** (2.0 * d)


def epsilon(d, x):
    """``epsilon = 1 - T^d`` (``<= 0`` during Genesis, ``0`` at the endpoint)."""
    T = T_of_x(x)
    return 1.0 - T ** d


def beta_alpha(d, x):
    """``beta_alpha = alpha_dot/(alpha H) = -2d/(1+d) T^d``."""
    T = T_of_x(x)
    return -(2.0 * d / (1.0 + d)) * T ** d


def E_eta(d, x):
    """Regular product ``epsilon*eta = epsilon'/H_conf = d/(1+d) T^(2d)``.

    ``eta`` itself diverges at ``epsilon(0)=0``; the product is finite and is
    the only combination entering the quadratic action.
    """
    T = T_of_x(x)
    return d / (1.0 + d) * T ** (2.0 * d)


# --------------------------------------------------------------------------- #
# one shared formula block (floats via ``math`` or arrays via ``numpy``)
# --------------------------------------------------------------------------- #
def _coefficient_block(kappa, alpha0, d, T, exp: Callable):
    """Return the tuple of all coefficients at ``T`` using one formula set.

    ``exp`` is ``math.exp`` (scalar fast path) or ``numpy.exp`` (arrays).
    Returned order::

        (a, Ht, al, eps, beta, Ee, A1, A2, B1, B2, N, D, z2, cR2, L1, L2, zxx_z, axx_a)
    """
    onepd = 1.0 + d
    Tmd = T ** (-d)
    Td = T ** d
    T2d = Td * Td
    lna = onepd / d * Tmd
    a = exp(lna)
    Ht = onepd * T ** (-1.0 - d)
    Ht2 = Ht * Ht
    al = alpha0 * T2d
    eps = 1.0 - Td
    beta = -(2.0 * d / onepd) * Td
    Ee = d / onepd * T2d

    # quadratic-action coefficients (manuscript definitions with E_eta)
    A1 = 2.0 * al * (12.0 + al - 2.0 * eps)
    A2 = 3.0 * al * al * (6.0 + al - 2.0 * eps)
    B1 = 2.0 * (al * (-4.0 * beta + 2.0 * eps + al)
                + 4.0 * eps * (6.0 + beta - 2.0 * eps)
                + 4.0 * Ee)
    B2 = al * (-(al - 2.0 * eps) ** 2 + 36.0 * eps
               + 6.0 * al * (-1.0 - beta + eps)
               + 12.0 * eps * (1.0 - eps)
               + 12.0 * Ee)

    k2 = kappa * kappa
    k4 = k2 * k2
    # z^2 = 2 a^2 alpha N / D,  N = 2 kappa^2 + 3 alpha Ht^2,  D = 4 kappa^2 + alpha Ht^2 (6+alpha-2eps)
    N = 2.0 * k2 + 3.0 * al * Ht2
    D = 4.0 * k2 + al * Ht2 * (6.0 + al - 2.0 * eps)
    z2 = 2.0 * a * a * al * N / D
    cR2 = (8.0 * k4 + B1 * Ht2 * k2 + B2 * Ht2 * Ht2) / (8.0 * k4 + A1 * Ht2 * k2 + A2 * Ht2 * Ht2)

    # closed-form derivatives of ln z^2 with respect to x (= T)
    # alpha*Ht^2 = alpha0 (1+d)^2 T^-2 ; S = 6 + alpha - 2 eps = 4 + 2 T^d + alpha0 T^{2d}
    c = alpha0 * onepd * onepd
    Tm2 = 1.0 / (T * T)
    Tm3 = Tm2 / T
    Tm4 = Tm2 * Tm2
    N_p = -6.0 * c * Tm3
    N_pp = 18.0 * c * Tm4
    D_p = c * (-8.0 * Tm3 + 2.0 * (d - 2.0) * Td * Tm3 + alpha0 * (2.0 * d - 2.0) * T2d * Tm3)
    D_pp = c * (24.0 * Tm4 + 2.0 * (d - 2.0) * (d - 3.0) * Td * Tm4
                + alpha0 * (2.0 * d - 2.0) * (2.0 * d - 3.0) * T2d * Tm4)
    lnN_p = N_p / N
    lnD_p = D_p / D
    lnN_pp = N_pp / N - lnN_p * lnN_p
    lnD_pp = D_pp / D - lnD_p * lnD_p
    lna_p = -onepd * Tmd / T              # (ln a)' = -(1+d) T^{-d-1}
    lna_pp = onepd * onepd * Tmd * Tm2    # (ln a)'' = (1+d)^2 T^{-d-2}
    L1 = 2.0 * lna_p + 2.0 * d / T + lnN_p - lnD_p
    L2 = 2.0 * lna_pp - 2.0 * d * Tm2 + lnN_pp - lnD_pp
    zxx_z = 0.5 * L2 + 0.25 * L1 * L1
    axx_a = lna_pp + lna_p * lna_p        # = (1+d)^2 (T^{-2-2d} + T^{-2-d})
    return (a, Ht, al, eps, beta, Ee, A1, A2, B1, B2, N, D, z2, cR2, L1, L2, zxx_z, axx_a)


def _block_np(kappa, alpha0, d, x):
    check_parameters(alpha0, d, kappa, x)
    return _coefficient_block(float(kappa), float(alpha0), float(d), T_of_x(x), np.exp)


# --------------------------------------------------------------------------- #
# public array API
# --------------------------------------------------------------------------- #
def A1(alpha0, d, x):
    return _block_np(1.0, alpha0, d, x)[6]


def A2(alpha0, d, x):
    return _block_np(1.0, alpha0, d, x)[7]


def B1(alpha0, d, x):
    return _block_np(1.0, alpha0, d, x)[8]


def B2(alpha0, d, x):
    return _block_np(1.0, alpha0, d, x)[9]


def z_squared(kappa, alpha0, d, x):
    """``z^2 = 2 a^2 alpha (2 kappa^2 + 3 alpha Ht^2) / (4 kappa^2 + alpha Ht^2 (6 + alpha - 2 eps))``."""
    return _block_np(kappa, alpha0, d, x)[12]


def cR_squared(kappa, alpha0, d, x):
    """Exact ``c_R^2``; may be negative at finite ``kappa`` (never clamped)."""
    return _block_np(kappa, alpha0, d, x)[13]


def log_z2_derivatives(kappa, alpha0, d, x):
    """Return ``((ln z^2)_x, (ln z^2)_xx)``."""
    blk = _block_np(kappa, alpha0, d, x)
    return blk[14], blk[15]


def z_xx_over_z(kappa, alpha0, d, x):
    """Scalar pump ``z_xx/z`` (closed form)."""
    return _block_np(kappa, alpha0, d, x)[16]


def a_xx_over_a(d, x):
    """Tensor pump ``a_xx/a = (1+d)^2 (T^{-2-2d} + T^{-2-d})``."""
    check_parameters(d=d, x=x)
    T = T_of_x(x)
    return (1.0 + d) ** 2 * (T ** (-2.0 - 2.0 * d) + T ** (-2.0 - d))


def omega_squared(kappa, alpha0, d, x):
    """Scalar mode frequency squared ``c_R^2 kappa^2 - z_xx/z`` (sign preserved)."""
    blk = _block_np(kappa, alpha0, d, x)
    return blk[13] * kappa * kappa - blk[16]


def tensor_omega_squared(kappa, d, x):
    """Tensor mode frequency squared ``kappa^2 - a_xx/a``."""
    check_parameters(d=d, kappa=kappa, x=x)
    return kappa * kappa - a_xx_over_a(d, x)


# --------------------------------------------------------------------------- #
# scalar fast path (ODE right-hand sides)
# --------------------------------------------------------------------------- #
def scalar_coefficients_scalar(kappa: float, alpha0: float, d: float, x: float):
    """Full coefficient tuple at one ``x`` using ``math`` (see ``_coefficient_block``)."""
    return _coefficient_block(kappa, alpha0, d, 1.0 + x, math.exp)


def omega_squared_scalar(kappa: float, alpha0: float, d: float, x: float) -> float:
    blk = _coefficient_block(kappa, alpha0, d, 1.0 + x, math.exp)
    return blk[13] * kappa * kappa - blk[16]


def tensor_omega_squared_scalar(kappa: float, d: float, x: float) -> float:
    T = 1.0 + x
    onepd = 1.0 + d
    return kappa * kappa - onepd * onepd * (T ** (-2.0 - 2.0 * d) + T ** (-2.0 - d))


def z_scalar(kappa: float, alpha0: float, d: float, x: float) -> float:
    """``z = sqrt(z^2)`` at one ``x`` (``z^2 > 0`` throughout Genesis)."""
    z2 = _coefficient_block(kappa, alpha0, d, 1.0 + x, math.exp)[12]
    if not z2 > 0.0:
        raise FloatingPointError(f"z^2 = {z2!r} is not positive at x={x!r}")
    return math.sqrt(z2)
