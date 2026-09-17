"""Power-spectrum normalization conventions (scalar and tensor).

Dimensionless powers stored in the scan table and used by the fit::

    S_R = b^2 M_Pl^2 P_R = kappa^3/(2 pi^2) |u_hat / z|^2
    S_h = b^2 M_Pl^2 P_h = 4 kappa^3/pi^2 |u_hat_T / a|^2

with ``u_hat = u / sqrt(b)`` (vacuum amplitude ``(2 kappa)^{-1/2}``) and
``kappa = b k``.  The tensor expression sums the two equal polarizations of
``u^lambda = a M_Pl h^lambda / 2`` with unit polarization norm.

The scan-table column ``Ps`` is ``S_R`` (``tau_B = -1``, ``M_Pl = 1`` convention).
Physical amplitudes are restored by the explicit factor ``(b^2 M_Pl^2)^{-1}``.

The functions in this module are pure algebra; mode evolution lives in
``vcdm_genesis.modes``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "u_hat_from_u", "u_from_u_hat", "tensor_mode_from_h", "h_from_tensor_mode",
    "scalar_power_dimensionless", "tensor_power_dimensionless",
    "tensor_power_from_h_physical", "to_physical", "from_physical",
    "vacuum_plane_wave", "wronskian",
]

PI2 = np.pi ** 2


# --------------------------------------------------------------------------- #
# variable rescalings
# --------------------------------------------------------------------------- #
def u_hat_from_u(u, b):
    """``u_hat = u / sqrt(b)``; ``u`` carries the dimensionful canonical normalization."""
    if not b > 0:
        raise ValueError("b = -tau_B must be positive")
    return np.asarray(u) / np.sqrt(b)


def u_from_u_hat(u_hat, b):
    if not b > 0:
        raise ValueError("b = -tau_B must be positive")
    return np.asarray(u_hat) * np.sqrt(b)


def tensor_mode_from_h(h_lambda, a, M_Pl=1.0):
    """Canonical tensor mode ``u^lambda = a M_Pl h^lambda / 2`` (one polarization)."""
    return a * M_Pl * np.asarray(h_lambda) / 2.0


def h_from_tensor_mode(u_lambda, a, M_Pl=1.0):
    """Inverse of :func:`tensor_mode_from_h`: ``h = 2 u / (a M_Pl)``."""
    return 2.0 * np.asarray(u_lambda) / (a * M_Pl)


# --------------------------------------------------------------------------- #
# dimensionless powers
# --------------------------------------------------------------------------- #
def scalar_power_dimensionless(kappa, u_hat, z):
    """``S_R = kappa^3/(2 pi^2) |u_hat/z|^2``.

    ``u_hat`` must be the rescaled mode ``u/sqrt(b)``.  Passing the unscaled
    physical ``u`` overestimates ``S_R`` by the factor ``b``.
    """
    return kappa ** 3 / (2.0 * PI2) * np.abs(np.asarray(u_hat) / z) ** 2


def tensor_power_dimensionless(kappa, u_hat_T, a):
    """``S_h = 4 kappa^3/pi^2 |u_hat_T/a|^2`` (both polarizations, canonical ``u = a M_Pl h/2``).

    ``a`` is the scale factor itself.  Do not pass ``a/2`` here: the factor
    ``4`` already converts ``|u|^2`` into ``a^2 M_Pl^2 |h|^2/4`` summed over
    two polarizations.
    """
    return 4.0 * kappa ** 3 / PI2 * np.abs(np.asarray(u_hat_T) / a) ** 2


def tensor_power_from_h_physical(k, h_lambda):
    """Independent physical route: ``P_h = k^3/pi^2 |h_lambda|^2`` for two equal polarizations.

    (``k^3/(2 pi^2) sum_lambda |h_lambda|^2`` with ``|h_+| = |h_x|``.)
    """
    return k ** 3 / PI2 * np.abs(np.asarray(h_lambda)) ** 2


# --------------------------------------------------------------------------- #
# dimensionless <-> physical
# --------------------------------------------------------------------------- #
def to_physical(S, b, M_Pl=1.0):
    """``P = S / (b^2 M_Pl^2)``."""
    if not (b > 0 and M_Pl > 0):
        raise ValueError("b and M_Pl must be positive")
    return np.asarray(S) / (b * b * M_Pl * M_Pl)


def from_physical(P, b, M_Pl=1.0):
    """``S = b^2 M_Pl^2 P``."""
    if not (b > 0 and M_Pl > 0):
        raise ValueError("b and M_Pl must be positive")
    return np.asarray(P) * (b * b * M_Pl * M_Pl)


# --------------------------------------------------------------------------- #
# vacuum data and normalization diagnostics
# --------------------------------------------------------------------------- #
def vacuum_plane_wave(kappa, x):
    """Asymptotic vacuum ``u_hat = e^{i kappa x}/sqrt(2 kappa)`` and ``u_hat_x = i kappa u_hat``."""
    u = np.exp(1j * kappa * np.asarray(x, dtype=float)) / np.sqrt(2.0 * kappa)
    return u, 1j * kappa * u


def wronskian(u, du):
    """``u du* - u* du``; equals ``-i`` for the normalized vacuum in the ``x`` convention."""
    u = np.asarray(u)
    du = np.asarray(du)
    return u * np.conj(du) - np.conj(u) * du
