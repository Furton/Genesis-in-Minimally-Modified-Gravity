"""Vacuum selection and scalar/tensor mode evolution.

Equations (in ``x``, physical time runs from large ``x`` down to the endpoint)::

    u_hat_xx   + (c_R^2 kappa^2 - z_xx/z) u_hat   = 0        (scalar, canonical)
    u_hat_T,xx + (kappa^2      - a_xx/a) u_hat_T = 0        (tensor, one polarization)

with the exact coefficients from :mod:`vcdm_genesis.background`; no interpolated
coefficients and no Hankel/inverse-wavenumber expansion are used.

Initial data are the normalized finite-time adiabatic (WKB) vacuum imposed at a
selected time ``x_ini``::

    Omega = +sqrt(omega^2) > 0,   u_hat = (2 Omega)^{-1/2},
    u_hat_x = (i Omega - Omega_x / (2 Omega)) u_hat,

which reduces to ``e^{i kappa x}/sqrt(2 kappa)``, ``u_x = i kappa u`` in the
asymptotic past and satisfies ``u u_x^* - u^* u_x = -i``.

Initial-time selection (``select_initial_time``) requires **at the returned
time** (after the safety multiplier and cap) and on a bracket around it:
``omega^2 > 0``, ``|c_R^2 - 1| <= cR2_tol``, ``|c_R^2| kappa^2 >= pump_ratio_min |z_xx/z|``
(absolute pump test), and ``|Omega_x|/Omega^2 <= adiabatic_tol``.  The search
is bounded and fails explicitly (``SelectionError``); the cap ``x_search_max`` is
never returned as a valid start unless it passes the same checks.

An independent cross-check formulation (``evolve_curvature_mode``) integrates
the curvature variable ``R = u_hat / z`` directly from the quadratic action,
``R_xx + (ln z^2)_x R_x + c_R^2 kappa^2 R = 0``, which uses only the first
derivative of ``ln z^2`` and never the pump ``z_xx/z``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.integrate import solve_ivp

from . import background as bg
from .spectra import scalar_power_dimensionless, tensor_power_dimensionless, wronskian

__all__ = [
    "SelectionError", "SolverConfig", "VacuumChecks", "ScalarModeResult", "TensorModeResult",
    "omega_of_x", "omega_derivative", "vacuum_checks", "select_initial_time",
    "adiabatic_initial_data", "evolve_scalar_mode", "evolve_curvature_mode",
    "tensor_omega_derivative", "tensor_vacuum_checks", "select_tensor_initial_time",
    "evolve_tensor_mode", "scalar_power_at_endpoint", "convergence_study",
]

X_FINAL_PAPER = 1e-7   # paper endpoint x_f; the exact Genesis endpoint is x = 0


class SelectionError(RuntimeError):
    """No initial time satisfying all vacuum checks was found in the bounded search."""


@dataclass
class SolverConfig:
    # vacuum-selection checks (all must hold at the returned time and on its bracket)
    pump_ratio_min: float = 1e5      # |c_R^2| kappa^2 >= pump_ratio_min * |z_xx/z|
    tensor_pump_ratio_min: float = 1e7  # kappa^2 >= tensor_pump_ratio_min * |a_xx/a| (slower-decaying tensor pump)
    cR2_tol: float = 1e-3            # |c_R^2 - 1| <= cR2_tol
    adiabatic_tol: float = 1e-3      # |Omega_x| / Omega^2 <= adiabatic_tol
    x_search_min: float = 1e-2
    x_search_max: float = 1e14
    n_search: int = 800              # logarithmic search grid
    multiplier: float = 2.0          # safety factor applied to the first passing time
    bracket: tuple = (0.5, 1.0 / math.sqrt(2.0), 1.0, math.sqrt(2.0), 2.0)
    # integration
    method: str = "DOP853"
    rtol: float = 1e-10
    atol_factor: float = 1e-3        # atol = atol_factor * rtol * (initial component scale)
    max_steps_per_period: float = 0.0  # 0 = let the integrator choose the step

    def to_dict(self):
        d = asdict(self)
        d["bracket"] = list(self.bracket)
        return d


@dataclass
class VacuumChecks:
    x: float
    omega2: float
    cR2: float
    pump: float
    ratio: float
    adiabaticity: float
    omega2_positive: bool
    cR2_near_unity: bool
    pump_small: bool
    adiabatic: bool

    @property
    def passed(self) -> bool:
        return self.omega2_positive and self.cR2_near_unity and self.pump_small and self.adiabatic


# --------------------------------------------------------------------------- #
# frequency and its derivative
# --------------------------------------------------------------------------- #
def omega_of_x(kappa, alpha0, d, x) -> float:
    w2 = bg.omega_squared_scalar(kappa, alpha0, d, x)
    if not w2 > 0.0:
        raise ValueError(f"omega^2 = {w2!r} <= 0 at x={x!r}: not an oscillatory point")
    return math.sqrt(w2)


def omega_derivative(kappa, alpha0, d, x, rel_step: float = 1e-4) -> float:
    """``dOmega/dx`` by a five-point stencil in ``ln x`` (validated against mpmath in the tests).

    Uses ``Omega = +sqrt(omega^2)`` and raises if ``omega^2 <= 0`` at any stencil node;
    negative ``omega^2`` is never replaced by ``|omega^2|`` or zero.
    """
    if not x > 0.0:
        raise ValueError("x must be positive")
    h = rel_step
    # f(s) = Omega(x e^s); dOmega/dx = f'(0)/x
    f = [omega_of_x(kappa, alpha0, d, x * math.exp(k * h)) for k in (-2, -1, 1, 2)]
    dfds = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h)
    return dfds / x


def vacuum_checks(kappa, alpha0, d, x, cfg: SolverConfig) -> VacuumChecks:
    blk = bg.scalar_coefficients_scalar(kappa, alpha0, d, x)
    cR2, pump = blk[13], blk[16]
    w2 = cR2 * kappa * kappa - pump
    ratio = abs(cR2 * kappa * kappa) / abs(pump) if pump != 0.0 else math.inf
    pos = w2 > 0.0
    if pos:
        try:
            adiab = abs(omega_derivative(kappa, alpha0, d, x)) / w2
        except ValueError:
            adiab = math.inf
    else:
        adiab = math.inf
    return VacuumChecks(
        x=float(x), omega2=float(w2), cR2=float(cR2), pump=float(pump), ratio=float(ratio),
        adiabaticity=float(adiab),
        omega2_positive=bool(pos),
        cR2_near_unity=bool(abs(cR2 - 1.0) <= cfg.cR2_tol),
        pump_small=bool(ratio >= cfg.pump_ratio_min),
        adiabatic=bool(adiab <= cfg.adiabatic_tol),
    )


def _bracket_passes(check_fn, x, cfg):
    """All checks must pass at ``x`` and at ``x * f`` for every bracket factor ``f``.

    This guards against a large pump ratio produced only by a zero of the pump.
    """
    results = []
    for f in cfg.bracket:
        xt = x * f
        if xt < cfg.x_search_min or xt > cfg.x_search_max:
            return False, results
        c = check_fn(xt)
        results.append(c)
        if not c.passed:
            return False, results
    return True, results


def select_initial_time(kappa, alpha0, d, cfg: SolverConfig | None = None):
    """Return ``(x_ini, diagnostics)``; every returned time passes all checks after the multiplier.

    Algorithm: scan a logarithmic grid upward from ``x_search_min``; for the first
    grid point passing all checks, apply the multiplier (capped at
    ``x_search_max``) and **retest** the returned time and its bracket.  If that
    retest fails, continue with later grid points.  The pump ratio is not assumed
    monotonic.  Raises :class:`SelectionError` if nothing passes; ``x_search_max``
    is returned only when it itself passes the checks.
    """
    cfg = cfg or SolverConfig()
    bg.check_parameters(alpha0, d, kappa)
    grid = np.geomspace(cfg.x_search_min, cfg.x_search_max, cfg.n_search)
    check = lambda x: vacuum_checks(kappa, alpha0, d, float(x), cfg)
    tested = 0
    first_candidate = None
    for xg in grid:
        c = check(xg)
        tested += 1
        if not c.passed:
            continue
        if first_candidate is None:
            first_candidate = c
        x_ret = min(float(xg) * cfg.multiplier, cfg.x_search_max)
        ok, bracket_results = _bracket_passes(check, x_ret, cfg)
        if ok:
            ret = check(x_ret)
            diag = {
                "x_first_pass": float(xg), "ratio_at_first_pass": c.ratio,
                "x_returned": x_ret, "ratio_at_returned": ret.ratio, "cR2_at_returned": ret.cR2,
                "adiabaticity_at_returned": ret.adiabaticity, "omega2_at_returned": ret.omega2,
                "bracket_min_ratio": min(r.ratio for r in bracket_results),
                "grid_points_tested": tested, "capped": bool(x_ret == cfg.x_search_max),
            }
            return x_ret, diag
    msg = (f"no admissible initial time for kappa={kappa:g}, alpha0={alpha0:g}, d={d:g} in "
           f"[{cfg.x_search_min:g}, {cfg.x_search_max:g}] with {cfg.n_search} points")
    if first_candidate is not None:
        msg += f" (a grid point passed at x={first_candidate.x:g} but its returned time failed the retest)"
    raise SelectionError(msg)


# --------------------------------------------------------------------------- #
# initial data
# --------------------------------------------------------------------------- #
def adiabatic_initial_data(kappa, alpha0, d, x):
    """Normalized finite-time adiabatic vacuum ``(u_hat, u_hat_x)`` at ``x`` (requires ``omega^2 > 0``)."""
    Om = omega_of_x(kappa, alpha0, d, x)          # raises for omega^2 <= 0 (never abs())
    Om_x = omega_derivative(kappa, alpha0, d, x)
    u = 1.0 / math.sqrt(2.0 * Om)
    ux = (1j * Om - 0.5 * Om_x / Om) * u
    return complex(u), complex(ux), Om, Om_x


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class ScalarModeResult:
    kappa: float
    alpha0: float
    d: float
    x_ini: float
    x_final: float
    u_final: complex
    ux_final: complex
    z_final: float
    S_R: float
    wronskian_final: complex
    wronskian_ini: complex
    nfev: int
    n_steps: int
    success: bool
    message: str
    formulation: str
    selection: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def R_final(self) -> complex:
        return self.u_final / self.z_final

    def to_record(self) -> dict:
        r = asdict(self)
        for k in ("u_final", "ux_final", "wronskian_final", "wronskian_ini"):
            r[k] = [r[k].real, r[k].imag]
        return r


@dataclass
class TensorModeResult:
    kappa: float
    d: float
    x_ini: float
    x_final: float
    u_final: complex
    ux_final: complex
    a_final: float
    S_h: float
    wronskian_final: complex
    nfev: int
    n_steps: int
    success: bool
    message: str
    selection: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# scalar evolution (canonical variable)
# --------------------------------------------------------------------------- #
def _integrate(rhs, x_ini, x_final, y0, cfg: SolverConfig, atol):
    kwargs = dict(method=cfg.method, rtol=cfg.rtol, atol=atol, dense_output=False)
    sol = solve_ivp(rhs, (x_ini, x_final), y0, **kwargs)
    return sol


def evolve_scalar_mode(kappa, alpha0, d, x_ini=None, x_final: float = X_FINAL_PAPER,
                       cfg: SolverConfig | None = None) -> ScalarModeResult:
    """Evolve ``u_hat`` from the adiabatic vacuum at ``x_ini`` to ``x_final`` and return ``S_R``."""
    cfg = cfg or SolverConfig()
    bg.check_parameters(alpha0, d, kappa)
    if not (0.0 <= x_final):
        raise ValueError("x_final must be >= 0")
    selection = {}
    if x_ini is None:
        x_ini, selection = select_initial_time(kappa, alpha0, d, cfg)
    if not x_ini > x_final:
        raise ValueError("x_ini must exceed x_final")
    u0, ux0, Om0, _ = adiabatic_initial_data(kappa, alpha0, d, x_ini)
    y0 = np.array([u0.real, u0.imag, ux0.real, ux0.imag])
    A = abs(u0)
    atol = np.array([A, A, A * Om0, A * Om0]) * cfg.rtol * cfg.atol_factor
    k, a0, dd = float(kappa), float(alpha0), float(d)
    w2 = bg.omega_squared_scalar

    def rhs(x, y):
        w = w2(k, a0, dd, x)
        return [y[2], y[3], -w * y[0], -w * y[1]]

    sol = _integrate(rhs, float(x_ini), float(x_final), y0, cfg, atol)
    if not sol.success:
        raise RuntimeError(f"scalar mode integration failed: {sol.message}")
    yf = sol.y[:, -1]
    uf = complex(yf[0], yf[1]); uxf = complex(yf[2], yf[3])
    zf = bg.z_scalar(k, a0, dd, float(x_final))
    S = float(scalar_power_dimensionless(k, uf, zf))
    return ScalarModeResult(
        kappa=k, alpha0=a0, d=dd, x_ini=float(x_ini), x_final=float(x_final),
        u_final=uf, ux_final=uxf, z_final=zf, S_R=S,
        wronskian_final=complex(wronskian(uf, uxf)), wronskian_ini=complex(wronskian(u0, ux0)),
        nfev=int(sol.nfev), n_steps=int(len(sol.t) - 1), success=bool(sol.success), message=str(sol.message),
        formulation="canonical u_hat (DOP853, exact coefficients)", selection=selection, config=cfg.to_dict(),
    )


def evolve_curvature_mode(kappa, alpha0, d, x_ini=None, x_final: float = X_FINAL_PAPER,
                          cfg: SolverConfig | None = None) -> ScalarModeResult:
    """Independent formulation: evolve ``R = u_hat/z`` from the quadratic action.

    ``R_xx + (ln z^2)_x R_x + c_R^2 kappa^2 R = 0`` (no ``z_xx/z`` anywhere).
    Same adiabatic vacuum data, converted with ``R = u/z``, ``R_x = (u_x - u (ln z^2)_x / 2)/z``.
    """
    cfg = cfg or SolverConfig()
    bg.check_parameters(alpha0, d, kappa)
    selection = {}
    if x_ini is None:
        x_ini, selection = select_initial_time(kappa, alpha0, d, cfg)
    u0, ux0, Om0, _ = adiabatic_initial_data(kappa, alpha0, d, x_ini)
    k, a0, dd = float(kappa), float(alpha0), float(d)
    blk0 = bg.scalar_coefficients_scalar(k, a0, dd, float(x_ini))
    z0 = math.sqrt(blk0[12]); L10 = blk0[14]
    R0 = u0 / z0
    Rx0 = (ux0 - u0 * 0.5 * L10) / z0
    y0 = np.array([R0.real, R0.imag, Rx0.real, Rx0.imag])
    A = abs(R0)
    atol = np.array([A, A, A * Om0, A * Om0]) * cfg.rtol * cfg.atol_factor
    block = bg.scalar_coefficients_scalar

    def rhs(x, y):
        b = block(k, a0, dd, x)
        L1 = b[14]; c2k2 = b[13] * k * k
        return [y[2], y[3], -L1 * y[2] - c2k2 * y[0], -L1 * y[3] - c2k2 * y[1]]

    sol = _integrate(rhs, float(x_ini), float(x_final), y0, cfg, atol)
    if not sol.success:
        raise RuntimeError(f"curvature mode integration failed: {sol.message}")
    yf = sol.y[:, -1]
    Rf = complex(yf[0], yf[1]); Rxf = complex(yf[2], yf[3])
    blkf = bg.scalar_coefficients_scalar(k, a0, dd, float(x_final))
    zf = math.sqrt(blkf[12]); L1f = blkf[14]
    uf = Rf * zf
    uxf = (Rxf + Rf * 0.5 * L1f) * zf
    S = float(scalar_power_dimensionless(k, uf, zf))
    return ScalarModeResult(
        kappa=k, alpha0=a0, d=dd, x_ini=float(x_ini), x_final=float(x_final),
        u_final=uf, ux_final=uxf, z_final=zf, S_R=S,
        wronskian_final=complex(wronskian(uf, uxf)), wronskian_ini=complex(wronskian(u0, ux0)),
        nfev=int(sol.nfev), n_steps=int(len(sol.t) - 1), success=bool(sol.success), message=str(sol.message),
        formulation="curvature R from quadratic action (DOP853, first-derivative damping)", selection=selection,
        config=cfg.to_dict(),
    )


def scalar_power_at_endpoint(kappa, alpha0, d, x_final=X_FINAL_PAPER, cfg=None) -> float:
    return evolve_scalar_mode(kappa, alpha0, d, x_final=x_final, cfg=cfg).S_R


# --------------------------------------------------------------------------- #
# tensor evolution
# --------------------------------------------------------------------------- #
def tensor_omega_of_x(kappa, d, x) -> float:
    w2 = bg.tensor_omega_squared_scalar(kappa, d, x)
    if not w2 > 0.0:
        raise ValueError(f"tensor omega^2 = {w2!r} <= 0 at x={x!r}")
    return math.sqrt(w2)


def tensor_omega_derivative(kappa, d, x, rel_step: float = 1e-4) -> float:
    h = rel_step
    f = [tensor_omega_of_x(kappa, d, x * math.exp(k * h)) for k in (-2, -1, 1, 2)]
    return (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h) / x


def tensor_vacuum_checks(kappa, d, x, cfg: SolverConfig) -> VacuumChecks:
    pump = bg.a_xx_over_a(d, x)
    w2 = kappa * kappa - pump
    ratio = kappa * kappa / abs(pump) if pump != 0.0 else math.inf
    pos = w2 > 0.0
    adiab = abs(tensor_omega_derivative(kappa, d, x)) / w2 if pos else math.inf
    return VacuumChecks(x=float(x), omega2=float(w2), cR2=1.0, pump=float(pump), ratio=float(ratio),
                        adiabaticity=float(adiab), omega2_positive=bool(pos), cR2_near_unity=True,
                        pump_small=bool(ratio >= cfg.tensor_pump_ratio_min), adiabatic=bool(adiab <= cfg.adiabatic_tol))


def select_tensor_initial_time(kappa, d, cfg: SolverConfig | None = None):
    cfg = cfg or SolverConfig()
    bg.check_parameters(d=d, kappa=kappa)
    grid = np.geomspace(cfg.x_search_min, cfg.x_search_max, cfg.n_search)
    check = lambda x: tensor_vacuum_checks(kappa, d, float(x), cfg)
    for xg in grid:
        c = check(xg)
        if not c.passed:
            continue
        x_ret = min(float(xg) * cfg.multiplier, cfg.x_search_max)
        ok, br = _bracket_passes(check, x_ret, cfg)
        if ok:
            ret = check(x_ret)
            return x_ret, {"x_first_pass": float(xg), "x_returned": x_ret, "ratio_at_returned": ret.ratio,
                           "adiabaticity_at_returned": ret.adiabaticity, "capped": bool(x_ret == cfg.x_search_max)}
    raise SelectionError(f"no admissible tensor initial time for kappa={kappa:g}, d={d:g}")


def evolve_tensor_mode(kappa, d, x_ini=None, x_final: float = 0.0, cfg: SolverConfig | None = None) -> TensorModeResult:
    """Evolve one tensor polarization ``u_hat_T`` and return ``S_h`` (both polarizations, canonical convention)."""
    cfg = cfg or SolverConfig()
    bg.check_parameters(d=d, kappa=kappa)
    selection = {}
    if x_ini is None:
        x_ini, selection = select_tensor_initial_time(kappa, d, cfg)
    Om0 = tensor_omega_of_x(kappa, d, x_ini)
    Om0_x = tensor_omega_derivative(kappa, d, x_ini)
    u0 = 1.0 / math.sqrt(2.0 * Om0)
    ux0 = (1j * Om0 - 0.5 * Om0_x / Om0) * u0
    y0 = np.array([u0.real, u0.imag, ux0.real, ux0.imag])
    atol = np.array([u0, u0, u0 * Om0, u0 * Om0]) * cfg.rtol * cfg.atol_factor
    k, dd = float(kappa), float(d)
    w2 = bg.tensor_omega_squared_scalar

    def rhs(x, y):
        w = w2(k, dd, x)
        return [y[2], y[3], -w * y[0], -w * y[1]]

    sol = _integrate(rhs, float(x_ini), float(x_final), y0, cfg, atol)
    if not sol.success:
        raise RuntimeError(f"tensor mode integration failed: {sol.message}")
    yf = sol.y[:, -1]
    uf = complex(yf[0], yf[1]); uxf = complex(yf[2], yf[3])
    af = float(bg.scale_factor(dd, float(x_final)))
    S = float(tensor_power_dimensionless(k, uf, af))
    return TensorModeResult(kappa=k, d=dd, x_ini=float(x_ini), x_final=float(x_final), u_final=uf, ux_final=uxf,
                            a_final=af, S_h=S, wronskian_final=complex(wronskian(uf, uxf)),
                            nfev=int(sol.nfev), n_steps=int(len(sol.t) - 1), success=bool(sol.success),
                            message=str(sol.message), selection=selection, config=cfg.to_dict())


# --------------------------------------------------------------------------- #
# convergence study
# --------------------------------------------------------------------------- #
def convergence_study(kappa, alpha0, d, x_final=X_FINAL_PAPER, cfg: SolverConfig | None = None,
                      x_ini_factor: float = 2.0, rtol_factor: float = 0.1, x_final_alt: float | None = 0.0,
                      independent: bool = True) -> dict:
    """Vary initial time, tolerance and endpoint separately; optionally cross-check with the R formulation."""
    cfg = cfg or SolverConfig()
    base = evolve_scalar_mode(kappa, alpha0, d, x_final=x_final, cfg=cfg)
    out = {"base": base, "S_R": base.S_R, "x_ini": base.x_ini}
    earlier = evolve_scalar_mode(kappa, alpha0, d, x_ini=base.x_ini * x_ini_factor, x_final=x_final, cfg=cfg)
    out["rel_change_x_ini"] = abs(earlier.S_R / base.S_R - 1.0)
    tight = SolverConfig(**{**cfg.to_dict(), "rtol": cfg.rtol * rtol_factor, "bracket": cfg.bracket})
    tighter = evolve_scalar_mode(kappa, alpha0, d, x_ini=base.x_ini, x_final=x_final, cfg=tight)
    out["rel_change_rtol"] = abs(tighter.S_R / base.S_R - 1.0)
    if x_final_alt is not None:
        alt = evolve_scalar_mode(kappa, alpha0, d, x_ini=base.x_ini, x_final=x_final_alt, cfg=cfg)
        out["S_R_x_final_alt"] = alt.S_R
        out["ratio_x_final"] = base.S_R / alt.S_R
    if independent:
        indep = evolve_curvature_mode(kappa, alpha0, d, x_ini=base.x_ini, x_final=x_final, cfg=cfg)
        out["S_R_independent"] = indep.S_R
        out["rel_diff_independent"] = abs(indep.S_R / base.S_R - 1.0)
    return out
