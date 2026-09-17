# Numerical methods

This document describes the maintained numerical layer (`vcdm_genesis/`) that
produces the scan table, the derivative columns, the analytic-fit checks and
the figures of *Genesis in Minimally Modified Gravity*. Conventions follow the
manuscript: `b = -tau_B > 0`, `x = tau/tau_B`, `T = 1 + x`, `kappa = b k`.
Physical time runs from large `x` down to the Genesis endpoint `x = 0`; the
paper evaluates the spectra at `x_f = 1e-7`.

## Background and quadratic action (`background.py`)

Exact closed forms are used everywhere; nothing is interpolated:

    a = exp[(1+d)/d T^-d],  Htilde = (1+d) T^(-1-d),  alpha = alpha0 T^(2d),
    epsilon = 1 - T^d,  beta_alpha = -2d/(1+d) T^d,  E_eta = epsilon*eta = d/(1+d) T^(2d)

`eta` itself diverges at `x = 0`; only the regular product `E_eta` enters
`B1, B2`, so all coefficients are finite at the endpoint. `z^2` and `c_R^2`
are the manuscript expressions. The scalar pump `z_xx/z` is evaluated from
closed-form first and second derivatives of `ln z^2` (using
`alpha Htilde^2 = alpha0 (1+d)^2 T^-2`); the tensor pump is
`a_xx/a = (1+d)^2 (T^{-2-2d} + T^{-2-d})`. Both are verified against 40-digit
numerical differentiation in `tests/test_background.py`. Negative
finite-wavenumber `c_R^2` is reported as is (never clamped); the manuscript
counter-example `c_R^2 = -65.936...` at `d=1, alpha0=1e-5, kappa=1e-4, x=1e3`
is a regression test.

## Normalization (`spectra.py`)

Stored powers are dimensionless: `S_R = b^2 M_Pl^2 P_R = kappa^3/(2 pi^2) |u_hat/z|^2`
and `S_h = b^2 M_Pl^2 P_h = 4 kappa^3/pi^2 |u_hat_T/a|^2` (two equal
polarizations of `u = a M_Pl h/2`, unit polarization norm), with
`u_hat = u/sqrt(b)`. The `Ps` column of the scan table is `S_R`.

## Vacuum selection and mode evolution (`modes.py`)

The scalar equation `u_xx + (c_R^2 kappa^2 - z_xx/z) u = 0` (tensor:
`kappa^2 - a_xx/a`) is integrated with `scipy.integrate.solve_ivp`
(DOP853, `rtol = 1e-10`, component-scaled absolute tolerance) from a selected
initial time down to `x_final`.

Initial data are the normalized finite-time adiabatic vacuum
`u = (2 Omega)^{-1/2}`, `u_x = (i Omega - Omega_x/(2 Omega)) u` with
`Omega = +sqrt(omega^2)`; `omega^2 <= 0` raises instead of being replaced by
`|omega^2|`. `Omega_x` is a five-point stencil in `ln x` validated against
mpmath.

The initial time is the first point of a logarithmic search grid that passes
all checks, times a safety multiplier of 2, **retested at the returned time and
on the bracket `x * {1/2, 1/sqrt2, 1, sqrt2, 2}`**:
`omega^2 > 0`, `|c_R^2 - 1| <= 1e-3`, `|c_R^2| kappa^2 >= 1e5 |z_xx/z|`
(tensors: `1e7`), `|Omega_x|/Omega^2 <= 1e-3`. The bracket rejects spurious
passes caused by a zero of the pump: a test at a single time followed by a
fixed multiplier can land on such a zero (the R6 case in
`docs/VALIDATION.md`). If nothing passes, `SelectionError` is raised; the
search cap is never returned as a valid start.

An independent formulation, `R_xx + (ln z^2)_x R_x + c_R^2 kappa^2 R = 0`
for `R = u/z` (no pump term), agrees with the canonical evolution to
`<= 2.4e-10` on the regression fixture; initial-time doubling changes the
powers by `<= 1.8e-8` and tolerance tightening by `<= 7e-10`.

Commands: `python -m vcdm_genesis.mode_cli single|converge|tensor|benchmark`.

## Scan table (`table.py`, `table_derivatives.py`)

`python -m vcdm_genesis.table amplitudes` evolves one converged mode per
parameter key and checkpoints records atomically in chunk files; resuming
with a different configuration is refused (fingerprint over axes, endpoint,
solver settings and code hashes); duplicate keys are rejected; failed modes
are recorded with their error, never replaced by fit values. `assemble`
writes `alpha,d,kappa,Ps,status` in the canonical alpha-major order.

`python -m vcdm_genesis.table derivatives` adds `ns` and `alpha_s`: for each
`(alpha, d)` block three extra converged modes are evolved beyond each end of
the log-uniform `kappa` axis and `ln Ps` is differentiated by centred
seven-point finite differences in `ln kappa` (truncation `O(h^6)`, no
smoothing). `Ps` is copied verbatim, so the same command post-processes an
existing amplitude table. With a per-point log-power uncertainty
`delta ~ 2e-9` and the table step `h = ln(10^5)/79 = 0.1457`, the
conservative noise bounds are `~1e-7` (first derivative) and `~6e-7`
(running); a five-point comparison on the same data is written to the
manifest as a truncation diagnostic.

Exact axes: `data/grid_axes.json` holds the stored decimal coordinates parsed
with round-trip float conversion (`pandas.read_csv` with its default parser
is one ulp off).

## Fit template (`fit.py`)

The amplitude template and its coefficient record `data/fit_coefficients.json`
are described in the manuscript appendix. `fit_derivatives` differentiates the
closed form (never evolved modes). `split_by_d_kappa` reproduces the grouped
split of `Fitting.ipynb` (seed 1234); `minimax_fit` is the linear programme.

## Potential figure (`potential.py`, `figures/plot_u.py`)

`U_hat(y; d) = b^2 U/chi_1^2 = C^-p [Gamma(p, y)/d + Gamma(p+1, y)]`,
`C = 2(1+d)/d`, `p = 2/d`, evaluated in log space with `gammaln` and the
regularized `gammaincc`. Validated against independent quadrature of the
canonical matter equation and the elementary polynomials for `d = 1, 1/2, 1/3`
(maximum relative deviation 3e-15 on the tested points). Supported domain
`0.05 <= d <= 10`, `0 <= y <= 300`; outside it the function raises.

## Reproduction

    python -m pytest                                              # full test suite
    python -m vcdm_genesis.figures.plot_u --outdir figures        # Figure 1
    python -m vcdm_genesis.figures.spectral_index --outdir figures --processes 8   # Figure 5
    python -m vcdm_genesis.mode_cli single --alpha0 1e-23 --d 0.3 --kappa 1e-7 --x-final 0 --outdir out
    python -m vcdm_genesis.table amplitudes --outdir out/small --alpha0 3e-27 7e-24 --d 0.22 0.33 --kappa-logspace 1e-8 1e-5 9
    python -m vcdm_genesis.table assemble --outdir out/small
    python -m vcdm_genesis.table derivatives --amplitudes out/small/amplitudes.csv --workdir out/small/deriv --output out/small/table.csv

The full `80^3` regeneration (`--axes data/grid_axes.json`) is an explicit
opt-in run of about 2.5 hours (9,123 s measured) on 18
processes of the reference machine, plus about 12 minutes for the derivative stage.
