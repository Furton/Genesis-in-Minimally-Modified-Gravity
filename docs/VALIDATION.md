# Validation record (version 1.0.0)

Every number below was produced by the code in this repository on the
reference machine (Intel Core i7-14700, Windows 11, Python 3.14.6,
NumPy 2.5.1, SciPy 1.18.0, matplotlib 3.11.0). Tolerances are engineering
acceptance targets, not uncertainty estimates claimed by the paper. Four
distinct error notions are kept apart: table-relative fitting error,
mode-integration error, derivative error and post-Genesis physical
uncertainty (the last is outside the scope of this repository).

## 1. Exact coefficients and normalization (`tests/test_background.py`, `tests/test_spectra.py`)

| Check | Result |
|---|---|
| Background, `A1, A2, B1, B2, z^2, c_R^2` vs manuscript definitions in 40-digit mpmath at 72 points (8 parameter sets incl. corners and the R6 tuple, `x` from 0 to 1e12) | relative deviation `< 1e-12` (`c_R^2 < 1e-11`) |
| Endpoint regularity through `E_eta = epsilon*eta` at `x = 0` | all coefficients finite; explicit endpoint limits reproduced |
| Closed-form pump `z_xx/z` vs numerical differentiation of `ln z^2` | `<= 1e-9` relative (guarded at pump zeros) |
| Tensor pump `a_xx/a` | `< 1e-10` relative |
| `c_R^2` counter-example `d=1, alpha0=1e-5, kappa=1e-4, x=1e3` | `-65.9361815185`, `kappa/Htilde = 50.10005` |
| Fixed-time UV limit, small-`alpha0` frequency expansion | pass |
| Powers of `b` and `M_Pl`, vacuum Wronskian `-i`, derivative sign, tensor two-route agreement | pass |

## 2. Potential and Figure 1 (`tests/test_potential.py`)

Independent quadrature of `dU_hat/dy = -C^-p e^-y y^(p-1) (1/d + y)` with
`U(y -> inf) = 0` (mpmath) and the elementary polynomials for `d = 1, 1/2, 1/3`
were compared with the production function at `y = 0.01, 0.1, 1, C, C/2, 2C, 10, 20, 32`
for `d = 1, 1/2, 1/3, 0.3, 0.7`: **maximum relative deviation 3.0e-15**
(target 1e-11). Plateaus `3/16`, `1/36`, `135/32768` and the reference values
at `y = 0.01` are reproduced to `1e-12`. The generator writes 1601 samples per
curve including `y = C` exactly; a 6400-sample regeneration agrees at common
coordinates. Visual inspection of `figures/Plot_U.pdf`: three curves,
correct plateaus, endpoint markers at `y = 4, 6, 8`, dashed beyond-Genesis
extension, tails resolved to `1e-12`, axis labels correct; the figure carries
no legend (the paper's caption describes the curves; `--legend` adds one).
Curve values computed with SciPy 1.18.0 and 1.17.1 differ by at most
`3.6e-15` relative (75 of 4803 samples).

## 3. Mode solver (`tests/test_modes.py`, `tests/test_regression_fixture.py`)

| Case | Reference | This release | Difference |
|---|---|---|---|
| R1 `d=0.3, alpha0=1e-23, kappa=1e-7, x_f=0` | `S_R = 6.91139484127e19` | `6.911394834946e19` | `-9.2e-10` rel |
| R2 endpoint ratio `S_R(1e-7)/S_R(0)` | `0.9999999773106846` | `0.9999999773106829` | `1.7e-15` |
| R2 endpoint `d ln|R|/dx` | `-0.1134465522` | `-0.1134465522` | exact to printed digits |
| R5 tensor `d=0.5, kappa=1e-7` | `S_h = 2.0196314518e-15` | `2.019631451507e-15` | `-1.5e-10` rel |
| R7 `alpha0=1e-30, d=0.13417721518987344, kappa=1.2263306841775643e-7, x_f=1e-7` | `1.48036818276e24` | `1.480368179788e24` | `-2.0e-9` rel |

The reference values are prior independent computations. Convergence on the
regression fixture (regression cases, grid corners, interior point):
initial-time doubling `<= 1.8e-8`, tolerance tightening `<= 7e-10`,
independent curvature-variable formulation `<= 2.4e-10` (target `1e-6`
each). Intentionally invalid starts (`omega^2 < 0`, `x_ini < x_final`,
search cap) raise instead of recovering silently.

Pump-zero rejection (R6, `alpha0=1e-30, d=0.15316455696202533, kappa=2.2616759492228647e-6`):
`z_xx/z` changes sign near `x = 1.368e6`, so the pump ratio passes the `1e5`
threshold at `x = 1.365114e6` (ratio `137962.99`) although it fails at twice
that time (ratio `2446.93`). A selector that tests a single time and returns
a fixed multiple of it would start there. The bracket retest rejects the
spike and returns `x = 6.807e7` with ratio `3.78e5` at the returned time and
at twice it.

Benchmark (45-mode fixed sample, `x_f = 1e-7`, `rtol = 1e-10`): serial mean
92 ms, median 103 ms, p90 114 ms, max 121 ms per mode; 18 processes,
steady-state 13 ms wall per mode (efficiency 0.39). Projection for the
512,000-mode scan: 1.9 h (measured rate) to 2.5 h (max cost), 13 h serial.

## 4. Spectral derivatives (`tests/test_derivatives.py`, `tests/test_spectral_derivatives.py`)

Finite-difference weights, constant/polynomial exactness and `h^6` scaling are
tested analytically. From independently evolved modes at `alpha0 = 1e-23`:

| Case | Reference | `h = 0.1457` (table step) | `h = 0.05` |
|---|---|---|---|
| R3 `d=0.35, kappa=1e-4`: `n_s` | `0.90964681576` | `0.90964681551` | `0.90964681352` |
| R3 `alpha_s` | `-0.03030813031` | `-0.03030816333` | `-0.03030832022` |
| R4 `d=0.10, kappa=1e-4`: `n_s` | `0.48958130079` | `0.48958130003` | `0.48958129975` |

All within the `2e-6` target; five- and seven-point stencils agree; the
larger scatter at `h = 0.05` is the expected `delta/h^2` noise amplification.
Noise budget at the table step with `delta = 2e-9`: `1e-7` (index),
`6e-7` (running).

Figure 5 (`figures/spectral_index_manifest.json`): 41 log-uniform `kappa`
points per `d` plus 3 pads per side. Measured maxima of the fit-minus-numerical
differences over the displayed range, all at `kappa = 1e-4`:
`|Delta n_s| = 1.38e-3, 8.00e-4, 3.93e-4` and
`|Delta alpha_s| = 1.23e-3, 1.02e-3, 7.29e-4` for `d = 0.30, 0.35, 0.40`
(about 3 percent of `alpha_s` there). The fit therefore tracks the converged
derivatives closely and no separate derivative fit is needed.

## 5. Scan table (`data/genesis_scan_table_provenance.json`)

Generation (`python -m vcdm_genesis.table amplitudes --axes data/grid_axes.json --x-final 1e-7`,
18 processes, 9,123 s): 512,000 records, 0 failed, 0 missing, all finite and
positive; every returned start passed all vacuum checks after the safety
factor (pump ratio minimum `2.63e5`, median `4.15e5`; none at the search cap;
`kappa * x_ini` between 95 and 324). Structural checks: `80^3` unique keys,
axes equal to `numpy.logspace/linspace` exactly, alpha-major order,
coordinate strings identical to `repr` of the axes in `data/grid_axes.json`.

Derivative columns (`python -m vcdm_genesis.table derivatives`, 739 s):
38,400 pad modes, step `h = 0.14573`; the largest five- versus seven-point
differences are `4.57e-7` (`ns`) and `3.72e-7` (`alpha_s`); ranges `ns` in
`[0.48958, 1.00130]`, `alpha_s` in `[-0.048757, -0.000237]`.

Reserved final-check sample (`tests/test_final_check_sample.py`: 24 grid
points, seed 20260918, drawn before the scan finished): the independent
curvature-variable formulation agrees with the released table to `4.3e-10`,
and the canonical solver reproduces the released values exactly. Released
file SHA-256 `9da89f3f3e23cf4d178c2a154bb3f9e16dbd08f5b7effa48b6f7293442f9feda`
(LF-normalized bytes).

## 6. Fit (`tests/test_fit.py`, `data/fit_coefficients.json`)

The template reproduces the Wolfram evaluations at four stored rows to
`1e-11` and the paper's Table 1 at the exact grid points to `2e-7`; the Python
template agrees with the Wolfram evaluation of the same formula to `3e-14` at
seven points, and `Analytic fit.nb` evaluates headlessly with `wolframscript`
and reproduces `data/fit_sample_table.csv`. The grouped split (seed 1234)
gives 409,600 / 102,400 rows and is frozen in `data/fit_split_manifest.json`.
Per-row template residuals are checked against an independent 30-digit mpmath
transcription of the template (53 rows, agreement `< 1e-12`).

Table-relative errors of the released coefficients:

| set | rows | mean | median | 99th percentile | maximum | location of maximum |
|---|---:|---:|---:|---:|---:|---|
| training | 409,600 | 0.108564% | 0.091998% | 0.29955% | 0.328417% | `alpha0=7.69e-22, d=0.172152, kappa=1e-4` |
| verification | 102,400 | 0.109237% | 0.093461% | 0.30468% | 0.327623% | `alpha0=7.69e-22, d=0.168354, kappa=1e-4` |
| full grid | 512,000 | 0.108699% | 0.092237% | 0.30040% | 0.328417% | as training |

The coefficients are fixed and are the values quoted in the paper. They are
not the minimax optimum of the released training set: the linear programme on
the same training groups gives `c0 = 0.0139557668608, c1 = -0.275286738213,
q = 1.32186006083` with training maximum `0.278080%` (training mean
`0.114480%`; full grid: maximum `0.278160%`, mean `0.114412%`). That
alternative is recorded in the coefficient record for reference and is not
used. The heat maps `full_loss.png` and `verification_loss.png` are produced
from the released table by `Fitting.ipynb` (maximum over `alpha0` per
`(d, kappa)` pair, in percent).

## 7. Release checks

* Full test suite (`python -m pytest`): 132 tests passed on the reference
  machine and, with Python 3.12.10, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3,
  matplotlib 3.11.0, on a second Windows 11 environment.
* Isolated copy of the tree (no repository metadata, bytecode writing
  disabled): the full suite passed, the single-mode command reproduced R1,
  Figure 5 curves rebuilt identical to the released data, and the documented
  small `2x2x9` grid ran through the interrupted-and-resumed amplitude stage,
  assembly and the derivative stage with `Ps` preserved verbatim.
* `figures/Plot_U.pdf`, its curve data and manifest were generated with
  Python 3.12.10, NumPy 2.4.6, SciPy 1.17.1, matplotlib 3.11.0; the written
  curve data agree with the elementary polynomials to `1e-11`
  (`tests/test_potential.py`).
* `Fitting.ipynb` executed from a clean kernel without errors; the committed
  `verification_loss.png` and `full_loss.png` are byte-identical to a
  regeneration from the released table with the notebook's code.
* The tree contains no personal absolute paths. Hashes of text files in the
  manifests and provenance records are of LF-normalized bytes and equal the
  committed Git blobs (a CRLF checkout on Windows yields different raw-file
  hashes).
