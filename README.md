# SSCI

Implementation of **Subset-Simultaneous Conformal Inversion for Budgeted Cross-Sensitive Gas Concentration Measurement**. Complete-vector calibration protects subsequent channel subsets. A conservative continuous-state inversion retains concentration and nuisance alternatives, while acquisition targets channels that separate concentration explanations.

## Install

Use Python 3.10 or later:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install -e .
```

## Prepare complete response records

```bash
ssci generate --output data/records.npz --records 2000 --channels 14 --seed 13
ssci prepare --input data/records.npz --output data/gas
```

The included source is a defined quadratic gas-array simulator with concentration in 0–20 ppm, humidity in 20–80 percent, a pre-acquisition protocol context, cross-sensitive channels, and correlated additive noise. It supplies an executable data path. The complete generated response vector refers to one compatible state.

For GAS487, dynamic gas mixtures, or other records, export stabilized, phase-compatible exposure features to numeric NPZ arrays:

| Key | Shape | Definition |
| --- | --- | --- |
| `y` | N × (1+P) | Concentration first, then nuisance references such as humidity |
| `z` | N × D | Complete compatible transformed response vector |
| `x` | N × C | Optional context available before channel selection |
| `group` | N | Optional integer independent exposure/record group |

Without context use N × 0; omitted `x` is filled that way. The response transformation, heater phase, aggregation window, reference units, and transition purging must be fixed when constructing records. Rows from a common exposure remain together. `prepare` partitions ordered groups into 50% fitting, 15% development, 15% calibration, and 20% test. Chronological drift stress retains chronological order rather than shuffling individual samples.

## Fit and calibrate

```bash
ssci train --data data/gas --output runs/model --domain 0 20 0 100 --baseline-channels 6
ssci calibrate --model runs/model/model.joblib --input data/gas/calibration.npz --output runs/model/calibrated.joblib --alpha 0.05
```

`--domain` supplies one lower/upper pair for concentration and every nuisance coordinate. These bounds describe the admissible physical/reference domain and must be fixed before testing. Domain limits are not inferred from test labels.

The forward model fits standardized quadratic features by ridge regression. Development residual means supply frozen channel-bias corrections, and centered development residual standard deviations supply positive channel scales. The response-scale floor is 1e-6. The model also fits fixed-first-channel RF and SVR inverse baselines using exactly `--baseline-channels` responses and the shared context.

Calibration scores the maximum standardized response residual over the complete array. Repeated records in a group are combined by a maximum. The quantile rank is `ceil((n+1)*(1-alpha))`; a rank beyond the available independent groups gives infinity. RF/SVR intervals use separately calibrated group maxima of absolute concentration residuals. Load only trusted joblib models.

## Acquire channels and execute inversion

```bash
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/separation --budget 6 --epsilon 0.5 --subdivisions 32 16
ssci plot --input runs --output runs/figures
```

The state domain is covered by continuous cells. Quadratic interval arithmetic encloses each channel's response over every cell, including squared terms crossing zero. A cell is deleted only when a revealed response is farther from its complete enclosed range than the calibrated tolerance. Unresolved cells remain in the returned set. Concentration projections are merged into connected components; their hull supplies the interval and midpoint report.

The acquisition policy accesses a response only through `ResponseService.query`. It chooses using surviving cells and frozen response ranges. It never receives target or nuisance references. Utilities count sampled pairs of concentration-separated cells whose response ranges the channel can distinguish. The policy samples 2048 pairs with the declared seed; the conservative inversion uses all cells. If no utility is positive, acquisition selects the cheapest remaining affordable channel.

The default unit-cost policy accepts a point when concentration hull width is at most `2*epsilon`. Otherwise it continues until budget exhaustion and returns an unresolved set. An empty inversion is marked incompatible with no point. Every revealed channel is charged once.

Key parameters:

| Parameter | Meaning |
| --- | --- |
| `--budget` | Maximum logical channel-readout cost; unit costs in the CLI |
| `--epsilon` | Accepted midpoint-error resolution in concentration units |
| `--subdivisions` | Number of continuous cells per state coordinate |
| `--policy` | `separation`, `fixed`, or `random` |
| `--variant` | Conservative outer inversion, mechanism ablation, or inverse baseline |
| `--reference-bound` | Optional independently justified physical/reference discrepancy expansion |
| `--range-edges` | Descriptive concentration strata for reporting |

Library callers can pass positive nonuniform channel costs to `invert`. Increasing subdivisions tightens enclosures while increasing memory and computation; it does not change the calibration threshold.

## Comparisons and ablations

```bash
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/fixed --policy fixed
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/random --policy random
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/plugin --variant plugin
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/point_grid --variant point_grid
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/rf --variant rf
ssci evaluate --model runs/model/calibrated.joblib --input data/gas/test.npz --output runs/svr --variant svr
```

`plugin` freezes unknown nuisance coordinates at declared-domain midpoints. `point_grid` replaces each continuous cell by its center. These are mechanism controls; they discard states that the conservative outer method retains. RF and SVR use the fixed channel count configured during training. For a matched-budget comparison, set that count equal to the outer method's budget; for an acquisition-rich reference, fit and evaluate a separate full-channel model.

Sensitivity experiments can vary channel budget, state resolution, calibration group count, and test drift with the same frozen calibration. Do not duplicate correlated windows to increase the apparent number of calibration units.

## Metrics and outputs

`records.json` contains selected channel indices, charged cost, projected components, hull, midpoint, acceptance, incompatibility, error, inclusion, retained cell count, local nuisance-orthogonal sensitivity, inversion cache size, and measured latency. Infinite baseline interval endpoints are encoded as JSON null; they remain unbounded rather than being clipped to the concentration domain.

`metrics.json` reports concentration MAE/RMSE/bias on records with a point, coverage over all records, hull width, component count, abstention and incompatibility rates, readout cost/fraction, latency, repeatability for repeated groups with a fixed reference, and accepted-error rates. Incompatible records count as uncovered and remain in the overall denominators. Accepted-error output gives both the unconditional frequency over all records and the conditional frequency among accepted records, together with the accepted-record count. An absence of accepted records gives a null conditional rate.

`ranges.json` provides concentration-stratified RMSE, coverage, widths, and record counts. `plot` builds charts directly from existing metrics.

The local geometry diagnostic standardizes the concentration and nuisance Jacobians, projects concentration sensitivity away from nuisance directions, and reports the residual norm. It explains first-order identifiability; the actual returned set still comes from nonlinear cell inversion.

## Implementation conventions

Forward models are quadratic to make continuous response enclosures explicit and computable. The supplied inverse baselines are RF and SVR. They are distinct from the task-driven, minimax, mask-conditional, ensemble, and temporal methods discussed in the manuscript. Uniform cells implement the conservative cover; refinement is controlled by rerunning at a finer declared subdivision rather than deleting unresolved cells.

The physical interval expands the reference interval by the supplied reference bound. Its validity requires an independently justified bound; the program does not infer that bound from point-prediction accuracy. Readout fractions describe logical response revelation, and inversion cache bytes describe the retained numerical arrays. They do not estimate heater energy or process peak memory.
