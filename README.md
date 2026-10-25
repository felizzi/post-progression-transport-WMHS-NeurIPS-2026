# Post-progression treatment regimes and the limits of cross-trial OS transport

Simulation code, results and figures for a study of how **post-progression
treatment regimes** break the transport of overall survival (OS) effects from
one trial population to another.

## The argument in brief

When an OS effect is carried from a source trial to a target population, the
answer depends on the post-progression therapy each population receives, and
not only on who is enrolled at baseline. The two standard method families each
miss this, for different reasons:

- **Crossover adjustment** (RPSFT, IPCW, two-stage) corrects *differential*
  switching, meaning an imbalance between arms in post-progression therapy. A
  regime can be balanced overall, leaving nothing to detect or correct, while
  differing sharply within biomarker strata in opposite directions. Even a
  perfect adjustment targets a world with no post-progression therapy, which
  is not the world the target population lives in.
- **Population adjustment** (MAIC, STC, AIPW, ML density-ratio weighting)
  matches *baseline* covariates. Post-progression therapy happens after
  randomization, so it cannot be part of the matching set. Two trials can
  match on every baseline covariate and still run under different regimes.

Treating the regime as a *policy* fixes this. Off-policy evaluation estimators
(g-computation / direct method, policy-ratio IPW, doubly robust) recover the
target estimand, provided the target regime is known and the source regime
overlaps it.

## Model

OS is built as PFS plus post-progression survival (PPS), so OS ≥ PFS holds by
construction:

```
log S   = log S0 + δ·T + b_X'X + ε_S                              (PFS)
log PPS = κ + λ_m·log(S/S0) + θT_m·T + θA·A + c_X'X + ε_P         (post-progression)
Y       = S + PPS                                                 (OS)
```

| Symbol | Meaning |
|---|---|
| `T` | randomized treatment |
| `M` | biomarker stratum (0 = wild type, 1 = mutant) |
| `X` | baseline covariates (age, ECOG, prior lines, metastatic sites, visceral disease) |
| `A ~ Bernoulli(q[m][t])` | post-progression targeted therapy; the table `q` is the **regime** |
| `λ_m` | elasticity of PPS to PFS (0 = PFS gain passes fully to OS, −1 = fully absorbed) |
| `θT_m` | direct treatment effect on PPS |
| `θA` | OS benefit of post-progression therapy |

The estimand is the **RMST difference at τ = 36 months**. RMST is used instead
of the hazard ratio because it is collapsible across strata.

Three regimes, as post-progression therapy rates by stratum and arm:

| Regime | wt, control | wt, treatment | mut, control | mut, treatment | Description |
|---|---|---|---|---|---|
| S1 balanced | 0.20 | 0.20 | 0.60 | 0.60 | equal in both arms within each stratum |
| S2 offsetting | 0.25 | 0.40 | 0.68 | 0.40 | balanced overall, opposite imbalances within strata |
| S3 classic | 0.50 | 0.10 | 0.80 | 0.30 | textbook crossover: control arm gets more |

The source population has biomarker prevalence 0.35 and the target has 0.70,
with moderate shifts in the baseline covariates.

## Repository layout

```
src/
  transport_sim.py             main simulation and analysis script (current version)
  transport_sim_26082026.py    earlier snapshot (26 Aug 2026)
  transport_sim_29082026.py    earlier snapshot (29 Aug 2026)
tests/
  test_transport_sim.py        sanity tests for the model and estimators
results/
  results.json                 every reported number, in one file
  part*.csv                    one CSV per results table
figures/
  km_curves.{pdf,png}          PFS and OS Kaplan-Meier curves by regime
  policy_values.{pdf,png}      value of each post-progression policy at the target
  estimator_bias.{pdf,png}     bias of MAIC and the off-policy estimators
  sensitivity.{pdf,png}        transported effect across assumed target regimes
requirements.txt
```

## Installation

Requires Python 3.10+ (developed on 3.12).

```bash
pip install -r requirements.txt     # numpy, scipy
pip install matplotlib              # optional: figures
pip install scikit-learn            # optional: ML density-ratio weights in Part 5
pip install pytest                  # optional: tests
```

The core simulation needs only numpy and scipy. Without matplotlib, the script
skips the figures. Without scikit-learn, the ML-ratio column in Part 5 shows
`n/a`.

## Usage

Run from the repository root so that `figures/` and `results/` are written
there:

```bash
python src/transport_sim.py               # full run
python src/transport_sim.py --quick       # smaller samples and fewer replications, for a fast check
python src/transport_sim.py --validate    # only check the weighted KM estimator against Monte Carlo
python src/transport_sim.py --outdir out  # write figures/ and results/ under out/
```

Every run uses fixed seeds, so results are reproducible. The full run takes
noticeably longer than `--quick` because of the bootstrap in Part 3 and the
replicated experiments in Parts 3b and 5–8.

### Tests

```bash
python -m pytest tests/ -q
```

The tests check the structural properties the argument rests on: OS ≥ PFS, the
weighted KM RMST against a Monte Carlo integral, pooling of tied event times,
S2 being balanced overall but imbalanced within strata, PFS being unaffected by
the regime, MAIC reaching covariate balance, oracle crossover adjustment
recovering the no-therapy counterfactual, and a mismatched regime producing
transport bias.

## What the script runs

| Part | Content | Output |
|---|---|---|
| Validation | weighted KM RMST (unit weights) against direct Monte Carlo | console |
| 0 | simulated trials reported as a publication would (PFS/OS Cox HRs, overall and by stratum) | `part0_published.csv` |
| 1 | what a standard switching assessment detects in each regime | `part1_switching.csv` |
| 2 | within-trial bias relative to the no-therapy counterfactual | console |
| 3 | MAIC transport across regime pairs, with bootstrap CIs and ESS | console |
| 3b | MAIC transport bias over 20 replications | `part3b_transport_replicated.csv` |
| 4 | crossover adjustment, within-trial and under transport | console |
| 5 | MAIC, STC, AIPW and ML-ratio weighting, plus a double-robustness check | `part5_estimator_families.csv`, `part5_double_robustness.csv` |
| 6 | value of each policy (no therapy, S1, S2, S3) at source and target | `part6_policy_values.csv` |
| 7 | regime-aware transport (g-computation over the target policy) | `part7_regime_aware.csv` |
| 8 | off-policy estimators: direct method, policy IPW, doubly robust | `part8_ope_estimators.csv` |
| 9 | sensitivity of the transported effect to an uncertain target regime | `part9_sensitivity.csv` |

## Key results

These figures come from the committed `results/` files (full run). Bias is in
months of RMST at τ = 36.

**Population adjustment works only when the regimes match** (Part 7, mean
over 6 replications):

| Source → target | MAIC bias | Regime-aware bias |
|---|---|---|
| S1 → S1 | +0.025 | −0.013 |
| S2 → S2 | +0.006 | −0.022 |
| S1 → S2 | +0.904 | −0.052 |
| S2 → S1 | −0.874 | +0.031 |
| S3 → S1 | −2.745 | +0.018 |

The same pattern holds for STC, AIPW and ML density-ratio weights (Part 5).
Making the covariate model more flexible does not help, because the regime is
not a covariate.

**Each method returns a different policy value** (Part 6, RMST difference in
the target population):

| Policy | Value at target |
|---|---|
| q = 0 (no therapy): what crossover adjustment returns | +1.70 |
| q = S1 balanced | +1.49 |
| q = S2 offsetting | +0.66 |
| q = S3 classic | −1.26 |

Crossover adjustment returns the first row. Population adjustment returns the
row the *source* ran under. The estimand is the row the *target* runs under.

**Practical implication.** Regime-aware transport needs `q_target[m][t]`,
the subsequent-therapy rates by biomarker stratum and arm in the target
setting. Trials could report these but generally do not. When they are
uncertain, Part 9 sweeps a plausible range and reports the resulting band.
Enrolling more patients does not narrow that band; reporting the regime does.

## Limitations

- `regime_aware_transport` and the off-policy estimators recover PPS from the
  true (uncensored) PFS and OS times, so they are a best case. A
  censoring-aware version is future work.
- Regime-aware transport assumes the PPS mechanism is shared between source
  and target, with only the policy differing. It also needs overlap: the
  source must give subsequent therapy in every stratum-arm cell the target
  policy reaches. `policy_overlap()` checks this.
- Hazard ratios appear only to show that the simulated trials look like
  published ones. All transport analyses use RMST.

## References

- Robins, J. (1986). A new approach to causal inference in mortality studies with a sustained exposure period. *Mathematical Modelling*, 7, 1393–1512.
- Signorovitch, J. E. et al. (2010). Comparative effectiveness without head-to-head trials: a method for matching-adjusted indirect comparisons. *PharmacoEconomics*, 28(10), 935–945.
- Phillippo, D. M. et al. (2018). Methods for population-adjusted indirect comparisons in health technology appraisal. *Medical Decision Making*, 38(2), 200–211.
