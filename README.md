# Post-progression treatment regimes and the limits of cross-trial OS transport

Simulation code, results and figures for the paper

> **Post-Progression Treatment Regimes and the Limits of Cross-Trial Survival Transport**
> Federico Felizzi (SIIAM, Rome; enGene, Basel) and Noman Paracha (Bayer AG, Basel).
> NeurIPS 2026 Workshop on World Models for High-Stakes Health (WMHS).

Every number in the paper is produced by `src/transport_sim.py` with fixed seeds.
The table [Paper ↔ code](#paper--code) shows where each one comes from.

## The argument in brief

When an overall survival (OS) effect is carried from a source trial to a target
population, the answer depends on the **post-progression treatment regime**,
the rates at which patients receive subsequent therapy by biomarker stratum and
arm, and not only on who is enrolled at baseline. The two standard method
families each miss this, for different reasons:

- **Crossover adjustment** (RPSFT, IPCW, two-stage) corrects *differential*
  switching, an imbalance between arms in post-progression therapy. A regime can
  be balanced overall, leaving nothing to detect or correct, while differing
  sharply within biomarker strata in opposite directions. Even a perfect
  adjustment targets a world with no post-progression therapy, which is not the
  world the target population lives in.
- **Population adjustment** (MAIC, STC, AIPW, ML density-ratio weighting)
  matches *baseline* covariates. Post-progression therapy happens after
  randomization, so it cannot be part of the matching set. Two trials can match
  on every baseline covariate and still run under different regimes.

Over plausible target regimes the transported effect spans zero, so without the
target regime the *sign* of the effect is undetermined. Conversely, treating the
regime as a *policy* fixes the problem: regime-aware and off-policy estimators
recover the target effect, provided the target regime is known and the source
regime overlaps it.

## Model

OS is built as PFS plus post-progression survival (PPS), so OS ≥ PFS holds by
construction (accelerated failure time model, log-normal errors):

```
log S   = log S0 + δ·T + b_X'X + ε_S                              (PFS)
log PPS = κ + λ_m·log(S/S0) + θT_m·T + θA·A + c_X'X + ε_P         (post-progression)
Y       = S + PPS                                                 (OS)
```

| Symbol | Meaning | Value |
|---|---|---|
| `T` | randomized treatment | Bernoulli(0.5) |
| `M` | biomarker stratum (0 = wild type, 1 = mutant) | prevalence 0.35 source, 0.70 target |
| `X` | baseline covariates: age, ECOG, prior lines, metastatic sites, visceral disease | moderate source → target shift |
| `S0` | control-arm median PFS | 6 months |
| `δ` | treatment effect on log PFS | 0.234 |
| `A ~ Bernoulli(q[m][t])` | post-progression targeted therapy; the table `q` is the **regime** | see below |
| `λ_m` | elasticity of PPS to PFS (0 = PFS gain passes fully to OS, −1 = fully absorbed) | 0.00 (wt), −0.85 (mut) |
| `θT_m` | direct treatment effect on PPS | 0.35 (wt), 0.05 (mut) |
| `θA` | log-scale PPS benefit of post-progression therapy | 0.50 |
| `ε_S`, `ε_P` | Gaussian errors | sd 0.45, 0.55 |

**Estimand:** the restricted mean survival time (RMST) difference at 36 months.
RMST is used instead of the hazard ratio because it is collapsible across
strata. The code calls the horizon `TAU`; the paper writes it t\* and reserves τ
for the treatment effect.

**Censoring:** uniform accrual over 24 months and analysis at 30.4 months, so
follow-up never exceeds 30.4 months and RMST at 36 months extrapolates the
Kaplan–Meier curve flat over the last 5.6 months (see [Limitations](#limitations)).

Three regimes, as post-progression therapy rates by stratum and arm:

| Regime | wt, control | wt, treatment | mut, control | mut, treatment | Description |
|---|---|---|---|---|---|
| S1 balanced | 0.20 | 0.20 | 0.60 | 0.60 | equal in both arms within each stratum |
| S2 offsetting | 0.25 | 0.40 | 0.68 | 0.40 | balanced overall, opposite imbalances within strata |
| S3 classic | 0.50 | 0.10 | 0.80 | 0.30 | textbook crossover: control arm gets more |

## Repository layout

```
src/
  transport_sim.py             main simulation and analysis script (produces every paper result)
  transport_sim_26082026.py    earlier snapshot (26 Aug 2026), not used for the paper
  transport_sim_29082026.py    earlier snapshot (29 Aug 2026), not used for the paper
tests/
  test_transport_sim.py        sanity tests for the model and estimators
results/
  results.json                 all tabulated results, in one file
  part*.csv                    one CSV per tabulated part
run_log.txt                    full console output of the reference run (Parts 2, 3 and 4 appear only here)
figures/
  km_curves.{pdf,png}          PFS and OS Kaplan-Meier curves by regime (paper Figure 1)
  policy_values.{pdf,png}      value of each post-progression policy at the target
  estimator_bias.{pdf,png}     bias of MAIC and the off-policy estimators
  sensitivity.{pdf,png}        transported effect across assumed target regimes (preliminary)
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
python src/transport_sim.py               # full run (reproduces the paper)
python src/transport_sim.py --quick       # smaller samples and fewer replications, for a fast check
python src/transport_sim.py --validate    # only check the weighted KM estimator against Monte Carlo
python src/transport_sim.py --outdir out  # write figures/ and results/ under out/
```

Parts 2, 3 and 4 print to the console only. To keep a log of the full run:

```bash
# macOS / Linux
python -u src/transport_sim.py 2>&1 | tee run_log.txt
```

```powershell
# Windows PowerShell
python -u src\transport_sim.py 2>&1 | Tee-Object -FilePath run_log.txt
```

Every run uses fixed seeds; rerunning reproduces `results/` exactly. The full
run takes noticeably longer than `--quick` because of the bootstrap in Part 3
and the replicated experiments in Parts 3b and 5–8.

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
| 3 | MAIC transport across regime pairs, single run with bootstrap CIs and ESS | console |
| 3b | MAIC transport bias over 20 replications | `part3b_transport_replicated.csv` |
| 4 | crossover adjustment, within-trial and under transport | console |
| 5 | MAIC, STC, AIPW and ML-ratio weighting (10 replications), plus a double-robustness check | `part5_estimator_families.csv`, `part5_double_robustness.csv` |
| 6 | value of each policy (no therapy, S1, S2, S3) at source and target, 6 replications | `part6_policy_values.csv` |
| 7 | regime-aware transport (g-computation over the target policy), 6 replications | `part7_regime_aware.csv` |
| 8 | off-policy estimators: direct method, policy-ratio IPW, doubly robust; 6 replications | `part8_ope_estimators.csv` |
| 9 | transported effect across an assumed target regime (preliminary, not in the paper) | `part9_sensitivity.csv` |

## Paper ↔ code

| Paper | Source | Output |
|---|---|---|
| Table 1 (regimes, marginal differences) | Part 1 | `part1_switching.csv` |
| Table 2 (published-style HRs) | Part 0 | `part0_published.csv` |
| Figure 1 (Kaplan–Meier curves, n = 120,000) | `plot_km` | `figures/km_curves.pdf` |
| Section 3, within-trial biases (−0.054, −0.111, −2.693) | Part 2 | `run_log.txt` |
| Table 3 (MAIC transport, 20 replications) | Part 3b | `part3b_transport_replicated.csv` |
| Table 4 (four adjustment approaches) | Part 5 | `part5_estimator_families.csv` |
| Table 5 (crossover adjustment) | Part 4 | `run_log.txt` |
| Section 7, set over target regimes | Part 6 | `part6_policy_values.csv` |
| Section 7, regime-aware and off-policy estimators | Parts 7 and 8 | `part7_regime_aware.csv`, `part8_ope_estimators.csv` |
| Section 9, KM extrapolation bias (0.16 months) | Validation | `run_log.txt` |
| Appendix A, double-robustness check | Part 5 | `part5_double_robustness.csv` |

## Key results

All values are months of RMST difference at 36 months, from the committed
reference run.

**Population adjustment fails when the regimes differ** (Part 3b, MAIC, mean
over 20 replications; paper Table 3):

| Source → target | Mean bias | SD |
|---|---|---|
| S1 → S1 | +0.092 | 0.143 |
| S2 → S2 | +0.022 | 0.168 |
| S1 → S2 | +0.949 | 0.132 |
| S2 → S1 | −0.835 | 0.179 |
| S3 → S1 | −2.748 | 0.169 |

The same pattern holds for STC, AIPW and ML density-ratio weights (Part 5).
Making the covariate model more flexible does not help, because the regime is
not a covariate.

**Without the target regime, the sign is undetermined** (Part 6, true effect in
the target population under each policy, mean over 6 replications):

| Policy | Value at target | Value at source |
|---|---|---|
| q = 0 (no therapy) | +1.70 | +3.78 |
| q = S1 balanced | +1.49 | +3.64 |
| q = S2 offsetting | +0.66 | +3.62 |
| q = S3 classic | −1.26 | +1.04 |

Over the plausible target regimes S1–S3 the effect ranges from −1.26 to +1.49
months (width 2.75), even with the mechanism known exactly.

What each method returns: crossover adjustment *targets* the q = 0 row; applied
in the source it returns the source value (+3.78; paper Table 5 gives +3.818 from
a separate run), and only combined with population adjustment would it give the
target's q = 0 value (+1.70). Population adjustment returns the row the
*source* ran under, evaluated at the target covariates. The estimand is the row
the *target* runs under.

**With the target regime, the effect is recovered** (Part 7, mean over 6
replications; MAIC re-estimated in the same replications, hence slightly
different from Part 3b):

| Source → target | MAIC bias | Regime-aware bias |
|---|---|---|
| S1 → S1 | +0.025 | −0.013 |
| S2 → S2 | +0.006 | −0.022 |
| S1 → S2 | +0.904 | −0.052 |
| S2 → S1 | −0.874 | +0.031 |
| S3 → S1 | −2.745 | +0.018 |

The three off-policy estimators of Part 8 (direct method, policy-ratio IPW,
doubly robust) give mean biases between −0.12 and +0.05, each within about one
standard error of zero at 6 replications.

**Practical implication.** Regime-aware transport needs `q_target[m][t]`, the
subsequent-therapy rates by biomarker stratum and arm in the target setting.
Trials could report these but generally do not.

**Part 9 (preliminary, not in the paper).** Part 9 sweeps an assumed
biomarker-positive target rate from 0 to 0.9. The transported estimate falls
from +1.73 to +1.29, while the simulated truth stays between +1.50 and +1.72
with no clear trend. Interpret with care until the sweep design is revisited.

## Limitations

- **Uncensored outcomes.** The crossover adjustment (Part 4), the STC and AIPW
  outcome models (Part 5), and the regime-aware and off-policy estimators
  (Parts 7–8) use the true, uncensored PFS and OS times, so they are best cases.
  A censoring-aware version is future work.
- **Horizon beyond follow-up.** Follow-up ends at 30.4 months but the RMST
  horizon is 36 months, so the Kaplan–Meier curve is extrapolated flat. At
  n = 200,000 this overstates the RMST difference by about 0.16 months against a
  direct Monte Carlo integral (validation step). The bias is common-mode and
  largely cancels in bias columns.
- **Shared mechanism and overlap.** Regime-aware transport assumes the PPS
  mechanism is shared between source and target, with only the policy differing.
  It also needs overlap: the source must give subsequent therapy in every
  stratum-arm cell the target policy reaches. `policy_overlap()` checks this.
- **Few replications for Parts 6–8.** These use 6 replications, so their biases
  are means with standard errors of roughly 0.07–0.14 months.
- **Mechanism held fixed.** The set over target regimes holds the mechanism at
  its true value, so it understates the full identified set.
- **Hazard ratios are descriptive.** They appear only to show that the simulated
  trials look like published ones. All transport analyses use RMST.

## Citation

```bibtex
@inproceedings{felizzi2026postprogression,
  title     = {Post-Progression Treatment Regimes and the Limits of Cross-Trial Survival Transport},
  author    = {Felizzi, Federico and Paracha, Noman},
  booktitle = {NeurIPS 2026 Workshop on World Models for High-Stakes Health (WMHS)},
  year      = {2026}
}
```

## References

- Robins, J. (1986). A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Robins, J. M., & Tsiatis, A. A. (1991). Correcting for non-compliance in randomized trials using rank preserving structural failure time models. *Communications in Statistics – Theory and Methods*, 20(8), 2609–2631.
- White, I. R., Babiker, A. G., Walker, S., & Darbyshire, J. H. (1999). Randomization-based methods for correcting for treatment changes: examples from the Concorde trial. *Statistics in Medicine*, 18(19), 2617–2634.
- Robins, J. M., & Finkelstein, D. M. (2000). Correcting for noncompliance and dependent censoring in an AIDS clinical trial with inverse probability of censoring weighted (IPCW) log-rank tests. *Biometrics*, 56(3), 779–788.
- Signorovitch, J. E. et al. (2010). Comparative effectiveness without head-to-head trials: a method for matching-adjusted indirect comparisons applied to psoriasis treatment with adalimumab or etanercept. *PharmacoEconomics*, 28(10), 935–945.
- Phillippo, D. M. et al. (2018). Methods for population-adjusted indirect comparisons in health technology appraisal. *Medical Decision Making*, 38(2), 200–211.
- Olivier, T., Haslam, A., & Prasad, V. (2023). Post-progression treatment in cancer randomized trials: a cross-sectional study of trials leading to FDA approval and published trials between 2018 and 2020. *BMC Cancer*, 23, 448.
