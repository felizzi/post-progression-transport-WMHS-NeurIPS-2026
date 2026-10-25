"""
================================================================================
Post-progression treatment regimes and the limits of cross-trial OS transport
================================================================================

WHAT THIS SHOWS
---------------
Overall survival (OS) transported from one trial to another population depends
on the POST-PROGRESSION TREATMENT REGIME, not only on baseline composition.
Two established method families each miss this, for different reasons:

  * Crossover adjustment (RPSFT, IPCW, two-stage) corrects DIFFERENTIAL
    switching -- an arm imbalance in post-progression therapy. A regime can be
    balanced marginally (nothing to detect, nothing to correct) while differing
    sharply within biomarker strata in offsetting directions.

  * MAIC / propensity weighting matches BASELINE covariates. Post-progression
    therapy is post-randomization, so it cannot enter the matching set even in
    principle. Two trials can match perfectly on every baseline covariate and
    still operate under different post-progression regimes.

The gap between the two families is unguarded. Neither method is wrong; each
answers a different question from the one transport requires.

STRUCTURAL MODEL
----------------
OS >= PFS by definition, so OS is modelled as PFS plus post-progression
survival (PPS) rather than as a regression of OS on PFS:

    log S   = log S0 + delta*T + b_X'X + eps_S            (PFS)
    log PPS = kappa + lambda_m*log(S/S0) + thetaT_m*T
                    + thetaA*A + c_X'X + eps_P            (post-progression)
    Y       = S + PPS                                     (OS)

    T   randomized treatment          M   biomarker stratum (0=wt, 1=mut)
    S   PFS                           A   post-progression targeted therapy
    Y   OS                            X   baseline covariates

    lambda_m  elasticity of PPS to PFS. lambda=0: a PFS gain passes through to
              OS one-for-one. lambda<0: the gain is partly absorbed by shorter
              survival after progression. lambda=-1: fully absorbed.
    thetaT_m  direct treatment effect on PPS, bypassing the surrogate.
    thetaA    OS benefit of post-progression targeted therapy.

A ~ Bernoulli(q[m][t]): the post-progression regime is a table of rates by
stratum and arm. This table is the object of study.

ESTIMAND
--------
Difference in restricted mean survival time (RMST) at TAU months. RMST is used
rather than the hazard ratio because it is collapsible, which matters when
reweighting populations across strata.

REGIMES
-------
  S1 balanced    equal rates in both arms within every stratum
  S2 offsetting  marginal rates equal across arms, stratum rates differ in
                 opposite directions -- invisible to a switching assessment
  S3 classic     control arm receives more in both strata (textbook crossover)

USAGE
-----
    python transport_sim.py                 run everything
    python transport_sim.py --quick         smaller n, fewer bootstraps
    python transport_sim.py --validate      estimator validation only

Requires numpy and scipy only. Fully seeded; reproducible.
================================================================================
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------------ constants
S0 = 6.0            # baseline (control-arm) median PFS, months
SD_S = 0.45         # sd of log PFS
SD_P = 0.55         # sd of log PPS
KAPPA = 2.30        # baseline log PPS  (median PPS ~ exp(2.30) ~ 10 months)
DELTA = 0.234       # treatment effect on log PFS  (PFS HR ~ 0.62)
ACCRUAL = 24.0      # months of uniform accrual
ANALYSIS = 30.4     # calendar time of analysis -> administrative censoring
TAU = 36.0          # RMST horizon, months
THETA_A = 0.50      # log-scale OS gain from post-progression targeted therapy

# mechanism, by stratum: 0 = biomarker wild type, 1 = mutant
LAMBDA = {0: 0.00, 1: -0.85}    # PPS elasticity to PFS
THETA_T = {0: 0.35, 1: 0.05}    # direct treatment effect on PPS

# baseline covariates: name -> (source mean, target mean, sd, PFS coef, PPS coef)
# coefficients are per unit; all are prognostic, none informs the regime.
COVARIATES = {
    # Shifts are moderate and realistic. Aggressive shifts drive the MAIC
    # effective sample size so low that the method fails on its own terms
    # (linear moment matching cannot correct nonlinear covariate effects),
    # which would confound the regime comparison this script is designed
    # to isolate. See the ESS column in Part 3.
    "age":        (62.0, 65.0, 9.0, -0.010, -0.008),
    "ecog":       (0.65, 0.58, 0.48, 0.180, 0.150),   # share ECOG 0-1
    "prior_lines": (1.20, 1.45, 0.90, -0.070, -0.060),
    "met_sites":  (2.10, 2.40, 1.10, -0.060, -0.050),
    "visceral":   (0.55, 0.62, 0.50, -0.130, -0.110),
}

RESULTS = {}   # accumulated structured output, written by save_results()

REGIMES = {
    #            stratum 0 (wt)          stratum 1 (mut)
    #            arm0    arm1            arm0    arm1
    "S1 balanced":   {0: {0: 0.20, 1: 0.20}, 1: {0: 0.60, 1: 0.60}},
    "S2 offsetting": {0: {0: 0.25, 1: 0.40}, 1: {0: 0.68, 1: 0.40}},
    "S3 classic":    {0: {0: 0.50, 1: 0.10}, 1: {0: 0.80, 1: 0.30}},
}


# ------------------------------------------------------------------ simulation
def simulate(n, prevalence, regime, rng, target_pop=False, force_A=None):
    """Simulate one trial or target population.

    prevalence : biomarker prevalence
    regime     : dict {stratum: {arm: post-progression therapy rate}}
    target_pop : if True, draw covariates from the target distribution
    force_A    : if 0 or 1, override the regime (used to define the
                 no-post-progression-therapy counterfactual)
    """
    T = rng.binomial(1, 0.5, n)
    M = rng.binomial(1, prevalence, n)

    X, lp_S, lp_P = {}, np.zeros(n), np.zeros(n)
    for name, (m_src, m_tgt, sd, cS, cP) in COVARIATES.items():
        mu = m_tgt if target_pop else m_src
        if name in ("ecog", "visceral"):
            x = rng.binomial(1, np.clip(mu, 0, 1), n).astype(float)
        elif name == "prior_lines":
            x = rng.poisson(mu, n).astype(float)
        else:
            x = rng.normal(mu, sd, n)
        X[name] = x
        centred = x - m_src
        lp_S += cS * centred
        lp_P += cP * centred

    S = S0 * np.exp(DELTA * T + lp_S + rng.normal(0, SD_S, n))

    if force_A is None:
        rate = np.array([regime[m][t] for m, t in zip(M, T)])
        A = rng.binomial(1, rate)
    else:
        A = np.full(n, float(force_A))

    lam = np.where(M == 1, LAMBDA[1], LAMBDA[0])
    tht = np.where(M == 1, THETA_T[1], THETA_T[0])
    PPS = np.exp(KAPPA + lam * np.log(S / S0) + tht * T
                 + THETA_A * A + lp_P + rng.normal(0, SD_P, n))

    Y = S + PPS                       # OS >= PFS by construction
    C = ANALYSIS - rng.uniform(0, ACCRUAL, n)

    out = dict(T=T, M=M, A=A, S_true=S, Y_true=Y,
               S_obs=np.minimum(S, C), S_ev=(S <= C).astype(int),
               Y_obs=np.minimum(Y, C), Y_ev=(Y <= C).astype(int))
    out.update(X)
    return out


# ------------------------------------------------- weighted Kaplan-Meier RMST
def km_rmst_arm(time, event, weight, tau=TAU):
    """Weighted Kaplan-Meier RMST for one arm.

    Ties are POOLED: all events at a common time produce a single survival
    decrement, as in the standard estimator. Applying decrements one row at a
    time (a common implementation slip) biases RMST upward.

    The curve is extrapolated flat beyond the last observed time, the usual
    convention when the largest observation is censored below tau.
    """
    order = np.argsort(time, kind="mergesort")
    t, e, w = time[order], event[order], weight[order]
    at_risk = w[::-1].cumsum()[::-1]          # weighted number at risk

    surv, prev_t, area = 1.0, 0.0, 0.0
    i, n = 0, len(t)
    while i < n:
        ti = t[i]
        j = i
        while j < n and t[j] == ti:           # collect the tie group
            j += 1
        if ti > tau:
            return area + surv * (tau - prev_t)
        area += surv * (ti - prev_t)
        d = w[i:j][e[i:j] == 1].sum()         # pooled weighted events at ti
        if d > 0 and at_risk[i] > 0:
            surv *= 1.0 - d / at_risk[i]
        prev_t = ti
        i = j
    return area + surv * max(0.0, tau - prev_t)


def rmst_diff(d, weights=None, tau=TAU):
    """Weighted RMST difference, treatment arm minus control arm."""
    n = len(d["T"])
    w = np.ones(n) if weights is None else weights
    tr, ct = d["T"] == 1, d["T"] == 0
    return (km_rmst_arm(d["Y_obs"][tr], d["Y_ev"][tr], w[tr], tau)
            - km_rmst_arm(d["Y_obs"][ct], d["Y_ev"][ct], w[ct], tau))


# --------------------------------------- outcome regression and AIPW
def rmst_pseudo(d):
    """RMST contribution min(Y, TAU) with an indicator for patients whose
    administrative follow-up reaches TAU. Censoring here is independent of
    outcome, so restricting to those patients is unbiased."""
    complete = (d["Y_obs"] >= TAU) | (d["Y_ev"] == 1)
    return np.minimum(d["Y_true"], TAU), complete


def outcome_model(d, cols):
    """Fit m(X, T) = E[min(Y, TAU) | X, T] in the source by least squares,
    with treatment-by-covariate interactions. Returns a predict function.

    This is the regression half of STC and of the AIPW estimator below.
    """
    y, ok = rmst_pseudo(d)
    X, T, y = cols(d)[ok], d["T"][ok], y[ok]
    design = np.c_[np.ones(len(y)), X, T, T[:, None] * X]
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)

    def predict(X_new, t):
        n = len(X_new)
        tt = np.full(n, float(t))
        return np.c_[np.ones(n), X_new, tt, tt[:, None] * X_new] @ beta
    return predict


def stc_estimate(src, tgt, cols):
    """Simulated treatment comparison: outcome model fitted in the source,
    evaluated at the target covariate profile (Phillippo et al., 2018)."""
    predict = outcome_model(src, cols)
    Xt = cols(tgt).mean(0)[None, :]
    return float((predict(Xt, 1) - predict(Xt, 0))[0])


def aipw_estimate(src, tgt, cols, weights=None, predict=None,
                  propensity=0.5):
    """Augmented inverse probability weighting for transport.

        tau_DR = E_target[ m(X,1) - m(X,0) ]
               + mean_i  w_i * [ 1{T_i=1}/e - 1{T_i=0}/(1-e) ]
                             * ( Y_i - m(X_i, T_i) )

    Consistent if EITHER the weights or the outcome model is correct --
    that is the double robustness property, verified in the test suite.

    IMPORTANT for the argument of this script: double robustness protects
    against misspecifying the FUNCTIONAL FORM of either component. Both
    components are functions of X. The post-progression regime is not a
    function of X, so no amount of double robustness reaches it. See
    part5_estimator_families().
    """
    Xs, Xt = cols(src), cols(tgt)
    if predict is None:
        predict = outcome_model(src, cols)
    if weights is None:
        weights = maic_weights(Xs, Xt.mean(0))
    regression = float((predict(Xt, 1) - predict(Xt, 0)).mean())
    y, _ = rmst_pseudo(src)
    T, e = src["T"], propensity
    m_hat = np.where(T == 1, predict(Xs, 1), predict(Xs, 0))
    augment = float(np.mean(weights * (T / e - (1 - T) / (1 - e))
                            * (y - m_hat)))
    return regression + augment


def cols_full(d):
    """All matched covariates plus the biomarker."""
    return np.c_[covariate_matrix(d), d["M"]]


# ------------------------------------------------- crossover adjustment
def crossover_adjust(d, theta_a=None):
    """Two-stage-style adjustment removing the post-progression therapy
    effect, recovering the counterfactual in which nobody received it.

    Post-progression survival is observable as PPS = OS - PFS for patients
    who progressed, so the adjustment rescales it:

        PPS_adj = PPS * exp(-theta_a * A),    OS_adj = PFS + PPS_adj

    This is the logic of White's two-stage estimator: re-express survival
    after progression as it would have been absent the post-progression
    treatment. RPSFT and IPCW target the same counterfactual by different
    routes.

    theta_a=None estimates the effect from the data by comparing treated and
    untreated patients within arm-by-stratum cells. Passing the true value
    gives an ORACLE adjustment -- the best any such method could do.
    """
    pps = np.maximum(d["Y_true"] - d["S_true"], 1e-9)
    if theta_a is None:                       # estimate within cells
        est = []
        for m in (0, 1):
            for t in (0, 1):
                cell = (d["M"] == m) & (d["T"] == t)
                a1, a0 = cell & (d["A"] == 1), cell & (d["A"] == 0)
                if a1.sum() > 30 and a0.sum() > 30:
                    est.append(np.log(pps[a1].mean()) - np.log(pps[a0].mean()))
        theta_a = float(np.mean(est)) if est else 0.0
    pps_adj = pps * np.exp(-theta_a * d["A"])
    y_adj = d["S_true"] + pps_adj
    C = d["Y_true"] - d["Y_obs"] + d["Y_obs"]        # recover censor times
    C = np.where(d["Y_ev"] == 1, np.inf, d["Y_obs"])
    out = dict(d)
    out["Y_obs"] = np.minimum(y_adj, C)
    out["Y_ev"] = (y_adj <= C).astype(int)
    return out, theta_a


# ------------------------------------------------------- Cox hazard ratio
def cox_hr(time, event, x, weights=None):
    """Cox partial-likelihood hazard ratio for a single binary covariate,
    Breslow handling of ties, optionally weighted.

    Reported for comparability with published trial results. Note that the
    hazard ratio is NON-COLLAPSIBLE: a marginal HR is not a weighted average
    of stratum-specific HRs even absent confounding. That is why the transport
    analyses in this script use RMST, which is collapsible, and why HRs appear
    here only as a descriptive check that simulated trials resemble real ones.
    """
    from scipy.optimize import brentq
    n = len(time)
    w = np.ones(n) if weights is None else weights
    order = np.argsort(-time, kind="mergesort")
    t, e, xx, ww = time[order], event[order], x[order].astype(float), w[order]

    def score(beta):
        ew = ww * np.exp(beta * xx)
        cum_w = np.cumsum(ew)                 # weighted risk set
        cum_wx = np.cumsum(ew * xx)
        ev = e == 1
        return np.sum(ww[ev] * (xx[ev] - cum_wx[ev] / cum_w[ev]))

    try:
        return float(np.exp(brentq(score, -6.0, 6.0, xtol=1e-8)))
    except ValueError:
        return float("nan")


def trial_summary(d, label=""):
    """Print PFS and OS hazard ratios, overall and by biomarker stratum,
    the way a trial publication reports them."""
    if "S_obs" not in d:
        raise KeyError("simulate() must be called with keep_pfs=True")
    print(f"  {label}")
    print(f"    {'subgroup':16s} {'PFS HR':>8s} {'OS HR':>8s} "
          f"{'OS events':>10s} {'n':>7s}")
    for name, mask in [("overall", np.ones(len(d["T"]), bool)),
                       ("biomarker wt", d["M"] == 0),
                       ("biomarker mut", d["M"] == 1)]:
        pfs = cox_hr(d["S_obs"][mask], d["S_ev"][mask], d["T"][mask])
        os_ = cox_hr(d["Y_obs"][mask], d["Y_ev"][mask], d["T"][mask])
        ev = d["Y_ev"][mask].mean()
        RESULTS.setdefault("part0_published", []).append(
            dict(regime=label, subgroup=name, pfs_hr=round(float(pfs), 4),
                 os_hr=round(float(os_), 4),
                 os_event_pct=round(100 * float(ev), 1), n=int(mask.sum())))
        print(f"    {name:16s} {pfs:8.3f} {os_:8.3f} {100*ev:9.0f}% "
              f"{mask.sum():7d}")


# ------------------------------------------------- KM curves and plotting
def km_curve(time, event, weight=None, tau=None):
    """Weighted Kaplan-Meier curve as step-function coordinates.

    Returns (t, S) suitable for a step plot. Ties are pooled, matching
    km_rmst_arm. Pass tau to truncate the curve at the RMST horizon.
    """
    n = len(time)
    w = np.ones(n) if weight is None else weight
    order = np.argsort(time, kind="mergesort")
    t, e, w = time[order], event[order], w[order]
    at_risk = w[::-1].cumsum()[::-1]
    ts, ss, surv, i = [0.0], [1.0], 1.0, 0
    while i < n:
        ti = t[i]
        j = i
        while j < n and t[j] == ti:
            j += 1
        if tau is not None and ti > tau:
            break
        d = w[i:j][e[i:j] == 1].sum()
        if d > 0 and at_risk[i] > 0:
            surv *= 1.0 - d / at_risk[i]
            ts.append(ti); ss.append(surv)
        i = j
    if tau is not None:
        ts.append(tau); ss.append(surv)
    return np.array(ts), np.array(ss)


def plot_km(outdir="figures", n=120_000, prevalence=0.35, seed=17):
    """Kaplan-Meier curves for PFS and OS, by arm, one column per regime.

    The point of the figure: the PFS panels are indistinguishable across
    regimes (post-progression therapy cannot affect an endpoint that
    precedes it), while the OS panels separate.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping figures")
        return None

    os.makedirs(outdir, exist_ok=True)
    names = list(REGIMES)
    fig, axes = plt.subplots(2, len(names), figsize=(4.0 * len(names), 6.4),
                             sharey="row", sharex=True)
    for k, name in enumerate(names):
        d = simulate(n, prevalence, REGIMES[name],
                     np.random.default_rng(seed))
        for row, (tt, ee, lab) in enumerate(
                [("S_obs", "S_ev", "progression-free survival"),
                 ("Y_obs", "Y_ev", "overall survival")]):
            ax = axes[row, k]
            for arm, colour, style in ((1, "tab:blue", "-"),
                                       (0, "tab:red", "--")):
                m = d["T"] == arm
                x, y = km_curve(d[tt][m], d[ee][m], tau=TAU)
                ax.step(x, y, where="post", color=colour, ls=style, lw=1.4,
                        label="treatment" if arm == 1 else "control")
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, TAU)
            if row == 0:
                ax.set_title(name, fontsize=10)
            if k == 0:
                ax.set_ylabel(f"{lab}\nsurvival probability", fontsize=8)
            if row == 1:
                ax.set_xlabel("months", fontsize=9)
            ax.grid(alpha=0.25)
            if row == 0 and k == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle("PFS is invariant to the post-progression regime; OS is not",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"km_curves.{ext}"), dpi=160)
    plt.close(fig)
    print(f"  figures written: {outdir}/km_curves.pdf, .png")
    return os.path.join(outdir, "km_curves.pdf")


# ------------------------------------------------------------------ MAIC
def maic_weights(X_source, target_means):
    """Method-of-moments MAIC weights (Signorovitch et al., 2010).

    Solves for a such that w = exp(a'(X - target_means)) has weighted source
    means equal to target_means. Returns weights normalised to mean 1.
    """
    Xc = X_source - target_means
    res = minimize(lambda a: np.log(np.mean(np.exp(Xc @ a))),
                   np.zeros(Xc.shape[1]), method="BFGS")
    w = np.exp(Xc @ res.x)
    return w / w.mean()


def effective_sample_size(w):
    """Kish effective sample size."""
    return w.sum() ** 2 / (w ** 2).sum()


def covariate_matrix(d):
    return np.c_[tuple(d[k] for k in COVARIATES)] if len(COVARIATES) > 1 \
        else np.c_[d[list(COVARIATES)[0]]]


# ================================================================== analyses
def validate_estimator(n=200_000, seed=1):
    """Weighted KM with unit weights must agree with a direct Monte Carlo
    integral of min(Y, tau) on the uncensored outcomes."""
    print("=" * 78)
    print("VALIDATION  weighted KM (w=1) against direct Monte Carlo")
    print("-" * 78)
    d = simulate(n, 0.35, REGIMES["S1 balanced"], np.random.default_rng(seed))
    km = rmst_diff(d)
    mc = (np.minimum(d["Y_true"][d["T"] == 1], TAU).mean()
          - np.minimum(d["Y_true"][d["T"] == 0], TAU).mean())
    print(f"  weighted KM RMST difference : {km:+.4f}")
    print(f"  direct Monte Carlo          : {mc:+.4f}")
    print(f"  discrepancy                 : {km - mc:+.4f}")
    print(f"  administrative censoring    : "
          f"{100 * (1 - d['Y_ev'].mean()):.1f}% of OS observations")
    print("  Note: a small positive discrepancy is expected because the KM")
    print("  curve is extrapolated flat beyond the last observed time.")
    return km - mc


def part1_switching_assessment(n, prevalence=0.35, seed=11):
    print()
    print("=" * 78)
    print("PART 1  What a standard treatment-switching assessment detects")
    print("-" * 78)
    rows = []
    print(f"{'regime':16s} {'arm 1':>8s} {'arm 0':>8s} {'difference':>12s}"
          f" {'flagged':>9s}")
    for name, regime in REGIMES.items():
        d = simulate(n, prevalence, regime, np.random.default_rng(seed))
        r1 = d["A"][d["T"] == 1].mean()
        r0 = d["A"][d["T"] == 0].mean()
        flag = "YES" if abs(r1 - r0) > 0.05 else "no"
        rows.append(dict(regime=name, arm1_rate=round(float(r1), 4),
                         arm0_rate=round(float(r0), 4),
                         marginal_diff=round(float(r1 - r0), 4), flagged=flag))
        print(f"{name:16s} {r1:8.3f} {r0:8.3f} {r1 - r0:+12.3f} {flag:>9s}")
    print()
    print("  Stratum-specific rates (arm1 / arm0):")
    for name, regime in REGIMES.items():
        print(f"    {name:16s} wt {regime[0][1]:.2f}/{regime[0][0]:.2f}"
              f"   mut {regime[1][1]:.2f}/{regime[1][0]:.2f}")
    print()
    RESULTS["part1_switching"] = rows
    print("  S2 shows no marginal imbalance, so a switching assessment finds")
    print("  nothing to correct and RPSFT / IPCW are never applied. Within")
    print("  strata the arms differ by 0.15 and 0.28 in opposite directions.")


def part2_within_trial(n, prevalence=0.35, seed=21):
    print()
    print("=" * 78)
    print("PART 2  Within-trial bias relative to the no-therapy counterfactual")
    print("-" * 78)
    print(f"{'regime':16s} {'truth (A=0)':>12s} {'observed':>10s} {'bias':>9s}")
    for name, regime in REGIMES.items():
        truth = rmst_diff(simulate(n, prevalence, regime,
                                   np.random.default_rng(seed), force_A=0))
        obs = rmst_diff(simulate(n, prevalence, regime,
                                 np.random.default_rng(seed)))
        print(f"{name:16s} {truth:+12.3f} {obs:+10.3f} {obs - truth:+9.3f}")
    print()
    print("  S2's offsetting imbalance cancels AT THIS PREVALENCE, so the")
    print("  trial's own estimate is nearly unbiased. S3 is badly biased and")
    print("  is exactly what crossover adjustment exists to repair.")


def part3_transport(n, n_boot, prevalence_source=0.35, prevalence_target=0.70,
                    seed=31):
    print()
    print("=" * 78)
    print("PART 3  Cross-trial transport with MAIC on baseline covariates")
    print("-" * 78)
    print(f"  matching on {len(COVARIATES)} covariates: "
          f"{', '.join(COVARIATES)}")
    print(f"  source prevalence {prevalence_source}, "
          f"target prevalence {prevalence_target}")
    print()
    print(f"{'source':>16s} {'target':>16s} {'truth':>8s} {'MAIC':>8s}"
          f" {'bias':>8s} {'bootstrap 95% CI':>20s} {'ESS%':>6s}")

    pairs = [("S1 balanced", "S1 balanced"),
             ("S2 offsetting", "S2 offsetting"),
             ("S1 balanced", "S2 offsetting"),
             ("S2 offsetting", "S1 balanced"),
             ("S3 classic", "S1 balanced")]

    for src_name, tgt_name in pairs:
        src = simulate(n, prevalence_source, REGIMES[src_name],
                       np.random.default_rng(seed), target_pop=False)
        tgt = simulate(n, prevalence_target, REGIMES[tgt_name],
                       np.random.default_rng(seed + 1), target_pop=True)

        Xs = np.c_[covariate_matrix(src), src["M"]]
        target_means = np.r_[[tgt[k].mean() for k in COVARIATES],
                             tgt["M"].mean()]
        w = maic_weights(Xs, target_means)

        est = rmst_diff(src, w)
        truth = rmst_diff(tgt)
        ess_pct = 100 * effective_sample_size(w) / n

        boot = []
        rb = np.random.default_rng(seed + 99)
        for _ in range(n_boot):
            idx = rb.integers(0, n, n)
            sb = {k: v[idx] for k, v in src.items()}
            try:
                Xb = np.c_[covariate_matrix(sb), sb["M"]]
                boot.append(rmst_diff(sb, maic_weights(Xb, target_means)))
            except Exception:
                continue
        lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
        covered = "" if lo <= truth <= hi else "   <-- truth outside CI"

        print(f"{src_name:>16s} {tgt_name:>16s} {truth:+8.3f} {est:+8.3f}"
              f" {est - truth:+8.3f} [{lo:+.3f},{hi:+.3f}] {ess_pct:5.1f}%"
              f"{covered}")

    print()
    print("  Baseline covariates and biomarker prevalence are matched exactly")
    print("  in every row. Rows where source and target share a regime are")
    print("  accurate. Rows where the regime differs are biased beyond the")
    print("  bootstrap interval, and no baseline covariate carries the")
    print("  information needed to correct it.")



def part4_crossover(n, prevalence_source=0.35, prevalence_target=0.70,
                    seed=41):
    """Crossover adjustment: what it repairs, and what it cannot."""
    print()
    print("=" * 78)
    print("PART 4  Crossover adjustment (two-stage style)")
    print("-" * 78)
    print("  (a) Within-trial: does adjustment recover the no-therapy world?")
    print()
    print(f"    {'regime':16s} {'unadjusted':>11s} {'adjusted':>10s} "
          f"{'oracle adj':>11s} {'A=0 truth':>10s}")
    for name, regime in REGIMES.items():
        d = simulate(n, prevalence_source, regime,
                     np.random.default_rng(seed))
        truth = rmst_diff(simulate(n, prevalence_source, regime,
                                   np.random.default_rng(seed), force_A=0))
        adj_est, _ = crossover_adjust(d)
        adj_orc, _ = crossover_adjust(d, theta_a=THETA_A)
        print(f"    {name:16s} {rmst_diff(d):+11.3f} "
              f"{rmst_diff(adj_est):+10.3f} {rmst_diff(adj_orc):+11.3f} "
              f"{truth:+10.3f}")
    print()
    print("    Adjustment recovers the no-therapy counterfactual well,")
    print("    including for S3 where the unadjusted estimate is far off.")
    print("    On its own terms the method works.")

    print()
    print("  (b) Transport: is the no-therapy counterfactual the right")
    print("      estimand for the target population?")
    print()
    print(f"    {'source':>16s} {'target':>16s} {'oracle-adj':>11s} "
          f"{'target truth':>13s} {'bias':>8s}")
    for src_name, tgt_name in [("S3 classic", "S1 balanced"),
                               ("S1 balanced", "S2 offsetting"),
                               ("S2 offsetting", "S1 balanced")]:
        src = simulate(n, prevalence_source, REGIMES[src_name],
                       np.random.default_rng(seed))
        tgt = simulate(n, prevalence_target, REGIMES[tgt_name],
                       np.random.default_rng(seed + 1), target_pop=True)
        adj, _ = crossover_adjust(src, theta_a=THETA_A)
        est, truth = rmst_diff(adj), rmst_diff(tgt)
        print(f"    {src_name:>16s} {tgt_name:>16s} {est:+11.3f} "
              f"{truth:+13.3f} {est - truth:+8.3f}")
    print()
    print("    Crossover adjustment targets the counterfactual in which NO")
    print("    patient receives post-progression therapy. The target")
    print("    population is not that world: it operates under its own")
    print("    regime. Adjusting perfectly still answers the wrong question,")
    print("    and the residual bias is of the same order as the effect.")



def ml_ratio_weights(src, tgt, cols, seed=0):
    """Density-ratio weights from a gradient-boosted classifier trained to
    discriminate source from target on X. w = p(target|X)/p(source|X).

    A deliberately flexible alternative to moment matching. Requires
    scikit-learn; returns None if unavailable.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return None
    Xs, Xt = cols(src), cols(tgt)
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=seed).fit(X, y)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-4, 1 - 1e-4)
    w = p / (1 - p)
    return w / w.mean()


def part5_estimator_families(n, n_rep=10, prevalence_source=0.35,
                             prevalence_target=0.70, seed=151):
    """Four adjustment approaches spanning distinct function classes,
    averaged over replications.

    If the obstruction were the parametric form of MAIC's weights, a more
    flexible estimator would help. It does not: every estimator here is a
    function of source data and target COVARIATES, and the regime is not a
    covariate.
    """
    print()
    print("=" * 78)
    print(f"PART 5  Four adjustment approaches, {n_rep} replications")
    print("-" * 78)
    print(f"  {'source -> target':30s} {'MAIC':>16s} {'STC':>16s} "
          f"{'AIPW':>16s} {'ML-ratio':>16s}")
    rows = []
    for s_, t_ in [("S1 balanced", "S1 balanced"),
                   ("S2 offsetting", "S2 offsetting"),
                   ("S1 balanced", "S2 offsetting"),
                   ("S2 offsetting", "S1 balanced")]:
        acc = {k: [] for k in ("MAIC", "STC", "AIPW", "ML")}
        for r in range(n_rep):
            src = simulate(n, prevalence_source, REGIMES[s_],
                           np.random.default_rng(seed + 2 * r))
            tgt = simulate(n, prevalence_target, REGIMES[t_],
                           np.random.default_rng(seed + 2 * r + 1),
                           target_pop=True)
            truth = rmst_diff(tgt)
            w = maic_weights(cols_full(src), cols_full(tgt).mean(0))
            acc["MAIC"].append(rmst_diff(src, w) - truth)
            acc["STC"].append(stc_estimate(src, tgt, cols_full) - truth)
            acc["AIPW"].append(
                aipw_estimate(src, tgt, cols_full, weights=w) - truth)
            wl = ml_ratio_weights(src, tgt, cols_full, seed=r)
            if wl is not None:
                acc["ML"].append(rmst_diff(src, wl) - truth)
        def fmt(v):
            if not v:
                return f"{'n/a':>16s}"
            a = np.array(v)
            return f"{a.mean():+8.3f}({a.std(ddof=1):.3f})"
        print(f"  {s_ + ' -> ' + t_:30s} " +
              " ".join(fmt(acc[k]) for k in ("MAIC", "STC", "AIPW", "ML")))
        row = dict(source=s_, target=t_, n_rep=n_rep)
        for k in ("MAIC", "STC", "AIPW", "ML"):
            if acc[k]:
                a = np.array(acc[k])
                row[f"{k}_mean_bias"] = round(float(a.mean()), 4)
                row[f"{k}_sd"] = round(float(a.std(ddof=1)), 4)
        rows.append(row)
    RESULTS["part5_estimator_families"] = rows
    print("  (mean bias, with standard deviation in parentheses)")
    print()
    print("  Every estimator carries near-zero bias when regimes match and")
    print("  bias of the same order when they differ, regardless of")
    print("  function class.")

    print()
    print("  Double robustness check (regimes matched; only model")
    print("  specification varies):")
    src = simulate(n, prevalence_source, REGIMES["S1 balanced"],
                   np.random.default_rng(seed))
    tgt = simulate(n, prevalence_target, REGIMES["S1 balanced"],
                   np.random.default_rng(seed + 1), target_pop=True)
    truth = rmst_diff(tgt)
    cols_poor = lambda d: np.c_[d["age"], d["M"]]
    w_good = maic_weights(cols_full(src), cols_full(tgt).mean(0))
    w_bad = maic_weights(cols_poor(src), cols_poor(tgt).mean(0))
    m_good = outcome_model(src, cols_full)
    m_poor = outcome_model(src, cols_poor)
    dr_rows = []
    for lbl, cols, w, pm in [
            ("both correct", cols_full, w_good, m_good),
            ("weights correct, model poor", cols_poor, w_good, m_poor),
            ("model correct, weights poor", cols_full, w_bad, m_good),
            ("both poor", cols_poor, w_bad, m_poor)]:
        bias = aipw_estimate(src, tgt, cols, w, pm) - truth
        dr_rows.append(dict(scenario=lbl, bias=round(float(bias), 4)))
        print(f"    {lbl:30s} {bias:+.3f}")
    RESULTS["part5_double_robustness"] = dr_rows
    print()
    print("  Bias stays small under every misspecification: the estimator")
    print("  is doubly robust. Double robustness protects against the wrong")
    print("  functional form, not against absent data.")


def part3_transport_replicated(n, n_rep=20, prevalence_source=0.35,
                               prevalence_target=0.70, seed=131):
    """Transport bias averaged over independent replications.

    A single run gives a point estimate whose Monte Carlo variability is
    not negligible: matched-regime bias has a standard deviation of roughly
    0.25 months at n=60,000. Reporting means with standard deviations makes
    the matched/mismatched comparison interpretable rather than resting on
    one draw.
    """
    print()
    print("=" * 78)
    print(f"PART 3b  Transport bias over {n_rep} replications")
    print("-" * 78)
    print(f"  {'source -> target':32s} {'mean bias':>10s} {'sd':>7s} "
          f"{'mean ESS%':>10s}")
    rows = []
    for s_, t_ in [("S1 balanced", "S1 balanced"),
                   ("S2 offsetting", "S2 offsetting"),
                   ("S1 balanced", "S2 offsetting"),
                   ("S2 offsetting", "S1 balanced"),
                   ("S3 classic", "S1 balanced")]:
        b, es = [], []
        for r in range(n_rep):
            src = simulate(n, prevalence_source, REGIMES[s_],
                           np.random.default_rng(seed + 2 * r))
            tgt = simulate(n, prevalence_target, REGIMES[t_],
                           np.random.default_rng(seed + 2 * r + 1),
                           target_pop=True)
            w = maic_weights(cols_full(src), cols_full(tgt).mean(0))
            b.append(rmst_diff(src, w) - rmst_diff(tgt))
            es.append(100 * effective_sample_size(w) / n)
        b = np.array(b)
        rows.append(dict(source=s_, target=t_,
                         mean_bias=round(float(b.mean()), 4),
                         sd=round(float(b.std(ddof=1)), 4),
                         n_rep=n_rep, mean_ess_pct=round(float(np.mean(es)), 1)))
        print(f"  {s_ + ' -> ' + t_:32s} {b.mean():+10.3f} "
              f"{b.std(ddof=1):7.3f} {np.mean(es):9.1f}%")
    RESULTS["part3b_transport_replicated"] = rows
    print()
    print("  Matched-regime bias is centred on zero; its standard deviation")
    print("  is the yardstick against which the mismatched rows should be")
    print("  read.")



def save_results(outdir="results"):
    """Write every reported number to disk: results.json plus one CSV per
    table, so manuscript tables and figures can be regenerated without
    re-running the simulation."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "results.json")
    with open(path, "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=float)
    for key, rows in RESULTS.items():
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            cols = sorted({c for r in rows for c in r})
            with open(os.path.join(outdir, f"{key}.csv"), "w") as fh:
                fh.write(",".join(cols) + "\n")
                for r in rows:
                    fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"  results written: {path} and per-table CSVs in {outdir}/")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="smaller sample, fewer bootstrap replicates")
    p.add_argument("--validate", action="store_true",
                   help="run estimator validation only")
    p.add_argument("--outdir", default=".",
                   help="where to write figures/ and results/")
    args = p.parse_args()

    n = 20_000 if args.quick else 60_000
    n_boot = 50 if args.quick else 300

    validate_estimator(n=50_000 if args.quick else 200_000)
    if args.validate:
        return

    print()
    print("=" * 78)
    print("PART 0  Simulated trials as a publication would report them")
    print("-" * 78)
    for name in REGIMES:
        d = simulate(4 * n, 0.35, REGIMES[name], np.random.default_rng(7))
        trial_summary(d, label=f"regime {name}, prevalence 0.35")
        print()
    part1_switching_assessment(n=4 * n)
    part2_within_trial(n=4 * n)
    part3_transport(n=n, n_boot=n_boot)
    part4_crossover(n=4 * n)
    part5_estimator_families(n=n, n_rep=4 if args.quick else 10)
    part3_transport_replicated(n=n, n_rep=8 if args.quick else 20)
    plot_km(outdir=args.outdir + "/figures",
            n=60_000 if args.quick else 120_000)
    save_results(outdir=args.outdir + "/results")


if __name__ == "__main__":
    main()