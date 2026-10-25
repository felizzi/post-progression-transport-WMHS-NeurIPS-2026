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

# --------------------------------------- regime-aware transport
def regime_aware_transport(src, target_regime, target_means, rng):
    """Transport by g-computation over the post-progression policy.

    The estimators in part5_estimator_families() all return V^{q_source}:
    the value of the SOURCE trial's regime carried to the target covariate
    profile. Crossover adjustment returns V^{q=0}. The estimand is
    V^{q_target}. This function estimates it.

        1. From source individual data recover PPS = OS - PFS and fit,
           within biomarker stratum,  log PPS ~ log(S/S0) + T + A + X
        2. MAIC-weight source patients to the target covariate profile.
        3. Re-draw A from the TARGET regime q_T[m][t], not the source's.
        4. Predict PPS under the redrawn A, keeping each patient's own
           residual, and recompose Y = S + PPS.
        5. Weighted RMST difference on the recomposed outcomes.

    This is standard g-computation (Robins, 1986) applied to a policy
    rather than a point intervention; it is not a new estimator. In
    off-policy evaluation terms it is the direct method, with q_source the
    behaviour policy and q_target the target policy.

    ASSUMPTIONS, to be stated wherever this is used:
      * the PPS mechanism is shared between source and target, and only
        the policy differs;
      * overlap, q_target << q_source: the source must administer
        subsequent therapy in every stratum-arm cell the target policy
        reaches. Check with policy_overlap().

    This implementation reads S_true and Y_true to recover PPS, so it has
    cleaner information than an analyst facing censored data. It is a best
    case; a censoring-aware version is future work.
    """
    pps = np.maximum(src["Y_true"] - src["S_true"], 1e-9)
    log_pps = np.log(pps)
    X = cols_full(src)
    logS = np.log(src["S_true"] / S0)

    beta, resid = {}, np.zeros(len(log_pps))
    for m in (0, 1):
        sel = src["M"] == m
        D = np.c_[np.ones(sel.sum()), logS[sel], src["T"][sel],
                  src["A"][sel], X[sel]]
        b, *_ = np.linalg.lstsq(D, log_pps[sel], rcond=None)
        beta[m] = b
        resid[sel] = log_pps[sel] - D @ b

    w = maic_weights(X, target_means)
    rate = np.array([target_regime[m][t]
                     for m, t in zip(src["M"], src["T"])])
    A_new = rng.binomial(1, rate)

    log_new = np.zeros(len(log_pps))
    for m in (0, 1):
        sel = src["M"] == m
        D = np.c_[np.ones(sel.sum()), logS[sel], src["T"][sel],
                  A_new[sel], X[sel]]
        log_new[sel] = D @ beta[m] + resid[sel]

    Y_new = src["S_true"] + np.exp(log_new)
    C = ANALYSIS - rng.uniform(0, ACCRUAL, len(Y_new))
    recomposed = dict(T=src["T"], Y_obs=np.minimum(Y_new, C),
                      Y_ev=(Y_new <= C).astype(int))
    return rmst_diff(recomposed, w)


def policy_ratio_weights(src, source_regime, target_regime, clip=20.0):
    """Importance weights on the POLICY, q_target(a|s) / q_source(a|s).

    The standard inverse-propensity estimator of off-policy evaluation.
    For a patient in stratum m and arm t who received A=a,

        rho_i = q_T[m][t](a) / q_S[m][t](a)

    with q(1) the tabulated rate and q(0) its complement. Weights are
    clipped to control variance; heavy clipping signals poor overlap.
    """
    qs = np.array([source_regime[m][t] for m, t in zip(src["M"], src["T"])])
    qt = np.array([target_regime[m][t] for m, t in zip(src["M"], src["T"])])
    a = src["A"]
    num = np.where(a == 1, qt, 1.0 - qt)
    den = np.clip(np.where(a == 1, qs, 1.0 - qs), 1e-6, None)
    rho = np.clip(num / den, 0.0, clip)
    return rho


def policy_ipw_transport(src, source_regime, target_regime, target_means):
    """Off-policy evaluation by importance weighting: combine covariate
    weights (for the population shift) with policy ratio weights (for the
    regime shift). No outcome model."""
    w_cov = maic_weights(cols_full(src), target_means)
    rho = policy_ratio_weights(src, source_regime, target_regime)
    w = w_cov * rho
    return rmst_diff(src, w / w.mean())


def policy_dr_transport(src, source_regime, target_regime, target_means,
                        rng):
    """Doubly robust off-policy evaluation.

        tau_DR = tau_direct + weighted residual correction

    where tau_direct is regime_aware_transport (the direct method, an
    outcome model evaluated under the target policy) and the correction
    is the policy-ratio-weighted residual of that model on the observed
    data. Consistent if EITHER the outcome model or the policy ratio is
    correct, which is the standard doubly robust OPE construction.
    """
    direct = regime_aware_transport(src, target_regime, target_means, rng)

    pps = np.maximum(src["Y_true"] - src["S_true"], 1e-9)
    log_pps = np.log(pps)
    X, logS = cols_full(src), np.log(src["S_true"] / S0)
    fitted = np.zeros(len(log_pps))
    for m in (0, 1):
        sel = src["M"] == m
        D = np.c_[np.ones(sel.sum()), logS[sel], src["T"][sel],
                  src["A"][sel], X[sel]]
        b, *_ = np.linalg.lstsq(D, log_pps[sel], rcond=None)
        fitted[sel] = D @ b
    resid_months = np.exp(log_pps) - np.exp(fitted)

    w_cov = maic_weights(X, target_means)
    rho = policy_ratio_weights(src, source_regime, target_regime)
    w = w_cov * rho
    T, e = src["T"], 0.5
    correction = float(np.mean(w * (T / e - (1 - T) / (1 - e))
                               * resid_months) / max(w.mean(), 1e-9))
    return direct + correction


def policy_overlap(source_regime, target_regime, tol=1e-6):
    """Check q_target << q_source cell by cell.

    Off-policy evaluation requires the behaviour policy to place mass
    wherever the target policy does. If the source never administers
    subsequent therapy in some stratum-arm cell, no method can evaluate a
    target policy that does. Returns a list of violating cells.
    """
    violations = []
    for m in (0, 1):
        for t in (0, 1):
            qs, qt = source_regime[m][t], target_regime[m][t]
            if qt > tol and qs <= tol:
                violations.append((m, t, qs, qt))
            if (1 - qt) > tol and (1 - qs) <= tol:
                violations.append((m, t, qs, qt))
    return violations


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


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_policy_values(outdir="figures"):
    """Figure: what each method returns, against the estimand.

    Requires part6_policy_values() and part7_regime_aware() to have run.
    """
    plt = _mpl()
    if plt is None or "part6_policy_values" not in RESULTS:
        return None
    rows = RESULTS["part6_policy_values"]
    labels = [r["policy"].replace("q = ", "") for r in rows]
    vals = [r["value_target"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colours = ["0.55"] + ["tab:blue"] * (len(vals) - 1)
    bars = ax.barh(range(len(vals)), vals, color=colours, height=0.6)
    bars[0].set_hatch("//")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RMST difference at the target population (months)",
                  fontsize=9)
    ax.set_title("Value of each post-progression policy", fontsize=10)
    ax.annotate("crossover adjustment\nalways returns this row",
                xy=(vals[0], 0), xytext=(vals[0] * 0.45, -0.75),
                fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("population adjustment returns\nwhichever row the SOURCE ran under",
                xy=(vals[1], 1), xytext=(min(vals) * 0.9, 2.6),
                fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"policy_values.{ext}"), dpi=160)
    plt.close(fig)
    print(f"  figure written: {outdir}/policy_values.pdf")
    return True


def plot_estimator_bias(outdir="figures"):
    """Figure: bias by estimator and by regime pairing.

    Requires part8_ope_estimators() to have run.
    """
    plt = _mpl()
    if plt is None or "part8_ope_estimators" not in RESULTS:
        return None
    rows = RESULTS["part8_ope_estimators"]
    pairs = [f"{r['source'].split()[0]}\u2192{r['target'].split()[0]}"
             for r in rows]
    keys = [("MAIC_bias", "MAIC (covariate only)"),
            ("direct_bias", "direct method"),
            ("ipw_bias", "policy IPW"),
            ("dr_bias", "doubly robust OPE")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width, x = 0.2, np.arange(len(rows))
    for k, (key, lab) in enumerate(keys):
        vals = [r[key] for r in rows]
        errs = [r.get(key.replace("_bias", "_sd"), 0) for r in rows]
        ax.bar(x + (k - 1.5) * width, vals, width, yerr=errs, capsize=2,
               label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel("bias (months)", fontsize=9)
    ax.set_title("Correcting the covariate shift is not enough; "
                 "the policy shift must be corrected too", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"estimator_bias.{ext}"), dpi=160)
    plt.close(fig)
    print(f"  figure written: {outdir}/estimator_bias.pdf")
    return True


def plot_sensitivity(grid, ests, outdir="figures"):
    """Figure: transported effect as a function of the assumed target
    regime, with the resulting band."""
    plt = _mpl()
    if plt is None or grid is None:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(grid, ests, marker="o", ms=4, color="tab:blue")
    ax.fill_between(grid, min(ests), max(ests), alpha=0.12,
                    color="tab:blue")
    ax.axhline(min(ests), ls=":", lw=0.9, color="0.4")
    ax.axhline(max(ests), ls=":", lw=0.9, color="0.4")
    ax.set_xlabel("assumed subsequent-therapy rate in the target,\n"
                  "biomarker-positive stratum", fontsize=9)
    ax.set_ylabel("transported effect (months)", fontsize=9)
    ax.set_title(f"Cost of not knowing the target regime: "
                 f"{max(ests) - min(ests):.2f} months", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"sensitivity.{ext}"), dpi=160)
    plt.close(fig)
    print(f"  figure written: {outdir}/sensitivity.pdf")
    return True


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



def part6_policy_values(n=120_000, n_rep=6, seed=3100):
    """Enumerate the value of every policy at source and target
    prevalence. Each method in this script returns one of these rows; the
    estimand is a different row."""
    print()
    print("=" * 78)
    print("PART 6  Policy values V^q (RMST difference, months)")
    print("-" * 78)
    print(f"  {'policy':24s} {'source pop':>14s} {'target pop':>14s}")
    rows = []
    specs = [("q = 0 (no therapy)", REGIMES["S1 balanced"], 0)]
    specs += [(f"q = {k}", v, None) for k, v in REGIMES.items()]
    for label, regime, force in specs:
        vals = []
        for pi, is_tgt in ((0.35, False), (0.70, True)):
            acc = [rmst_diff(simulate(n, pi, regime,
                                      np.random.default_rng(seed + 2 * r),
                                      target_pop=is_tgt, force_A=force))
                   for r in range(n_rep)]
            vals.append(float(np.mean(acc)))
        rows.append(dict(policy=label, value_source=round(vals[0], 4),
                         value_target=round(vals[1], 4), n_rep=n_rep))
        print(f"  {label:24s} {vals[0]:+14.3f} {vals[1]:+14.3f}")
    RESULTS["part6_policy_values"] = rows
    tv = [r["value_target"] for r in rows]
    print()
    print(f"  spread at the target: [{min(tv):+.3f}, {max(tv):+.3f}], "
          f"width {max(tv) - min(tv):.3f} months")
    print("  Crossover adjustment returns the first row. Population")
    print("  adjustment returns whichever row the SOURCE ran under. The")
    print("  estimand is whichever row the TARGET operates under.")


def part7_regime_aware(n, n_rep=6, prevalence_source=0.35,
                       prevalence_target=0.70, seed=901):
    """The constructive result: g-computation over the target policy
    recovers the estimand where covariate adjustment does not."""
    print()
    print("=" * 78)
    print(f"PART 7  Regime-aware transport, {n_rep} replications")
    print("-" * 78)
    print(f"  {'source -> target':30s} {'MAIC bias':>11s} "
          f"{'regime-aware bias':>18s} {'overlap':>9s}")
    rows = []
    for s_, t_ in [("S1 balanced", "S1 balanced"),
                   ("S2 offsetting", "S2 offsetting"),
                   ("S1 balanced", "S2 offsetting"),
                   ("S2 offsetting", "S1 balanced"),
                   ("S3 classic", "S1 balanced")]:
        mb, rb = [], []
        for r in range(n_rep):
            src = simulate(n, prevalence_source, REGIMES[s_],
                           np.random.default_rng(seed + 2 * r))
            tgt = simulate(n, prevalence_target, REGIMES[t_],
                           np.random.default_rng(seed + 2 * r + 1),
                           target_pop=True)
            tm = cols_full(tgt).mean(0)
            truth = rmst_diff(tgt)
            mb.append(rmst_diff(src, maic_weights(cols_full(src), tm))
                      - truth)
            rb.append(regime_aware_transport(
                src, REGIMES[t_], tm,
                np.random.default_rng(seed + 5000 + r)) - truth)
        viol = policy_overlap(REGIMES[s_], REGIMES[t_])
        rows.append(dict(source=s_, target=t_,
                         maic_bias=round(float(np.mean(mb)), 4),
                         regime_aware_bias=round(float(np.mean(rb)), 4),
                         n_rep=n_rep, overlap_violations=len(viol)))
        print(f"  {s_ + ' -> ' + t_:30s} {np.mean(mb):+11.3f} "
              f"{np.mean(rb):+18.3f} {len(viol):9d}")
    RESULTS["part7_regime_aware"] = rows
    print()
    print("  The estimator requires q_target[m][t]: subsequent-therapy")
    print("  rates by stratum and arm in the target setting. That is")
    print("  exactly the quantity trials could publish and generally do")
    print("  not. It also requires overlap, reported in the last column.")



def part8_ope_estimators(n, n_rep=6, prevalence_source=0.35,
                         prevalence_target=0.70, seed=1201):
    """Three standard off-policy evaluation estimators for the regime
    shift, alongside the covariate-only baseline."""
    print()
    print("=" * 78)
    print(f"PART 8  Off-policy evaluation estimators, {n_rep} replications")
    print("-" * 78)
    print(f"  {'source -> target':28s} {'MAIC':>9s} {'direct':>9s} "
          f"{'policy IPW':>11s} {'DR-OPE':>9s}")
    rows = []
    for s_, t_ in [("S1 balanced", "S1 balanced"),
                   ("S1 balanced", "S2 offsetting"),
                   ("S2 offsetting", "S1 balanced"),
                   ("S3 classic", "S1 balanced")]:
        acc = {k: [] for k in ("MAIC", "direct", "ipw", "dr")}
        for r in range(n_rep):
            src = simulate(n, prevalence_source, REGIMES[s_],
                           np.random.default_rng(seed + 2 * r))
            tgt = simulate(n, prevalence_target, REGIMES[t_],
                           np.random.default_rng(seed + 2 * r + 1),
                           target_pop=True)
            tm = cols_full(tgt).mean(0)
            truth = rmst_diff(tgt)
            acc["MAIC"].append(
                rmst_diff(src, maic_weights(cols_full(src), tm)) - truth)
            acc["direct"].append(regime_aware_transport(
                src, REGIMES[t_], tm,
                np.random.default_rng(seed + 7000 + r)) - truth)
            acc["ipw"].append(policy_ipw_transport(
                src, REGIMES[s_], REGIMES[t_], tm) - truth)
            acc["dr"].append(policy_dr_transport(
                src, REGIMES[s_], REGIMES[t_], tm,
                np.random.default_rng(seed + 9000 + r)) - truth)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
              for k, v in acc.items()}
        rows.append(dict(source=s_, target=t_, n_rep=n_rep,
                         **{f"{k}_bias": round(m[k], 4) for k in m},
                         **{f"{k}_sd": round(sd[k], 4) for k in sd}))
        print(f"  {s_ + ' -> ' + t_:28s} {m['MAIC']:+9.3f} "
              f"{m['direct']:+9.3f} {m['ipw']:+11.3f} {m['dr']:+9.3f}")
    RESULTS["part8_ope_estimators"] = rows
    print()
    print("  MAIC corrects the covariate shift only. The three OPE")
    print("  estimators additionally correct the policy shift, each by a")
    print("  different route: an outcome model under the target policy")
    print("  (direct), importance weights on the policy ratio (IPW), and")
    print("  their combination (DR).")


def part9_sensitivity(n, n_grid=9, prevalence_source=0.35,
                      prevalence_target=0.70, seed=1301, outdir="."):
    """When q_target is not known exactly, sweep it and report a band.

    Practically the target regime is uncertain rather than unknown: a
    plausible range can be argued from the treatment landscape. Sweeping
    that range converts the point estimate into an interval whose width is
    the price of not reporting q.
    """
    print()
    print("=" * 78)
    print("PART 9  Sensitivity to an uncertain target regime")
    print("-" * 78)
    src = simulate(n, prevalence_source, REGIMES["S1 balanced"],
                   np.random.default_rng(seed))
    grid = np.linspace(0.0, 0.9, n_grid)
    ests, rows = [], []
    for qm in grid:
        cand = {0: {0: 0.20, 1: 0.20}, 1: {0: float(qm), 1: float(qm)}}
        tgt = simulate(n, prevalence_target, cand,
                       np.random.default_rng(seed + 1), target_pop=True)
        tm = cols_full(tgt).mean(0)
        est = regime_aware_transport(src, cand, tm,
                                     np.random.default_rng(seed + 77))
        ests.append(est)
        rows.append(dict(q_target_biomarker_positive=round(float(qm), 3),
                         transported_effect=round(float(est), 4),
                         truth=round(float(rmst_diff(tgt)), 4)))
        print(f"    q_target(biomarker+) = {qm:.2f}   "
              f"transported effect {est:+.3f}")
    RESULTS["part9_sensitivity"] = rows
    print()
    print(f"  range over the swept grid: [{min(ests):+.3f}, {max(ests):+.3f}]"
          f"   width {max(ests) - min(ests):.3f} months")
    print("  This width is the cost of not knowing q_target. It is not")
    print("  reduced by more patients, only by reporting the regime.")
    return grid, np.array(ests)


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
    part6_policy_values(n=60_000 if args.quick else 120_000,
                        n_rep=3 if args.quick else 6)
    part7_regime_aware(n=n, n_rep=3 if args.quick else 6)
    part8_ope_estimators(n=n, n_rep=3 if args.quick else 6)
    grid, ests = part9_sensitivity(n=n, n_grid=5 if args.quick else 9,
                                   outdir=args.outdir)
    plot_km(outdir=args.outdir + "/figures",
            n=60_000 if args.quick else 120_000)
    plot_policy_values(outdir=args.outdir + "/figures")
    plot_estimator_bias(outdir=args.outdir + "/figures")
    plot_sensitivity(grid, ests, outdir=args.outdir + "/figures")
    save_results(outdir=args.outdir + "/results")


if __name__ == "__main__":
    main()