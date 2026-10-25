"""Sanity tests. Run with: python -m pytest tests/ -q"""
import sys, os
import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import transport_sim as ts


def test_os_exceeds_pfs():
    """OS >= PFS is structural, not approximate."""
    d = ts.simulate(20_000, 0.35, ts.REGIMES["S1 balanced"],
                    np.random.default_rng(0))
    assert (d["Y_true"] >= d["S_true"]).all()


def test_km_matches_monte_carlo():
    """Weighted KM with unit weights should track a direct MC integral.
    A small positive gap is expected from flat tail extrapolation."""
    d = ts.simulate(100_000, 0.35, ts.REGIMES["S1 balanced"],
                    np.random.default_rng(1))
    km = ts.rmst_diff(d)
    mc = (np.minimum(d["Y_true"][d["T"] == 1], ts.TAU).mean()
          - np.minimum(d["Y_true"][d["T"] == 0], ts.TAU).mean())
    assert abs(km - mc) < 0.30, f"KM {km:.3f} vs MC {mc:.3f}"


def test_km_handles_ties():
    """Tied event times must produce a single pooled decrement."""
    t = np.array([1.0, 1.0, 1.0, 2.0])
    e = np.array([1, 1, 1, 0])
    w = np.ones(4)
    got = ts.km_rmst_arm(t, e, w, tau=2.0)
    # S(t)=1 on [0,1); after 3 pooled events out of 4 at risk, S=0.25
    assert abs(got - (1.0 + 0.25)) < 1e-9, got


def test_s2_is_marginally_balanced():
    """S2 must be undetectable by a marginal switching assessment."""
    d = ts.simulate(200_000, 0.35, ts.REGIMES["S2 offsetting"],
                    np.random.default_rng(2))
    r1 = d["A"][d["T"] == 1].mean()
    r0 = d["A"][d["T"] == 0].mean()
    assert abs(r1 - r0) < 0.02, f"marginal diff {r1 - r0:.3f}"


def test_s2_is_imbalanced_within_strata():
    """...while differing sharply within strata, in opposite directions."""
    reg = ts.REGIMES["S2 offsetting"]
    wt = reg[0][1] - reg[0][0]
    mut = reg[1][1] - reg[1][0]
    assert wt > 0.10 and mut < -0.10, (wt, mut)


def test_pfs_unaffected_by_regime():
    """Post-progression therapy cannot affect PFS."""
    hrs = []
    for name in ts.REGIMES:
        d = ts.simulate(60_000, 0.35, ts.REGIMES[name],
                        np.random.default_rng(3))
        hrs.append(ts.cox_hr(d["S_obs"], d["S_ev"], d["T"]))
    assert max(hrs) - min(hrs) < 0.03, hrs


def test_maic_balances_covariates():
    """After weighting, source covariate means must match the target."""
    src = ts.simulate(40_000, 0.35, ts.REGIMES["S1 balanced"],
                      np.random.default_rng(4), target_pop=False)
    tgt = ts.simulate(40_000, 0.70, ts.REGIMES["S1 balanced"],
                      np.random.default_rng(5), target_pop=True)
    Xs = np.c_[ts.covariate_matrix(src), src["M"]]
    tm = np.r_[[tgt[k].mean() for k in ts.COVARIATES], tgt["M"].mean()]
    w = ts.maic_weights(Xs, tm)
    achieved = (Xs * w[:, None]).sum(0) / w.sum()
    assert np.allclose(achieved, tm, atol=0.02), (achieved, tm)


def test_crossover_adjustment_recovers_no_therapy_world():
    """Oracle adjustment should recover the A=0 counterfactual."""
    d = ts.simulate(60_000, 0.35, ts.REGIMES["S3 classic"],
                    np.random.default_rng(6))
    truth = ts.rmst_diff(ts.simulate(60_000, 0.35, ts.REGIMES["S3 classic"],
                                     np.random.default_rng(6), force_A=0))
    adj, _ = ts.crossover_adjust(d, theta_a=ts.THETA_A)
    assert abs(ts.rmst_diff(adj) - truth) < 0.20


def test_regime_mismatch_creates_bias():
    """The central claim: matched regimes transport, mismatched do not."""
    def transported(src_reg, tgt_reg):
        src = ts.simulate(40_000, 0.35, ts.REGIMES[src_reg],
                          np.random.default_rng(7), target_pop=False)
        tgt = ts.simulate(40_000, 0.70, ts.REGIMES[tgt_reg],
                          np.random.default_rng(8), target_pop=True)
        Xs = np.c_[ts.covariate_matrix(src), src["M"]]
        tm = np.r_[[tgt[k].mean() for k in ts.COVARIATES], tgt["M"].mean()]
        w = ts.maic_weights(Xs, tm)
        return ts.rmst_diff(src, w) - ts.rmst_diff(tgt)

    matched = abs(transported("S1 balanced", "S1 balanced"))
    mismatched = abs(transported("S1 balanced", "S2 offsetting"))
    assert mismatched > matched + 0.4, (matched, mismatched)