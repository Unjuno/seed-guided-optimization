from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPS = tuple(range(1800, 1830))
MARGIN = 0.05
CALIPER = 0.05


def estimate(x: np.ndarray, level: float = 0.95) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x); mean = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p = 0.0 if mean > 0 else (0.5 if mean == 0 else 1.0); lo = hi = mean
    else:
        p = float(stats.t.sf(mean / se, n - 1)); crit = float(stats.t.ppf(0.5 + level / 2.0, n - 1)); lo = mean - crit * se; hi = mean + crit * se
    return {"mean": mean, "se": se, "ci_low": float(lo), "ci_high": float(hi), "p": p, "positive": int((x > 0).sum())}


def tost(x: np.ndarray, margin: float = MARGIN) -> dict:
    x = np.asarray(x, dtype=np.float64); n = len(x); mean = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf(0.95, n - 1)); lo = mean - crit * se; hi = mean + crit * se
    if se == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0; p_upper = 0.0 if mean < margin else 1.0
    else:
        p_lower = float(stats.t.sf((mean + margin) / se, n - 1)); p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))
    return {"mean": mean, "se": se, "ci90_low": float(lo), "ci90_high": float(hi), "p_lower": p_lower, "p_upper": p_upper, "pass": bool(lo > -margin and hi < margin and p_lower < .05 and p_upper < .05)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input-dir", default="results"); a = ap.parse_args(); root = Path(a.input_dir)
    paired = pd.read_csv(root / "parameter_matched_paired30.csv")
    decision = pd.read_csv(root / "parameter_matched_decision30.csv").iloc[0]
    calibration = pd.read_csv(root / "parameter_match_calibration_decision.csv").iloc[0]
    if len(paired) != 30 or set(paired.rep.astype(int)) != set(REPS) or paired.rep.duplicated().any(): raise ValueError("replicate grid")
    numeric = [c for c in paired.columns if c != "rep"]
    if not np.isfinite(paired[numeric].to_numpy(float)).all(): raise ValueError("nonfinite")
    if (paired.gradient_gap < -1e-12).any(): raise ValueError("gradient ordering")
    if (paired.mean_abs_parameter_z_diff > CALIPER + 1e-12).any(): raise ValueError("mean parameter caliper diagnostic impossible")
    if str(calibration.decision) != "MATCHING CALIBRATION PASS" or abs(float(calibration.selected_caliper) - CALIPER) > 1e-12: raise ValueError("calibration seal")

    g = estimate(paired.gradient_gap.to_numpy()); h = tost(paired.hardness_diff.to_numpy()); p = tost(paired.parameter_z_diff.to_numpy()); perf = estimate(paired.heldout_benefit.to_numpy())
    checks = {
        "mean_gradient_gap": g["mean"], "gradient_gap_se": g["se"], "gradient_gap_ci95_low": g["ci_low"], "gradient_gap_ci95_high": g["ci_high"], "gradient_gap_p_one_sided": g["p"],
        "mean_hardness_diff": h["mean"], "hardness_se": h["se"], "hardness_ci90_low": h["ci90_low"], "hardness_ci90_high": h["ci90_high"], "hardness_tost_p_lower": h["p_lower"], "hardness_tost_p_upper": h["p_upper"],
        "mean_parameter_z_diff": p["mean"], "parameter_z_se": p["se"], "parameter_z_ci90_low": p["ci90_low"], "parameter_z_ci90_high": p["ci90_high"], "parameter_tost_p_lower": p["p_lower"], "parameter_tost_p_upper": p["p_upper"],
        "mean_heldout_benefit": perf["mean"], "heldout_benefit_se": perf["se"], "heldout_benefit_ci95_low": perf["ci_low"], "heldout_benefit_ci95_high": perf["ci_high"], "heldout_benefit_p_one_sided": perf["p"],
    }
    for key, value in checks.items():
        if abs(float(decision[key]) - value) > 5e-13: raise ValueError(f"decision mismatch {key}: {decision[key]} != {value}")
    if not (g["mean"] > 0 and g["p"] < .05): raise ValueError("gradient manipulation")
    if not h["pass"]: raise ValueError("hardness equivalence")
    if not p["pass"]: raise ValueError("parameter equivalence")
    if not (perf["mean"] > 0 and perf["p"] < .05): raise ValueError("performance")
    if str(decision.decision) != "PARAMETER-MATCHED GRADIENT NONREDUNDANCY SUPPORT": raise ValueError("frozen decision")
    print("GRADIENT", g); print("HARDNESS_TOST", h); print("PARAMETER_TOST", p); print("HELDOUT", perf); print("RAW_PARAMETER_DIFF", float(paired.raw_parameter_diversity_diff.mean())); print("PASS")


if __name__ == "__main__": main()
