from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPS = tuple(range(1600, 1630))
MARGIN = 0.10


def estimate(x: np.ndarray, level: float = 0.95) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    t = mean / se if se else np.inf
    p = float(stats.t.sf(t, n - 1)) if se else (0.0 if mean > 0 else 0.5)
    crit = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
    return {
        "mean": mean,
        "se": se,
        "ci_low": float(mean - crit * se),
        "ci_high": float(mean + crit * se),
        "p_one_sided": p,
        "positive": int((x > 0).sum()),
    }


def tost(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    crit90 = float(stats.t.ppf(0.95, n - 1))
    lo, hi = mean - crit90 * se, mean + crit90 * se
    p_lower = float(stats.t.sf((mean + MARGIN) / se, n - 1))
    p_upper = float(stats.t.cdf((mean - MARGIN) / se, n - 1))
    return {
        "mean": mean,
        "se": se,
        "ci90_low": float(lo),
        "ci90_high": float(hi),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "pass": bool(lo > -MARGIN and hi < MARGIN and p_lower < .05 and p_upper < .05),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results")
    a = ap.parse_args()
    root = Path(a.input_dir)
    paired = pd.read_csv(root / "nonredundancy_paired30.csv")
    decision = pd.read_csv(root / "nonredundancy_decision30.csv").iloc[0]
    param = pd.read_csv(root / "nonredundancy_parameter_diversity_posthoc.csv")

    if len(paired) != 30 or set(paired.rep.astype(int)) != set(REPS) or paired.rep.duplicated().any():
        raise ValueError("paired replicate grid")
    if len(param) != 30 or set(param.rep.astype(int)) != set(REPS) or param.rep.duplicated().any():
        raise ValueError("parameter diagnostic replicate grid")
    numerical = [c for c in paired.columns if c != "rep"]
    if not np.isfinite(paired[numerical].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite paired value")
    if (paired.novelty_diff < -1e-12).any():
        raise ValueError("negative MAXNOV-MINNOV manipulation")

    novelty = estimate(paired.novelty_diff.to_numpy())
    hardness = tost(paired.hardness_diff.to_numpy())
    performance = estimate(paired.heldout_benefit.to_numpy())
    param_est = estimate(param.parameter_diversity_diff.to_numpy())

    checks = {
        "mean_novelty_diff": novelty["mean"],
        "novelty_se": novelty["se"],
        "novelty_ci95_low": novelty["ci_low"],
        "novelty_ci95_high": novelty["ci_high"],
        "novelty_p_one_sided": novelty["p_one_sided"],
        "mean_hardness_diff": hardness["mean"],
        "hardness_se": hardness["se"],
        "hardness_ci90_low": hardness["ci90_low"],
        "hardness_ci90_high": hardness["ci90_high"],
        "hardness_tost_p_lower": hardness["p_lower"],
        "hardness_tost_p_upper": hardness["p_upper"],
        "mean_heldout_benefit": performance["mean"],
        "heldout_benefit_se": performance["se"],
        "heldout_benefit_ci95_low": performance["ci_low"],
        "heldout_benefit_ci95_high": performance["ci_high"],
        "heldout_benefit_p_one_sided": performance["p_one_sided"],
    }
    for key, value in checks.items():
        if abs(float(decision[key]) - value) > 5e-13:
            raise ValueError(f"decision mismatch: {key} {decision[key]} != {value}")

    if not (novelty["mean"] > 0 and novelty["p_one_sided"] < .05):
        raise ValueError("novelty manipulation does not pass")
    if not hardness["pass"]:
        raise ValueError("hardness TOST does not pass")
    if not (performance["mean"] > 0 and performance["p_one_sided"] < .05):
        raise ValueError("performance does not pass")
    if str(decision["decision"]) != "LOSS-STRATIFIED NONREDUNDANCY SUPPORT":
        raise ValueError("unexpected frozen decision")

    print("NOVELTY", novelty)
    print("HARDNESS_TOST", hardness)
    print("HELDOUT", performance)
    print("POSTHOC_PARAMETER_DIVERSITY", param_est)
    print("PASS")


if __name__ == "__main__":
    main()
