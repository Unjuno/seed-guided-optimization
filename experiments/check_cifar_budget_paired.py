from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPS = tuple(range(50, 80))
QS = (2, 4, 6, 8)
TOL = 5e-15


def estimate(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p1 = 0.0 if mean > 0 else (0.5 if mean == 0 else 1.0)
        lo = hi = mean
    else:
        t = mean / se
        p1 = float(stats.t.sf(t, n - 1))
        crit = float(stats.t.ppf(0.975, n - 1))
        lo, hi = mean - crit * se, mean + crit * se
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_one_sided": p1,
        "positive_pairs": int((x > 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results")
    a = ap.parse_args()
    root = Path(a.input_dir)
    paired = pd.read_csv(root / "cifar_budget_paired30.csv")
    decision = pd.read_csv(root / "cifar_budget_decision30.csv")
    mismatch = pd.read_csv(root / "cifar_budget_q8_mismatches.csv")

    if len(paired) != 30 or set(paired.rep.astype(int)) != set(REPS) or paired.rep.duplicated().any():
        raise ValueError("replicate grid mismatch")
    required = ["benefit_q2", "benefit_q4", "benefit_q6", "benefit_q8"]
    if not np.isfinite(paired[required].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite benefit")
    if len(mismatch) != 0:
        raise ValueError("published Q8 mismatch table is non-empty")
    if np.max(np.abs(paired["benefit_q8"].to_numpy(dtype=np.float64))) > TOL:
        raise ValueError("Q8 benefit is not zero within CSV round-trip tolerance")

    low = (paired.benefit_q2 + paired.benefit_q4) / 2.0
    high = (paired.benefit_q6 + paired.benefit_q8) / 2.0
    attenuation = low - high
    primary = estimate(attenuation.to_numpy())

    arithmetic = {
        "benefit_low": low,
        "benefit_high": high,
        "benefit_attenuation": attenuation,
        "posthoc_low_minus_q6": low - paired.benefit_q6,
    }
    for key, expected in arithmetic.items():
        if np.max(np.abs(paired[key].to_numpy(dtype=np.float64) - expected.to_numpy(dtype=np.float64))) > TOL:
            raise ValueError(f"paired arithmetic mismatch: {key}")

    row = decision.iloc[0]
    checks = {
        "mean_benefit_low": float(low.mean()),
        "mean_benefit_high": float(high.mean()),
        "mean_benefit_attenuation": primary["mean"],
        "benefit_attenuation_se": primary["se"],
        "benefit_attenuation_ci95_low": primary["ci95_low"],
        "benefit_attenuation_ci95_high": primary["ci95_high"],
        "benefit_attenuation_p_one_sided": primary["p_one_sided"],
    }
    for key, value in checks.items():
        if abs(float(row[key]) - value) > 5e-13:
            raise ValueError(f"decision arithmetic mismatch {key}: {row[key]} vs {value}")
    if not bool(row["q8_identity_pass"]) or int(row["q8_mismatch_count"]) != 0:
        raise ValueError("published Q8 identity decision failed")
    if str(row["decision"]) != "CIFAR RESNET FINITE-BUDGET COVERAGE REPLICATES":
        raise ValueError("unexpected frozen decision")

    # Post-hoc endpoint sensitivity only; it never replaces the preregistered primary.
    nonfull = estimate((low - paired.benefit_q6).to_numpy())

    print("FROZEN_PRIMARY", primary)
    print("POSTHOC_LOW_MINUS_Q6", nonfull)
    print("Q_MEAN_BENEFITS", {q: float(paired[f"benefit_q{q}"].mean()) for q in QS})
    print("PASS")


if __name__ == "__main__":
    main()
