"""Read-only replication of five LOCAL-FROZEN 2026-10-31 switch/seed contrasts.

Statistical-unit n=12 environment pools; does NOT audit trained checkpoint
hashes, training data, optimizer histories, or raw-image inference.
No experimental decisions or hyperparameters are changed.
"""
from collections import Counter
from itertools import product
from pathlib import Path
import csv
import math
import statistics
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "H1_adaptive_vs_A",
    "H2_adaptive_vs_M",
    "H3_adaptive_vs_schedule",
    "H4_seed_selection",
    "H5_joint_vs_bestfixed",
)


def read(name):
    with (ROOT / "results" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def check_close(actual, expected, field, tol=1e-11):
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        raise ValueError(f"{field} mismatch: {actual} vs {expected}")


def holm(pvalues):
    order = sorted(pvalues, key=lambda key: (pvalues[key], NAMES.index(key)))
    result, running = {}, 0.0
    for j, name in enumerate(order):
        running = max(running, (len(order) - j) * pvalues[name])
        result[name] = min(1.0, running)
    return result


def main():
    raw = read("switch_seed_20261031_pool_differences.csv")
    published = read("switch_seed_20261031_primary_decisions.csv")
    if len(raw) != 60 or len(published) != len(NAMES):
        raise ValueError("incomplete or duplicated study")
    keys = [(r["contrast"], int(r["pool"])) for r in raw]
    if Counter(keys) != Counter(product(NAMES, range(12))):
        raise ValueError("pool/contrast grid mismatch")
    names = [r["contrast"] for r in published]
    if len(set(names)) != 5 or set(names) != set(NAMES):
        raise ValueError("primary hypotheses changed")

    computed, p_raw, p_sign = {}, {}, {}
    for name in NAMES:
        vals = [float(next(r["ce_improvement"] for r in raw
                           if r["contrast"] == name and int(r["pool"]) == i))
                for i in range(12)]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("nonfinite pool CE")
        mean = statistics.fmean(vals)
        se = statistics.stdev(vals) / math.sqrt(12)
        if se <= 0 or not math.isfinite(se):
            raise ValueError("invalid SE")
        p = float(t.sf(mean / se, df=11))
        ci = float(t.ppf(.975, df=11) * se)
        # A post-hoc sensitivity check, assuming pool-level sign symmetry.
        p_sf = sum(
            (sum(s * v for s, v in zip(signs, vals)) / 12)
            >= mean - 1e-13
            for signs in product((-1, 1), repeat=12)
        ) / 4096
        computed[name] = dict(
            n_pools=12, mean_ce_improvement=mean, median=statistics.median(vals),
            se=se, ci95_low=mean-ci, ci95_high=mean+ci, p_raw=p,
            p_signflip=p_sf, positive_pools=sum(v > 0 for v in vals))
        p_raw[name], p_sign[name] = p, p_sf
    adj, sign_adj = holm(p_raw), holm(p_sign)
    for row in published:
        name = row["contrast"]
        derived = dict(computed[name], p_holm=adj[name], p_signflip_holm=sign_adj[name])
        for field, value in derived.items():
            check_close(float(row[field]), float(value), field)
        status = ("SUPPORT" if computed[name]["mean_ce_improvement"] >= .01
                  and adj[name] < .05 else "NOT SUPPORTED")
        if row["decision"] != status:
            raise ValueError(f"{name} decision changed")
    print("SWITCH SEED PUBLIC STATISTICS VERIFIED: 12 pools, 5 tests, 60 differences.")
    print("Supported: " + ", ".join(
        name for name in NAMES
        if computed[name]["mean_ce_improvement"] >= .01 and adj[name] < .05))
    print("This does NOT validate 576 checkpoint digests or an external preregistration.")


if __name__ == "__main__":
    main()
