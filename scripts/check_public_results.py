"""Read-only check of two published Issue #91 performance summaries.

No training, network access, checkpoints, or changes to scientific decisions.
This is not the full matching/support/provenance audit. Values in CSVs are
accuracy fractions; only the human-readable output is converted to percentage
points. Numerical comparison uses rtol=1e-10, atol=1e-12 in CSV units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = {
    "u_mean_test_benefit": "Reference loss + total diversity",
    "m_mean_test_benefit": "Also translation-balanced on supported steps",
}
FIELDS = ("n", "mean", "se", "k", "ci95_low", "ci95_high", "p_one_sided", "positive_pairs")


def verify(input_dir: Path) -> dict:
    """Reject malformed inputs and compare recomputed statistics to the CSV."""
    primary = input_dir / "translation_matched_primary30.csv"
    summary = input_dir / "translation_matched_summary30.csv"
    paired = pd.read_csv(primary, float_precision="round_trip")
    saved = pd.read_csv(summary, float_precision="round_trip")
    needed = {"rep", *ENDPOINTS}
    if not needed.issubset(paired.columns):
        raise ValueError(f"missing paired columns: {sorted(needed - set(paired.columns))}")
    if not {"endpoint", *FIELDS}.issubset(saved.columns):
        raise ValueError("summary columns are incomplete")
    if len(paired) != 30 or paired.rep.duplicated().any() or set(paired.rep) != set(range(2200, 2230)):
        raise ValueError("expected exactly one row for each rep 2200-2229")
    if not np.isfinite(paired[list(needed)].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite paired values")
    if (paired[list(ENDPOINTS)].abs() > 1).any().any():
        raise ValueError("accuracy differences must be fractions in [-1, 1]")

    tests = {}
    for endpoint in ENDPOINTS:
        rows = saved.loc[saved.endpoint == endpoint]
        if len(rows) != 1:
            raise ValueError(f"expected one saved summary for {endpoint}")
        x = paired[endpoint].to_numpy(dtype=float)
        mean, se = float(x.mean()), float(stats.sem(x))
        if se <= 0:
            raise ValueError(f"degenerate sample for {endpoint}")
        k = float(stats.t.ppf(0.975, len(x) - 1))
        values = {
            "n": len(x), "mean": mean, "se": se, "k": k,
            "ci95_low": mean - k * se, "ci95_high": mean + k * se,
            "p_one_sided": float(stats.ttest_1samp(x, 0, alternative="greater").pvalue),
            "positive_pairs": int((x > 0).sum()),
        }
        for field, actual in values.items():
            expected = float(rows.iloc[0][field])
            if not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"summary mismatch: {endpoint}.{field}")
        tests[endpoint] = values
    return {
        "verification": "PUBLIC STATISTICS VERIFIED",
        "scope": "two public performance summaries only; not a full experiment audit or new replication",
        "csv_unit": "accuracy fraction",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (primary, summary)},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
        "tests": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--json", action="store_true", help="emit verification metadata and statistics as JSON")
    args = parser.parse_args()
    try:
        report = verify(args.input_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"PUBLIC CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        for endpoint, label in ENDPOINTS.items():
            s = report["tests"][endpoint]
            print(f"{label}: {100*s['mean']:+.6f} pp; "
                  f"95% CI [{100*s['ci95_low']:+.6f}, {100*s['ci95_high']:+.6f}]; "
                  f"one-sided p={s['p_one_sided']:.6g}; positive={s['positive_pairs']}/{s['n']}")
        print("PUBLIC STATISTICS VERIFIED (not a full experiment audit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
