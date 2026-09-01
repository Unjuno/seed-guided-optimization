from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXPECTED_REPS = tuple(range(500, 530))
FAMILIES = ("structured", "unstructured")
METHODS = ("loss_hard", "gradnov")
HELDOUT_METRICS = ("mean_test", "sd_test", "p10_test", "min_test", "clean_test")


def load_parts(input_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(input_dir.rglob(prefix + "*.csv"))
    if not files:
        raise FileNotFoundError(f"no files for {prefix}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def validate_grid(frame: pd.DataFrame, label: str) -> None:
    expected = {
        (rep, family, method)
        for rep in EXPECTED_REPS
        for family in FAMILIES
        for method in METHODS
    }
    observed = set(zip(
        frame.rep.astype(int), frame.family.astype(str), frame.method.astype(str)
    ))
    if len(frame) != len(expected) or observed != expected:
        raise ValueError(f"{label} grid mismatch")
    if frame.duplicated(["rep", "family", "method"]).any():
        raise ValueError(f"{label} duplicate rows")


def one_sided_positive(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    if np.all(x == x[0]):
        if x[0] > 0:
            return float("inf"), 0.0
        if x[0] < 0:
            return float("-inf"), 1.0
        return 0.0, 0.5
    r = stats.ttest_1samp(x, 0.0)
    t = float(r.statistic)
    p2 = float(r.pvalue)
    return t, (p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sealed-severity", required=True, type=float)
    args = ap.parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    diag = load_parts(inp, "matched_novelty_diagnostics_")
    held = load_parts(inp, "matched_novelty_heldout_")
    validate_grid(diag, "diagnostics")
    validate_grid(held, "heldout")

    unstructured_severities = diag.loc[
        diag.family == "unstructured", "severity"
    ].dropna().to_numpy(float)
    if len(unstructured_severities) == 0 or not np.all(
        unstructured_severities == args.sealed_severity
    ):
        raise ValueError("confirmatory training severity not identical to sealed severity")
    held_sev = held.loc[
        held.family == "unstructured", "severity"
    ].dropna().to_numpy(float)
    if len(held_sev) == 0 or not np.all(held_sev == args.sealed_severity):
        raise ValueError("heldout severity mismatch")

    train_rows = []
    outcome_rows = []
    conversion_rows = []
    for rep in EXPECTED_REPS:
        rep_train = diag[diag.rep == rep]
        rep_held = held[held.rep == rep]
        family_benefits = {}
        for family in FAMILIES:
            tr = rep_train[rep_train.family == family].set_index("method")
            ho = rep_held[rep_held.family == family].set_index("method")
            novelty_gain = float(
                tr.loc["gradnov", "selected_pairwise_novelty"]
                - tr.loc["loss_hard", "selected_pairwise_novelty"]
            )
            candidate_loss_level = float(tr["mean_candidate_loss"].mean())
            selected_loss_level = float(tr["mean_selected_loss"].mean())
            train_rows.append({
                "rep": rep,
                "family": family,
                "novelty_gain": novelty_gain,
                "candidate_loss_level": candidate_loss_level,
                "selected_loss_level": selected_loss_level,
            })
            row = {"rep": rep, "family": family}
            for metric in HELDOUT_METRICS:
                row["delta_" + metric] = float(
                    ho.loc["gradnov", metric] - ho.loc["loss_hard", metric]
                )
            outcome_rows.append(row)
            family_benefits[family] = row["delta_mean_test"]
        conversion_rows.append({
            "rep": rep,
            "structured_benefit": family_benefits["structured"],
            "unstructured_benefit": family_benefits["unstructured"],
            "conversion_contrast": (
                family_benefits["structured"] - family_benefits["unstructured"]
            ),
        })

    train_summary = pd.DataFrame(train_rows)
    outcomes = pd.DataFrame(outcome_rows)
    conversion = pd.DataFrame(conversion_rows)

    s = train_summary[train_summary.family == "structured"]
    u = train_summary[train_summary.family == "unstructured"]
    mean_nov_s = float(s.novelty_gain.mean())
    mean_nov_u = float(u.novelty_gain.mean())
    mean_loss_s = float(s.candidate_loss_level.mean())
    mean_loss_u = float(u.candidate_loss_level.mean())
    nov_mismatch = abs(mean_nov_s - mean_nov_u)
    loss_mismatch = abs(mean_loss_s - mean_loss_u)
    match_pass = bool(nov_mismatch <= 0.03 and loss_mismatch <= 0.10)

    vals = conversion.conversion_contrast.to_numpy(float)
    t, p = one_sided_positive(vals)
    conversion_pass = bool(vals.mean() > 0 and p < 0.05)

    if not match_pass:
        decision = "MATCH FAILURE / INCONCLUSIVE"
    elif conversion_pass:
        decision = "REUSABILITY-CONVERSION PASS"
    else:
        decision = "NO CONVERSION SEPARATION"

    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    decision_df = pd.DataFrame([{
        "n": len(vals),
        "sealed_severity": args.sealed_severity,
        "mean_structured_novelty_gain": mean_nov_s,
        "mean_unstructured_novelty_gain": mean_nov_u,
        "novelty_mismatch": nov_mismatch,
        "novelty_tolerance": 0.03,
        "mean_structured_candidate_loss": mean_loss_s,
        "mean_unstructured_candidate_loss": mean_loss_u,
        "candidate_loss_mismatch": loss_mismatch,
        "candidate_loss_tolerance": 0.10,
        "match_pass": match_pass,
        "mean_structured_benefit": float(conversion.structured_benefit.mean()),
        "mean_unstructured_benefit": float(conversion.unstructured_benefit.mean()),
        "mean_conversion_contrast": float(vals.mean()),
        "conversion_se": se,
        "conversion_t": t,
        "conversion_p_one_sided": p,
        "conversion_pass": conversion_pass,
        "decision": decision,
    }])

    train_summary.to_csv(out / "matched_novelty_confirm_training30.csv", index=False)
    outcomes.to_csv(out / "matched_novelty_family_deltas30.csv", index=False)
    conversion.to_csv(out / "matched_novelty_conversion30.csv", index=False)
    decision_df.to_csv(out / "matched_novelty_decision30.csv", index=False)
    print(decision_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
