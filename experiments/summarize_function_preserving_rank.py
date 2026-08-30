from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_REPS = tuple(range(200, 220))
METHODS = ("loss_hard", "gradnov")
INTERVENTIONS = ("native", "spread", "concentrate")
METRICS = ("mean_test", "sd_test", "p10_test", "min_test", "clean_test")
RANK_UP_MIN = 0.5
RANK_DOWN_MAX = -0.5
SIGN_COUNT_MIN = 18
LOGIT_TOL = 1e-5


def read_concat(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def validate_grid(df: pd.DataFrame, name: str) -> None:
    expected = {
        (rep, method, intervention)
        for rep in EXPECTED_REPS
        for method in METHODS
        for intervention in INTERVENTIONS
    }
    actual = {
        (int(r.rep), str(r.method), str(r.intervention))
        for r in df.itertuples(index=False)
    }
    if actual != expected or len(df) != len(expected):
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise RuntimeError(
            f"{name} grid mismatch: rows={len(df)} expected={len(expected)} "
            f"missing={missing} extra={extra}"
        )


def manipulation_summary(diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = diagnostics.pivot(
        index=["rep", "method"], columns="intervention", values="rep_eff_rank"
    )
    std_pivot = diagnostics.pivot(
        index=["rep", "method"], columns="intervention", values="std_rep_eff_rank"
    )
    per_pair = pivot.reset_index()[["rep", "method"]].copy()
    per_pair["rank_native"] = pivot["native"].to_numpy()
    per_pair["rank_spread"] = pivot["spread"].to_numpy()
    per_pair["rank_concentrate"] = pivot["concentrate"].to_numpy()
    per_pair["delta_rank_spread"] = (pivot["spread"] - pivot["native"]).to_numpy()
    per_pair["delta_rank_concentrate"] = (
        pivot["concentrate"] - pivot["native"]
    ).to_numpy()
    per_pair["std_rank_native"] = std_pivot["native"].to_numpy()
    per_pair["std_rank_spread"] = std_pivot["spread"].to_numpy()
    per_pair["std_rank_concentrate"] = std_pivot["concentrate"].to_numpy()
    per_pair["delta_std_rank_spread"] = (
        std_pivot["spread"] - std_pivot["native"]
    ).to_numpy()
    per_pair["delta_std_rank_concentrate"] = (
        std_pivot["concentrate"] - std_pivot["native"]
    ).to_numpy()

    rows = []
    for method in METHODS:
        sub = per_pair[per_pair["method"] == method]
        mean_up = float(sub["delta_rank_spread"].mean())
        mean_down = float(sub["delta_rank_concentrate"].mean())
        up_sign = int((sub["delta_rank_spread"] > 0).sum())
        down_sign = int((sub["delta_rank_concentrate"] < 0).sum())
        method_pass = (
            mean_up >= RANK_UP_MIN
            and mean_down <= RANK_DOWN_MAX
            and up_sign >= SIGN_COUNT_MIN
            and down_sign >= SIGN_COUNT_MIN
        )
        rows.append(
            {
                "method": method,
                "n": len(sub),
                "mean_delta_rank_spread": mean_up,
                "spread_positive_count": up_sign,
                "mean_delta_rank_concentrate": mean_down,
                "concentrate_negative_count": down_sign,
                "mean_delta_std_rank_spread": float(
                    sub["delta_std_rank_spread"].mean()
                ),
                "mean_delta_std_rank_concentrate": float(
                    sub["delta_std_rank_concentrate"].mean()
                ),
                "manipulation_pass_method": bool(method_pass),
            }
        )
    return per_pair, pd.DataFrame(rows)


def function_identity(heldout: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    native = heldout[heldout["intervention"] == "native"].set_index(["rep", "method"])
    rows = []
    for intervention in ("spread", "concentrate"):
        sub = heldout[heldout["intervention"] == intervention].set_index(["rep", "method"])
        for key, row in sub.iterrows():
            ref = native.loc[key]
            metric_exact = all(float(row[m]) == float(ref[m]) for m in METRICS)
            pred_exact = bool(row["predictions_identical_to_native"])
            logit_diff = float(row["max_abs_logit_diff"])
            pass_row = pred_exact and metric_exact and logit_diff <= LOGIT_TOL
            rows.append(
                {
                    "rep": int(key[0]),
                    "method": str(key[1]),
                    "intervention": intervention,
                    "predictions_identical": pred_exact,
                    "metrics_exact": bool(metric_exact),
                    "max_abs_logit_diff": logit_diff,
                    "function_identity_pass_row": bool(pass_row),
                }
            )
    frame = pd.DataFrame(rows)
    summary = {
        "n_intervention_rows": len(frame),
        "prediction_mismatch_rows": int((~frame["predictions_identical"]).sum()),
        "metric_mismatch_rows": int((~frame["metrics_exact"]).sum()),
        "max_abs_logit_diff": float(frame["max_abs_logit_diff"].max()),
        "function_identity_pass": bool(frame["function_identity_pass_row"].all()),
    }
    return frame, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = read_concat(input_dir, "rank_intervention_diagnostics_*.csv")
    heldout = read_concat(input_dir, "rank_intervention_heldout_*.csv")
    validate_grid(diagnostics, "diagnostics")
    validate_grid(heldout, "heldout")

    diagnostics = diagnostics.sort_values(["rep", "method", "intervention"]).reset_index(drop=True)
    heldout = heldout.sort_values(["rep", "method", "intervention"]).reset_index(drop=True)

    per_pair, manip = manipulation_summary(diagnostics)
    identity_rows, identity = function_identity(heldout)

    manipulation_pass = bool(manip["manipulation_pass_method"].all())
    identity_pass = bool(identity["function_identity_pass"])
    if not identity_pass:
        decision = "REPARAMETERIZATION FAILURE"
    elif manipulation_pass:
        decision = "COORDINATE-DEPENDENCE PASS"
    else:
        decision = "INCONCLUSIVE / WEAK MANIPULATION"

    decision_row = {
        "n_reps": len(EXPECTED_REPS),
        "manipulation_pass": manipulation_pass,
        **identity,
        "rank_up_min": RANK_UP_MIN,
        "rank_down_max": RANK_DOWN_MAX,
        "sign_count_min": SIGN_COUNT_MIN,
        "logit_tolerance": LOGIT_TOL,
        "decision": decision,
    }

    diagnostics.to_csv(output_dir / "function_preserving_rank_all_diagnostics20.csv", index=False)
    heldout.to_csv(output_dir / "function_preserving_rank_all_heldout20.csv", index=False)
    per_pair.to_csv(output_dir / "function_preserving_rank_deltas20.csv", index=False)
    manip.to_csv(output_dir / "function_preserving_rank_manipulation_summary20.csv", index=False)
    identity_rows.to_csv(output_dir / "function_preserving_rank_identity_rows20.csv", index=False)
    pd.DataFrame([decision_row]).to_csv(
        output_dir / "function_preserving_rank_decision20.csv", index=False
    )

    print(pd.DataFrame([decision_row]).to_string(index=False))
    print("\nManipulation summary:\n" + manip.to_string(index=False))


if __name__ == "__main__":
    main()
