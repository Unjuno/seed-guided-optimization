from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_REPS = tuple(range(100, 130))
Q_VALUES = (2, 4, 8, 12, 16)
METHODS = ("loss_hard", "gradnov")
DIAGNOSTIC_METRICS = (
    "rep_eff_rank",
    "selected_pairwise_novelty",
    "mean_selected_loss",
    "mean_candidate_loss",
)
HELDOUT_METRICS = (
    "mean_test",
    "sd_test",
    "p10_test",
    "min_test",
    "clean_test",
)
SCIENTIFIC_METRICS = DIAGNOSTIC_METRICS + HELDOUT_METRICS


def one_sided_positive_ttest(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return float("nan"), float("nan")
    if np.all(x == x[0]):
        if x[0] > 0:
            return float("inf"), 0.0
        if x[0] < 0:
            return float("-inf"), 1.0
        return 0.0, 0.5
    result = stats.ttest_1samp(x, popmean=0.0)
    t = float(result.statistic)
    p2 = float(result.pvalue)
    p1 = p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0
    return t, p1


def exact_scalar_equal(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        aa = np.asarray([a], dtype=np.float64).view(np.uint64)[0]
        bb = np.asarray([b], dtype=np.float64).view(np.uint64)[0]
        return bool(aa == bb)
    return bool(a == b)


def load_parts(input_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(input_dir.rglob(prefix + "*.csv"))
    if not files:
        raise FileNotFoundError(f"no files matching {prefix}*.csv in {input_dir}")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def validate_grid(frame: pd.DataFrame, label: str) -> None:
    expected = {
        (rep, method, q)
        for rep in EXPECTED_REPS
        for q in Q_VALUES
        for method in METHODS
    }
    observed = set(
        zip(
            frame["rep"].astype(int),
            frame["method"].astype(str),
            frame["q"].astype(int),
        )
    )
    if len(frame) != len(expected):
        raise ValueError(f"{label}: expected {len(expected)} rows, got {len(frame)}")
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise ValueError(f"{label}: grid mismatch missing={missing} extra={extra}")
    if frame.duplicated(["rep", "method", "q"]).any():
        raise ValueError(f"{label}: duplicate rep/method/q rows")


def build_deltas(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for rep in EXPECTED_REPS:
        for q in Q_VALUES:
            part = all_rows[(all_rows.rep == rep) & (all_rows.q == q)].set_index("method")
            row = {
                "rep": rep,
                "q": q,
                "coverage_ratio": q / 16.0,
            }
            for metric in SCIENTIFIC_METRICS:
                row["delta_" + metric] = float(
                    part.loc["gradnov", metric] - part.loc["loss_hard", metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def q16_identity(all_rows: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    mismatches: list[dict] = []
    compare_fields = (
        "candidate_k",
        "coverage_ratio",
        "probe_size",
        *SCIENTIFIC_METRICS,
    )
    for rep in EXPECTED_REPS:
        part = all_rows[(all_rows.rep == rep) & (all_rows.q == 16)].set_index("method")
        for field in compare_fields:
            a = part.loc["loss_hard", field]
            b = part.loc["gradnov", field]
            if not exact_scalar_equal(a, b):
                mismatches.append(
                    {
                        "rep": rep,
                        "field": field,
                        "loss_hard": a,
                        "gradnov": b,
                    }
                )
    return len(mismatches) == 0, pd.DataFrame(mismatches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = load_parts(input_dir, "budget_coverage_diagnostics_")
    heldout = load_parts(input_dir, "budget_coverage_heldout_")
    validate_grid(diagnostics, "diagnostics")
    validate_grid(heldout, "heldout")

    keys = ["rep", "method", "q", "candidate_k", "coverage_ratio"]
    all_rows = diagnostics.merge(
        heldout,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).sort_values(["rep", "q", "method"]).reset_index(drop=True)
    validate_grid(all_rows, "merged")

    deltas = build_deltas(all_rows)

    attenuation_rows: list[dict] = []
    for rep in EXPECTED_REPS:
        part = deltas[deltas.rep == rep].set_index("q")
        b_low = float(part.loc[[2, 4], "delta_mean_test"].mean())
        b_high = float(part.loc[[12, 16], "delta_mean_test"].mean())
        r_low = float(part.loc[[2, 4], "delta_rep_eff_rank"].mean())
        r_high = float(part.loc[[12, 16], "delta_rep_eff_rank"].mean())
        x = part.loc[list(Q_VALUES), "coverage_ratio"].to_numpy(dtype=np.float64)
        b = part.loc[list(Q_VALUES), "delta_mean_test"].to_numpy(dtype=np.float64)
        r = part.loc[list(Q_VALUES), "delta_rep_eff_rank"].to_numpy(dtype=np.float64)
        attenuation_rows.append(
            {
                "rep": rep,
                "benefit_low": b_low,
                "benefit_high": b_high,
                "benefit_attenuation": b_low - b_high,
                "rank_low": r_low,
                "rank_high": r_high,
                "rank_attenuation": r_low - r_high,
                "benefit_slope_vs_coverage": float(np.polyfit(x, b, 1)[0]),
                "rank_slope_vs_coverage": float(np.polyfit(x, r, 1)[0]),
            }
        )
    attenuation = pd.DataFrame(attenuation_rows)

    t_b, p_b = one_sided_positive_ttest(
        attenuation["benefit_attenuation"].to_numpy()
    )
    t_r, p_r = one_sided_positive_ttest(
        attenuation["rank_attenuation"].to_numpy()
    )
    coverage_pass = bool(
        attenuation["benefit_attenuation"].mean() > 0 and p_b < 0.05
    )
    representation_pass = bool(
        attenuation["rank_attenuation"].mean() > 0 and p_r < 0.05
    )

    identity_pass, identity_mismatches = q16_identity(all_rows)

    if not identity_pass:
        decision = "INVALID / REPRO FAILURE"
    elif not coverage_pass:
        decision = "THEORY FAIL"
    elif representation_pass:
        decision = "STRONG THEORY PASS"
    else:
        decision = "PARTIAL PASS"

    if len(attenuation) >= 3 and attenuation["benefit_attenuation"].std(ddof=1) > 0 and attenuation["rank_attenuation"].std(ddof=1) > 0:
        corr = stats.pearsonr(
            attenuation["benefit_attenuation"], attenuation["rank_attenuation"]
        )
        attenuation_corr = float(corr.statistic)
        attenuation_corr_p = float(corr.pvalue)
    else:
        attenuation_corr = float("nan")
        attenuation_corr_p = float("nan")

    q_summary_rows: list[dict] = []
    for q in Q_VALUES:
        part = deltas[deltas.q == q]
        row = {"q": q, "coverage_ratio": q / 16.0, "n": len(part)}
        for metric in (
            "delta_mean_test",
            "delta_rep_eff_rank",
            "delta_selected_pairwise_novelty",
            "delta_p10_test",
            "delta_min_test",
            "delta_clean_test",
        ):
            vals = part[metric].to_numpy(dtype=np.float64)
            row["mean_" + metric] = float(vals.mean())
            row["se_" + metric] = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        q_summary_rows.append(row)
    q_summary = pd.DataFrame(q_summary_rows)

    decision_row = pd.DataFrame(
        [
            {
                "n": len(attenuation),
                "identity_pass": identity_pass,
                "identity_mismatch_count": len(identity_mismatches),
                "mean_benefit_low": float(attenuation["benefit_low"].mean()),
                "mean_benefit_high": float(attenuation["benefit_high"].mean()),
                "mean_benefit_attenuation": float(
                    attenuation["benefit_attenuation"].mean()
                ),
                "benefit_attenuation_t": t_b,
                "benefit_attenuation_p_one_sided": p_b,
                "coverage_pass": coverage_pass,
                "mean_rank_low": float(attenuation["rank_low"].mean()),
                "mean_rank_high": float(attenuation["rank_high"].mean()),
                "mean_rank_attenuation": float(
                    attenuation["rank_attenuation"].mean()
                ),
                "rank_attenuation_t": t_r,
                "rank_attenuation_p_one_sided": p_r,
                "representation_coupling_pass": representation_pass,
                "attenuation_pearson_r": attenuation_corr,
                "attenuation_pearson_p": attenuation_corr_p,
                "decision": decision,
            }
        ]
    )

    all_rows.to_csv(output_dir / "budget_coverage_all30.csv", index=False)
    deltas.to_csv(output_dir / "budget_coverage_deltas30.csv", index=False)
    attenuation.to_csv(output_dir / "budget_coverage_attenuation30.csv", index=False)
    q_summary.to_csv(output_dir / "budget_coverage_q_summary30.csv", index=False)
    decision_row.to_csv(output_dir / "budget_coverage_decision30.csv", index=False)
    identity_mismatches.to_csv(
        output_dir / "budget_coverage_q16_identity_mismatches.csv", index=False
    )

    print(decision_row.to_string(index=False), flush=True)
    print("\nQ summary\n" + q_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
