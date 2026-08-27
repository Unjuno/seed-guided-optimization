from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

METHODS = {"loss4", "gradnov4"}


def load_inputs(root: Path, pattern: str, start: int, end: int) -> pd.DataFrame:
    files = sorted(root.glob(pattern))
    files = [p for p in files if not p.stem.endswith("_paired")]
    if not files:
        raise FileNotFoundError(f"no shard CSVs found under {root} for {pattern}")
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame = frame.sort_values(["rep", "method"]).reset_index(drop=True)
    expected_reps = list(range(start, end))
    reps = sorted(frame.rep.unique().tolist())
    if reps != expected_reps:
        raise ValueError(f"expected reps {expected_reps}, got {reps}")
    if len(frame) != 2 * len(expected_reps):
        raise ValueError(f"expected {2 * len(expected_reps)} rows, got {len(frame)}")
    methods = set(frame.method.unique().tolist())
    if methods != METHODS:
        raise ValueError(f"expected methods {METHODS}, got {methods}")
    per_rep_methods = frame.groupby("rep").method.apply(lambda x: set(x.tolist()))
    if not per_rep_methods.apply(lambda x: x == METHODS).all():
        raise ValueError("each replicate must contain exactly loss4 and gradnov4")
    counts = frame.groupby(["rep", "method"]).size()
    if not (counts == 1).all():
        raise ValueError("duplicate method rows")
    return frame


def pair(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    base = frame[frame.method == "loss4"].set_index("rep")
    novel = frame[frame.method == "gradnov4"].set_index("rep")
    rows = []
    for rep in range(start, end):
        rows.append(
            {
                "rep": rep,
                "delta_rep_eff_rank": float(novel.loc[rep, "rep_eff_rank"] - base.loc[rep, "rep_eff_rank"]),
                "delta_mean_test": float(novel.loc[rep, "mean_test"] - base.loc[rep, "mean_test"]),
                "delta_p10_test": float(novel.loc[rep, "p10_test"] - base.loc[rep, "p10_test"]),
                "delta_min_test": float(novel.loc[rep, "min_test"] - base.loc[rep, "min_test"]),
                "delta_clean_test": float(novel.loc[rep, "clean_test"] - base.loc[rep, "clean_test"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, paired: pd.DataFrame, tolerance: float) -> dict:
    rank_delta = float(paired.delta_rep_eff_rank.mean())
    mean_delta = float(paired.delta_mean_test.mean())
    if abs(rank_delta) < tolerance:
        predicted = "UNCERTAIN"
        decision = "UNCERTAIN"
    else:
        predicted = "positive" if rank_delta > 0 else "nonpositive"
        observed = "positive" if mean_delta > 0 else "nonpositive"
        decision = "PASS" if predicted == observed else "FAIL"

    base = frame[frame.method == "loss4"].sort_values("rep")
    novel = frame[frame.method == "gradnov4"].sort_values("rep")
    p_rank = float(ttest_rel(novel.rep_eff_rank, base.rep_eff_rank).pvalue)
    p_mean = float(ttest_rel(novel.mean_test, base.mean_test).pvalue)
    n = len(paired)
    return {
        "n": n,
        "mean_delta_rep_eff_rank": rank_delta,
        "rank_practical_tolerance": tolerance,
        "registered_prediction": predicted,
        "mean_delta_mean_test": mean_delta,
        "observed_mean_direction": "positive" if mean_delta > 0 else "nonpositive",
        "decision": decision,
        "p_rank_unadjusted": p_rank,
        "p_mean_unadjusted": p_mean,
        "rank_delta_se": float(paired.delta_rep_eff_rank.std(ddof=1) / np.sqrt(n)),
        "mean_delta_se": float(paired.delta_mean_test.std(ddof=1) / np.sqrt(n)),
    }


def main(args):
    root = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_inputs(root, args.pattern, args.start, args.end)
    paired = pair(frame, args.start, args.end)
    summary = summarize(frame, paired, args.tolerance)
    n = args.end - args.start
    frame.to_csv(out / f"{args.prefix}_all{n}.csv", index=False)
    paired.to_csv(out / f"{args.prefix}_deltas{n}.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / f"{args.prefix}_decision{n}.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--tolerance", type=float, default=0.01)
    main(p.parse_args())
