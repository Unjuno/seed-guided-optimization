from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

EXPECTED_REPS = list(range(40, 50))
TOLERANCE = 0.01


def load_inputs(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("cifar_resnet_rep_rank_reps*.csv"))
    files = [p for p in files if not p.stem.endswith("_paired")]
    if not files:
        raise FileNotFoundError(f"no shard CSVs found under {root}")
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame = frame.sort_values(["rep", "method"]).reset_index(drop=True)
    reps = sorted(frame.rep.unique().tolist())
    if reps != EXPECTED_REPS:
        raise ValueError(f"expected reps {EXPECTED_REPS}, got {reps}")
    counts = frame.groupby(["rep", "method"]).size()
    if not (counts == 1).all():
        raise ValueError("duplicate or missing method rows")
    return frame


def pair(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame.method == "loss4"].set_index("rep")
    novel = frame[frame.method == "gradnov4"].set_index("rep")
    rows = []
    for rep in EXPECTED_REPS:
        rows.append(
            {
                "rep": rep,
                "delta_rep_eff_rank": float(
                    novel.loc[rep, "rep_eff_rank"] - base.loc[rep, "rep_eff_rank"]
                ),
                "delta_mean_test": float(
                    novel.loc[rep, "mean_test"] - base.loc[rep, "mean_test"]
                ),
                "delta_p10_test": float(
                    novel.loc[rep, "p10_test"] - base.loc[rep, "p10_test"]
                ),
                "delta_min_test": float(
                    novel.loc[rep, "min_test"] - base.loc[rep, "min_test"]
                ),
                "delta_clean_test": float(
                    novel.loc[rep, "clean_test"] - base.loc[rep, "clean_test"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, paired: pd.DataFrame) -> dict:
    rank_delta = float(paired.delta_rep_eff_rank.mean())
    mean_delta = float(paired.delta_mean_test.mean())

    if abs(rank_delta) < TOLERANCE:
        predicted = "UNCERTAIN"
        decision = "UNCERTAIN"
    else:
        predicted = "positive" if rank_delta > 0 else "nonpositive"
        observed = "positive" if mean_delta > 0 else "nonpositive"
        decision = "PASS" if predicted == observed else "FAIL"

    base = frame[frame.method == "loss4"].set_index("rep")
    novel = frame[frame.method == "gradnov4"].set_index("rep")
    p_rank = float(ttest_rel(novel.rep_eff_rank, base.rep_eff_rank).pvalue)
    p_mean = float(ttest_rel(novel.mean_test, base.mean_test).pvalue)

    return {
        "n": len(EXPECTED_REPS),
        "mean_delta_rep_eff_rank": rank_delta,
        "rank_practical_tolerance": TOLERANCE,
        "registered_prediction": predicted,
        "mean_delta_mean_test": mean_delta,
        "observed_mean_direction": "positive" if mean_delta > 0 else "nonpositive",
        "decision": decision,
        "p_rank_unadjusted": p_rank,
        "p_mean_unadjusted": p_mean,
        "rank_delta_se": float(paired.delta_rep_eff_rank.std(ddof=1) / np.sqrt(len(paired))),
        "mean_delta_se": float(paired.delta_mean_test.std(ddof=1) / np.sqrt(len(paired))),
    }


def main(args):
    root = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_inputs(root)
    paired = pair(frame)
    summary = summarize(frame, paired)
    frame.to_csv(out / "cifar_resnet_rep_rank_all10.csv", index=False)
    paired.to_csv(out / "cifar_resnet_rep_rank_deltas10.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        out / "cifar_resnet_rep_rank_decision10.csv", index=False
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    main(p.parse_args())
