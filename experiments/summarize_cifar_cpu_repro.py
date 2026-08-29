from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

REPS = (45, 46)
METHODS = ("loss4", "gradnov4")
REPEATS = ("A", "B")
RANK_COLUMNS = (
    "pre_rep_eff_rank",
    "rep_eff_rank",
    "rep_eff_rank_from_pretrain",
    "rep_delta_rep_eff_rank",
)
ACCURACY_COLUMNS = (
    "mean_test",
    "sd_test",
    "p10_test",
    "min_test",
    "clean_test",
)
TIMING_COLUMNS = {"train_seconds"}
FILENAME = re.compile(r"cifar_cpu_repro_rep(45|46)_([AB])\.csv$")


def f64_bits(value: float) -> bytes:
    return struct.pack(">d", float(value))


def load(root: Path) -> dict[tuple[int, str], pd.DataFrame]:
    runs: dict[tuple[int, str], pd.DataFrame] = {}
    for path in sorted(root.rglob("cifar_cpu_repro_rep*_?.csv")):
        if path.stem.endswith("_paired"):
            continue
        m = FILENAME.search(path.name)
        if not m:
            continue
        key = (int(m.group(1)), m.group(2))
        frame = pd.read_csv(path).sort_values("method").reset_index(drop=True)
        if set(frame.method) != set(METHODS) or len(frame) != 2:
            raise ValueError(f"unexpected rows in {path}: {frame[['rep','method']].to_dict('records')}")
        if set(frame.rep.astype(int)) != {key[0]}:
            raise ValueError(f"rep mismatch in {path}")
        runs[key] = frame
    expected = {(rep, repeat) for rep in REPS for repeat in REPEATS}
    if set(runs) != expected:
        raise ValueError(f"expected {sorted(expected)}, got {sorted(runs)}")
    return runs


def compare(runs: dict[tuple[int, str], pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    details = []
    all_bitwise = True
    max_rank_drift = 0.0
    max_accuracy_drift = 0.0
    metadata_exact = True

    for rep in REPS:
        a = runs[(rep, "A")].set_index("method")
        b = runs[(rep, "B")].set_index("method")
        for method in METHODS:
            columns = [c for c in a.columns if c not in TIMING_COLUMNS]
            if columns != [c for c in b.columns if c not in TIMING_COLUMNS]:
                raise ValueError("scientific column mismatch between repeats")
            for col in columns:
                av = a.loc[method, col]
                bv = b.loc[method, col]
                if col == "method":
                    exact = str(av) == str(bv)
                    drift = 0.0 if exact else float("inf")
                elif pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                    af = float(av)
                    bf = float(bv)
                    exact = f64_bits(af) == f64_bits(bf)
                    drift = abs(af - bf)
                else:
                    exact = str(av) == str(bv)
                    drift = 0.0 if exact else float("inf")
                all_bitwise &= exact
                if col in RANK_COLUMNS:
                    max_rank_drift = max(max_rank_drift, drift)
                elif col in ACCURACY_COLUMNS:
                    max_accuracy_drift = max(max_accuracy_drift, drift)
                elif col not in TIMING_COLUMNS:
                    metadata_exact &= exact
                details.append(
                    {
                        "rep": rep,
                        "method": method,
                        "column": col,
                        "repeat_A": av,
                        "repeat_B": bv,
                        "bitwise_equal": exact,
                        "abs_drift": drift,
                    }
                )

    if all_bitwise:
        decision = "BITWISE PASS"
    elif metadata_exact and max_rank_drift <= 1e-8 and max_accuracy_drift <= 1e-8:
        decision = "NUMERIC-STABLE"
    else:
        decision = "DRIFT PERSISTS"

    direction_rows = []
    for repeat in REPEATS:
        rows = pd.concat([runs[(rep, repeat)] for rep in REPS], ignore_index=True)
        base = rows[rows.method == "loss4"].set_index("rep")
        novel = rows[rows.method == "gradnov4"].set_index("rep")
        delta_rank = float((novel.rep_eff_rank - base.rep_eff_rank).mean())
        delta_mean = float((novel.mean_test - base.mean_test).mean())
        direction_rows.append(
            {
                "repeat": repeat,
                "mean_delta_rep_eff_rank": delta_rank,
                "mean_delta_mean_test": delta_mean,
                "rank_direction": "positive" if delta_rank > 0 else "nonpositive",
                "mean_direction": "positive" if delta_mean > 0 else "nonpositive",
            }
        )
    directions = pd.DataFrame(direction_rows)
    direction_stable = (
        directions.rank_direction.nunique() == 1
        and directions.mean_direction.nunique() == 1
    )

    summary = {
        "reps": "45,46",
        "n_independent_repeats": 2,
        "torch_threads": 1,
        "bitwise_equal_all_scientific_fields": bool(all_bitwise),
        "metadata_exact": bool(metadata_exact),
        "max_abs_rank_drift": float(max_rank_drift),
        "max_abs_accuracy_drift": float(max_accuracy_drift),
        "tolerance": 1e-8,
        "decision": decision,
        "aggregate_direction_stable": bool(direction_stable),
        "repeat_A_rank_direction": directions.loc[directions.repeat == "A", "rank_direction"].iloc[0],
        "repeat_B_rank_direction": directions.loc[directions.repeat == "B", "rank_direction"].iloc[0],
        "repeat_A_mean_direction": directions.loc[directions.repeat == "A", "mean_direction"].iloc[0],
        "repeat_B_mean_direction": directions.loc[directions.repeat == "B", "mean_direction"].iloc[0],
    }
    return pd.DataFrame(details), directions, summary


def main(a: argparse.Namespace) -> None:
    root = Path(a.input_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(root)
    details, directions, summary = compare(runs)
    details.to_csv(out / "cifar_cpu_repro_comparison.csv", index=False)
    directions.to_csv(out / "cifar_cpu_repro_directions.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "cifar_cpu_repro_decision.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    main(p.parse_args())
