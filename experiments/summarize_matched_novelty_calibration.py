from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_REPS = tuple(range(450, 460))
LAMBDA_GRID = (0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10)


def load_parts(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.rglob("matched_novelty_calibration_*.csv"))
    if not files:
        raise FileNotFoundError("no calibration csv files")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def summarize_condition(frame: pd.DataFrame) -> tuple[float, float]:
    gains = []
    loss_levels = []
    for rep in EXPECTED_REPS:
        part = frame[frame.rep == rep].set_index("method")
        gains.append(float(
            part.loc["gradnov", "selected_pairwise_novelty"]
            - part.loc["loss_hard", "selected_pairwise_novelty"]
        ))
        loss_levels.append(float(part["mean_candidate_loss"].mean()))
    return float(np.mean(gains)), float(np.mean(loss_levels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_parts(inp)
    expected_rows = len(EXPECTED_REPS) * (1 + len(LAMBDA_GRID)) * 2
    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} calibration rows, got {len(df)}")

    structured = df[df.family == "structured"]
    if len(structured) != len(EXPECTED_REPS) * 2:
        raise ValueError("structured calibration grid incomplete")
    s_nov, s_loss = summarize_condition(structured)

    rows = []
    for severity in LAMBDA_GRID:
        part = df[(df.family == "unstructured") & np.isclose(df.severity, severity)]
        if len(part) != len(EXPECTED_REPS) * 2:
            raise ValueError(f"unstructured severity {severity} incomplete")
        u_nov, u_loss = summarize_condition(part)
        nov_mismatch = abs(u_nov - s_nov)
        loss_mismatch = abs(u_loss - s_loss)
        score = nov_mismatch / 0.03 + loss_mismatch / 0.10
        rows.append({
            "severity": severity,
            "structured_novelty_gain": s_nov,
            "unstructured_novelty_gain": u_nov,
            "novelty_mismatch": nov_mismatch,
            "structured_candidate_loss": s_loss,
            "unstructured_candidate_loss": u_loss,
            "candidate_loss_mismatch": loss_mismatch,
            "calibration_score": score,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["calibration_score", "severity"]
    ).reset_index(drop=True)
    selected = summary.iloc[0]
    decision = pd.DataFrame([{
        "selected_severity": float(selected.severity),
        "calibration_score": float(selected.calibration_score),
        "calibration_novelty_mismatch": float(selected.novelty_mismatch),
        "calibration_candidate_loss_mismatch": float(selected.candidate_loss_mismatch),
        "heldout_used": False,
    }])
    summary.to_csv(out / "matched_novelty_calibration_summary.csv", index=False)
    decision.to_csv(out / "matched_novelty_calibration_decision.csv", index=False)
    print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
