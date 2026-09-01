from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_REPS = tuple(range(550, 560))
GRID = (0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90)


def load_parts(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("highdim_calibration_*.csv"))
    if not files:
        raise FileNotFoundError("no highdim calibration files")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def summarize(frame: pd.DataFrame) -> tuple[float,float]:
    novelty = []
    losses = []
    for rep in EXPECTED_REPS:
        part = frame[frame.rep == rep].set_index("method")
        novelty.append(float(
            part.loc["gradnov","selected_pairwise_novelty"]
            - part.loc["loss_hard","selected_pairwise_novelty"]
        ))
        losses.append(float(part["mean_candidate_loss"].mean()))
    return float(np.mean(novelty)), float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_parts(Path(a.input_dir))
    expected = len(EXPECTED_REPS) * (1 + len(GRID)) * 2
    if len(df) != expected:
        raise ValueError(f"expected {expected} rows got {len(df)}")
    structured = df[df.family == "structured"]
    if len(structured) != len(EXPECTED_REPS)*2:
        raise ValueError("structured grid incomplete")
    s_nov, s_loss = summarize(structured)
    rows=[]
    for severity in GRID:
        part = df[(df.family=="highdim") & np.isclose(df.severity,severity)]
        if len(part) != len(EXPECTED_REPS)*2:
            raise ValueError(f"severity {severity} incomplete")
        u_nov,u_loss=summarize(part)
        nm=abs(u_nov-s_nov)
        lm=abs(u_loss-s_loss)
        rows.append({
            "severity":severity,
            "structured_novelty_gain":s_nov,
            "highdim_novelty_gain":u_nov,
            "novelty_mismatch":nm,
            "structured_candidate_loss":s_loss,
            "highdim_candidate_loss":u_loss,
            "candidate_loss_mismatch":lm,
            "calibration_score":nm/0.03 + lm/0.10,
        })
    summary=pd.DataFrame(rows).sort_values(["calibration_score","severity"]).reset_index(drop=True)
    best=summary.iloc[0]
    passed=bool(best.novelty_mismatch <= 0.03 and best.candidate_loss_mismatch <= 0.10)
    decision=pd.DataFrame([{
        "selected_severity":float(best.severity),
        "calibration_novelty_mismatch":float(best.novelty_mismatch),
        "novelty_tolerance":0.03,
        "calibration_candidate_loss_mismatch":float(best.candidate_loss_mismatch),
        "candidate_loss_tolerance":0.10,
        "calibration_pass":passed,
        "heldout_used":False,
        "decision":"CALIBRATION GATE PASS" if passed else "CALIBRATION MATCH FAILURE",
    }])
    summary.to_csv(out/"highdim_calibration_summary.csv",index=False)
    decision.to_csv(out/"highdim_calibration_decision.csv",index=False)
    print(decision.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
