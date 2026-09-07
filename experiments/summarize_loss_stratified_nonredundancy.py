"""Frozen Issue #80 decision for shared-reference nonredundancy intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPS = tuple(range(1600, 1630))
ARMS = ("maxnov", "minnov")
MARGIN = 0.10
PROTOCOL = {
    "issue": 80,
    "reps": REPS,
    "arms": ARMS,
    "K": 16,
    "Q": 4,
    "batch": 128,
    "epochs": 10,
    "lr": 5e-3,
    "wd": 1e-3,
    "train_seeds": tuple(range(51000, 51064)),
    "heldout_seeds": tuple(range(52000, 52080)),
    "base_seed_offset": 2010000000,
    "base_seed_stride": 4099,
    "hardness_pairs": ((2, 3), (4, 5), (6, 7)),
    "hardness_margin_sd": MARGIN,
    "reference_update": "loss_hard_top4",
    "evaluation": "canonical_nontraining_union_809",
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_archives(root: Path) -> None:
    arm_manifests = sorted(root.rglob("arm_manifest.json"))
    if len(arm_manifests) != 6:
        raise ValueError(f"expected 6 arm manifests, got {len(arm_manifests)}")
    covered = []
    source0 = None
    split0 = None
    for apath in arm_manifests:
        shard = apath.parent
        arm = json.loads(apath.read_text())
        rpath = shard / "reference_manifest.json"
        if not rpath.exists():
            raise ValueError("missing reference manifest")
        ref = json.loads(rpath.read_text())
        if arm.get("protocol_hash") != PROTOCOL_HASH or ref.get("protocol_hash") != PROTOCOL_HASH:
            raise ValueError("protocol hash mismatch")
        if (arm.get("start"), arm.get("end")) != (ref.get("start"), ref.get("end")):
            raise ValueError("reference/arm range mismatch")
        covered.extend(range(int(arm["start"]), int(arm["end"])))
        if arm.get("source_hashes") != ref.get("source_hashes") or arm.get("split_hashes") != ref.get("split_hashes"):
            raise ValueError("reference/arm source or split mismatch")
        if source0 is None:
            source0 = ref["source_hashes"]
            split0 = ref["split_hashes"]
        elif ref["source_hashes"] != source0 or ref["split_hashes"] != split0:
            raise ValueError("cross-shard source/split mismatch")
        for name, digest in ref["schedule_hashes"].items():
            path = shard / "schedules" / name
            if not path.exists() or sha256(path) != digest:
                raise ValueError(f"schedule hash mismatch: {name}")
        for name, digest in arm["checkpoint_hashes"].items():
            path = shard / "states" / name
            if not path.exists() or sha256(path) != digest:
                raise ValueError(f"checkpoint hash mismatch: {name}")
        for name, digest in ref["source_hashes"].items():
            path = shard / "source" / name
            if not path.exists() or sha256(path) != digest:
                raise ValueError(f"archived source hash mismatch: {name}")
    if sorted(covered) != list(REPS):
        raise ValueError("manifest replicate coverage mismatch")


def estimate(x: np.ndarray, level: float = 0.95) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p1 = 0.0 if mean > 0 else (0.5 if mean == 0 else 1.0)
        lo = hi = mean
    else:
        t = mean / se
        p1 = float(stats.t.sf(t, n - 1))
        crit = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
        lo, hi = mean - crit * se, mean + crit * se
    return {
        "n": n,
        "mean": mean,
        "se": se,
        f"ci{int(level*100)}_low": float(lo),
        f"ci{int(level*100)}_high": float(hi),
        "p_one_sided_positive": p1,
        "positive_pairs": int((x > 0).sum()),
    }


def tost(x: np.ndarray, margin: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        lo90 = hi90 = mean
    else:
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = float(stats.t.sf(t_lower, n - 1))
        p_upper = float(stats.t.cdf(t_upper, n - 1))
        crit90 = float(stats.t.ppf(0.95, n - 1))
        lo90, hi90 = mean - crit90 * se, mean + crit90 * se
    passed = bool(lo90 > -margin and hi90 < margin and p_lower < 0.05 and p_upper < 0.05)
    return {
        "mean": mean,
        "se": se,
        "ci90_low": float(lo90),
        "ci90_high": float(hi90),
        "margin": margin,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "pass": passed,
    }


def load_all(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    root = Path(a.input_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_archives(root)

    steps = load_all(root, "nonredundancy_reference_steps_*.csv.gz")
    arm_train = load_all(root, "nonredundancy_arm_training_*.csv")
    held = load_all(root, "nonredundancy_heldout_*.csv")

    expected_steps = set(REPS)
    if set(steps.rep.astype(int)) != expected_steps:
        raise ValueError("reference replicate coverage")
    counts = steps.groupby("rep").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 80:
        raise ValueError(f"reference step count mismatch: {counts.to_dict()}")
    if not np.isfinite(
        steps[["novelty_diff", "hardness_diff", "raw_selected_loss_diff", "schedule_overlap"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("nonfinite reference diagnostics")
    if (steps.novelty_diff < -1e-12).any():
        raise ValueError("MAXNOV below MINNOV in reference schedule")
    if ((steps.schedule_overlap < 0.25 - 1e-12) | (steps.schedule_overlap > 1.0 + 1e-12)).any():
        raise ValueError("schedule overlap outside feasible bounds")

    arm_expected = {(r, arm) for r in REPS for arm in ARMS}
    if (
        len(arm_train) != 60
        or set(zip(arm_train.rep.astype(int), arm_train.arm.astype(str))) != arm_expected
        or arm_train.duplicated(["rep", "arm"]).any()
    ):
        raise ValueError("arm training grid")
    if not np.isfinite(arm_train[["mean_arm_selected_loss"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite arm training metrics")
    if (
        len(held) != 60
        or set(zip(held.rep.astype(int), held.arm.astype(str))) != arm_expected
        or held.duplicated(["rep", "arm"]).any()
    ):
        raise ValueError("heldout grid")
    metric_cols = ["mean_test", "sd_test", "p10_test", "min_test", "clean_test"]
    train_digest = arm_train.set_index(["rep", "arm"])["state_digest"]
    held_digest = held.set_index(["rep", "arm"])["state_digest"]
    if not train_digest.sort_index().equals(held_digest.sort_index()):
        raise ValueError("training/evaluation state digest mismatch")
    if not np.isfinite(held[metric_cols].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite heldout metrics")

    ref = steps.groupby("rep", as_index=False).agg(
        novelty_diff=("novelty_diff", "mean"),
        hardness_diff=("hardness_diff", "mean"),
        raw_selected_loss_diff=("raw_selected_loss_diff", "mean"),
        schedule_overlap=("schedule_overlap", "mean"),
    )
    pivot = held.pivot(index="rep", columns="arm")
    paired = ref.set_index("rep")
    paired["heldout_benefit"] = pivot["mean_test"]["maxnov"] - pivot["mean_test"]["minnov"]
    paired["clean_benefit"] = pivot["clean_test"]["maxnov"] - pivot["clean_test"]["minnov"]
    paired["p10_benefit"] = pivot["p10_test"]["maxnov"] - pivot["p10_test"]["minnov"]
    paired["min_benefit"] = pivot["min_test"]["maxnov"] - pivot["min_test"]["minnov"]
    paired = paired.reset_index()

    novelty = estimate(paired.novelty_diff.to_numpy())
    performance = estimate(paired.heldout_benefit.to_numpy())
    hardness = tost(paired.hardness_diff.to_numpy(), MARGIN)
    novelty_pass = bool(novelty["mean"] > 0 and novelty["p_one_sided_positive"] < 0.05)
    performance_pass = bool(performance["mean"] > 0 and performance["p_one_sided_positive"] < 0.05)

    if not novelty_pass:
        decision = "NOVELTY MANIPULATION FAIL"
    elif not hardness["pass"]:
        decision = "HARDNESS EQUIVALENCE FAIL / CONFOUNDED"
    elif performance_pass:
        decision = "LOSS-STRATIFIED NONREDUNDANCY SUPPORT"
    else:
        decision = "DIRECT NONREDUNDANCY TEST FAIL"

    decision_row = {
        "decision": decision,
        "n": 30,
        "novelty_manipulation_pass": novelty_pass,
        "hardness_equivalence_pass": hardness["pass"],
        "performance_pass": performance_pass,
        "mean_novelty_diff": novelty["mean"],
        "novelty_se": novelty["se"],
        "novelty_ci95_low": novelty["ci95_low"],
        "novelty_ci95_high": novelty["ci95_high"],
        "novelty_p_one_sided": novelty["p_one_sided_positive"],
        "mean_hardness_diff": hardness["mean"],
        "hardness_se": hardness["se"],
        "hardness_ci90_low": hardness["ci90_low"],
        "hardness_ci90_high": hardness["ci90_high"],
        "hardness_margin": hardness["margin"],
        "hardness_tost_p_lower": hardness["p_lower"],
        "hardness_tost_p_upper": hardness["p_upper"],
        "mean_heldout_benefit": performance["mean"],
        "heldout_benefit_se": performance["se"],
        "heldout_benefit_ci95_low": performance["ci95_low"],
        "heldout_benefit_ci95_high": performance["ci95_high"],
        "heldout_benefit_p_one_sided": performance["p_one_sided_positive"],
        "positive_heldout_pairs": performance["positive_pairs"],
        "mean_schedule_overlap": float(paired.schedule_overlap.mean()),
        "mean_raw_selected_loss_diff": float(paired.raw_selected_loss_diff.mean()),
        "mean_clean_benefit_secondary": float(paired.clean_benefit.mean()),
        "mean_p10_benefit_secondary": float(paired.p10_benefit.mean()),
        "mean_min_benefit_secondary": float(paired.min_benefit.mean()),
    }

    paired.to_csv(out / "nonredundancy_paired30.csv", index=False)
    ref.to_csv(out / "nonredundancy_reference_summary30.csv", index=False)
    pd.DataFrame([decision_row]).to_csv(out / "nonredundancy_decision30.csv", index=False)
    print(pd.DataFrame([decision_row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
