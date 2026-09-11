"""Issue96 Stage A: calibrate gradient-span separation at matched mean novelty.

Training-reference-only calibration. No intervention arm or heldout environment is
constructed in this stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import parameter_matched_novelty_calibration as cal
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS = tuple(range(2300, 2310))
SEEDS = tuple(range(65000, 65064))
G_GRID = (0.005, 0.010, 0.020, 0.050)
H_CALIPER = 0.05
P_CALIPER = 0.05
T_CALIPER = 0.20
SLACK = 1e-12
MIN_POOLED_SUPPORT = 0.90
MIN_REP_SUPPORT = 0.80
MIN_MEAN_SPAN_GAP = 0.20
MIN_REP_MEAN_SPAN_GAP = 0.10
NEG_EIG_TOL = 1e-10
OFFSET = 2710000000

PROTOCOL = {
    "issue": 96,
    "stage": "A",
    "reps": REPS,
    "train_seeds": SEEDS,
    "offset": OFFSET,
    "stride": 4099,
    "K": 16,
    "Q": 4,
    "steps": 80,
    "epochs": 10,
    "batch": 128,
    "lr": 0.005,
    "wd": 0.001,
    "h_caliper": H_CALIPER,
    "p_caliper": P_CALIPER,
    "t_caliper": T_CALIPER,
    "g_grid": G_GRID,
    "min_pooled_support": MIN_POOLED_SUPPORT,
    "min_rep_support": MIN_REP_SUPPORT,
    "min_mean_span_gap": MIN_MEAN_SPAN_GAP,
    "min_rep_mean_span_gap": MIN_REP_MEAN_SPAN_GAP,
    "span_metric": "spectral_effective_rank_of_normalized_head_gradient_gram",
    "negative_eigenvalue_tolerance": NEG_EIG_TOL,
    "threads": 1,
}
PH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
SOURCES = tuple(dict.fromkeys((*tc.SOURCES, "gradient_span_calibration.py")))


def source_hashes() -> dict[str, str]:
    return {name: fd.sha256(Path(__file__).parent / name) for name in SOURCES}


def spectral_effective_rank(v: torch.Tensor) -> float:
    """Entropy effective rank of the row Gram spectrum, computed in float64."""
    a = v.detach().double().cpu().numpy()
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[0] < 2:
        raise ValueError("invalid signature matrix")
    gram = a @ a.T
    eig = np.linalg.eigvalsh(gram)
    if float(eig.min()) < -NEG_EIG_TOL:
        raise ValueError(f"Gram matrix has materially negative eigenvalue {eig.min()}")
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("degenerate Gram spectrum")
    p = eig / total
    pos = p > 0
    entropy = -float(np.sum(p[pos] * np.log(p[pos])))
    out = float(np.exp(entropy))
    if not np.isfinite(out) or out < 1.0 - 1e-10 or out > a.shape[0] + 1e-10:
        raise ValueError(f"effective rank out of range: {out}")
    return out


def mean_pairwise_novelty(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    gram = v @ v.T
    ii, jj = np.triu_indices(len(v), 1)
    return float(np.mean(1.0 - gram[ii, jj]))


def add_span(rows: list[dict], gd: torch.Tensor) -> list[dict]:
    for r in rows:
        idx = torch.tensor(r["local"], dtype=torch.long)
        r["spectral_erank"] = spectral_effective_rank(gd.index_select(0, idx))
    return rows


def choose_span_pair(rows: list[dict], g_caliper: float) -> dict | None:
    fields = ("spectral_erank", "gradient_novelty", "z_hard", "z_param", "translation_score")
    a = np.asarray([[r[f] for f in fields] for r in rows], dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(fields) or not np.isfinite(a).all():
        raise ValueError("nonfinite subset table")
    if not np.isfinite(g_caliper) or g_caliper <= 0:
        raise ValueError("bad G caliper")
    ii, jj = np.triu_indices(len(rows), 1)
    diff = a[ii] - a[jj]
    keep = (
        (np.abs(diff[:, 2]) <= H_CALIPER + SLACK)
        & (np.abs(diff[:, 3]) <= P_CALIPER + SLACK)
        & (np.abs(diff[:, 4]) <= T_CALIPER + SLACK)
        & (np.abs(diff[:, 1]) <= g_caliper + SLACK)
    )
    eligible = np.flatnonzero(keep)
    if not len(eligible):
        return None
    k = min(
        eligible,
        key=lambda z: (
            -abs(diff[z, 0]),
            abs(diff[z, 1]),
            abs(diff[z, 4]),
            abs(diff[z, 3]),
            abs(diff[z, 2]),
            tuple(rows[ii[z]]["local"]),
            tuple(rows[jj[z]]["local"]),
        ),
    )
    i, j = int(ii[k]), int(jj[k])
    high, low = (i, j) if a[i, 0] >= a[j, 0] else (j, i)
    return {
        "high": high,
        "low": low,
        "eligible_pairs": int(len(eligible)),
        "span_gap": float(a[high, 0] - a[low, 0]),
        "gradient_novelty_diff": float(a[high, 1] - a[low, 1]),
        "gradient_novelty_abs_diff": float(abs(a[high, 1] - a[low, 1])),
        "hardness_diff": float(a[high, 2] - a[low, 2]),
        "parameter_z_diff": float(a[high, 3] - a[low, 3]),
        "translation_diff": float(a[high, 4] - a[low, 4]),
    }


def check_range(start: int, end: int) -> None:
    if start >= end or not set(range(start, end)).issubset(REPS):
        raise ValueError("unregistered Stage-A range")


def run(start: int, end: int, out: Path) -> None:
    check_range(start, end)
    base.configure_determinism(1)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "calibration_manifest.json").exists():
        raise ValueError("overwrite refused")
    x, y, *_ = cr.data_split()
    ty = torch.tensor(y)
    envs = base.geometric_envs(x, SEEDS)
    par = tc.parameters(SEEDS)
    pair_rows: list[dict] = []
    subset_rows: list[dict] = []

    for rep in range(start, end):
        seed = OFFSET + 4099 * rep
        sched = pm.build_schedule(len(ty), seed)
        if len(sched) != 80:
            raise ValueError("step count")
        seed_everything(seed)
        model = SmallCNN()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

        for step, (b, cand) in enumerate(sched):
            logits, h = model(torch.cat([envs[e][b] for e in cand]))
            per = torch.nn.functional.cross_entropy(
                logits, ty[b].repeat(16), reduction="none"
            ).reshape(16, -1)
            losses = per.mean(1)
            gd = head_gradient_directions(logits, h, ty[b], 16)
            subsets = tc.add_translation(
                cal.feasible_subsets(losses, gd @ gd.T, cand, par), par
            )
            subsets = add_span(subsets, gd)

            for idx, s in enumerate(subsets):
                subset_rows.append(
                    {
                        "rep": rep,
                        "step": step,
                        "subset_id": idx,
                        "local": ";".join(map(str, s["local"])),
                        "envs": ";".join(map(str, s["envs"])),
                        **{
                            f: s[f]
                            for f in (
                                "gradient_novelty",
                                "spectral_erank",
                                "z_hard",
                                "z_param",
                                "translation_score",
                                "physical_diversity",
                            )
                        },
                    }
                )

            for c in G_GRID:
                pair = choose_span_pair(subsets, c)
                pair_rows.append(
                    {
                        "rep": rep,
                        "step": step,
                        "g_caliper": c,
                        "feasible": pair is not None,
                        **(pair or {}),
                    }
                )

            selected = sorted(
                sorted(range(16), key=lambda j: (-float(losses[j].detach()), j))[:4]
            )
            opt.zero_grad(set_to_none=True)
            per[selected].mean().backward()
            opt.step()
        print(f"SPAN_CALIBRATION_REFERENCE_COMPLETE rep={rep}; no arms, no heldout", flush=True)

    p = out / f"gradient_span_pairs_{start}_{end}.csv"
    s = out / f"gradient_span_subsets_{start}_{end}.csv.gz"
    pd.DataFrame(pair_rows).to_csv(p, index=False)
    pd.DataFrame(subset_rows).to_csv(s, index=False, compression="gzip")
    manifest = {
        "protocol": PROTOCOL,
        "protocol_hash": PH,
        "start": start,
        "end": end,
        "source_hashes": source_hashes(),
        "split_hashes": cr.split_hashes(),
        "files": {q.name: fd.sha256(q) for q in (p, s)},
        "runtime": fd.runtime(),
        "heldout_constructed": False,
        "intervention_arms_trained": 0,
    }
    (out / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )


def summarize(root: Path, out: Path) -> None:
    covered: list[int] = []
    frames: list[pd.DataFrame] = []
    for mp in sorted(root.rglob("calibration_manifest.json")):
        m = json.loads(mp.read_text())
        if m["protocol_hash"] != PH or m["source_hashes"] != source_hashes():
            raise ValueError("source/protocol provenance")
        if m["split_hashes"] != cr.split_hashes():
            raise ValueError("split provenance")
        if m["heldout_constructed"] or m["intervention_arms_trained"] != 0:
            raise ValueError("Stage-A boundary violated")
        covered.extend(range(m["start"], m["end"]))
        for name, digest in m["files"].items():
            if fd.sha256(mp.parent / name) != digest:
                raise ValueError("artifact hash mismatch")
        frames.append(
            pd.read_csv(
                mp.parent / f"gradient_span_pairs_{m['start']}_{m['end']}.csv",
                float_precision="round_trip",
            )
        )
    if sorted(covered) != list(REPS):
        raise ValueError("replicate coverage")

    df = pd.concat(frames, ignore_index=True)
    keys = ["rep", "step", "g_caliper"]
    expected = {(r, t, c) for r in REPS for t in range(80) for c in G_GRID}
    observed = set(df[keys].itertuples(index=False, name=None))
    if len(df) != len(expected) or df.duplicated(keys).any() or observed != expected:
        raise ValueError("calibration grid")

    rows = []
    selected = None
    for c in G_GRID:
        v = df[np.isclose(df.g_caliper, c)].copy()
        feasible = v.feasible.astype(bool)
        valid = v[feasible]
        cols = [
            "span_gap",
            "gradient_novelty_abs_diff",
            "hardness_diff",
            "parameter_z_diff",
            "translation_diff",
        ]
        if len(valid) and not np.isfinite(valid[cols].to_numpy(dtype=np.float64)).all():
            raise ValueError("nonfinite selected pair")
        if (
            (valid.hardness_diff.abs() > H_CALIPER + SLACK).any()
            or (valid.parameter_z_diff.abs() > P_CALIPER + SLACK).any()
            or (valid.translation_diff.abs() > T_CALIPER + SLACK).any()
            or (valid.gradient_novelty_abs_diff > c + SLACK).any()
        ):
            raise ValueError("caliper violation")
        if (valid.span_gap < -SLACK).any():
            raise ValueError("span orientation")

        support_by_rep = v.groupby("rep").feasible.mean().reindex(REPS, fill_value=0.0)
        span_by_rep = (
            valid.groupby("rep").span_gap.mean().reindex(REPS, fill_value=0.0)
        )
        pooled = float(feasible.mean())
        min_support = float(support_by_rep.min())
        mean_span = float(valid.span_gap.mean()) if len(valid) else 0.0
        min_rep_span = float(span_by_rep.min())
        passed = bool(
            pooled >= MIN_POOLED_SUPPORT
            and min_support >= MIN_REP_SUPPORT
            and mean_span >= MIN_MEAN_SPAN_GAP
            and min_rep_span >= MIN_REP_MEAN_SPAN_GAP
        )
        row = {
            "g_caliper": c,
            "supported_steps": int(feasible.sum()),
            "total_steps": len(v),
            "pooled_support": pooled,
            "minimum_replicate_support": min_support,
            "mean_span_gap_supported": mean_span,
            "minimum_replicate_mean_span_gap_supported": min_rep_span,
            "mean_abs_gradient_novelty_diff_supported": float(valid.gradient_novelty_abs_diff.mean()) if len(valid) else None,
            "mean_abs_hardness_diff_supported": float(valid.hardness_diff.abs().mean()) if len(valid) else None,
            "mean_abs_parameter_z_diff_supported": float(valid.parameter_z_diff.abs().mean()) if len(valid) else None,
            "mean_abs_translation_diff_supported": float(valid.translation_diff.abs().mean()) if len(valid) else None,
            "pass": passed,
        }
        rows.append(row)
        if passed and selected is None:
            selected = c

    decision = {
        "decision": "SPAN MATCH CALIBRATION PASS" if selected is not None else "SPAN MATCH CALIBRATION FAIL",
        "selected_g_caliper": selected,
        "protocol_hash": PH,
        "heldout_constructed": False,
        "intervention_arms_trained": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "gradient_span_calibration_summary.csv", index=False)
    (out / "gradient_span_calibration_decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False)
    )
    print(json.dumps(decision), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def selftest() -> None:
    # Exact equal-G counterexample from the algebra note: one-axis opposition has
    # rank 1; two opposing orthogonal axes have rank 2. Both have G=4/3.
    a = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    b = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    ga, gb = mean_pairwise_novelty(a), mean_pairwise_novelty(b)
    sa = spectral_effective_rank(torch.tensor(a))
    sb = spectral_effective_rank(torch.tensor(b))
    assert abs(ga - 4.0 / 3.0) < 1e-12 and abs(gb - 4.0 / 3.0) < 1e-12
    assert abs(sa - 1.0) < 1e-12 and abs(sb - 2.0) < 1e-12

    rows = [
        {"local": (0, 1, 2, 3), "spectral_erank": sa, "gradient_novelty": ga, "z_hard": 0.0, "z_param": 0.0, "translation_score": 0.0},
        {"local": (0, 1, 2, 4), "spectral_erank": sb, "gradient_novelty": gb, "z_hard": 0.0, "z_param": 0.0, "translation_score": 0.0},
    ]
    pair = choose_span_pair(rows, 0.005)
    assert pair is not None and pair["high"] == 1 and pair["low"] == 0
    assert abs(pair["span_gap"] - 1.0) < 1e-12
    assert pair["gradient_novelty_abs_diff"] < 1e-12
    for bad in (float("nan"), 0.0, -0.1):
        try:
            choose_span_pair(rows, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid G caliper accepted")
    print("GRADIENT_SPAN_SYNTHETIC_PASS", PH)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("selftest", "run", "summarize"))
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--input-dir")
    p.add_argument("--output-dir")
    a = p.parse_args()
    if a.mode == "selftest":
        selftest()
    elif a.mode == "run":
        if a.start is None or a.end is None or a.output_dir is None:
            raise ValueError("run args required")
        run(a.start, a.end, Path(a.output_dir))
    else:
        if a.input_dir is None or a.output_dir is None:
            raise ValueError("summarize args required")
        summarize(Path(a.input_dir), Path(a.output_dir))


if __name__ == "__main__":
    main()
