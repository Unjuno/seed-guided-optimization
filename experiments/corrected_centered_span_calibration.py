"""Issue100 Stage A: corrected centered-span calibration.

Scientific matching/manipulation gates are unchanged from Issue98. Only the
numerical centered-energy identity is replaced by the exact general identity for
returned, approximately-unit float32 signatures. Stage A constructs no heldout
set and trains no intervention arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import centered_span_calibration as old
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS = tuple(range(2700, 2710))
SEEDS = tuple(range(73000, 73064))
G_GRID = old.G_GRID
H_CALIPER = old.H_CALIPER
L_CALIPER = old.L_CALIPER
P_CALIPER = old.P_CALIPER
T_CALIPER = old.T_CALIPER
SLACK = old.SLACK
NEG_EIG_TOL = old.NEG_EIG_TOL
MAX_GENERAL_IDENTITY_ERROR = 1e-10
MIN_POOLED_SUPPORT = old.MIN_POOLED_SUPPORT
MIN_REP_SUPPORT = old.MIN_REP_SUPPORT
MIN_MEAN_SPAN_GAP = old.MIN_MEAN_SPAN_GAP
MIN_REP_MEAN_SPAN_GAP = old.MIN_REP_MEAN_SPAN_GAP
OFFSET = 3110000000
K = 16
Q = 4

PROTOCOL = {
    "issue": 100,
    "stage": "A",
    "reps": REPS,
    "train_seeds": SEEDS,
    "offset": OFFSET,
    "stride": 4099,
    "K": K,
    "Q": Q,
    "steps": 80,
    "epochs": 10,
    "batch": 128,
    "lr": 0.005,
    "wd": 0.001,
    "subset_family": "loss-rank1 plus any3 from ranks2-16",
    "subset_count": 455,
    "h_caliper": H_CALIPER,
    "loss_variance_z_caliper": L_CALIPER,
    "p_caliper": P_CALIPER,
    "t_caliper": T_CALIPER,
    "g_grid": G_GRID,
    "negative_eigenvalue_tolerance": NEG_EIG_TOL,
    "max_general_identity_error": MAX_GENERAL_IDENTITY_ERROR,
    "min_pooled_support": MIN_POOLED_SUPPORT,
    "min_rep_support": MIN_REP_SUPPORT,
    "min_mean_centered_span_gap": MIN_MEAN_SPAN_GAP,
    "min_rep_mean_centered_span_gap": MIN_REP_MEAN_SPAN_GAP,
    "g_numeric": "float64 from returned head_gradient_directions rows",
    "identity": "Ec=(q-1)*(G+mean_squared_row_norm-1)",
    "threads": 1,
}
PH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
SOURCES = (
    "common.py",
    "centered_span_calibration.py",
    "parameter_matched_novelty_calibration.py",
    "parameter_matched_novelty_confirmatory.py",
    "translation_matched_calibration.py",
    "cnn_regime_interaction.py",
    "fixed_dose_response.py",
    "transfer_specificity.py",
    "corrected_centered_span_calibration.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: fd.sha256(root / name) for name in SOURCES}


def geometry(v: torch.Tensor) -> dict[str, float]:
    a = v.detach().double().cpu().numpy()
    if a.ndim != 2 or a.shape[0] != Q or not np.isfinite(a).all():
        raise ValueError("invalid signature matrix")
    c = a - a.mean(axis=0, keepdims=True)
    gram_c = c @ c.T
    eig = np.linalg.eigvalsh(gram_c)
    if float(eig.min()) < -NEG_EIG_TOL:
        raise ValueError(f"materially negative centered eigenvalue {eig.min()}")
    eig = np.clip(eig, 0.0, None)
    energy = float(eig.sum())
    if not np.isfinite(energy) or energy <= 1e-12:
        raise ValueError("degenerate centered energy")
    p = eig / energy
    pos = p > 0
    erank = float(np.exp(-np.sum(p[pos] * np.log(p[pos]))))
    if not np.isfinite(erank) or erank < 1.0 - 1e-9 or erank > 3.0 + 1e-8:
        raise ValueError(f"centered effective rank out of range: {erank}")
    gram = a @ a.T
    ii, jj = np.triu_indices(Q, 1)
    novelty = float(np.mean(1.0 - gram[ii, jj]))
    m2 = float(np.mean(np.sum(a * a, axis=1)))
    expected_energy = float((Q - 1) * (novelty + m2 - 1.0))
    identity_error = float(energy - expected_energy)
    return {
        "centered_erank": erank,
        "centered_energy": energy,
        "gradient_novelty": novelty,
        "mean_squared_row_norm": m2,
        "general_identity_error": identity_error,
    }


def expanded_subsets(losses, gd, candidate_envs, params) -> list[dict]:
    # Reuse Issue98's prospectively fixed 455-subset H/L/P/T construction,
    # then replace all gradient-geometry quantities consistently in float64.
    rows = old.expanded_subsets(losses, gd, candidate_envs, params)
    for r in rows:
        idx = torch.tensor(r["local"], dtype=torch.long)
        q = geometry(gd.index_select(0, idx))
        r.update(q)
    return rows


def choose_pair(rows: list[dict], g_caliper: float) -> dict | None:
    return old.choose_pair(rows, g_caliper)


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
    params = tc.parameters(SEEDS)
    pair_rows: list[dict] = []
    subset_rows: list[dict] = []
    max_identity_error = 0.0
    max_norm_deviation = 0.0

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
                logits, ty[b].repeat(K), reduction="none"
            ).reshape(K, -1)
            losses = per.mean(1)
            gd = head_gradient_directions(logits, h, ty[b], K)
            subsets = expanded_subsets(losses, gd, cand, params)
            step_err = max(abs(float(s["general_identity_error"])) for s in subsets)
            step_norm = max(abs(float(s["mean_squared_row_norm"]) - 1.0) for s in subsets)
            max_identity_error = max(max_identity_error, step_err)
            max_norm_deviation = max(max_norm_deviation, step_norm)

            for idx, s in enumerate(subsets):
                subset_rows.append(
                    {
                        "rep": rep,
                        "step": step,
                        "subset_id": idx,
                        "local": ";".join(map(str, s["local"])),
                        "ranks": ";".join(map(str, s["ranks"])),
                        "envs": ";".join(map(str, s["envs"])),
                        **{
                            f: s[f]
                            for f in (
                                "gradient_novelty",
                                "centered_erank",
                                "centered_energy",
                                "mean_squared_row_norm",
                                "general_identity_error",
                                "z_hard",
                                "loss_variance",
                                "z_loss_variance",
                                "physical_diversity",
                                "z_param",
                                "translation_score",
                            )
                        },
                    }
                )

            for c in G_GRID:
                pair = choose_pair(subsets, c)
                pair_rows.append(
                    {
                        "rep": rep,
                        "step": step,
                        "g_caliper": c,
                        "feasible": pair is not None,
                        "step_max_general_identity_error": step_err,
                        "step_max_mean_norm_deviation": step_norm,
                        **(pair or {}),
                    }
                )

            selected = sorted(
                sorted(range(K), key=lambda j: (-float(losses[j].detach()), j))[:Q]
            )
            opt.zero_grad(set_to_none=True)
            per[selected].mean().backward()
            opt.step()

        print(
            f"CORRECTED_CENTERED_REFERENCE_COMPLETE rep={rep}; identity={max_identity_error:.3g}; normdev={max_norm_deviation:.3g}; no arms, no heldout",
            flush=True,
        )

    p = out / f"corrected_centered_pairs_{start}_{end}.csv"
    s = out / f"corrected_centered_subsets_{start}_{end}.csv.gz"
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
        "maximum_general_identity_error": max_identity_error,
        "maximum_mean_squared_norm_deviation": max_norm_deviation,
        "heldout_constructed": False,
        "intervention_arms_trained": 0,
    }
    (out / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )


def summarize(root: Path, out: Path) -> None:
    covered, frames = [], []
    max_identity_error = 0.0
    max_norm_deviation = 0.0
    for mp in sorted(root.rglob("calibration_manifest.json")):
        m = json.loads(mp.read_text())
        if m["protocol_hash"] != PH or m["source_hashes"] != source_hashes():
            raise ValueError("source/protocol provenance")
        if m["split_hashes"] != cr.split_hashes():
            raise ValueError("split provenance")
        if m["heldout_constructed"] or m["intervention_arms_trained"] != 0:
            raise ValueError("Stage-A boundary violated")
        max_identity_error = max(max_identity_error, abs(float(m["maximum_general_identity_error"])))
        max_norm_deviation = max(max_norm_deviation, abs(float(m["maximum_mean_squared_norm_deviation"])))
        covered.extend(range(m["start"], m["end"]))
        for name, digest in m["files"].items():
            if fd.sha256(mp.parent / name) != digest:
                raise ValueError("artifact hash mismatch")
        frames.append(pd.read_csv(mp.parent / f"corrected_centered_pairs_{m['start']}_{m['end']}.csv", float_precision="round_trip"))
    if sorted(covered) != list(REPS):
        raise ValueError("replicate coverage")

    df = pd.concat(frames, ignore_index=True)
    keys = ["rep", "step", "g_caliper"]
    expected = {(r, t, c) for r in REPS for t in range(80) for c in G_GRID}
    if len(df) != len(expected) or df.duplicated(keys).any() or set(df[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("calibration grid")

    identity_pass = max_identity_error <= MAX_GENERAL_IDENTITY_ERROR
    rows, selected = [], None
    for c in G_GRID:
        v = df[np.isclose(df.g_caliper, c)].copy()
        feasible = v.feasible.astype(bool)
        valid = v[feasible]
        cols = ["centered_span_gap", "gradient_novelty_abs_diff", "hardness_diff", "loss_variance_z_diff", "parameter_z_diff", "translation_diff"]
        if len(valid) and not np.isfinite(valid[cols].to_numpy(dtype=np.float64)).all():
            raise ValueError("nonfinite selected pair")
        if (
            (valid.hardness_diff.abs() > H_CALIPER + SLACK).any()
            or (valid.loss_variance_z_diff.abs() > L_CALIPER + SLACK).any()
            or (valid.parameter_z_diff.abs() > P_CALIPER + SLACK).any()
            or (valid.translation_diff.abs() > T_CALIPER + SLACK).any()
            or (valid.gradient_novelty_abs_diff > c + SLACK).any()
        ):
            raise ValueError("caliper violation")
        support_by_rep = v.assign(feas=feasible).groupby("rep").feas.mean().reindex(REPS, fill_value=0.0)
        span_by_rep = valid.groupby("rep").centered_span_gap.mean().reindex(REPS, fill_value=0.0)
        pooled = float(feasible.mean())
        min_support = float(support_by_rep.min())
        mean_span = float(valid.centered_span_gap.mean()) if len(valid) else 0.0
        min_rep_span = float(span_by_rep.min())
        passed = bool(
            identity_pass
            and pooled >= MIN_POOLED_SUPPORT
            and min_support >= MIN_REP_SUPPORT
            and mean_span >= MIN_MEAN_SPAN_GAP
            and min_rep_span >= MIN_REP_MEAN_SPAN_GAP
        )
        rows.append({
            "g_caliper": c,
            "supported_steps": int(feasible.sum()),
            "total_steps": len(v),
            "pooled_support": pooled,
            "minimum_replicate_support": min_support,
            "mean_centered_span_gap_supported": mean_span,
            "minimum_replicate_mean_centered_span_gap_supported": min_rep_span,
            "mean_abs_gradient_novelty_diff_supported": float(valid.gradient_novelty_abs_diff.mean()) if len(valid) else None,
            "mean_abs_hardness_diff_supported": float(valid.hardness_diff.abs().mean()) if len(valid) else None,
            "mean_abs_loss_variance_z_diff_supported": float(valid.loss_variance_z_diff.abs().mean()) if len(valid) else None,
            "mean_abs_parameter_z_diff_supported": float(valid.parameter_z_diff.abs().mean()) if len(valid) else None,
            "mean_abs_translation_diff_supported": float(valid.translation_diff.abs().mean()) if len(valid) else None,
            "maximum_general_identity_error": max_identity_error,
            "maximum_mean_squared_norm_deviation": max_norm_deviation,
            "identity_pass": identity_pass,
            "pass": passed,
        })
        if passed and selected is None:
            selected = c

    decision = {
        "decision": "CORRECTED CENTERED SPAN CALIBRATION PASS" if selected is not None else "CORRECTED CENTERED SPAN CALIBRATION FAIL",
        "selected_g_caliper": selected,
        "protocol_hash": PH,
        "maximum_general_identity_error": max_identity_error,
        "maximum_mean_squared_norm_deviation": max_norm_deviation,
        "heldout_constructed": False,
        "intervention_arms_trained": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "corrected_centered_span_calibration_summary.csv", index=False)
    (out / "corrected_centered_span_calibration_decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False))
    print(json.dumps(decision), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def selftest() -> None:
    # General identity must hold even for deliberately non-unit rows.
    x = torch.tensor([[2.0, 0.0], [-0.5, 0.0], [0.0, 3.0], [0.0, -1.5]], dtype=torch.float64)
    q = geometry(x)
    assert abs(q["general_identity_error"]) < 1e-12
    # Equal-G unit-norm counterexample with different centered rank.
    a = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    qa, qb = geometry(a), geometry(b)
    assert abs(qa["gradient_novelty"] - qb["gradient_novelty"]) < 1e-12
    assert abs(qa["centered_erank"] - 1.0) < 1e-12
    assert abs(qb["centered_erank"] - 2.0) < 1e-12
    print("CORRECTED_CENTERED_SPAN_SYNTHETIC_PASS", PH)


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
