"""Issue98 Stage A: centered residual-span calibration with expanded support.

Training-reference-only. No intervention arm and no heldout environment are
constructed unless a later, separately implemented Stage B is allowed by the
frozen Stage-A decision.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import parameter_matched_novelty_calibration as pmc
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS = tuple(range(2500, 2510))
SEEDS = tuple(range(69000, 69064))
G_GRID = (0.005, 0.010, 0.020, 0.050)
H_CALIPER = 0.05
L_CALIPER = 0.05
P_CALIPER = 0.05
T_CALIPER = 0.20
SLACK = 1e-12
NEG_EIG_TOL = 1e-10
MAX_IDENTITY_ERROR = 1e-8
MIN_POOLED_SUPPORT = 0.90
MIN_REP_SUPPORT = 0.80
MIN_MEAN_SPAN_GAP = 0.20
MIN_REP_MEAN_SPAN_GAP = 0.10
OFFSET = 2910000000
K = 16
Q = 4

PROTOCOL = {
    "issue": 98,
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
    "max_centered_energy_identity_error": MAX_IDENTITY_ERROR,
    "min_pooled_support": MIN_POOLED_SUPPORT,
    "min_rep_support": MIN_REP_SUPPORT,
    "min_mean_centered_span_gap": MIN_MEAN_SPAN_GAP,
    "min_rep_mean_centered_span_gap": MIN_REP_MEAN_SPAN_GAP,
    "threads": 1,
}
PH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
SOURCES = (
    "common.py",
    "parameter_matched_novelty_calibration.py",
    "parameter_matched_novelty_confirmatory.py",
    "translation_matched_calibration.py",
    "cnn_regime_interaction.py",
    "fixed_dose_response.py",
    "transfer_specificity.py",
    "centered_span_calibration.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: fd.sha256(root / name) for name in SOURCES}


def centered_geometry(v: torch.Tensor) -> tuple[float, float, float]:
    """Return centered effective rank, centered energy, and mean pairwise G."""
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
    identity_error = float(energy - (Q - 1) * novelty)
    return erank, energy, identity_error


def expanded_subsets(
    losses: torch.Tensor,
    gd: torch.Tensor,
    candidate_envs: list[int],
    params: np.ndarray,
) -> list[dict]:
    order = sorted(range(K), key=lambda i: (-float(losses[i].detach()), i))
    anchor = order[0]
    candidate_mean = float(losses.mean().detach())
    candidate_sd = float(losses.detach().double().std(unbiased=False))
    cos = gd @ gd.T
    rows: list[dict] = []

    for others in itertools.combinations(order[1:], 3):
        local = tuple(sorted((anchor, *others)))
        envs = tuple(int(candidate_envs[i]) for i in local)
        lv = losses[list(local)].detach().double().cpu().numpy()
        erank, energy, identity_error = centered_geometry(gd[list(local)])
        rows.append(
            {
                "local": local,
                "ranks": tuple(order.index(i) + 1 for i in local),
                "envs": envs,
                "gradient_novelty": pmc.pairnov(cos, local),
                "centered_erank": erank,
                "centered_energy": energy,
                "identity_error": identity_error,
                "z_hard": (float(lv.mean()) - candidate_mean) / (candidate_sd + 1e-8),
                "loss_variance": float(lv.var(ddof=0)),
                "physical_diversity": pmc.physical_diversity(envs, params),
            }
        )

    if len(rows) != 455:
        raise ValueError(f"expected 455 subsets, got {len(rows)}")

    pv = np.asarray([r["physical_diversity"] for r in rows], dtype=np.float64)
    lv = np.asarray([r["loss_variance"] for r in rows], dtype=np.float64)
    psd = float(pv.std(ddof=0))
    lsd = float(lv.std(ddof=0))
    if not np.isfinite(psd) or psd <= 1e-12:
        raise ValueError("physical diversity SD degenerate")
    if not np.isfinite(lsd) or lsd <= 1e-12:
        raise ValueError("loss variance SD degenerate")
    pmu, lmu = float(pv.mean()), float(lv.mean())
    for r in rows:
        r["z_param"] = (r["physical_diversity"] - pmu) / psd
        r["z_loss_variance"] = (r["loss_variance"] - lmu) / lsd
        v = params[list(r["envs"])].var(axis=0, ddof=0)
        r["translation_score"] = float(v[1] + v[2])
    return rows


def choose_pair(rows: list[dict], g_caliper: float) -> dict | None:
    fields = (
        "centered_erank",
        "gradient_novelty",
        "z_hard",
        "z_loss_variance",
        "z_param",
        "translation_score",
    )
    a = np.asarray([[r[f] for f in fields] for r in rows], dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(fields) or not np.isfinite(a).all():
        raise ValueError("nonfinite subset table")
    if not np.isfinite(g_caliper) or g_caliper <= 0:
        raise ValueError("bad G caliper")
    ii, jj = np.triu_indices(len(rows), 1)
    d = a[ii] - a[jj]
    keep = (
        (np.abs(d[:, 2]) <= H_CALIPER + SLACK)
        & (np.abs(d[:, 3]) <= L_CALIPER + SLACK)
        & (np.abs(d[:, 4]) <= P_CALIPER + SLACK)
        & (np.abs(d[:, 5]) <= T_CALIPER + SLACK)
        & (np.abs(d[:, 1]) <= g_caliper + SLACK)
    )
    loc = np.flatnonzero(keep)
    if not len(loc):
        return None
    k = min(
        loc,
        key=lambda z: (
            -abs(d[z, 0]),
            abs(d[z, 1]),
            abs(d[z, 5]),
            abs(d[z, 4]),
            abs(d[z, 3]),
            abs(d[z, 2]),
            tuple(rows[ii[z]]["local"]),
            tuple(rows[jj[z]]["local"]),
        ),
    )
    i, j = int(ii[k]), int(jj[k])
    hi, lo = (i, j) if a[i, 0] >= a[j, 0] else (j, i)
    return {
        "high": hi,
        "low": lo,
        "eligible_pairs": int(len(loc)),
        "centered_span_gap": float(a[hi, 0] - a[lo, 0]),
        "gradient_novelty_diff": float(a[hi, 1] - a[lo, 1]),
        "gradient_novelty_abs_diff": float(abs(a[hi, 1] - a[lo, 1])),
        "hardness_diff": float(a[hi, 2] - a[lo, 2]),
        "loss_variance_z_diff": float(a[hi, 3] - a[lo, 3]),
        "parameter_z_diff": float(a[hi, 4] - a[lo, 4]),
        "translation_diff": float(a[hi, 5] - a[lo, 5]),
        "high_local": ";".join(map(str, rows[hi]["local"])),
        "low_local": ";".join(map(str, rows[lo]["local"])),
        "high_ranks": ";".join(map(str, rows[hi]["ranks"])),
        "low_ranks": ";".join(map(str, rows[lo]["ranks"])),
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
    params = tc.parameters(SEEDS)
    pair_rows: list[dict] = []
    subset_rows: list[dict] = []
    max_identity_error = 0.0

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
            step_err = max(abs(float(s["identity_error"])) for s in subsets)
            max_identity_error = max(max_identity_error, step_err)

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
                                "identity_error",
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
                        "step_max_identity_error": step_err,
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
            f"CENTERED_SPAN_REFERENCE_COMPLETE rep={rep}; max_identity_error={max_identity_error:.3g}; no arms, no heldout",
            flush=True,
        )

    p = out / f"centered_span_pairs_{start}_{end}.csv"
    s = out / f"centered_span_subsets_{start}_{end}.csv.gz"
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
        "maximum_identity_error": max_identity_error,
        "heldout_constructed": False,
        "intervention_arms_trained": 0,
    }
    (out / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )


def summarize(root: Path, out: Path) -> None:
    covered: list[int] = []
    frames: list[pd.DataFrame] = []
    maximum_identity_error = 0.0
    for mp in sorted(root.rglob("calibration_manifest.json")):
        m = json.loads(mp.read_text())
        if m["protocol_hash"] != PH or m["source_hashes"] != source_hashes():
            raise ValueError("source/protocol provenance")
        if m["split_hashes"] != cr.split_hashes():
            raise ValueError("split provenance")
        if m["heldout_constructed"] or m["intervention_arms_trained"] != 0:
            raise ValueError("Stage-A boundary violated")
        maximum_identity_error = max(maximum_identity_error, abs(float(m["maximum_identity_error"])))
        covered.extend(range(m["start"], m["end"]))
        for name, digest in m["files"].items():
            if fd.sha256(mp.parent / name) != digest:
                raise ValueError("artifact hash mismatch")
        frames.append(
            pd.read_csv(
                mp.parent / f"centered_span_pairs_{m['start']}_{m['end']}.csv",
                float_precision="round_trip",
            )
        )
    if sorted(covered) != list(REPS):
        raise ValueError("replicate coverage")

    df = pd.concat(frames, ignore_index=True)
    keys = ["rep", "step", "g_caliper"]
    expected = {(r, t, c) for r in REPS for t in range(80) for c in G_GRID}
    if len(df) != len(expected) or df.duplicated(keys).any() or set(df[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("calibration grid")

    rows = []
    selected = None
    identity_pass = maximum_identity_error <= MAX_IDENTITY_ERROR
    for c in G_GRID:
        v = df[np.isclose(df.g_caliper, c)].copy()
        feasible = v.feasible.astype(bool)
        valid = v[feasible]
        cols = [
            "centered_span_gap",
            "gradient_novelty_abs_diff",
            "hardness_diff",
            "loss_variance_z_diff",
            "parameter_z_diff",
            "translation_diff",
        ]
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
        if (valid.centered_span_gap < -SLACK).any():
            raise ValueError("span orientation")

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
        rows.append(
            {
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
                "maximum_identity_error": maximum_identity_error,
                "identity_pass": identity_pass,
                "pass": passed,
            }
        )
        if passed and selected is None:
            selected = c

    decision = {
        "decision": "CENTERED SPAN CALIBRATION PASS" if selected is not None else "CENTERED SPAN CALIBRATION FAIL",
        "selected_g_caliper": selected,
        "protocol_hash": PH,
        "maximum_identity_error": maximum_identity_error,
        "heldout_constructed": False,
        "intervention_arms_trained": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "centered_span_calibration_summary.csv", index=False)
    (out / "centered_span_calibration_decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False)
    )
    print(json.dumps(decision), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def selftest() -> None:
    a = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    ea, ena, erra = centered_geometry(a)
    eb, enb, errb = centered_geometry(b)
    assert abs(ena - 4.0) < 1e-12 and abs(enb - 4.0) < 1e-12
    assert abs(erra) < 1e-12 and abs(errb) < 1e-12
    assert abs(ea - 1.0) < 1e-12 and abs(eb - 2.0) < 1e-12

    rng = np.random.default_rng(123)
    for _ in range(100):
        x = rng.normal(size=(Q, 9))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        _, _, err = centered_geometry(torch.tensor(x, dtype=torch.float64))
        assert abs(err) < 1e-10
    print("CENTERED_SPAN_SYNTHETIC_PASS", PH)


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
