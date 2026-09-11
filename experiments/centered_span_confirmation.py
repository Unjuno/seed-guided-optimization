"""Issue100 Stage B: G-matched centered residual-span confirmation.

Global barriers are enforced by the workflow:
1. build/seal every reference schedule before any intervention arm trains;
2. train/seal all high/low intervention states plus reference states;
3. only then construct fresh heldout environments and evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

import corrected_centered_span_calibration as cal
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS = tuple(range(2800, 2830))
TRAIN_SEEDS = tuple(range(75000, 75064))
HELDOUT_SEEDS = tuple(range(76000, 76080))
ARMS = ("high", "low")
ALL_ARMS = ARMS + ("reference",)
OFFSET = 3210000000
K = 16
Q = 4
STEPS = 80
SELECTED_G_CALIPER = 0.05
H_MARGIN = 0.05
L_MARGIN = 0.05
P_MARGIN = 0.05
T_MARGIN = 0.20
G_MARGIN = SELECTED_G_CALIPER
SPAN_MIN = 0.20
MIN_POOLED_SUPPORT = 0.90
MIN_REP_SUPPORT = 0.80
MAX_GENERAL_IDENTITY_ERROR = 1e-10
STAGE_A_PROTOCOL_HASH = "61717ae7a410a2ca64601821affea83427937c23d165f83c0f4b92ab185b1d6d"

PROTOCOL = {
    "issue": 100,
    "stage": "B",
    "reps": REPS,
    "arms": ARMS,
    "baseline": "reference_loss_hard",
    "train_seeds": TRAIN_SEEDS,
    "heldout_seeds": HELDOUT_SEEDS,
    "base_seed_offset": OFFSET,
    "base_seed_stride": 4099,
    "K": K,
    "Q": Q,
    "steps": STEPS,
    "epochs": 10,
    "batch": 128,
    "lr": 0.005,
    "wd": 0.001,
    "subset_family": "loss-rank1 plus any3 from ranks2-16",
    "subset_count": 455,
    "selected_g_caliper": SELECTED_G_CALIPER,
    "h_equivalence_margin": H_MARGIN,
    "loss_variance_equivalence_margin": L_MARGIN,
    "physical_diversity_equivalence_margin": P_MARGIN,
    "translation_equivalence_margin": T_MARGIN,
    "g_equivalence_margin": G_MARGIN,
    "minimum_span_manipulation": SPAN_MIN,
    "minimum_pooled_support": MIN_POOLED_SUPPORT,
    "minimum_replicate_support": MIN_REP_SUPPORT,
    "max_general_identity_error": MAX_GENERAL_IDENTITY_ERROR,
    "fallback": "identical_reference_loss_hard_top4",
    "stage_a_protocol_hash": STAGE_A_PROTOCOL_HASH,
    "threads": 1,
}
PH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
SOURCES = (
    "common.py",
    "centered_span_calibration.py",
    "corrected_centered_span_calibration.py",
    "parameter_matched_novelty_calibration.py",
    "parameter_matched_novelty_confirmatory.py",
    "translation_matched_calibration.py",
    "cnn_regime_interaction.py",
    "fixed_dose_response.py",
    "transfer_specificity.py",
    "centered_span_confirmation.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: fd.sha256(root / name) for name in SOURCES}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False))


def check_range(start: int, end: int) -> None:
    if start is None or end is None or start >= end or not set(range(start, end)).issubset(REPS):
        raise ValueError("unregistered Stage-B range")


def state_digest(state: dict[str, torch.Tensor]) -> str:
    return pm.state_digest(state)


def save_state(out: Path, rep: int, arm: str, state: dict[str, torch.Tensor], schedule_sha: str) -> Path:
    path = out / "states" / f"rep{rep}_{arm}.pt"
    if path.exists():
        raise ValueError("state overwrite refused")
    torch.save(
        {
            "state_dict": state,
            "rep": rep,
            "arm": arm,
            "protocol_hash": PH,
            "schedule_sha256": schedule_sha,
        },
        path,
    )
    return path


def validate_files(root: Path, manifest: dict) -> None:
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.exists() or fd.sha256(path) != digest:
            raise ValueError(f"sealed file mismatch: {name}")


def validate_reference_shard(root: Path, start: int, end: int) -> dict:
    m = json.loads((root / "reference_manifest.json").read_text())
    if (m["start"], m["end"], m["protocol_hash"]) != (start, end, PH):
        raise ValueError("reference protocol")
    if m["source_hashes"] != source_hashes() or m["split_hashes"] != cr.split_hashes():
        raise ValueError("reference source/split changed")
    if m["heldout_constructed"] or m["intervention_arms_trained"] != 0:
        raise ValueError("reference barrier violated")
    if abs(float(m["maximum_general_identity_error"])) > MAX_GENERAL_IDENTITY_ERROR:
        raise ValueError("general identity gate")
    validate_files(root, m)
    return m


def validate_global_seal(path: Path, stage: str) -> dict:
    seal = json.loads(path.read_text())
    if seal.get("protocol_hash") != PH or seal.get("stage") != stage or seal.get("reps") != list(REPS):
        raise ValueError(f"invalid global {stage} seal")
    return seal


def reference(start: int, end: int, out: Path) -> None:
    check_range(start, end)
    base.configure_determinism(1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "schedules").mkdir(exist_ok=True)
    (out / "states").mkdir(exist_ok=True)
    if (out / "reference_manifest.json").exists():
        raise ValueError("reference overwrite refused")

    x, y, *_ = cr.data_split()
    ty = torch.tensor(y)
    envs = base.geometric_envs(x, TRAIN_SEEDS)
    params = tc.parameters(TRAIN_SEEDS)
    step_rows: list[dict] = []
    subset_rows: list[dict] = []
    files: dict[str, str] = {}
    schedule_hashes: dict[str, str] = {}
    max_identity_error = 0.0

    for rep in range(start, end):
        seed = OFFSET + 4099 * rep
        sched = pm.build_schedule(len(ty), seed)
        if len(sched) != STEPS:
            raise ValueError("step count")
        seed_everything(seed)
        model = SmallCNN()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
        selections = {"high": [], "low": []}
        fallback_flags = []

        for step, (b, cand) in enumerate(sched):
            logits, h = model(torch.cat([envs[e][b] for e in cand]))
            per = torch.nn.functional.cross_entropy(
                logits, ty[b].repeat(K), reduction="none"
            ).reshape(K, -1)
            losses = per.mean(1)
            gd = head_gradient_directions(logits, h, ty[b], K)
            subsets = cal.expanded_subsets(losses, gd, cand, params)
            step_err = max(abs(float(s["general_identity_error"])) for s in subsets)
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
                            field: s[field]
                            for field in (
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

            pair = cal.choose_pair(subsets, SELECTED_G_CALIPER)
            top4_local = sorted(
                sorted(range(K), key=lambda j: (-float(losses[j].detach()), j))[:Q]
            )
            top4_envs = [int(cand[j]) for j in top4_local]
            if pair is None:
                hi_envs = list(top4_envs)
                lo_envs = list(top4_envs)
                fallback = True
                rec = {
                    "centered_span_gap": 0.0,
                    "gradient_novelty_diff": 0.0,
                    "gradient_novelty_abs_diff": 0.0,
                    "hardness_diff": 0.0,
                    "loss_variance_z_diff": 0.0,
                    "parameter_z_diff": 0.0,
                    "translation_diff": 0.0,
                    "eligible_pairs": 0,
                    "high_ranks": ";".join(str(i + 1) for i in range(Q)),
                    "low_ranks": ";".join(str(i + 1) for i in range(Q)),
                }
            else:
                hi = subsets[pair["high"]]
                lo = subsets[pair["low"]]
                hi_envs = list(map(int, hi["envs"]))
                lo_envs = list(map(int, lo["envs"]))
                fallback = False
                rec = dict(pair)
            if len(hi_envs) != Q or len(lo_envs) != Q:
                raise ValueError("selected schedule size")
            selections["high"].append(hi_envs)
            selections["low"].append(lo_envs)
            fallback_flags.append(bool(fallback))
            step_rows.append(
                {
                    "rep": rep,
                    "step": step,
                    "supported": not fallback,
                    "fallback": fallback,
                    "high_envs": ";".join(map(str, hi_envs)),
                    "low_envs": ";".join(map(str, lo_envs)),
                    "overlap": len(set(hi_envs) & set(lo_envs)) / Q,
                    "step_max_general_identity_error": step_err,
                    **{k: v for k, v in rec.items() if k not in ("high", "low")},
                }
            )

            opt.zero_grad(set_to_none=True)
            per[top4_local].mean().backward()
            opt.step()

        payload = {
            "rep": rep,
            "protocol_hash": PH,
            "base_schedule_digest": pm.schedule_digest(sched),
            "selected_g_caliper": SELECTED_G_CALIPER,
            "fallback": fallback_flags,
            **selections,
        }
        sp = out / "schedules" / f"rep{rep}.json"
        write_json(sp, payload)
        schedule_hashes[sp.name] = fd.sha256(sp)
        files[str(sp.relative_to(out))] = fd.sha256(sp)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state_path = save_state(out, rep, "reference", state, fd.sha256(sp))
        files[str(state_path.relative_to(out))] = fd.sha256(state_path)
        print(f"CENTERED_SPAN_REFERENCE_SEALED rep={rep}; arms=0 heldout=0", flush=True)

    step_path = out / f"centered_span_reference_{start}_{end}.csv.gz"
    subset_path = out / f"centered_span_subsets_{start}_{end}.csv.gz"
    pd.DataFrame(step_rows).to_csv(step_path, index=False, compression="gzip")
    pd.DataFrame(subset_rows).to_csv(subset_path, index=False, compression="gzip")
    files[step_path.name] = fd.sha256(step_path)
    files[subset_path.name] = fd.sha256(subset_path)
    write_json(
        out / "reference_manifest.json",
        {
            "protocol": PROTOCOL,
            "protocol_hash": PH,
            "start": start,
            "end": end,
            "files": files,
            "schedule_hashes": schedule_hashes,
            "source_hashes": source_hashes(),
            "split_hashes": cr.split_hashes(),
            "runtime": fd.runtime(),
            "maximum_general_identity_error": max_identity_error,
            "heldout_constructed": False,
            "intervention_arms_trained": 0,
        },
    )


def load_frames(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(p, float_precision="round_trip") for p in paths], ignore_index=True)


def reference_seal(root: Path, out: Path) -> None:
    covered = []
    shards = []
    max_identity = 0.0
    for mp in sorted(root.rglob("reference_manifest.json")):
        m = json.loads(mp.read_text())
        start, end = int(m["start"]), int(m["end"])
        check_range(start, end)
        validate_reference_shard(mp.parent, start, end)
        covered.extend(range(start, end))
        max_identity = max(max_identity, abs(float(m["maximum_general_identity_error"])))
        shards.append({"start": start, "end": end, "manifest_sha256": fd.sha256(mp)})
    if sorted(covered) != list(REPS):
        raise ValueError("global reference coverage")

    steps = load_frames(root, "centered_span_reference_*.csv.gz")
    expected = {(r, t) for r in REPS for t in range(STEPS)}
    if len(steps) != len(expected) or steps.duplicated(["rep", "step"]).any() or set(steps[["rep", "step"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("reference step grid")
    supported = steps.supported.astype(bool)
    if not (supported == ~steps.fallback.astype(bool)).all():
        raise ValueError("support/fallback mismatch")
    support_by_rep = steps.assign(supported_bool=supported).groupby("rep").supported_bool.mean().reindex(REPS)
    pooled = float(supported.mean())
    minimum = float(support_by_rep.min())
    if pooled < MIN_POOLED_SUPPORT or minimum < MIN_REP_SUPPORT:
        raise RuntimeError(f"CENTERED SPAN OVERLAP SUPPORT FAILURE pooled={pooled} minimum={minimum}")
    live = steps[supported]
    if (
        (live.gradient_novelty_abs_diff > G_MARGIN + 1e-12).any()
        or (live.hardness_diff.abs() > H_MARGIN + 1e-12).any()
        or (live.loss_variance_z_diff.abs() > L_MARGIN + 1e-12).any()
        or (live.parameter_z_diff.abs() > P_MARGIN + 1e-12).any()
        or (live.translation_diff.abs() > T_MARGIN + 1e-12).any()
    ):
        raise ValueError("supported-step caliper violation")
    fallback = steps[~supported]
    zero_fields = [
        "centered_span_gap",
        "gradient_novelty_diff",
        "gradient_novelty_abs_diff",
        "hardness_diff",
        "loss_variance_z_diff",
        "parameter_z_diff",
        "translation_diff",
    ]
    if len(fallback) and (
        not fallback.high_envs.eq(fallback.low_envs).all()
        or not fallback[zero_fields].eq(0).all().all()
    ):
        raise ValueError("fallback asymmetry")
    if max_identity > MAX_GENERAL_IDENTITY_ERROR:
        raise ValueError("general identity failure")
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "reference_seal.json",
        {
            "protocol_hash": PH,
            "stage": "reference",
            "reps": list(REPS),
            "selected_g_caliper": SELECTED_G_CALIPER,
            "pooled_support": pooled,
            "minimum_replicate_support": minimum,
            "supported_steps": int(supported.sum()),
            "fallback_steps": int((~supported).sum()),
            "maximum_general_identity_error": max_identity,
            "shards": sorted(shards, key=lambda z: z["start"]),
        },
    )
    print(f"GLOBAL_REFERENCE_SEAL support={pooled:.6f} min={minimum:.6f}", flush=True)


def train(start: int, end: int, out: Path, global_seal: Path) -> None:
    check_range(start, end)
    validate_global_seal(global_seal, "reference")
    base.configure_determinism(1)
    ref = validate_reference_shard(out, start, end)
    if (out / "arm_manifest.json").exists():
        raise ValueError("arm overwrite refused")
    x, y, *_ = cr.data_split()
    ty = torch.tensor(y)
    envs = base.geometric_envs(x, TRAIN_SEEDS)
    rows = []
    files = {}

    for rep in range(start, end):
        seed = OFFSET + 4099 * rep
        sched = pm.build_schedule(len(ty), seed)
        sp = out / "schedules" / f"rep{rep}.json"
        payload = json.loads(sp.read_text())
        if payload["rep"] != rep or payload["protocol_hash"] != PH or payload["selected_g_caliper"] != SELECTED_G_CALIPER:
            raise ValueError("schedule metadata")
        if pm.schedule_digest(sched) != payload["base_schedule_digest"]:
            raise ValueError("base schedule changed")
        for arm in ARMS:
            if len(payload[arm]) != STEPS:
                raise ValueError("schedule truncation")
            state, diag = pm.train_arm(envs, ty, sched, payload[arm], seed)
            p = save_state(out, rep, arm, state, fd.sha256(sp))
            files[str(p.relative_to(out))] = fd.sha256(p)
            rows.append({"rep": rep, "arm": arm, **diag})
        rp = out / "states" / f"rep{rep}_reference.pt"
        files[str(rp.relative_to(out))] = fd.sha256(rp)
        print(f"CENTERED_SPAN_ARMS_SEALED rep={rep}; heldout=0", flush=True)

    tp = out / f"centered_span_training_{start}_{end}.csv"
    pd.DataFrame(rows).to_csv(tp, index=False)
    files[tp.name] = fd.sha256(tp)
    write_json(
        out / "arm_manifest.json",
        {
            "protocol_hash": PH,
            "start": start,
            "end": end,
            "files": files,
            "reference_manifest_sha256": fd.sha256(out / "reference_manifest.json"),
            "global_reference_seal_sha256": fd.sha256(global_seal),
            "source_hashes": source_hashes(),
            "split_hashes": cr.split_hashes(),
            "runtime": fd.runtime(),
            "heldout_constructed": False,
            "intervention_arms_trained": 2 * (end - start),
        },
    )


def validate_arm_shard(root: Path, start: int, end: int) -> dict:
    ref = validate_reference_shard(root, start, end)
    m = json.loads((root / "arm_manifest.json").read_text())
    if (m["start"], m["end"], m["protocol_hash"]) != (start, end, PH):
        raise ValueError("arm protocol")
    if m["source_hashes"] != source_hashes() or m["split_hashes"] != cr.split_hashes():
        raise ValueError("arm source/split changed")
    if m["reference_manifest_sha256"] != fd.sha256(root / "reference_manifest.json"):
        raise ValueError("arm reference provenance")
    if m["heldout_constructed"]:
        raise ValueError("heldout constructed before arm seal")
    validate_files(root, m)
    return m


def arms_seal(root: Path, out: Path) -> None:
    covered = []
    shards = []
    state_count = 0
    for mp in sorted(root.rglob("arm_manifest.json")):
        m = json.loads(mp.read_text())
        start, end = int(m["start"]), int(m["end"])
        check_range(start, end)
        validate_arm_shard(mp.parent, start, end)
        covered.extend(range(start, end))
        shards.append({"start": start, "end": end, "manifest_sha256": fd.sha256(mp)})
        for rep in range(start, end):
            for arm in ALL_ARMS:
                p = mp.parent / "states" / f"rep{rep}_{arm}.pt"
                ck = torch.load(p, map_location="cpu", weights_only=True)
                if (ck["rep"], ck["arm"], ck["protocol_hash"]) != (rep, arm, PH):
                    raise ValueError("state metadata")
                state_count += 1
    if sorted(covered) != list(REPS):
        raise ValueError("global arm coverage")
    if state_count != len(REPS) * len(ALL_ARMS):
        raise ValueError("global state count")
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "arms_seal.json",
        {
            "protocol_hash": PH,
            "stage": "arms",
            "reps": list(REPS),
            "state_count": state_count,
            "intervention_state_count": len(REPS) * len(ARMS),
            "reference_state_count": len(REPS),
            "shards": sorted(shards, key=lambda z: z["start"]),
        },
    )
    print(f"GLOBAL_ARMS_SEAL states={state_count}; heldout still not constructed", flush=True)


def evaluate(start: int, end: int, out: Path, global_seal: Path) -> None:
    check_range(start, end)
    validate_global_seal(global_seal, "arms")
    base.configure_determinism(1)
    validate_arm_shard(out, start, end)
    if (out / "evaluation_manifest.json").exists():
        raise ValueError("evaluation overwrite refused")

    _, _, x, y, *_ = cr.data_split()
    labels = torch.tensor(y)
    clean = torch.tensor(x)
    heldout = [torch.tensor(base.geometric_environment(x, int(s))) for s in HELDOUT_SEEDS]
    rows = []
    envrows = []
    files = {}

    for rep in range(start, end):
        for arm in ALL_ARMS:
            p = out / "states" / f"rep{rep}_{arm}.pt"
            ck = torch.load(p, map_location="cpu", weights_only=True)
            if (ck["rep"], ck["arm"], ck["protocol_hash"]) != (rep, arm, PH):
                raise ValueError("state metadata")
            model = SmallCNN()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            vals = []
            with torch.no_grad():
                clean_acc = float((model(clean)[0].argmax(1) == labels).float().mean())
                for s, xx in zip(HELDOUT_SEEDS, heldout):
                    acc = float((model(xx)[0].argmax(1) == labels).float().mean())
                    vals.append(acc)
                    envrows.append({"rep": rep, "arm": arm, "env_seed": s, "accuracy": acc})
            v = np.asarray(vals, np.float64)
            rows.append(
                {
                    "rep": rep,
                    "arm": arm,
                    "mean_test": float(v.mean()),
                    "sd_test": float(v.std(ddof=1)),
                    "p10_test": float(np.quantile(v, 0.1)),
                    "min_test": float(v.min()),
                    "clean_test": clean_acc,
                    "state_digest": state_digest(ck["state_dict"]),
                }
            )
        print(f"CENTERED_SPAN_EVALUATED rep={rep}", flush=True)

    hp = out / f"centered_span_heldout_{start}_{end}.csv"
    ep = out / f"centered_span_environment_{start}_{end}.csv.gz"
    pd.DataFrame(rows).to_csv(hp, index=False)
    pd.DataFrame(envrows).to_csv(ep, index=False, compression="gzip")
    files[hp.name] = fd.sha256(hp)
    files[ep.name] = fd.sha256(ep)
    write_json(
        out / "evaluation_manifest.json",
        {
            "protocol_hash": PH,
            "start": start,
            "end": end,
            "files": files,
            "arm_manifest_sha256": fd.sha256(out / "arm_manifest.json"),
            "global_arms_seal_sha256": fd.sha256(global_seal),
            "source_hashes": source_hashes(),
            "split_hashes": cr.split_hashes(),
            "runtime": fd.runtime(),
            "heldout_constructed": True,
        },
    )


def validate_evaluation_shard(root: Path, start: int, end: int) -> dict:
    arm = validate_arm_shard(root, start, end)
    m = json.loads((root / "evaluation_manifest.json").read_text())
    if (m["start"], m["end"], m["protocol_hash"]) != (start, end, PH):
        raise ValueError("evaluation protocol")
    if m["source_hashes"] != source_hashes() or m["split_hashes"] != cr.split_hashes():
        raise ValueError("evaluation source/split")
    if m["arm_manifest_sha256"] != fd.sha256(root / "arm_manifest.json") or not m["heldout_constructed"]:
        raise ValueError("evaluation provenance")
    validate_files(root, m)
    return m


def estimate(x) -> dict:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("invalid statistical sample")
    mean = float(a.mean())
    se = float(stats.sem(a))
    k = float(stats.t.ppf(0.975, len(a) - 1))
    if se > 0:
        p = float(stats.t.sf(mean / se, len(a) - 1))
    else:
        p = 0.0 if mean > 0 else 1.0 if mean < 0 else 0.5
    return {
        "n": len(a),
        "mean": mean,
        "se": se,
        "ci95_low": mean - k * se,
        "ci95_high": mean + k * se,
        "p_one_sided": p,
        "positive_pairs": int((a > 0).sum()),
    }


def equivalent(x, margin: float) -> dict:
    e = estimate(x)
    mean, se, n = e["mean"], e["se"], e["n"]
    k = float(stats.t.ppf(0.95, n - 1))
    lo, hi = mean - k * se, mean + k * se
    if se > 0:
        p_lower = float(stats.t.sf((mean + margin) / se, n - 1))
        p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))
    else:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    return {
        "mean": mean,
        "se": se,
        "margin": margin,
        "ci90_low": lo,
        "ci90_high": hi,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "pass": bool(lo > -margin and hi < margin and p_lower < 0.05 and p_upper < 0.05),
    }


def summarize(root: Path, out: Path, global_seal: Path) -> None:
    validate_global_seal(global_seal, "arms")
    covered = []
    state_count = 0
    for mp in sorted(root.rglob("evaluation_manifest.json")):
        m = json.loads(mp.read_text())
        start, end = int(m["start"]), int(m["end"])
        check_range(start, end)
        validate_evaluation_shard(mp.parent, start, end)
        covered.extend(range(start, end))
        for rep in range(start, end):
            for arm in ALL_ARMS:
                p = mp.parent / "states" / f"rep{rep}_{arm}.pt"
                ck = torch.load(p, map_location="cpu", weights_only=True)
                if (ck["rep"], ck["arm"], ck["protocol_hash"]) != (rep, arm, PH):
                    raise ValueError("summary state metadata")
                state_count += 1
    if sorted(covered) != list(REPS) or state_count != 90:
        raise ValueError("summary global coverage")

    steps = load_frames(root, "centered_span_reference_*.csv.gz")
    held = load_frames(root, "centered_span_heldout_*.csv")
    env = load_frames(root, "centered_span_environment_*.csv.gz")
    expected_steps = {(r, t) for r in REPS for t in range(STEPS)}
    expected_held = {(r, a) for r in REPS for a in ALL_ARMS}
    expected_env = {(r, a, s) for r in REPS for a in ALL_ARMS for s in HELDOUT_SEEDS}
    if len(steps) != len(expected_steps) or steps.duplicated(["rep", "step"]).any() or set(steps[["rep", "step"]].itertuples(index=False, name=None)) != expected_steps:
        raise ValueError("summary step grid")
    if len(held) != len(expected_held) or held.duplicated(["rep", "arm"]).any() or set(held[["rep", "arm"]].itertuples(index=False, name=None)) != expected_held:
        raise ValueError("summary heldout grid")
    if len(env) != len(expected_env) or env.duplicated(["rep", "arm", "env_seed"]).any() or set(env[["rep", "arm", "env_seed"]].itertuples(index=False, name=None)) != expected_env:
        raise ValueError("summary environment grid")

    numeric_step = [
        "centered_span_gap",
        "gradient_novelty_diff",
        "gradient_novelty_abs_diff",
        "hardness_diff",
        "loss_variance_z_diff",
        "parameter_z_diff",
        "translation_diff",
        "step_max_general_identity_error",
    ]
    if not np.isfinite(steps[numeric_step].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite reference quantities")
    if not np.isfinite(held[["mean_test", "sd_test", "p10_test", "min_test", "clean_test"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite heldout quantities")
    if not np.isfinite(env[["accuracy"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite environment quantities")
    if float(steps.step_max_general_identity_error.abs().max()) > MAX_GENERAL_IDENTITY_ERROR:
        raise ValueError("general identity failure")

    supported = steps.supported.astype(bool)
    live = steps[supported]
    if (
        (live.gradient_novelty_abs_diff > G_MARGIN + 1e-12).any()
        or (live.hardness_diff.abs() > H_MARGIN + 1e-12).any()
        or (live.loss_variance_z_diff.abs() > L_MARGIN + 1e-12).any()
        or (live.parameter_z_diff.abs() > P_MARGIN + 1e-12).any()
        or (live.translation_diff.abs() > T_MARGIN + 1e-12).any()
    ):
        raise ValueError("summary caliper violation")
    support_by_rep = steps.assign(sb=supported).groupby("rep").sb.mean().reindex(REPS)
    pooled_support = float(supported.mean())
    min_support = float(support_by_rep.min())
    if pooled_support < MIN_POOLED_SUPPORT or min_support < MIN_REP_SUPPORT:
        raise ValueError("Stage-B support failure")

    hz = held.set_index(["rep", "arm"])
    max_env_error = 0.0
    for (rep, arm), block in env.groupby(["rep", "arm"]):
        v = block.sort_values("env_seed").accuracy.to_numpy(dtype=np.float64)
        observed = np.array([v.mean(), v.std(ddof=1), np.quantile(v, 0.1), v.min()])
        reported = hz.loc[(rep, arm), ["mean_test", "sd_test", "p10_test", "min_test"]].to_numpy(dtype=float)
        max_env_error = max(max_env_error, float(np.max(np.abs(observed - reported))))
    if max_env_error > 1e-12:
        raise ValueError("environment reaggregation")

    ref = steps.groupby("rep").mean(numeric_only=True).reindex(REPS)
    paired = []
    metrics = ("mean_test", "sd_test", "p10_test", "min_test", "clean_test")
    for rep in REPS:
        row = {
            "rep": rep,
            "support_fraction": float(support_by_rep.loc[rep]),
            "centered_span_gap": float(ref.loc[rep, "centered_span_gap"]),
            "gradient_novelty_diff": float(ref.loc[rep, "gradient_novelty_diff"]),
            "hardness_diff": float(ref.loc[rep, "hardness_diff"]),
            "loss_variance_z_diff": float(ref.loc[rep, "loss_variance_z_diff"]),
            "parameter_z_diff": float(ref.loc[rep, "parameter_z_diff"]),
            "translation_diff": float(ref.loc[rep, "translation_diff"]),
        }
        for met in metrics:
            row[met + "_benefit"] = float(hz.loc[(rep, "high"), met] - hz.loc[(rep, "low"), met])
            row["high_" + met + "_vs_reference"] = float(hz.loc[(rep, "high"), met] - hz.loc[(rep, "reference"), met])
            row["low_" + met + "_vs_reference"] = float(hz.loc[(rep, "low"), met] - hz.loc[(rep, "reference"), met])
        paired.append(row)
    paired = pd.DataFrame(paired)

    balances = []
    gates = True
    for col, margin in (
        ("hardness_diff", H_MARGIN),
        ("loss_variance_z_diff", L_MARGIN),
        ("parameter_z_diff", P_MARGIN),
        ("translation_diff", T_MARGIN),
        ("gradient_novelty_diff", G_MARGIN),
    ):
        result = equivalent(paired[col], margin)
        balances.append({"endpoint": col, **result})
        gates &= bool(result["pass"])
    span = estimate(paired.centered_span_gap)
    span_pass = bool(span["mean"] >= SPAN_MIN and span["p_one_sided"] < 0.05)
    gates &= span_pass

    perf = estimate(paired.mean_test_benefit)
    perf_pass = bool(perf["mean"] > 0 and perf["p_one_sided"] < 0.05)
    if not gates:
        decision = "CENTERED SPAN MATCH OR MANIPULATION FAILURE"
    elif perf_pass:
        decision = "G-MATCHED CENTERED GRADIENT-SPAN SUPPORT"
    else:
        decision = "NO CENTERED GRADIENT-SPAN PERFORMANCE SUPPORT"

    summary_rows = []
    for col in paired.columns:
        if col != "rep":
            summary_rows.append({"endpoint": col, **estimate(paired[col])})
    decision_row = {
        "decision": decision,
        "n": len(REPS),
        "selected_g_caliper": SELECTED_G_CALIPER,
        "all_balance_manipulation_gates_pass": bool(gates),
        "span_manipulation_pass": span_pass,
        "performance_pass": perf_pass,
        "pooled_support": pooled_support,
        "minimum_replicate_support": min_support,
        "fallback_steps": int((~supported).sum()),
        "states_checked": state_count,
        "environment_rows": len(env),
        "max_environment_reaggregation_error": max_env_error,
        "maximum_general_identity_error": float(steps.step_max_general_identity_error.abs().max()),
        "protocol_hash": PH,
    }
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "centered_span_primary30.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "centered_span_summary30.csv", index=False)
    pd.DataFrame(balances).to_csv(out / "centered_span_balance30.csv", index=False)
    held.to_csv(out / "centered_span_heldout30.csv", index=False)
    ref.reset_index().to_csv(out / "centered_span_reference_summary30.csv", index=False)
    pd.DataFrame([decision_row]).to_csv(out / "centered_span_decision30.csv", index=False)
    print(json.dumps(decision_row), flush=True)
    print("PRIMARY", json.dumps(perf), flush=True)
    print("SPAN", json.dumps(span), flush=True)
    print(pd.DataFrame(balances).to_string(index=False), flush=True)


def selftest() -> None:
    cal.selftest()
    assert equivalent(np.zeros(30), 0.05)["pass"]
    assert not equivalent(np.full(30, 0.05), 0.05)["pass"]
    a = np.linspace(0.25, 0.35, 30)
    assert estimate(a)["mean"] >= SPAN_MIN and estimate(a)["p_one_sided"] < 0.05
    print("CENTERED_SPAN_CONFIRMATION_SYNTHETIC_PASS", PH)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("selftest", "reference", "reference-seal", "train", "arms-seal", "evaluate", "summarize"))
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--input-dir")
    p.add_argument("--output-dir")
    p.add_argument("--global-seal")
    a = p.parse_args()
    if a.mode == "selftest":
        selftest(); return
    if a.output_dir is None:
        raise ValueError("output-dir required")
    out = Path(a.output_dir)
    if a.mode == "reference":
        reference(a.start, a.end, out)
    elif a.mode == "reference-seal":
        reference_seal(Path(a.input_dir), out)
    elif a.mode == "train":
        train(a.start, a.end, out, Path(a.global_seal))
    elif a.mode == "arms-seal":
        arms_seal(Path(a.input_dir), out)
    elif a.mode == "evaluate":
        evaluate(a.start, a.end, out, Path(a.global_seal))
    elif a.mode == "summarize":
        summarize(Path(a.input_dir), out, Path(a.global_seal))


if __name__ == "__main__":
    main()
