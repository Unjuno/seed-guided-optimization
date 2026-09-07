"""Issue #83 Stage B: parameter-diversity-matched gradient-novelty reversal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cnn_regime_interaction as cr
import fixed_dose_response as fd
import parameter_matched_novelty_calibration as cal
import transfer_specificity as base
from common import SmallCNN, environment_parameters, head_gradient_directions, seed_everything

REPS = tuple(range(1800, 1830))
ARMS = ("maxnov", "minnov")
K = 16
Q = 4
BATCH = 128
EPOCHS = 10
LR = 5e-3
WD = 1e-3
TRAIN_SEEDS = tuple(range(55000, 55064))
HELDOUT_SEEDS = tuple(range(56000, 56080))
CALIPER = 0.05
HARDNESS_EQ_MARGIN = 0.05
PARAM_EQ_MARGIN = 0.05
STAGE_A_PROTOCOL_HASH = "444ed1dd41a200f03e883b0e86583b92e86f4f182226f4f6fac3f7fed3a5b9d3"
PROTOCOL = {
    "issue": 83,
    "stage": "B",
    "reps": REPS,
    "arms": ARMS,
    "K": K,
    "Q": Q,
    "batch": BATCH,
    "epochs": EPOCHS,
    "lr": LR,
    "wd": WD,
    "train_seeds": TRAIN_SEEDS,
    "heldout_seeds": HELDOUT_SEEDS,
    "selected_caliper": CALIPER,
    "hardness_equivalence_margin": HARDNESS_EQ_MARGIN,
    "parameter_equivalence_margin": PARAM_EQ_MARGIN,
    "stage_a_protocol_hash": STAGE_A_PROTOCOL_HASH,
    "base_seed_offset": 2210000000,
    "base_seed_stride": 4099,
    "feasible_ranks": tuple(range(2, 11)),
    "evaluation": "canonical_nontraining_union_809",
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
LOCAL_SOURCES = (
    "common.py",
    "cnn_regime_interaction.py",
    "fixed_dose_response.py",
    "transfer_specificity.py",
    "parameter_matched_novelty_calibration.py",
    "parameter_matched_novelty_confirmatory.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {n: fd.sha256(root / n) for n in LOCAL_SOURCES}


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def build_schedule(n: int, seed: int):
    tg = torch.Generator().manual_seed(seed + 1)
    er = np.random.default_rng(seed + 2)
    out = []
    for _ in range(EPOCHS):
        for b in torch.randperm(n, generator=tg).split(BATCH):
            out.append((b, er.choice(len(TRAIN_SEEDS), K, replace=False).tolist()))
    return out


def schedule_digest(sched) -> str:
    h = hashlib.sha256()
    for b, cand in sched:
        h.update(b.detach().cpu().numpy().astype(np.int64).tobytes())
        h.update(np.asarray(cand, dtype=np.int64).tobytes())
    return h.hexdigest()


def check_range(start: int, end: int):
    if start >= end or not set(range(start, end)).issubset(REPS):
        raise ValueError("unregistered confirmatory range")


def parameter_matrix() -> np.ndarray:
    raw = np.stack([environment_parameters(int(s)) for s in TRAIN_SEEDS]).astype(np.float64)
    return (raw - raw.mean(0, keepdims=True)) / (raw.std(0, ddof=0, keepdims=True) + 1e-12)


def parse_envs(text: str) -> list[int]:
    return [int(x) for x in text.split(";")]


def build_reference(start: int, end: int, out: Path):
    check_range(start, end); base.configure_determinism(1)
    x, y, _, _, _, _, _, _ = cr.data_split(); ty = torch.tensor(y)
    envs = base.geometric_envs(x, TRAIN_SEEDS); params = parameter_matrix()
    out.mkdir(parents=True, exist_ok=True); sdir = out / "schedules"; sdir.mkdir(exist_ok=True)
    if list(sdir.glob("*.json")): raise ValueError("schedule overwrite refused")
    steps = []; schedule_hashes = {}; base_hashes = {}
    for rep in range(start, end):
        seed = 2210000000 + 4099 * rep; sched = build_schedule(len(ty), seed); base_hashes[str(rep)] = schedule_digest(sched)
        seed_everything(seed); model = SmallCNN(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); model.train()
        hi_sched = []; lo_sched = []
        for step, (b, cand) in enumerate(sched):
            xb = torch.cat([envs[e][b] for e in cand]); logits, h = model(xb)
            per = torch.nn.functional.cross_entropy(logits, ty[b].repeat(K), reduction="none").reshape(K, -1)
            losses = per.mean(1); gd = head_gradient_directions(logits, h, ty[b], K); cos = gd @ gd.T
            subsets = cal.feasible_subsets(losses, cos, cand, params); pair = cal.best_pair(subsets, CALIPER)
            if pair is None:
                raise RuntimeError(f"CONFIRMATORY MATCH FEASIBILITY FAIL rep={rep} step={step} caliper={CALIPER}")
            high_envs = parse_envs(pair["high_envs"]); low_envs = parse_envs(pair["low_envs"])
            if len(high_envs) != Q or len(low_envs) != Q: raise ValueError("selected schedule size")
            hi_sched.append(high_envs); lo_sched.append(low_envs)
            p_hi = cal.physical_diversity(tuple(high_envs), params); p_lo = cal.physical_diversity(tuple(low_envs), params)
            local = {int(e): i for i, e in enumerate(cand)}
            hi_local = [local[e] for e in high_envs]; lo_local = [local[e] for e in low_envs]
            raw_loss_diff = float(losses[hi_local].mean().detach() - losses[lo_local].mean().detach())
            steps.append({
                "rep": rep, "step": step,
                "gradient_gap": pair["gradient_gap"],
                "hardness_diff": pair["hardness_diff"],
                "parameter_z_diff": pair["parameter_z_diff"],
                "parameter_abs_z_diff": pair["parameter_abs_z_diff"],
                "raw_parameter_diversity_diff": p_hi - p_lo,
                "raw_selected_loss_diff": raw_loss_diff,
                "high_envs": pair["high_envs"], "low_envs": pair["low_envs"],
                "schedule_overlap": len(set(high_envs) & set(low_envs)) / Q,
            })
            order = sorted(range(K), key=lambda i: (-float(losses[i].detach()), i))[:Q]; order = sorted(order)
            opt.zero_grad(set_to_none=True); per[order].mean().backward(); opt.step()
        payload = {"rep": rep, "protocol_hash": PROTOCOL_HASH, "base_schedule_digest": base_hashes[str(rep)], "maxnov": hi_sched, "minnov": lo_sched}
        sp = sdir / f"rep{rep}_schedules.json"; sp.write_text(json.dumps(payload, separators=(",", ":"))); schedule_hashes[sp.name] = fd.sha256(sp)
        print(f"MATCHED_REFERENCE_SEALED rep={rep}; arms not trained", flush=True)
    pd.DataFrame(steps).to_csv(out / f"parameter_matched_reference_steps_{start}_{end}.csv.gz", index=False, compression="gzip")
    manifest = {"protocol": PROTOCOL, "protocol_hash": PROTOCOL_HASH, "start": start, "end": end, "schedule_hashes": schedule_hashes, "base_schedule_hashes": base_hashes, "source_hashes": source_hashes(), "split_hashes": cr.split_hashes(), "runtime": fd.runtime(), "n_train": 988, "n_eval": 809, "heldout_constructed": False}
    (out / "reference_manifest.json").write_text(json.dumps(manifest, indent=2))


def train_arm(envs, y, sched, selected_schedule, seed):
    seed_everything(seed); model = SmallCNN(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); losses_out = []; model.train()
    for (b, cand), chosen in zip(sched, selected_schedule):
        if len(chosen) != Q or not set(chosen).issubset(set(cand)): raise ValueError("fixed schedule mismatch")
        xb = torch.cat([envs[e][b] for e in chosen]); logits, _ = model(xb)
        per = torch.nn.functional.cross_entropy(logits, y[b].repeat(Q), reduction="none").reshape(Q, -1)
        losses_out.append(float(per.mean().detach())); opt.zero_grad(set_to_none=True); per.mean().backward(); opt.step()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, {"mean_arm_selected_loss": float(np.mean(losses_out)), "state_digest": state_digest(state)}


def train_arms(start: int, end: int, out: Path):
    check_range(start, end); base.configure_determinism(1)
    ref = json.loads((out / "reference_manifest.json").read_text())
    if (ref["start"], ref["end"], ref["protocol_hash"]) != (start, end, PROTOCOL_HASH): raise ValueError("reference manifest")
    if ref["source_hashes"] != source_hashes() or ref["split_hashes"] != cr.split_hashes(): raise ValueError("source/split changed")
    for name, dig in ref["schedule_hashes"].items():
        if fd.sha256(out / "schedules" / name) != dig: raise ValueError("schedule changed")
    x, y, _, _, _, _, _, _ = cr.data_split(); ty = torch.tensor(y); envs = base.geometric_envs(x, TRAIN_SEEDS)
    states = out / "states"; states.mkdir(exist_ok=True)
    if list(states.glob("*.pt")): raise ValueError("checkpoint overwrite")
    rows = []; hashes = {}
    for rep in range(start, end):
        seed = 2210000000 + 4099 * rep; sched = build_schedule(len(ty), seed)
        if schedule_digest(sched) != ref["base_schedule_hashes"][str(rep)]: raise ValueError("base schedule changed")
        sp = out / "schedules" / f"rep{rep}_schedules.json"; payload = json.loads(sp.read_text())
        for arm in ARMS:
            state, diag = train_arm(envs, ty, sched, payload[arm], seed); p = states / f"rep{rep}_{arm}.pt"
            torch.save({"state_dict": state, "rep": rep, "arm": arm, "protocol_hash": PROTOCOL_HASH, "schedule_sha256": ref["schedule_hashes"][sp.name]}, p)
            hashes[p.name] = fd.sha256(p); rows.append({"rep": rep, "arm": arm, **diag})
        print(f"MATCHED_ARMS_SEALED rep={rep}; heldout not constructed", flush=True)
    pd.DataFrame(rows).to_csv(out / f"parameter_matched_arm_training_{start}_{end}.csv", index=False)
    (out / "arm_manifest.json").write_text(json.dumps({"protocol_hash": PROTOCOL_HASH, "start": start, "end": end, "checkpoint_hashes": hashes, "schedule_hashes": ref["schedule_hashes"], "base_schedule_hashes": ref["base_schedule_hashes"], "source_hashes": ref["source_hashes"], "split_hashes": ref["split_hashes"], "runtime": fd.runtime()}, indent=2))


def evaluate(start: int, end: int, out: Path):
    check_range(start, end); base.configure_determinism(1)
    ref = json.loads((out / "reference_manifest.json").read_text()); arm = json.loads((out / "arm_manifest.json").read_text())
    if (arm["start"], arm["end"], arm["protocol_hash"]) != (start, end, PROTOCOL_HASH): raise ValueError("arm manifest")
    if arm["source_hashes"] != source_hashes() or arm["split_hashes"] != cr.split_hashes(): raise ValueError("source/split mismatch")
    for n, d in ref["schedule_hashes"].items():
        if fd.sha256(out / "schedules" / n) != d: raise ValueError("schedule changed")
    for n, d in arm["checkpoint_hashes"].items():
        if fd.sha256(out / "states" / n) != d: raise ValueError("checkpoint changed")
    _, _, x, y, _, _, _, _ = cr.data_split(); labels = torch.tensor(y); clean = torch.tensor(x)
    heldout = [torch.tensor(base.geometric_environment(x, int(s))) for s in HELDOUT_SEEDS]
    rows = []; perenv = []
    for rep in range(start, end):
        for name in ARMS:
            p = out / "states" / f"rep{rep}_{name}.pt"; ck = torch.load(p, map_location="cpu", weights_only=True)
            model = SmallCNN(); model.load_state_dict(ck["state_dict"]); model.eval(); acc = []
            with torch.no_grad():
                clean_acc = float((model(clean)[0].argmax(1) == labels).float().mean())
                for s, xx in zip(HELDOUT_SEEDS, heldout):
                    a = float((model(xx)[0].argmax(1) == labels).float().mean()); acc.append(a); perenv.append({"rep": rep, "arm": name, "env_seed": s, "accuracy": a})
            a = np.asarray(acc, np.float64); rows.append({"rep": rep, "arm": name, "mean_test": float(a.mean()), "sd_test": float(a.std(ddof=1)), "p10_test": float(np.quantile(a, .1)), "min_test": float(a.min()), "clean_test": clean_acc, "state_digest": state_digest(ck["state_dict"])})
    for n, d in arm["checkpoint_hashes"].items():
        if fd.sha256(out / "states" / n) != d: raise ValueError("checkpoint mutated")
    pd.DataFrame(rows).to_csv(out / f"parameter_matched_heldout_{start}_{end}.csv", index=False)
    pd.DataFrame(perenv).to_csv(out / f"parameter_matched_environment_{start}_{end}.csv.gz", index=False, compression="gzip")
    print(f"MATCHED_EVALUATION_COMPLETE {start}:{end}", flush=True)


def selftest():
    base.configure_determinism(1); assert CALIPER == 0.05 and STAGE_A_PROTOCOL_HASH.startswith("444ed1dd")
    losses = torch.linspace(2.0, .5, K); g = torch.arange(K*7, dtype=torch.float32).reshape(K,7)+1; g = g/g.norm(dim=1, keepdim=True); cos = g@g.T
    params = np.arange(64*7, dtype=np.float64).reshape(64,7); params = (params-params.mean(0))/(params.std(0)+1e-12)
    subsets = cal.feasible_subsets(losses, cos, list(range(K)), params); assert len(subsets)==84
    pair = cal.best_pair(subsets, CALIPER)
    if pair is not None:
        assert abs(pair["hardness_diff"]) <= CALIPER + 1e-12 and abs(pair["parameter_z_diff"]) <= CALIPER + 1e-12
    print("SELFTEST PASS", PROTOCOL_HASH)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("selftest","reference","train","evaluate")); ap.add_argument("--start", type=int); ap.add_argument("--end", type=int); ap.add_argument("--output-dir"); a = ap.parse_args()
    if a.mode == "selftest": selftest(); return
    if a.start is None or a.end is None or a.output_dir is None: raise ValueError("range/output required")
    out = Path(a.output_dir)
    if a.mode == "reference": build_reference(a.start, a.end, out)
    elif a.mode == "train": train_arms(a.start, a.end, out)
    else: evaluate(a.start, a.end, out)

if __name__ == "__main__": main()
