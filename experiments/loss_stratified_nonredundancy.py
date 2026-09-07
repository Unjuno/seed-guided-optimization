"""Issue #80: shared-reference loss-stratified gradient-redundancy intervention."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS = tuple(range(1600, 1630))
ARMS = ("maxnov", "minnov")
K = 16
Q = 4
BATCH = 128
EPOCHS = 10
LR = 5e-3
WD = 1e-3
TRAIN_SEEDS = tuple(range(51000, 51064))
HELDOUT_SEEDS = tuple(range(52000, 52080))
HARDNESS_MARGIN = 0.10
EPS = 1e-8
PROTOCOL = {
    "issue": 80,
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
    "base_seed_offset": 2010000000,
    "base_seed_stride": 4099,
    "hardness_pairs": ((2, 3), (4, 5), (6, 7)),
    "hardness_margin_sd": HARDNESS_MARGIN,
    "reference_update": "loss_hard_top4",
    "evaluation": "canonical_nontraining_union_809",
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
LOCAL_SOURCES = (
    "common.py",
    "cnn_regime_interaction.py",
    "fixed_dose_response.py",
    "transfer_specificity.py",
    "loss_stratified_nonredundancy.py",
)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build_schedule(n: int, seed: int) -> list[tuple[torch.Tensor, list[int]]]:
    tg = torch.Generator().manual_seed(seed + 1)
    er = np.random.default_rng(seed + 2)
    out = []
    for _ in range(EPOCHS):
        for b in torch.randperm(n, generator=tg).split(BATCH):
            out.append((b, er.choice(len(TRAIN_SEEDS), K, replace=False).tolist()))
    return out


def base_schedule_digest(sched: list[tuple[torch.Tensor, list[int]]]) -> str:
    h = hashlib.sha256()
    for b, cand in sched:
        h.update(b.detach().cpu().numpy().astype(np.int64).tobytes())
        h.update(np.asarray(cand, dtype=np.int64).tobytes())
    return h.hexdigest()


def pairnov(cos: torch.Tensor, sel: list[int]) -> float:
    idx = torch.tensor(sel, dtype=torch.long)
    sub = cos.index_select(0, idx).index_select(1, idx)
    up = torch.triu_indices(len(sel), len(sel), offset=1)
    return float((1.0 - sub[up[0], up[1]]).mean()) if up.shape[1] else 0.0


def stratified_extremes(losses: torch.Tensor, cos: torch.Tensor) -> tuple[list[int], list[int], dict]:
    if len(losses) != K or cos.shape != (K, K):
        raise ValueError("candidate shape mismatch")
    order = sorted(range(K), key=lambda i: (-float(losses[i].detach()), i))
    anchor = order[0]
    strata = ((order[1], order[2]), (order[3], order[4]), (order[5], order[6]))
    feasible = []
    for bits in itertools.product((0, 1), repeat=3):
        sel = sorted([anchor] + [strata[j][bits[j]] for j in range(3)])
        feasible.append((pairnov(cos, sel), tuple(sel)))
    max_item = sorted(feasible, key=lambda x: (-x[0], x[1]))[0]
    min_item = sorted(feasible, key=lambda x: (x[0], x[1]))[0]
    max_sel, min_sel = list(max_item[1]), list(min_item[1])
    candidate_mean = float(losses.mean().detach())
    candidate_sd = float(losses.detach().double().std(unbiased=False))

    def diagnostics(sel: list[int], nov: float) -> dict:
        selected_loss = float(losses[sel].mean().detach())
        z = (selected_loss - candidate_mean) / (candidate_sd + EPS)
        return {"novelty": nov, "selected_loss": selected_loss, "hardness_z": z}

    dmax = diagnostics(max_sel, max_item[0])
    dmin = diagnostics(min_sel, min_item[0])
    return max_sel, min_sel, {
        "order": order,
        "strata": strata,
        "candidate_mean_loss": candidate_mean,
        "candidate_sd_loss": candidate_sd,
        "max": dmax,
        "min": dmin,
    }


def check_range(start: int, end: int) -> None:
    if start >= end or not set(range(start, end)).issubset(REPS):
        raise ValueError("unregistered range")


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: fd.sha256(root / name) for name in LOCAL_SOURCES}


def build_reference(start: int, end: int, out: Path) -> None:
    check_range(start, end)
    base.configure_determinism(1)
    x, y, _, _, _, _, _, _ = cr.data_split()
    ty = torch.tensor(y)
    envs = base.geometric_envs(x, TRAIN_SEEDS)
    out.mkdir(parents=True, exist_ok=True)
    sched_dir = out / "schedules"
    sched_dir.mkdir(exist_ok=True)
    if list(sched_dir.glob("*.json")):
        raise ValueError("refuse schedule overwrite")

    all_steps = []
    schedule_hashes = {}
    base_hashes = {}
    for rep in range(start, end):
        seed = 2010000000 + 4099 * rep
        sched = build_schedule(len(ty), seed)
        base_hashes[str(rep)] = base_schedule_digest(sched)
        seed_everything(seed)
        model = SmallCNN()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        model.train()
        max_schedule = []
        min_schedule = []
        for step, (b, cand) in enumerate(sched):
            xb = torch.cat([envs[e][b] for e in cand])
            logits, h = model(xb)
            per = torch.nn.functional.cross_entropy(
                logits, ty[b].repeat(K), reduction="none"
            ).reshape(K, -1)
            losses = per.mean(1)
            dirs = head_gradient_directions(logits, h, ty[b], K)
            cos = dirs @ dirs.T
            max_sel, min_sel, diag = stratified_extremes(losses, cos)
            max_envs = [int(cand[i]) for i in max_sel]
            min_envs = [int(cand[i]) for i in min_sel]
            max_schedule.append(max_envs)
            min_schedule.append(min_envs)
            all_steps.append({
                "rep": rep,
                "step": step,
                "max_novelty": diag["max"]["novelty"],
                "min_novelty": diag["min"]["novelty"],
                "novelty_diff": diag["max"]["novelty"] - diag["min"]["novelty"],
                "candidate_mean_loss": diag["candidate_mean_loss"],
                "candidate_sd_loss": diag["candidate_sd_loss"],
                "max_selected_loss": diag["max"]["selected_loss"],
                "min_selected_loss": diag["min"]["selected_loss"],
                "raw_selected_loss_diff": diag["max"]["selected_loss"] - diag["min"]["selected_loss"],
                "max_hardness_z": diag["max"]["hardness_z"],
                "min_hardness_z": diag["min"]["hardness_z"],
                "hardness_diff": diag["max"]["hardness_z"] - diag["min"]["hardness_z"],
                "schedule_overlap": len(set(max_envs) & set(min_envs)) / Q,
                "max_envs": ";".join(map(str, max_envs)),
                "min_envs": ";".join(map(str, min_envs)),
            })
            ref_sel = sorted(diag["order"][:Q])
            opt.zero_grad(set_to_none=True)
            per[ref_sel].mean().backward()
            opt.step()
        payload = {
            "rep": rep,
            "protocol_hash": PROTOCOL_HASH,
            "base_schedule_digest": base_hashes[str(rep)],
            "maxnov": max_schedule,
            "minnov": min_schedule,
        }
        path = sched_dir / f"rep{rep}_schedules.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        schedule_hashes[path.name] = fd.sha256(path)
        print(f"REFERENCE_SEALED rep={rep}; arms not trained", flush=True)

    pd.DataFrame(all_steps).to_csv(
        out / f"nonredundancy_reference_steps_{start}_{end}.csv.gz",
        index=False,
        compression="gzip",
    )
    manifest = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "start": start,
        "end": end,
        "schedule_hashes": schedule_hashes,
        "base_schedule_hashes": base_hashes,
        "source_hashes": source_hashes(),
        "split_hashes": cr.split_hashes(),
        "runtime": fd.runtime(),
        "n_train": 988,
        "n_eval": 809,
    }
    (out / "reference_manifest.json").write_text(json.dumps(manifest, indent=2))


def train_fixed_arm(
    envs: list[torch.Tensor],
    y: torch.Tensor,
    sched: list[tuple[torch.Tensor, list[int]]],
    selected_schedule: list[list[int]],
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    if len(sched) != len(selected_schedule):
        raise ValueError("schedule length mismatch")
    seed_everything(seed)
    model = SmallCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    selected_losses = []
    model.train()
    for (b, cand), selected_envs in zip(sched, selected_schedule):
        if len(selected_envs) != Q or not set(selected_envs).issubset(set(cand)):
            raise ValueError("selected environments not in candidate set")
        xb = torch.cat([envs[e][b] for e in selected_envs])
        logits, _ = model(xb)
        per = torch.nn.functional.cross_entropy(
            logits, y[b].repeat(Q), reduction="none"
        ).reshape(Q, -1)
        selected_losses.append(float(per.mean().detach()))
        opt.zero_grad(set_to_none=True)
        per.mean().backward()
        opt.step()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, {
        "mean_arm_selected_loss": float(np.mean(selected_losses)),
        "state_digest": state_digest(state),
    }


def train_arms(start: int, end: int, out: Path) -> None:
    check_range(start, end)
    base.configure_determinism(1)
    ref = json.loads((out / "reference_manifest.json").read_text())
    if (ref["start"], ref["end"], ref["protocol_hash"]) != (start, end, PROTOCOL_HASH):
        raise ValueError("reference manifest mismatch")
    if ref["split_hashes"] != cr.split_hashes() or ref["source_hashes"] != source_hashes():
        raise ValueError("reference source/split changed")
    for name, digest in ref["schedule_hashes"].items():
        if fd.sha256(out / "schedules" / name) != digest:
            raise ValueError("schedule changed after seal")

    x, y, _, _, _, _, _, _ = cr.data_split()
    ty = torch.tensor(y)
    envs = base.geometric_envs(x, TRAIN_SEEDS)
    states_dir = out / "states"
    states_dir.mkdir(exist_ok=True)
    if list(states_dir.glob("*.pt")):
        raise ValueError("refuse checkpoint overwrite")
    rows = []
    hashes = {}
    for rep in range(start, end):
        seed = 2010000000 + 4099 * rep
        sched = build_schedule(len(ty), seed)
        if base_schedule_digest(sched) != ref["base_schedule_hashes"][str(rep)]:
            raise ValueError("base schedule changed")
        spath = out / "schedules" / f"rep{rep}_schedules.json"
        payload = json.loads(spath.read_text())
        if (
            payload["rep"] != rep
            or payload["protocol_hash"] != PROTOCOL_HASH
            or payload["base_schedule_digest"] != ref["base_schedule_hashes"][str(rep)]
        ):
            raise ValueError("schedule metadata mismatch")
        for arm in ARMS:
            state, diag = train_fixed_arm(envs, ty, sched, payload[arm], seed)
            path = states_dir / f"rep{rep}_{arm}.pt"
            torch.save(
                {
                    "state_dict": state,
                    "rep": rep,
                    "arm": arm,
                    "protocol_hash": PROTOCOL_HASH,
                    "schedule_sha256": ref["schedule_hashes"][spath.name],
                },
                path,
            )
            hashes[path.name] = fd.sha256(path)
            rows.append({"rep": rep, "arm": arm, **diag})
        print(f"ARMS_SEALED rep={rep}; heldout not constructed", flush=True)

    pd.DataFrame(rows).to_csv(out / f"nonredundancy_arm_training_{start}_{end}.csv", index=False)
    manifest = {
        "protocol_hash": PROTOCOL_HASH,
        "start": start,
        "end": end,
        "checkpoint_hashes": hashes,
        "schedule_hashes": ref["schedule_hashes"],
        "base_schedule_hashes": ref["base_schedule_hashes"],
        "source_hashes": ref["source_hashes"],
        "split_hashes": ref["split_hashes"],
        "runtime": fd.runtime(),
    }
    (out / "arm_manifest.json").write_text(json.dumps(manifest, indent=2))


def evaluate(start: int, end: int, out: Path) -> None:
    check_range(start, end)
    base.configure_determinism(1)
    ref = json.loads((out / "reference_manifest.json").read_text())
    arm = json.loads((out / "arm_manifest.json").read_text())
    if (arm["start"], arm["end"], arm["protocol_hash"]) != (start, end, PROTOCOL_HASH):
        raise ValueError("arm manifest mismatch")
    if arm["split_hashes"] != cr.split_hashes() or arm["source_hashes"] != source_hashes():
        raise ValueError("source/split mismatch")
    for name, digest in ref["schedule_hashes"].items():
        if fd.sha256(out / "schedules" / name) != digest:
            raise ValueError("schedule changed")
    for name, digest in arm["checkpoint_hashes"].items():
        if fd.sha256(out / "states" / name) != digest:
            raise ValueError("checkpoint changed")

    _, _, x, y, _, _, _, _ = cr.data_split()
    labels = torch.tensor(y)
    clean = torch.tensor(x)
    heldout = [torch.tensor(base.geometric_environment(x, int(s))) for s in HELDOUT_SEEDS]
    rows = []
    perenv = []
    for rep in range(start, end):
        for arm_name in ARMS:
            path = out / "states" / f"rep{rep}_{arm_name}.pt"
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if (ck["rep"], ck["arm"], ck["protocol_hash"]) != (rep, arm_name, PROTOCOL_HASH):
                raise ValueError("checkpoint metadata")
            model = SmallCNN()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            acc = []
            with torch.no_grad():
                clean_acc = float((model(clean)[0].argmax(1) == labels).float().mean())
                for env_seed, xx in zip(HELDOUT_SEEDS, heldout):
                    pred = model(xx)[0].argmax(1)
                    a = float((pred == labels).float().mean())
                    acc.append(a)
                    perenv.append({"rep": rep, "arm": arm_name, "env_seed": env_seed, "accuracy": a})
            a = np.asarray(acc, dtype=np.float64)
            rows.append({
                "rep": rep,
                "arm": arm_name,
                "mean_test": float(a.mean()),
                "sd_test": float(a.std(ddof=1)),
                "p10_test": float(np.quantile(a, 0.1)),
                "min_test": float(a.min()),
                "clean_test": clean_acc,
                "state_digest": state_digest(ck["state_dict"]),
            })
    for name, digest in arm["checkpoint_hashes"].items():
        if fd.sha256(out / "states" / name) != digest:
            raise ValueError("checkpoint mutated during evaluation")
    pd.DataFrame(rows).to_csv(out / f"nonredundancy_heldout_{start}_{end}.csv", index=False)
    pd.DataFrame(perenv).to_csv(
        out / f"nonredundancy_environment_{start}_{end}.csv.gz",
        index=False,
        compression="gzip",
    )
    print(f"EVALUATION_COMPLETE {start}:{end}; n_eval=809", flush=True)


def selftest() -> None:
    base.configure_determinism(1)
    losses = torch.linspace(2.0, 0.5, K)
    g = torch.arange(K * 5, dtype=torch.float32).reshape(K, 5) + 1.0
    g = g / g.norm(dim=1, keepdim=True)
    cos = g @ g.T
    max_sel, min_sel, diag = stratified_extremes(losses, cos)
    order = diag["order"]
    assert len(max_sel) == len(min_sel) == Q
    assert order[0] in max_sel and order[0] in min_sel
    for lo, hi in ((1, 2), (3, 4), (5, 6)):
        pair = {order[lo], order[hi]}
        assert len(pair & set(max_sel)) == 1
        assert len(pair & set(min_sel)) == 1
    assert diag["max"]["novelty"] >= diag["min"]["novelty"]
    sched = build_schedule(988, 12345)
    assert base_schedule_digest(sched) == base_schedule_digest(build_schedule(988, 12345))
    print("SELFTEST PASS", PROTOCOL_HASH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "reference", "train", "evaluate"))
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--output-dir")
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest()
        return
    if a.start is None or a.end is None or a.output_dir is None:
        raise ValueError("range/output required")
    out = Path(a.output_dir)
    if a.mode == "reference":
        build_reference(a.start, a.end, out)
    elif a.mode == "train":
        train_arms(a.start, a.end, out)
    else:
        evaluate(a.start, a.end, out)


if __name__ == "__main__":
    main()
