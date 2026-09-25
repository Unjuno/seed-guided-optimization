"""Issue124: six-pool confirmation of online clean-versus-tail tradeoff.

Prospective protocol: docs/CLEAN_TAIL_CONFIRM_PROTOCOL.md registered at
bd81e737de8ec64bf88770495c8eb51b661da87f. This file is a dedicated
implementation; historical scientific sources are imported but not modified.
Modes: selftest, train, seal, evaluate, aggregate.
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

import common
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as envbase
import parameter_matched_novelty_confirmatory as pm

REPS = tuple(range(6000, 6060))
METHODS = ("gradnov", "anchor_random")
K, Q, BATCH, EPOCHS = 16, 4, 128, 10
OFFSET = 7500000000
ANCHOR_RNG_OFFSET = 102021
POOL_TRAIN_STARTS = tuple(300000 + 2000 * p for p in range(6))
POOL_HELDOUT_STARTS = tuple(301000 + 2000 * p for p in range(6))
REGISTRATION_COMMIT = "bd81e737de8ec64bf88770495c8eb51b661da87f"
REGISTRATION_BLOB_SHA = "3c573b2e63dffe192d4e18cbb700f85ab4235cd8"
REGISTRATION_PATH = "docs/CLEAN_TAIL_CONFIRM_PROTOCOL.md"

PROTOCOL = {
    "issue": 124,
    "name": "clean_tail_six_pool_confirmation",
    "registration_commit": REGISTRATION_COMMIT,
    "registration_path": REGISTRATION_PATH,
    "registration_blob_sha": REGISTRATION_BLOB_SHA,
    "reps": REPS,
    "methods": METHODS,
    "K": K,
    "Q": Q,
    "batch": BATCH,
    "epochs": EPOCHS,
    "steps": 80,
    "lr": 0.005,
    "weight_decay": 0.001,
    "novelty_weight": 0.6,
    "base_seed_offset": OFFSET,
    "base_seed_stride": 4099,
    "anchor_rng_offset": ANCHOR_RNG_OFFSET,
    "pool_train_starts": POOL_TRAIN_STARTS,
    "pool_heldout_starts": POOL_HELDOUT_STARTS,
    "blocks_per_pool": 10,
    "n_pools": 6,
    "train_environments": 64,
    "heldout_environments": 80,
    "primary_unit": "six_pool_level_contrasts",
    "primary_df": 5,
    "alpha": 0.05,
    "threads": 1,
}
PH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
SOURCES = (
    "common.py",
    "cnn_regime_interaction.py",
    "fixed_dose_response.py",
    "transfer_specificity.py",
    "parameter_matched_novelty_confirmatory.py",
    "parameter_matched_novelty_calibration.py",
    "clean_tail_confirm.py",
)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def write(path: Path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: fd.sha256(root / name) for name in SOURCES}


def protocol_file() -> Path:
    return Path(__file__).parent.parent / REGISTRATION_PATH


def check_registration() -> None:
    p = protocol_file()
    require(p.exists(), "registered protocol missing")
    require(git_blob_sha1(p) == REGISTRATION_BLOB_SHA, "registered protocol content changed")


def pool(rep: int) -> int:
    require(rep in REPS, "unregistered repetition")
    return (rep - 6000) // 10


def seeds(rep: int, heldout: bool = False) -> tuple[int, ...]:
    p = pool(rep)
    start = POOL_HELDOUT_STARTS[p] if heldout else POOL_TRAIN_STARTS[p]
    n = 80 if heldout else 64
    return tuple(range(start, start + n))


def check_range(start: int | None, end: int | None) -> None:
    require(start is not None and end is not None and start < end, "range")
    require(set(range(start, end)) <= set(REPS), "unregistered range")


def schedule(seed: int):
    tg = torch.Generator().manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 2)
    out = []
    for _ in range(EPOCHS):
        for b in torch.randperm(988, generator=tg).split(BATCH):
            out.append((b, rng.choice(64, K, replace=False).tolist()))
    return out


def choose(method: str, losses: torch.Tensor, gram: torch.Tensor, rng: np.random.Generator | None):
    require(method in METHODS, "method")
    require(len(losses) == K and tuple(gram.shape) == (K, K), "selector shape")
    if method == "gradnov":
        selected = common.select_hard_gradient_novel(losses, gram, Q, 0.6)
    else:
        require(rng is not None, "anchor RNG")
        anchor = int(torch.argmax(losses))
        rest = [i for i in range(K) if i != anchor]
        selected = [anchor, *map(int, rng.choice(rest, Q - 1, replace=False))]
    selected = sorted(map(int, selected))
    require(len(selected) == Q and len(set(selected)) == Q, "selected cardinality")
    require(0 <= min(selected) and max(selected) < K, "selected indices")
    return selected


def train(start: int, end: int, out: Path) -> None:
    check_registration()
    check_range(start, end)
    envbase.configure_determinism(1)
    require(not out.exists() or not any(out.iterdir()), "refuse overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir()

    x, y, *_ = cr.data_split()
    require(len(x) == 988, "training split")
    y = torch.as_tensor(y)
    files = {}
    records = []
    initials = {}
    schedules = {}
    cache = {}

    for rep in range(start, end):
        p = pool(rep)
        if p not in cache:
            cache[p] = envbase.geometric_envs(x, seeds(rep))
        envs = cache[p]
        base_seed = OFFSET + 4099 * rep
        sched = schedule(base_seed)
        require(len(sched) == 80, "schedule size")
        schedules[str(rep)] = pm.schedule_digest(sched)

        for method in METHODS:
            common.seed_everything(base_seed)
            model = common.SmallCNN()
            model.train()
            initial = pm.state_digest(model.state_dict())
            initials.setdefault(str(rep), initial)
            require(initials[str(rep)] == initial, "unpaired initialization")
            opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
            rng = np.random.default_rng(base_seed + ANCHOR_RNG_OFFSET) if method == "anchor_random" else None

            for step, (b, cand) in enumerate(sched):
                logits, h = model(torch.cat([envs[e][b] for e in cand]))
                per = torch.nn.functional.cross_entropy(
                    logits, y[b].repeat(K), reduction="none"
                ).reshape(K, -1)
                losses = per.mean(1)
                dirs = common.head_gradient_directions(logits, h, y[b], K)
                gram = dirs @ dirs.T
                require(torch.isfinite(losses).all() and torch.isfinite(gram).all(), "training finite")
                selected = choose(method, losses, gram, rng)
                order = sorted(range(K), key=lambda i: (-float(losses[i].detach()), i))
                ranks = {i: r + 1 for r, i in enumerate(order)}
                sub = gram[selected][:, selected].detach().double().cpu().numpy()
                ii, jj = np.triu_indices(Q, 1)
                records.append({
                    "rep": rep,
                    "pool": p,
                    "method": method,
                    "step": step,
                    "batch_size": len(b),
                    "selected_local": ";".join(map(str, selected)),
                    "selected_env_seed": ";".join(str(seeds(rep)[cand[i]]) for i in selected),
                    "mean_selected_rank": float(np.mean([ranks[i] for i in selected])),
                    "mean_candidate_loss": float(losses.mean().detach()),
                    "mean_selected_loss": float(losses[selected].mean().detach()),
                    "selected_pairwise_novelty": float(np.mean(1.0 - sub[ii, jj])),
                })
                opt.zero_grad(set_to_none=True)
                per[selected].mean().backward()
                opt.step()

            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            require(all(torch.isfinite(t).all() for t in state.values()), "state finite")
            state_digest = pm.state_digest(state)
            path = out / "states" / f"rep{rep}_{method}.pt"
            torch.save({
                "state_dict": state,
                "rep": rep,
                "method": method,
                "pool": p,
                "protocol_hash": PH,
                "state_digest": state_digest,
                "initial_digest": initial,
                "schedule_digest": schedules[str(rep)],
            }, path)
            files[str(path.relative_to(out))] = fd.sha256(path)
        print("CLEAN_TAIL_TRAINED", rep, "heldout_not_constructed", flush=True)

    step_path = out / "training_steps.csv.gz"
    pd.DataFrame(records).to_csv(step_path, index=False, compression="gzip")
    files[step_path.name] = fd.sha256(step_path)
    write(out / "training_manifest.json", {
        "protocol": PROTOCOL,
        "protocol_hash": PH,
        "registration_commit": REGISTRATION_COMMIT,
        "registration_blob_sha": REGISTRATION_BLOB_SHA,
        "registration_sha256": sha256(protocol_file()),
        "start": start,
        "end": end,
        "files": files,
        "source_hashes": source_hashes(),
        "split_hashes": cr.split_hashes(),
        "runtime": fd.runtime(),
        "initial_digests": initials,
        "schedule_digests": schedules,
        "heldout_constructed": False,
    })


def validate_train(out: Path):
    check_registration()
    path = out / "training_manifest.json"
    m = json.loads(path.read_text())
    check_range(m["start"], m["end"])
    require(m["protocol_hash"] == PH, "protocol hash")
    require(m["protocol"] == json.loads(json.dumps(PROTOCOL)), "protocol content")
    require(m["registration_commit"] == REGISTRATION_COMMIT, "registration commit")
    require(m["registration_blob_sha"] == REGISTRATION_BLOB_SHA, "registration blob")
    require(m["registration_sha256"] == sha256(protocol_file()), "registration sha256")
    require(m["source_hashes"] == source_hashes(), "training source provenance")
    require(m["split_hashes"] == cr.split_hashes(), "training split provenance")
    require(not m["heldout_constructed"], "training boundary")
    for name, digest in m["files"].items():
        require(fd.sha256(out / name) == digest, "training file hash " + name)
    return m


def seal(root: Path, out: Path) -> None:
    check_registration()
    reps = []
    shards = []
    state_entries = []
    common_sources = None
    common_splits = None

    for path in sorted(root.rglob("training_manifest.json")):
        m = validate_train(path.parent)
        reps.extend(range(m["start"], m["end"]))
        common_sources = m["source_hashes"] if common_sources is None else common_sources
        common_splits = m["split_hashes"] if common_splits is None else common_splits
        require(m["source_hashes"] == common_sources and m["split_hashes"] == common_splits, "cross-shard provenance")
        shards.append({
            "start": m["start"],
            "end": m["end"],
            "training_manifest_sha256": fd.sha256(path),
        })
        for rep in range(m["start"], m["end"]):
            for method in METHODS:
                sp = path.parent / "states" / f"rep{rep}_{method}.pt"
                ck = torch.load(sp, map_location="cpu", weights_only=True)
                require((ck["rep"], ck["method"], ck["protocol_hash"]) == (rep, method, PH), "state metadata")
                tensor_digest = pm.state_digest(ck["state_dict"])
                require(tensor_digest == ck["state_digest"], "state tensor digest")
                require(ck["initial_digest"] == m["initial_digests"][str(rep)], "initial digest chain")
                require(ck["schedule_digest"] == m["schedule_digests"][str(rep)], "schedule digest chain")
                state_entries.append({
                    "rep": rep,
                    "pool": pool(rep),
                    "method": method,
                    "state_file_sha256": fd.sha256(sp),
                    "state_tensor_digest": tensor_digest,
                    "initial_digest": ck["initial_digest"],
                    "schedule_digest": ck["schedule_digest"],
                })

    require(sorted(reps) == list(REPS), "global repetition coverage")
    require(len(state_entries) == 120, "global state coverage")
    require(len({(e["rep"], e["method"]) for e in state_entries}) == 120, "duplicate states")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "global_seal.json", {
        "protocol_hash": PH,
        "registration_commit": REGISTRATION_COMMIT,
        "registration_blob_sha": REGISTRATION_BLOB_SHA,
        "source_hashes": common_sources,
        "split_hashes": common_splits,
        "reps": list(REPS),
        "states": 120,
        "state_entries": state_entries,
        "shards": shards,
        "heldout_constructed": False,
    })
    print("GLOBAL_CLEAN_TAIL_STATE_SEAL", 120, PH, flush=True)


def validate_seal(path: Path, m):
    s = json.loads(path.read_text())
    require(s["protocol_hash"] == PH, "seal protocol")
    require(s["registration_commit"] == REGISTRATION_COMMIT and s["registration_blob_sha"] == REGISTRATION_BLOB_SHA, "seal registration")
    require(s["source_hashes"] == source_hashes(), "seal sources")
    require(s["split_hashes"] == cr.split_hashes(), "seal split")
    require(s["reps"] == list(REPS) and s["states"] == 120, "seal coverage")
    require(len(s["state_entries"]) == 120 and not s["heldout_constructed"], "seal boundary")
    entries = [e for e in s["shards"] if (e["start"], e["end"]) == (m["start"], m["end"])]
    require(len(entries) == 1, "seal shard")
    return s, entries[0]


def evaluate(start: int, end: int, out: Path, global_seal: Path) -> None:
    check_registration()
    check_range(start, end)
    envbase.configure_determinism(1)
    m = validate_train(out)
    require((m["start"], m["end"]) == (start, end), "evaluation range")
    seal_obj, shard = validate_seal(global_seal, m)
    require(shard["training_manifest_sha256"] == fd.sha256(out / "training_manifest.json"), "global seal manifest chain")
    require(not (out / "evaluation_manifest.json").exists(), "evaluation overwrite")

    # The global seal is fully validated above. Heldout construction starts only here.
    _, _, x, y, *_ = cr.data_split()
    require(len(x) == 809, "evaluation split")
    labels = torch.as_tensor(y)
    clean = torch.as_tensor(x)
    cache = {}
    rows = []
    envrows = []
    state_index = {(e["rep"], e["method"]): e for e in seal_obj["state_entries"]}

    for rep in range(start, end):
        p = pool(rep)
        if p not in cache:
            cache[p] = [torch.as_tensor(envbase.geometric_environment(x, s)) for s in seeds(rep, True)]
        for method in METHODS:
            sp = out / "states" / f"rep{rep}_{method}.pt"
            ck = torch.load(sp, map_location="cpu", weights_only=True)
            require((ck["rep"], ck["method"], ck["protocol_hash"]) == (rep, method, PH), "evaluation state metadata")
            digest = pm.state_digest(ck["state_dict"])
            require(digest == ck["state_digest"] == state_index[(rep, method)]["state_tensor_digest"], "evaluation tensor digest")
            require(fd.sha256(sp) == state_index[(rep, method)]["state_file_sha256"], "evaluation state file hash")

            model = common.SmallCNN()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            vals = []
            with torch.no_grad():
                clean_correct = int((model(clean)[0].argmax(1) == labels).sum())
                for env_seed, xx in zip(seeds(rep, True), cache[p]):
                    correct = int((model(xx)[0].argmax(1) == labels).sum())
                    acc = correct / len(labels)
                    vals.append(acc)
                    envrows.append({
                        "rep": rep,
                        "pool": p,
                        "method": method,
                        "env_seed": env_seed,
                        "correct": correct,
                        "n_images": len(labels),
                        "accuracy": acc,
                    })
            a = np.asarray(vals, np.float64)
            rows.append({
                "rep": rep,
                "pool": p,
                "method": method,
                "mean_test": float(a.mean()),
                "sd_test": float(a.std(ddof=1)),
                "p10_test": float(np.quantile(a, 0.1)),
                "min_test": float(a.min()),
                "clean_test": clean_correct / len(labels),
                "clean_correct": clean_correct,
                "n_clean": len(labels),
                "state_digest": digest,
            })
        print("CLEAN_TAIL_EVALUATED", rep, flush=True)

    validate_train(out)
    files = {}
    held_path = out / "heldout.csv"
    env_path = out / "environment.csv.gz"
    pd.DataFrame(rows).to_csv(held_path, index=False)
    pd.DataFrame(envrows).to_csv(env_path, index=False, compression="gzip")
    files[held_path.name] = fd.sha256(held_path)
    files[env_path.name] = fd.sha256(env_path)
    write(out / "evaluation_manifest.json", {
        "protocol_hash": PH,
        "start": start,
        "end": end,
        "files": files,
        "source_hashes": source_hashes(),
        "split_hashes": cr.split_hashes(),
        "training_manifest_sha256": fd.sha256(out / "training_manifest.json"),
        "global_seal_sha256": fd.sha256(global_seal),
        "runtime": fd.runtime(),
        "heldout_constructed": True,
    })


def estimate(x, direction: str):
    x = np.asarray(x, np.float64)
    require(x.ndim == 1 and len(x) == 6 and np.isfinite(x).all(), "primary sample")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    k = float(stats.t.ppf(0.975, 5))
    result = {
        "n_pools": 6,
        "df": 5,
        "mean": mean,
        "se": se,
        "k95": k,
        "ci95_low": mean - k * se,
        "ci95_high": mean + k * se,
        "direction": direction,
        "degenerate_zero_se": bool(se == 0.0),
    }
    if se == 0.0:
        result["t"] = None
        result["p_one_sided"] = None
        result["pass"] = False
        return result
    t = mean / se
    if direction == "positive":
        p = float(stats.t.sf(t, 5))
        passed = bool(mean > 0 and p < 0.05)
    elif direction == "negative":
        p = float(stats.t.cdf(t, 5))
        passed = bool(mean < 0 and p < 0.05)
    else:
        raise ValueError("direction")
    result.update({"t": float(t), "p_one_sided": p, "pass": passed})
    return result


def block_estimate(x):
    x = np.asarray(x, np.float64)
    require(x.ndim == 1 and len(x) == 60 and np.isfinite(x).all(), "block sample")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(60))
    k = float(stats.t.ppf(0.975, 59))
    return {
        "n_blocks": 60,
        "mean": mean,
        "se": se,
        "ci95_low": mean - k * se,
        "ci95_high": mean + k * se,
        "positive_blocks": int((x > 0).sum()),
        "negative_blocks": int((x < 0).sum()),
    }


def aggregate(root: Path, out: Path, global_seal: Path) -> None:
    check_registration()
    covered = []
    held = []
    environments = []
    training = []
    runtimes = []
    states_checked = 0
    seal_obj = json.loads(global_seal.read_text())
    require(seal_obj["states"] == 120 and seal_obj["reps"] == list(REPS), "aggregate seal")
    state_index = {(e["rep"], e["method"]): e for e in seal_obj["state_entries"]}
    require(len(state_index) == 120, "aggregate seal state index")

    for mp in sorted(root.rglob("evaluation_manifest.json")):
        d = mp.parent
        em = json.loads(mp.read_text())
        tm = validate_train(d)
        _, shard = validate_seal(global_seal, tm)
        require((em["start"], em["end"]) == (tm["start"], tm["end"]), "evaluation range")
        require(em["protocol_hash"] == PH and em["source_hashes"] == source_hashes(), "evaluation provenance")
        require(em["split_hashes"] == cr.split_hashes(), "evaluation split provenance")
        require(em["training_manifest_sha256"] == shard["training_manifest_sha256"] == fd.sha256(d / "training_manifest.json"), "manifest chain")
        require(em["global_seal_sha256"] == fd.sha256(global_seal), "global seal chain")
        for name, digest in em["files"].items():
            require(fd.sha256(d / name) == digest, "evaluation file hash " + name)
        covered.extend(range(tm["start"], tm["end"]))
        h = pd.read_csv(d / "heldout.csv", float_precision="round_trip")
        hidx = h.set_index(["rep", "method"])
        for rep in range(tm["start"], tm["end"]):
            for method in METHODS:
                sp = d / "states" / f"rep{rep}_{method}.pt"
                ck = torch.load(sp, map_location="cpu", weights_only=True)
                digest = pm.state_digest(ck["state_dict"])
                require(digest == hidx.loc[(rep, method), "state_digest"], "aggregate heldout digest")
                require(digest == state_index[(rep, method)]["state_tensor_digest"], "aggregate seal tensor digest")
                require(fd.sha256(sp) == state_index[(rep, method)]["state_file_sha256"], "aggregate seal file digest")
                states_checked += 1
        held.append(h)
        environments.append(pd.read_csv(d / "environment.csv.gz", float_precision="round_trip"))
        training.append(pd.read_csv(d / "training_steps.csv.gz", float_precision="round_trip"))
        runtimes.append({"start": tm["start"], "training": tm["runtime"], "evaluation": em["runtime"]})

    require(sorted(covered) == list(REPS) and states_checked == 120, "aggregate coverage")
    held = pd.concat(held, ignore_index=True)
    env = pd.concat(environments, ignore_index=True)
    train = pd.concat(training, ignore_index=True)
    require(len(held) == 120 and not held.duplicated(["rep", "method"]).any(), "heldout grid")
    require(set(held[["rep", "method"]].itertuples(index=False, name=None)) == {(r, m) for r in REPS for m in METHODS}, "heldout IDs")
    require(len(env) == 9600 and not env.duplicated(["rep", "method", "env_seed"]).any(), "environment grid")
    require(len(train) == 9600 and not train.duplicated(["rep", "method", "step"]).any(), "training grid")
    require(np.isfinite(held[["mean_test", "sd_test", "p10_test", "min_test", "clean_test"]].to_numpy(float)).all(), "heldout finite")

    hz = held.set_index(["rep", "method"])
    max_env_error = 0.0
    max_clean_error = 0.0
    for (rep, method), b in env.groupby(["rep", "method"]):
        require(set(b.env_seed) == set(seeds(int(rep), True)), "environment IDs")
        require((b.n_images == 809).all() and ((b.correct >= 0) & (b.correct <= 809)).all(), "environment counts")
        reconstructed = b.correct.to_numpy(float) / 809.0
        max_env_error = max(max_env_error, float(np.max(np.abs(b.accuracy.to_numpy(float) - reconstructed))))
        a = b.sort_values("env_seed").accuracy.to_numpy(float)
        obs = np.array([a.mean(), a.std(ddof=1), np.quantile(a, 0.1), a.min()])
        stored = hz.loc[(rep, method), ["mean_test", "sd_test", "p10_test", "min_test"]].to_numpy(float)
        max_env_error = max(max_env_error, float(np.max(np.abs(obs - stored))))
        clean_obs = float(hz.loc[(rep, method), "clean_correct"]) / float(hz.loc[(rep, method), "n_clean"])
        max_clean_error = max(max_clean_error, abs(clean_obs - float(hz.loc[(rep, method), "clean_test"])))
    require(max_env_error <= 1e-12 and max_clean_error <= 1e-15, "count reconstruction")

    blocks = []
    for rep in REPS:
        g = hz.loc[(rep, "gradnov")]
        a = hz.loc[(rep, "anchor_random")]
        blocks.append({
            "rep": rep,
            "pool": pool(rep),
            "minimum_contrast": float(g.min_test - a.min_test),
            "clean_contrast": float(g.clean_test - a.clean_test),
            "mean_contrast": float(g.mean_test - a.mean_test),
            "p10_contrast": float(g.p10_test - a.p10_test),
            "sd_contrast": float(g.sd_test - a.sd_test),
        })
    blocks = pd.DataFrame(blocks)
    require(len(blocks) == 60 and not blocks.duplicated("rep").any(), "block contrasts")
    require(blocks.groupby("pool").size().to_dict() == {p: 10 for p in range(6)}, "pool allocation")

    pools = blocks.groupby("pool", as_index=False)[["minimum_contrast", "clean_contrast", "mean_contrast", "p10_contrast", "sd_contrast"]].mean()
    require(len(pools) == 6 and set(pools.pool) == set(range(6)), "pool contrasts")

    minimum = estimate(pools.minimum_contrast.to_numpy(float), "positive")
    clean = estimate(pools.clean_contrast.to_numpy(float), "negative")
    if minimum["degenerate_zero_se"] or clean["degenerate_zero_se"]:
        decision = "INVALID / EXECUTION FAILURE"
    elif minimum["pass"] and clean["pass"]:
        decision = "CLEAN-TAIL POLICY TRADEOFF SUPPORT"
    elif minimum["pass"]:
        decision = "TAIL ADVANTAGE ONLY / CLEAN COST NOT CONFIRMED"
    elif clean["pass"]:
        decision = "CLEAN COST ONLY / TAIL ADVANTAGE NOT CONFIRMED"
    else:
        decision = "NO CLEAN-TAIL POLICY TRADEOFF CONFIRMATION"

    secondary = []
    for endpoint in ("minimum_contrast", "clean_contrast", "mean_contrast", "p10_contrast", "sd_contrast"):
        secondary.append({"endpoint": endpoint, **block_estimate(blocks[endpoint].to_numpy(float))})

    out.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(out / "clean_tail_block_contrasts60.csv", index=False)
    pools.to_csv(out / "clean_tail_pool_contrasts6.csv", index=False)
    pd.DataFrame([
        {"endpoint": "minimum_accuracy", **minimum},
        {"endpoint": "clean_accuracy", **clean},
    ]).to_csv(out / "clean_tail_primary6.csv", index=False)
    pd.DataFrame(secondary).to_csv(out / "clean_tail_secondary60.csv", index=False)
    train.groupby(["rep", "pool", "method"]).mean(numeric_only=True).reset_index().to_csv(out / "clean_tail_training_summary60.csv", index=False)
    held.to_csv(out / "clean_tail_heldout120.csv", index=False)
    write(out / "clean_tail_runtime.json", runtimes)
    write(out / "clean_tail_decision60.json", {
        "decision": decision,
        "protocol_hash": PH,
        "registration_commit": REGISTRATION_COMMIT,
        "n_training_blocks": 60,
        "n_primary_pools": 6,
        "primary_df": 5,
        "states_checked": states_checked,
        "environment_rows": len(env),
        "block_contrasts": len(blocks),
        "pool_contrasts": len(pools),
        "max_environment_count_reconstruction_error": max_env_error,
        "max_clean_count_reconstruction_error": max_clean_error,
        "minimum_primary": minimum,
        "clean_primary": clean,
        "raw_image_retraining_in_aggregate": False,
    })
    print(json.dumps({"decision": decision, "minimum": minimum, "clean": clean}, allow_nan=False), flush=True)
    print(pools.to_string(index=False), flush=True)


def selftest() -> None:
    check_registration()
    losses = torch.linspace(2.0, 0.1, K)
    a = torch.arange(K * 7, dtype=torch.float32).reshape(K, 7) + 1.0
    a = a / a.norm(dim=1, keepdim=True)
    gram = a @ a.T
    g1 = choose("gradnov", losses, gram, None)
    g2 = choose("gradnov", losses, gram, None)
    require(g1 == g2 and len(g1) == Q, "gradnov synthetic determinism")
    r1 = choose("anchor_random", losses, gram, np.random.default_rng(17))
    r2 = choose("anchor_random", losses, gram, np.random.default_rng(17))
    require(r1 == r2 and int(torch.argmax(losses)) in r1 and len(r1) == Q, "anchor synthetic determinism")
    require(len(schedule(1234)) == 80 and len(schedule(1234)[-1][0]) == 92, "schedule synthetic shape")
    require(pool(6000) == 0 and pool(6009) == 0 and pool(6010) == 1 and pool(6059) == 5, "pool boundaries")
    for p in range(6):
        r = 6000 + 10 * p
        require(set(seeds(r)).isdisjoint(seeds(r, True)), "train/heldout seed overlap")
    all_train = set().union(*(set(seeds(6000 + 10 * p)) for p in range(6)))
    all_test = set().union(*(set(seeds(6000 + 10 * p, True)) for p in range(6)))
    require(len(all_train) == 384 and len(all_test) == 480 and all_train.isdisjoint(all_test), "cross-pool seed uniqueness")
    x = np.array([0.01, 0.02, 0.03, 0.00, 0.01, 0.02])
    e = estimate(x, "positive")
    p = float(stats.ttest_1samp(x, 0.0, alternative="greater").pvalue)
    require(abs(e["p_one_sided"] - p) < 1e-12, "positive t test")
    en = estimate(-x, "negative")
    pn = float(stats.ttest_1samp(-x, 0.0, alternative="less").pvalue)
    require(abs(en["p_one_sided"] - pn) < 1e-12, "negative t test")
    z = estimate(np.zeros(6), "positive")
    require(z["degenerate_zero_se"] and z["p_one_sided"] is None and not z["pass"], "zero SE handling")
    print("CLEAN_TAIL_SYNTHETIC_PASS", PH, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("selftest", "train", "seal", "evaluate", "aggregate"))
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--global-seal", type=Path)
    a = p.parse_args()
    if a.mode == "selftest":
        selftest()
    elif a.mode == "train":
        train(a.start, a.end, a.output_dir)
    elif a.mode == "seal":
        seal(a.input_dir, a.output_dir)
    elif a.mode == "evaluate":
        evaluate(a.start, a.end, a.output_dir, a.global_seal)
    else:
        aggregate(a.input_dir, a.output_dir, a.global_seal)


if __name__ == "__main__":
    main()
