from __future__ import annotations

import argparse, random, time
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, rotate, shift
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from common import (
    MLP, configure_determinism, farthest_subset, head_gradient_directions,
    select_hard_gradient_novel,
)

REL = np.array([5, 13, 22, 31, 40, 49, 58], dtype=int)
OLD_TOP7 = np.array([1, 0, 2, 4, 3, 10, 26], dtype=int)
BATCH, EPOCHS, K, PREF, Q, LR = 128, 10, 16, 8, 4, 1e-2
METHODS = ['oracle7_8', 'raw64_8', 'learned_top7_8', 'weighted12_8', 'old_transfer7_8', 'random8']


def rng64(seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).random(64).astype(np.float32)


def params(seed: int) -> np.ndarray:
    u = rng64(seed)[REL]
    return np.array([
        (u[0] * 2 - 1) * 30,
        (u[1] * 2 - 1) * 1.2,
        (u[2] * 2 - 1) * 1.2,
        u[3] * .85,
        .78 + u[4] * .44,
        -.06 + u[5] * .12,
        u[6] * .08,
    ], dtype=np.float32)


def environment(images: np.ndarray, seed: int) -> np.ndarray:
    angle, dx, dy, blur, contrast, bright, noise = params(seed)
    out = np.empty_like(images)
    noise_rng = np.random.default_rng(int(seed) * 5003 + 29)
    for i, image in enumerate(images):
        z = rotate(image, float(angle), reshape=False, order=1, mode='constant', cval=0, prefilter=False)
        z = shift(z, (float(dy), float(dx)), order=1, mode='constant', cval=0, prefilter=False)
        if blur > 1e-6:
            z = gaussian_filter(z, float(blur), mode='nearest')
        z = (z - .5) * contrast + .5 + bright + noise_rng.normal(0, noise, z.shape)
        out[i] = np.clip(z, 0, 1)
    return out.reshape(len(out), -1).astype(np.float32)


def standardize(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(0)) / (a.std(0) + 1e-8)


def calibrate(xtr: np.ndarray, ytr: np.ndarray):
    cal_seeds = np.arange(15000, 15128, dtype=int)
    rng = np.random.default_rng(515151)
    cal_idx = rng.choice(len(ytr), 256, replace=False)
    cal_y = torch.tensor(ytr[cal_idx])
    torch.manual_seed(5151)
    model = MLP().eval()
    gradients = []
    for seed in cal_seeds:
        xb = torch.tensor(environment(xtr[cal_idx], int(seed)))
        with torch.no_grad():
            logits, h = model(xb)
        gradients.append(head_gradient_directions(logits, h, cal_y, 1)[0].numpy())
    gradients = np.stack(gradients)
    raw = np.stack([rng64(s) for s in cal_seeds])
    mu, sd = raw.mean(0), raw.std(0) + 1e-8
    x = (raw - mu) / sd
    pca = PCA(n_components=16, random_state=0)
    target = pca.fit_transform(gradients)
    target = standardize(target)
    ridge = RidgeCV(alphas=[.01, .1, 1., 10., 100.]).fit(x, target)
    coef = np.asarray(ridge.coef_)
    relevance = np.sqrt((coef ** 2).sum(0))
    order = np.argsort(relevance)[::-1]
    top7, top12 = order[:7], order[:12]
    true_ranks = [int(np.where(order == j)[0][0] + 1) for j in REL]
    print('ridge_alpha', ridge.alpha_, 'top7', top7.tolist(), 'true_ranks', true_ranks, flush=True)
    return mu, sd, relevance, top7, top12


def evaluate(model, unseen, labels, clean):
    model.eval()
    with torch.no_grad():
        acc = np.array([float((model(x)[0].argmax(1) == labels).float().mean()) for x in unseen])
        clean_acc = float((model(clean)[0].argmax(1) == labels).float().mean())
    return dict(mean_test=acc.mean(), sd_test=acc.std(ddof=1), p10_test=np.quantile(acc, .1), min_test=acc.min(), clean_test=clean_acc)


def run(start: int, end: int, output: str):
    configure_determinism(1)
    x, y = load_digits(return_X_y=True)
    x = (x.astype(np.float32) / 16.).reshape(-1, 8, 8)
    y = y.astype(np.int64)
    xtr, xtmp, ytr, ytmp = train_test_split(x, y, test_size=.45, random_state=314159, stratify=y)
    _, xte, _, yte = train_test_split(xtmp, ytmp, test_size=.55, random_state=271828, stratify=ytmp)

    mu, sd, relevance, top7, top12 = calibrate(xtr, ytr)
    seeds = np.arange(16000, 16064, dtype=int)
    raw = np.stack([rng64(s) for s in seeds])
    raw_z = (raw - mu) / sd
    oracle = standardize(raw_z[:, REL])
    raw64 = standardize(raw_z)
    learned7 = standardize(raw_z[:, top7])
    old7 = standardize(raw_z[:, OLD_TOP7])
    weights = relevance / relevance.max()
    weighted12 = raw_z[:, top12] * weights[top12][None, :]
    features = dict(oracle7_8=oracle, raw64_8=raw64, learned_top7_8=learned7,
                    weighted12_8=weighted12, old_transfer7_8=old7)

    train = [torch.tensor(environment(xtr, int(s))) for s in seeds]
    unseen = [torch.tensor(environment(xte, s)) for s in range(17000, 17080)]
    clean = torch.tensor(xte.reshape(len(xte), -1).astype(np.float32))
    ty, ey = torch.tensor(ytr), torch.tensor(yte)
    rows = []
    for rep in range(start, end):
        base = 18100000 + rep * 2213
        tg = torch.Generator().manual_seed(base + 1)
        er = np.random.default_rng(base + 2)
        schedule = []
        for _ in range(EPOCHS):
            for b in torch.randperm(len(ty), generator=tg).split(BATCH):
                schedule.append((b, er.choice(64, K, replace=False).tolist()))
        for mi, method in enumerate(METHODS):
            torch.manual_seed(base); np.random.seed(base % (2**32 - 1)); random.seed(base)
            random_selector = np.random.default_rng(base + 9000 + mi)
            model = MLP(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
            forwards = 0; st = time.time()
            for b, cand16 in schedule:
                if method == 'random8':
                    local = sorted(random_selector.choice(K, PREF, replace=False).tolist())
                else:
                    local = farthest_subset(features[method][np.array(cand16)], PREF)
                cand = [cand16[i] for i in local]
                m = len(cand); forwards += m
                xb = torch.cat([train[e][b] for e in cand])
                logits, h = model(xb)
                per = torch.nn.functional.cross_entropy(logits, ty[b].repeat(m), reduction='none').reshape(m, -1)
                losses = per.mean(1)
                gd = head_gradient_directions(logits, h, ty[b], m)
                chosen = select_hard_gradient_novel(losses, gd @ gd.T, Q)
                opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
            row = dict(rep=rep, method=method, train_seconds=time.time()-st, env_forwards=forwards,
                       **evaluate(model, unseen, ey, clean))
            rows.append(row)
        print('rep', rep, 'done', flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=20)
    p.add_argument('--output', default='learned_rng_cross_generator.csv')
    a = p.parse_args(); run(a.start, a.end, a.output)
