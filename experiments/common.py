from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, rotate, shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def configure_determinism(num_threads: int = 1) -> None:
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


def load_digits_split():
    x, y = load_digits(return_X_y=True)
    x = (x.astype(np.float32) / 16.0).reshape(-1, 8, 8)
    y = y.astype(np.int64)
    xtr, xtmp, ytr, ytmp = train_test_split(x, y, test_size=0.45, random_state=314159, stratify=y)
    _, xte, _, yte = train_test_split(xtmp, ytmp, test_size=0.55, random_state=271828, stratify=ytmp)
    return xtr, ytr, xte, yte


def environment_parameters(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([
        rng.uniform(-30, 30), rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2),
        rng.uniform(0, 0.85), rng.uniform(0.78, 1.22), rng.uniform(-0.06, 0.06), rng.uniform(0, 0.08),
    ], dtype=np.float32)


def rng_fingerprint(seed: int, length: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).random(length).astype(np.float32)


def geometric_environment(images: np.ndarray, seed: int) -> np.ndarray:
    angle, dx, dy, blur, contrast, bright, noise = environment_parameters(seed)
    out = np.empty_like(images)
    noise_rng = np.random.default_rng(seed * 4001 + 17)
    for i, image in enumerate(images):
        z = rotate(image, float(angle), reshape=False, order=1, mode="constant", cval=0, prefilter=False)
        z = shift(z, (float(dy), float(dx)), order=1, mode="constant", cval=0, prefilter=False)
        if blur > 1e-6:
            z = gaussian_filter(z, float(blur), mode="nearest")
        z = (z - 0.5) * contrast + 0.5 + bright + noise_rng.normal(0, noise, z.shape)
        out[i] = np.clip(z, 0, 1)
    return out.astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU())
        self.head = torch.nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.flatten(1)
        h = self.body(x)
        return self.head(h), h


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
        )
        self.project = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32 * 4 * 4, 64), torch.nn.ReLU())
        self.head = torch.nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x[:, None, :, :]
        h = self.project(self.features(x))
        return self.head(h), h


def zscore(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def head_gradient_directions(logits: torch.Tensor, features: torch.Tensor, y: torch.Tensor, n_env: int) -> torch.Tensor:
    b = len(y)
    logits = logits.detach().reshape(n_env, b, -1)
    features = features.detach().reshape(n_env, b, -1)
    probs = torch.softmax(logits, -1)
    onehot = torch.nn.functional.one_hot(y, 10).float()[None]
    residual = (probs - onehot) / b
    gw = torch.einsum("mbc,mbh->mch", residual, features).reshape(n_env, -1)
    gb = residual.sum(1)
    g = torch.cat([gw, gb], 1)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)


def select_loss_hard(loss: torch.Tensor, q: int = 4) -> list[int]:
    return torch.topk(loss, q).indices.tolist()


def select_hard_gradient_novel(loss: torch.Tensor, cosine: torch.Tensor, q: int = 4, novelty_weight: float = 0.6) -> list[int]:
    selected = [int(torch.argmax(loss))]
    while len(selected) < q:
        remaining = [i for i in range(len(loss)) if i not in selected]
        novelty = zscore((1 - cosine[:, selected]).min(1).values)
        score = zscore(loss.detach()) + novelty_weight * novelty
        selected.append(max(remaining, key=lambda i: float(score[i])))
    return selected


def select_hard_parameter_novel(loss: torch.Tensor, parameters: np.ndarray, q: int = 4, novelty_weight: float = 0.6) -> list[int]:
    selected = [int(torch.argmax(loss))]
    distances = torch.tensor(np.sqrt(((parameters[:, None, :] - parameters[None, :, :]) ** 2).sum(-1)), dtype=torch.float32)
    while len(selected) < q:
        remaining = [i for i in range(len(loss)) if i not in selected]
        novelty = zscore(distances[:, selected].min(1).values)
        score = zscore(loss.detach()) + novelty_weight * novelty
        selected.append(max(remaining, key=lambda i: float(score[i])))
    return selected


def farthest_subset(features: np.ndarray, k: int) -> list[int]:
    center = features.mean(0, keepdims=True)
    selected = [int(np.argmax(((features - center) ** 2).sum(1)))]
    while len(selected) < k:
        d = ((features[:, None, :] - features[np.array(selected)][None, :, :]) ** 2).sum(-1)
        md = d.min(1)
        md[np.array(selected)] = -1
        selected.append(int(np.argmax(md)))
    return selected


def environment_metrics(model: torch.nn.Module, environments: Sequence[torch.Tensor], labels: torch.Tensor, clean: torch.Tensor):
    model.eval()
    with torch.no_grad():
        acc = np.array([float((model(x)[0].argmax(1) == labels).float().mean()) for x in environments])
        clean_acc = float((model(clean)[0].argmax(1) == labels).float().mean())
    return {"mean_test": float(acc.mean()), "sd_test": float(acc.std(ddof=1)), "p10_test": float(np.quantile(acc, 0.1)), "min_test": float(acc.min()), "clean_test": clean_acc}
