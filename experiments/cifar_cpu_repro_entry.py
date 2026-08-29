from __future__ import annotations

import argparse
from argparse import Namespace

import torch

import cifar_resnet_rep_rank_audit as audit


def configure_single_thread() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def main(a: argparse.Namespace) -> None:
    # The scientific audit imports `configure` into its module namespace.
    # Replace only that runtime hook; all model/data/seed/protocol code remains unchanged.
    audit.configure = configure_single_thread
    args = Namespace(
        data_root=a.data_root,
        output=a.output,
        start=a.rep,
        end=a.rep + 1,
        pretrain_epochs=10,
        finetune_epochs=2,
        batch_size=128,
        train_per_class=600,
        test_per_class=300,
        test_envs=32,
        k=8,
        q=4,
        probe_size=256,
        pretrain_lr=3e-3,
        finetune_lr=1e-3,
    )
    audit.run(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rep", type=int, choices=[45, 46], required=True)
    p.add_argument("--data-root", default=".cache/cifar10")
    p.add_argument("--output", required=True)
    main(p.parse_args())
