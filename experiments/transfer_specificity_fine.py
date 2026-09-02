from __future__ import annotations

import numpy as np
import transfer_specificity as impl

# Preregistered Issue #56: only the fresh blocks and evaluation-only alpha grid differ.
impl.CAL_TRAIN_SEEDS = np.arange(31000, 31064, dtype=int)
impl.CAL_SHARED_SEEDS = np.arange(32000, 32040, dtype=int)
impl.CAL_NUISANCE_SEEDS = np.arange(32100, 32140, dtype=int)
impl.CONF_TRAIN_SEEDS = np.arange(33000, 33064, dtype=int)
impl.CONF_SHARED_SEEDS = np.arange(34000, 34080, dtype=int)
impl.CONF_NUISANCE_SEEDS = np.arange(35000, 35080, dtype=int)
impl.ALPHAS = tuple(np.round(np.arange(0.00, 0.101, 0.01), 2))

if __name__ == '__main__':
    impl.main()
