from __future__ import annotations

import numpy as np
import summarize_transfer_calibration as impl

impl.EXPECTED_REPS = tuple(range(950, 960))
impl.ALPHAS = tuple(np.round(np.arange(0.00, 0.101, 0.01), 2))

if __name__ == '__main__':
    impl.main()
