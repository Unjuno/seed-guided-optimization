from __future__ import annotations

import summarize_transfer_specificity as impl

impl.EXPECTED_REPS = tuple(range(1000, 1030))

if __name__ == '__main__':
    impl.main()
