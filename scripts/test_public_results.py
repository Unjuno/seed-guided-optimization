"""Input-rejection tests for the read-only publication checker."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from check_public_results import ROOT, verify


class PublicResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ("translation_matched_primary30.csv", "translation_matched_summary30.csv"):
            shutil.copyfile(ROOT / "results" / name, self.directory / name)
        self.primary = self.directory / "translation_matched_primary30.csv"
        self.summary = self.directory / "translation_matched_summary30.csv"

    def change_primary(self, fn):
        data = pd.read_csv(self.primary, float_precision="round_trip")
        fn(data).to_csv(self.primary, index=False)

    def test_original_passes(self):
        report = verify(self.directory)
        self.assertEqual(report["verification"], "PUBLIC STATISTICS VERIFIED")
        self.assertEqual(len(report["tests"]), 2)

    def test_missing_row(self):
        self.change_primary(lambda d: d.iloc[:-1])
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_duplicate_rep(self):
        self.change_primary(lambda d: d.assign(rep=[2200] * len(d)))
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_noninteger_rep(self):
        self.change_primary(lambda d: d.assign(rep=d.rep + 0.5))
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_missing_column(self):
        self.change_primary(lambda d: d.drop(columns="m_mean_test_benefit"))
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_nonfinite(self):
        self.change_primary(lambda d: d.assign(m_mean_test_benefit=np.nan))
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_fraction_range(self):
        self.change_primary(lambda d: d.assign(m_mean_test_benefit=2.0))
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_degenerate_sample(self):
        self.change_primary(lambda d: d.assign(m_mean_test_benefit=0.0))
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_modified_summary(self):
        d = pd.read_csv(self.summary)
        d.loc[d.endpoint == "m_mean_test_benefit", "mean"] += 0.01
        d.to_csv(self.summary, index=False)
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_duplicate_summary(self):
        d = pd.read_csv(self.summary)
        pd.concat([d, d[d.endpoint == "m_mean_test_benefit"]]).to_csv(self.summary, index=False)
        with self.assertRaises(ValueError):
            verify(self.directory)

    def test_missing_file(self):
        self.primary.unlink()
        with self.assertRaises(OSError):
            verify(self.directory)


if __name__ == "__main__":
    unittest.main()
