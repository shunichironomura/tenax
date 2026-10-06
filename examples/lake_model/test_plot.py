"""Regression checks: uv run --with xy==0.0.5 examples/lake_model/test_plot.py."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import plot


class PairwisePlotTests(unittest.TestCase):
    def test_exports_without_matplotlib(self) -> None:
        """Exercise message, singleton histogram, and multi-panel matrix exports."""
        for dimensions in (0, 1, 2):
            for all_cases in (False, True, None):
                with self.subTest(dimensions=dimensions, all_cases=all_cases):
                    initial = [
                        plot.FeatureLimit(name, 0.0, 1.0, 0.1, 0.9)
                        for name in ("first", "second")
                    ]
                    chosen = [
                        plot.FeatureLimit(
                            limit.feature,
                            0.0,
                            1.0,
                            0.3 if index < dimensions else 0.1,
                            0.7 if index < dimensions else 0.9,
                        )
                        for index, limit in enumerate(initial)
                    ]
                    experiments = [
                        plot.Experiment(
                            index,
                            {"first": value, "second": 1.0 - value},
                            index % 2 == 0 if all_cases is None else all_cases,
                        )
                        for index, value in enumerate((0.1, 0.3, 0.5, 0.7, 0.9))
                    ]
                    membership = [
                        plot.is_box_member(row, chosen) for row in experiments
                    ]
                    cases = sum(
                        member and row.case_of_interest
                        for row, member in zip(experiments, membership, strict=True)
                    )
                    points = sum(membership)
                    selected = plot.TrajectoryStep(
                        1,
                        plot.PrimPhase.PEEL,
                        0.0,
                        cases / points,
                        points / len(experiments),
                        dimensions,
                        points,
                        cases,
                    )
                    self.assertEqual(
                        plot.validate_membership(selected, chosen, experiments),
                        membership,
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "pairs.png"
                        plot.export_pairs_scatter(
                            path, experiments, initial, chosen, selected
                        )
                        png = path.read_bytes()
                        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                        width, height = struct.unpack(">II", png[16:24])
                        self.assertGreater(width, 100)
                        self.assertGreater(height, 100)
                        html = path.with_suffix(".html").read_text()
                        self.assertIn("PRIM step 1", html)
                        if dimensions:
                            self.assertIn("Selected box projection", html)
                        else:
                            self.assertIn("no restricted dimensions", html)
        self.assertFalse(
            any(
                name == "matplotlib" or name.startswith("matplotlib.")
                for name in sys.modules
            )
        )

    def test_inconsistent_restriction_count_is_rejected(self) -> None:
        limits = [plot.FeatureLimit("feature", 0.0, 1.0, 0.1, 0.9)]
        selected = plot.TrajectoryStep(1, plot.PrimPhase.PEEL, 1.0, 1.0, 1.0, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "limits imply 0"):
            plot.restricted_feature_limits(limits, limits, selected)


if __name__ == "__main__":
    unittest.main()
