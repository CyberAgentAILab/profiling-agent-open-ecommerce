"""Unit tests for the feasibility metrics."""

import sys
import unittest
from pathlib import Path

import pandas as pd

# model.py uses package-relative imports (from .dataset import ...), so it must
# be imported as part of its package rather than via a file-location spec.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from investigate_confidence_feasibility import model  # noqa: E402


class TestPrf(unittest.TestCase):
    def test_basic_values(self) -> None:
        result = model._prf(tp=2, selected=4, m=8)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall_of_m"], 0.25)
        self.assertAlmostEqual(result["f1"], 2 * 0.5 * 0.25 / 0.75)

    def test_zero_denominators(self) -> None:
        result = model._prf(tp=0, selected=0, m=0)
        self.assertEqual(result, {"precision": 0.0, "recall_of_m": 0.0, "f1": 0.0})


class TestThresholdSweep(unittest.TestCase):
    def test_selection_per_threshold(self) -> None:
        scored = [(0.9, 1), (0.6, 0), (0.2, 1)]
        rows = model._threshold_sweep(scored, m=2, thresholds=[0.0, 0.5])
        self.assertEqual(rows[0]["selected"], 3)
        self.assertEqual(rows[0]["tp"], 2)
        self.assertEqual(rows[1]["selected"], 2)
        self.assertEqual(rows[1]["tp"], 1)


class TestPrecisionRecallAtK(unittest.TestCase):
    def test_ranked_by_confidence(self) -> None:
        scored = [(0.2, 0), (0.9, 1), (0.5, 1)]
        rows = model._precision_recall_at_k(scored, m=2, ks=[1, 2])
        self.assertEqual(rows[0]["precision_at_k"], 1.0)  # top-1 = (0.9, 1)
        self.assertEqual(rows[1]["tp"], 2)
        self.assertEqual(rows[1]["recall_at_k_of_m"], 1.0)


class TestCalibrationBins(unittest.TestCase):
    def test_last_bin_includes_right_edge(self) -> None:
        scored = [(0.0, 0), (0.5, 1), (1.0, 1)]
        bins = model._calibration_bins(scored, edges=[0.0, 0.5, 1.0])
        self.assertEqual(bins[0]["n"], 1)
        self.assertEqual(bins[1]["n"], 2)  # 0.5 and 1.0 (right edge closed)
        self.assertEqual(bins[1]["n_pos"], 2)


class TestEvaluateTag(unittest.TestCase):
    def _survey(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Survey ResponseID": ["u1", "u2", "u3", "u4"],
                "Q-personal-wheelchair": ["Yes", "No", "Yes", "No"],
            }
        ).set_index("Survey ResponseID")

    def _spec(self) -> dict:
        return {
            "tag": "mobility",
            "category": "life_event",
            "name": "wheelchair",
            "survey_col": "Q-personal-wheelchair",
            "match": "yes_no",
            "positive_value": "Yes",
        }

    def test_full_block_with_separable_confidences(self) -> None:
        user_conf = {"u1": 0.9, "u2": 0.1, "u3": 0.8, "u4": 0.2, "u-no-survey": 0.5}
        block = model.evaluate_tag(
            spec=self._spec(),
            user_conf=user_conf,
            survey_indexed=self._survey(),
            thresholds=[0.0, 0.5],
            ks=[2],
            calibration_edges=[0.0, 0.5, 1.0],
            plot_dir=None,
        )
        self.assertEqual(block["n_tagged"], 5)
        self.assertEqual(block["n_tagged_evaluable"], 4)
        self.assertEqual(block["n_tagged_unlabeled"], 1)
        self.assertEqual(block["m_survey_positives"], 2)
        self.assertEqual(block["auc"], 1.0)  # positives strictly above negatives
        self.assertEqual(block["best_f1_operating_point"]["threshold"], 0.5)

    def test_single_class_cohort_has_no_auc(self) -> None:
        block = model.evaluate_tag(
            spec=self._spec(),
            user_conf={"u1": 0.9, "u3": 0.8},  # both survey-positive
            survey_indexed=self._survey(),
            thresholds=[0.0],
            ks=[1],
            calibration_edges=[0.0, 1.0],
            plot_dir=None,
        )
        self.assertIsNone(block["auc"])


if __name__ == "__main__":
    unittest.main()
