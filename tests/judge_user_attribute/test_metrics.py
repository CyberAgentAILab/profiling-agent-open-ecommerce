"""Unit tests for the judge_user_attribute evaluation logic: _score_block, evaluate_predictions,
attach_ground_truth and filter_users_with_any_yes_ground_truth (the paper's headline metrics)."""

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_module("judge_user_attribute/model.py", "judge_user_attribute_model_metrics")

SURVEY_COLUMNS = ["Survey ResponseID", *model.BINARY_TARGETS.values(), "Q-life-changes"]


def _survey(rows: list[dict]) -> pd.DataFrame:
    """Build a survey DataFrame; missing binary answers become NaN like an empty CSV cell."""
    df = pd.DataFrame(rows)
    for col in SURVEY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[SURVEY_COLUMNS]


class TestScoreBlock(unittest.TestCase):
    def test_threshold_metrics_match_known_values(self) -> None:
        y_true, y_pred = [0, 0, 1, 1], [0, 1, 1, 1]
        block = model._score_block(y_true, y_pred)
        self.assertEqual(block["n"], 4)
        self.assertEqual(block["n_pos_true"], 2)
        self.assertEqual(block["n_pos_pred"], 3)
        self.assertAlmostEqual(block["prevalence"], 0.5)
        self.assertAlmostEqual(block["accuracy"], 0.75)
        self.assertAlmostEqual(block["precision"], 2 / 3)
        self.assertAlmostEqual(block["recall"], 1.0)
        self.assertAlmostEqual(block["f1"], 0.8)
        # F1 of predicting every user positive = 2p / (1 + p)
        self.assertAlmostEqual(block["baseline_f1_all_positive"], 2 * 0.5 / 1.5)
        # no y_score -> no threshold-independent metrics
        for key in ("auc", "pr_auc", "f1_optimal", "threshold_optimal"):
            self.assertNotIn(key, block)

    def test_score_metrics_match_sklearn(self) -> None:
        y_true, y_pred = [0, 0, 1, 1], [0, 1, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        block = model._score_block(y_true, y_pred, y_score=y_score)
        self.assertAlmostEqual(block["auc"], 0.75)  # the classic sklearn example
        self.assertAlmostEqual(block["auc"], roc_auc_score(y_true, y_score))
        self.assertAlmostEqual(block["pr_auc"], average_precision_score(y_true, y_score))
        self.assertAlmostEqual(block["pr_auc_lift"], block["pr_auc"] / 0.5)
        # best F1 over the PR curve, recomputed independently
        p, r, thresholds = precision_recall_curve(y_true, y_score)
        f1 = np.where((p[:-1] + r[:-1]) > 0, 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12), 0.0)
        self.assertAlmostEqual(block["f1_optimal"], float(f1.max()))
        self.assertAlmostEqual(block["threshold_optimal"], float(thresholds[int(np.argmax(f1))]))
        # the PR-curve F1 carries a 1e-12 epsilon, hence the tolerance
        self.assertGreaterEqual(block["f1_optimal"], block["f1"] - 1e-9)
        self.assertAlmostEqual(
            block["f1_optimal_lift_over_baseline"], block["f1_optimal"] / block["baseline_f1_all_positive"]
        )
        self.assertEqual(block["score_n_unique"], 4)
        self.assertAlmostEqual(block["score_min"], 0.1)
        self.assertAlmostEqual(block["score_max"], 0.8)

    def test_tied_scores_pick_the_best_threshold(self) -> None:
        # thresholds 0.9 / 0.5 / 0.1 give F1 = 0.4 / 0.571 / 0.75 -> the lowest threshold wins
        y_true = [1, 0, 1, 0, 1]
        y_score = [0.9, 0.9, 0.5, 0.5, 0.1]
        block = model._score_block(y_true, [1, 1, 1, 1, 0], y_score=y_score)
        self.assertAlmostEqual(block["f1_optimal"], 0.75)
        self.assertAlmostEqual(block["threshold_optimal"], 0.1)
        self.assertAlmostEqual(block["precision_optimal"], 0.6)
        self.assertAlmostEqual(block["recall_optimal"], 1.0)
        self.assertEqual(block["score_n_unique"], 3)
        self.assertEqual(block["score_top5_modes"][0][1], 2)  # a mode with two users

    def test_single_class_omits_ranking_metrics(self) -> None:
        block = model._score_block([1, 1], [1, 0], y_score=[0.9, 0.2])
        self.assertAlmostEqual(block["recall"], 0.5)
        self.assertNotIn("auc", block)
        self.assertNotIn("f1_optimal", block)

    def test_empty_input(self) -> None:
        self.assertEqual(model._score_block([], []), {"n": 0})


class TestAttachGroundTruth(unittest.TestCase):
    def test_ground_truth_and_correctness(self) -> None:
        survey = _survey(
            [
                {
                    "Survey ResponseID": "u1",
                    "Q-substance-use-cigarettes": "Yes",
                    "Q-substance-use-marijuana": "No",
                    "Q-personal-diabetes": "Prefer not to say",
                    "Q-personal-wheelchair": "Stopped using in the last year",
                    "Q-life-changes": "Had a child; Moved place of residence",
                }
            ]
        )
        predictions = [
            {
                "user_id": "u1",
                "predicted": {"cigarettes": True, "marijuana": True, "had_child": True, "moved": False},
            },
            {"user_id": "u_not_in_survey", "predicted": {"cigarettes": True}},
        ]
        out = model.attach_ground_truth(predictions, survey)

        gt = out[0]["ground_truth"]
        self.assertEqual(list(gt), model.ATTRIBUTE_KEYS)
        self.assertIs(gt["cigarettes"], True)
        self.assertIs(gt["marijuana"], False)
        self.assertIsNone(gt["alcohol"])  # empty cell
        self.assertIsNone(gt["diabetes"])  # refused
        self.assertIs(gt["wheelchair"], True)  # "stopped ..." counts as yes
        self.assertIs(gt["had_child"], True)
        self.assertIs(gt["moved"], True)
        self.assertIs(gt["divorce"], False)
        c = out[0]["correctness"]
        self.assertEqual(c["cigarettes"], "correct")
        self.assertEqual(c["marijuana"], "wrong")
        self.assertEqual(c["alcohol"], "unknown")
        self.assertEqual(c["wheelchair"], "wrong")  # predicted False (absent) vs True
        self.assertEqual(c["moved"], "wrong")
        self.assertEqual(c["became_pregnant"], "correct")  # False vs False
        self.assertEqual(out[0]["correctness_summary"], {"n_scored": 8, "n_correct": 5, "accuracy": 5 / 8})

        # a user absent from the survey gets all-unknown ground truth
        self.assertTrue(all(v is None for v in out[1]["ground_truth"].values()))
        self.assertEqual(out[1]["correctness_summary"], {"n_scored": 0, "n_correct": 0, "accuracy": None})


class TestEvaluatePredictions(unittest.TestCase):
    def test_per_attribute_and_macro_metrics(self) -> None:
        survey = _survey(
            [
                {"Survey ResponseID": "u1", "Q-substance-use-cigarettes": "Yes", "Q-life-changes": "Had a child"},
                {"Survey ResponseID": "u2", "Q-substance-use-cigarettes": "No", "Q-life-changes": "None of the above"},
                {"Survey ResponseID": "u3", "Q-substance-use-cigarettes": "No", "Q-life-changes": "Divorce"},
            ]
        )
        predictions = [
            {
                "user_id": "u1",
                "predicted": {"cigarettes": True, "had_child": True},
                "predicted_proba": {"cigarettes": 0.9, "had_child": 0.8},
            },
            {
                "user_id": "u2",
                "predicted": {"cigarettes": True, "had_child": False},
                "predicted_proba": {"cigarettes": 0.6, "had_child": 0.2},
            },
            {
                "user_id": "u3",
                "predicted": {"cigarettes": False, "had_child": False},
                "predicted_proba": {"cigarettes": 0.1, "had_child": 0.3},
            },
            {"user_id": "u_not_in_survey", "predicted": {"cigarettes": True}, "predicted_proba": {"cigarettes": 0.99}},
        ]
        metrics = model.evaluate_predictions(predictions, survey)

        cig = metrics["cigarettes"]
        self.assertEqual(cig["n"], 3)  # the user absent from the survey is skipped
        self.assertEqual(cig["n_pos_true"], 1)
        self.assertEqual(cig["n_pos_pred"], 2)
        self.assertAlmostEqual(cig["precision"], 0.5)
        self.assertAlmostEqual(cig["recall"], 1.0)
        self.assertAlmostEqual(cig["auc"], 1.0)  # the positive user has the highest score
        self.assertAlmostEqual(cig["auc"], roc_auc_score([1, 0, 0], [0.9, 0.6, 0.1]))

        had_child = metrics["had_child"]
        self.assertEqual(had_child["n"], 3)
        self.assertAlmostEqual(had_child["f1"], 1.0)
        self.assertAlmostEqual(had_child["auc"], 1.0)
        self.assertAlmostEqual(metrics["divorce"]["recall"], 0.0)  # u3 divorced but predicted False

        # alcohol etc. have no survey answers -> empty blocks that do not enter the macro averages
        self.assertEqual(metrics["alcohol"], {"n": 0})
        overall = metrics["__overall__"]
        self.assertEqual(overall["n_users"], 4)
        scored = [m for k, m in metrics.items() if k != "__overall__" and isinstance(m.get("auc"), float)]
        self.assertAlmostEqual(overall["macro_auc"], sum(m["auc"] for m in scored) / len(scored))
        f1s = [m["f1"] for k, m in metrics.items() if k != "__overall__" and isinstance(m.get("f1"), float)]
        self.assertAlmostEqual(overall["macro_f1"], sum(f1s) / len(f1s))


class TestFilterUsersWithAnyYesGroundTruth(unittest.TestCase):
    def test_cohort_filter_and_counts(self) -> None:
        survey = _survey(
            [
                {
                    "Survey ResponseID": "yes_binary",
                    "Q-substance-use-alcohol": "Yes",
                    "Q-life-changes": "None of the above",
                },
                {"Survey ResponseID": "all_no", "Q-substance-use-alcohol": "No", "Q-life-changes": "None of the above"},
                {"Survey ResponseID": "yes_life", "Q-substance-use-alcohol": "No", "Q-life-changes": "Divorce"},
                {"Survey ResponseID": "all_unknown"},
            ]
        )
        kept, counts = model.filter_users_with_any_yes_ground_truth(
            ["yes_binary", "all_no", "yes_life", "all_unknown", "missing"], survey, mode="db"
        )
        self.assertEqual(kept, ["yes_binary", "yes_life"])
        self.assertEqual(counts, {"valid_user": 2, "invalid_user": 3, "n_input_users": 5, "n_missing_from_survey": 1})


if __name__ == "__main__":
    unittest.main()
