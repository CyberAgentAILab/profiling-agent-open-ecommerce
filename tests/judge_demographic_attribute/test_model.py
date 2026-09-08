"""Unit tests for judge_demographic_attribute pure helpers."""

import importlib.util
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_module("judge_demographic_attribute/model.py", "judge_demographic_attribute_model")


class TestFillUnknownDemographics(unittest.TestCase):
    def _user_demo(self) -> dict:
        return {
            "u1": {cat: Counter() for cat in model._FILL_CATEGORIES},
            "u2": {"signal:gender": Counter({"female": 3})},
        }

    def test_drop_strategy_is_a_no_op(self) -> None:
        user_demo = self._user_demo()
        result = model.fill_unknown_demographics(user_demo, strategy="drop")
        self.assertFalse(result["u1"]["signal:gender"])

    def test_majority_fills_missing_with_observed_mode(self) -> None:
        user_demo = self._user_demo()
        result = model.fill_unknown_demographics(user_demo, strategy="majority")
        # u2's observed gender (female) is the majority, so u1 is filled with it.
        self.assertEqual(result["u1"]["signal:gender"], Counter({"female": 1}))
        # u2's existing prediction is untouched.
        self.assertEqual(result["u2"]["signal:gender"], Counter({"female": 3}))

    def test_random_fill_is_deterministic_for_a_seed(self) -> None:
        first = model.fill_unknown_demographics(self._user_demo(), strategy="random", seed=7)
        second = model.fill_unknown_demographics(self._user_demo(), strategy="random", seed=7)
        self.assertEqual(first["u1"], second["u1"])

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            model.fill_unknown_demographics(self._user_demo(), strategy="magic")


class TestAgeRepresentative(unittest.TestCase):
    def test_range_maps_to_midpoint(self) -> None:
        self.assertEqual(model.age_representative("25-34"), 30)

    def test_open_ended_bin(self) -> None:
        self.assertIsNotNone(model.age_representative("65+"))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(model.age_representative("unknown"))


class TestNormalizeSurveyGender(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertEqual(model.normalize_survey_gender("Female"), "female")
        self.assertEqual(model.normalize_survey_gender("MALE"), "male")

    def test_missing_value_returns_none(self) -> None:
        self.assertIsNone(model.normalize_survey_gender(None))


if __name__ == "__main__":
    unittest.main()
