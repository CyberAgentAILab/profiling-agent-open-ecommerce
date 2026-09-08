"""Unit tests for base_agent.config utilities."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

# Make first-party packages (common, ...) importable when running from a
# plain checkout, then load config.py directly so that importing it does not
# pull in base_agent/__init__.py (which imports vLLM via model.py).
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = Path(__file__).parent.parent.parent / "src" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load_module("base_agent/config.py", "base_agent_config")


class TestValidateJson(unittest.TestCase):
    def test_keys_match_sets_is_json_true(self) -> None:
        result = config.validate_json(json.dumps({"a": 1, "b": "x"}), ["a", "b"])
        self.assertTrue(result["is_json"])
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], "x")

    def test_keys_mismatch_sets_is_json_false(self) -> None:
        result = config.validate_json(json.dumps({"a": 1}), ["a", "b"])
        self.assertFalse(result["is_json"])

    def test_extra_keys_set_is_json_false(self) -> None:
        result = config.validate_json(json.dumps({"a": 1, "b": 2, "c": 3}), ["a", "b"])
        self.assertFalse(result["is_json"])

    def test_parse_failure_returns_is_json_false(self) -> None:
        result = config.validate_json("not a json", ["a"])
        self.assertEqual(result, {"is_json": False})

    def test_attribute_valid_items(self) -> None:
        payload = {"items": [{"a": "x", "b": None}, {"a": "y", "b": True}]}
        result = config.validate_json(json.dumps(payload), ["a", "b"], "items")
        self.assertTrue(result["is_json"])

    def test_attribute_not_a_list(self) -> None:
        payload = {"items": {"a": "x"}}
        result = config.validate_json(json.dumps(payload), ["a"], "items")
        self.assertFalse(result["is_json"])

    def test_attribute_item_not_a_dict(self) -> None:
        payload = {"items": ["plain string"]}
        result = config.validate_json(json.dumps(payload), ["a"], "items")
        self.assertFalse(result["is_json"])

    def test_attribute_item_missing_required_key(self) -> None:
        payload = {"items": [{"a": "x"}]}
        result = config.validate_json(json.dumps(payload), ["a", "b"], "items")
        self.assertFalse(result["is_json"])

    def test_attribute_item_value_with_invalid_type(self) -> None:
        payload = {"items": [{"a": 123}]}
        result = config.validate_json(json.dumps(payload), ["a"], "items")
        self.assertFalse(result["is_json"])

    def test_attribute_name_absent_falls_back_to_key_match(self) -> None:
        result = config.validate_json(json.dumps({"a": 1}), ["a"], "items")
        self.assertTrue(result["is_json"])


class TestSaveAsJson(unittest.TestCase):
    def test_dict_is_wrapped_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            saved = config.save_as_json({"k": "v"}, out_dir, "ts")
            self.assertTrue(Path(saved).is_absolute())
            self.assertEqual(Path(saved).name, "output-ts.json")
            with open(saved, encoding="utf-8") as f:
                self.assertEqual(json.load(f), [{"k": "v"}])

    def test_job_id_in_filename(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            saved = config.save_as_json([{"k": "v"}], out_dir, "ts", job_id="42")
            self.assertEqual(Path(saved).name, "output-ts-42.json")

    def test_non_ascii_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            saved = config.save_as_json({"title": "Café au Lait"}, out_dir, "ts")
            with open(saved, encoding="utf-8") as f:
                self.assertIn("Café au Lait", f.read())

    def test_creates_missing_out_dir(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            nested = str(Path(out_dir) / "not" / "yet" / "there")
            saved = config.save_as_json({"k": "v"}, nested, "ts")
            self.assertTrue(Path(saved).exists())


class TestLoadYamlConfig(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("out_dir: ./results\nmin_unique_buyers: 2\n")
        loaded = config.load_yaml_config(f.name)
        self.assertEqual(loaded, {"out_dir": "./results", "min_unique_buyers": 2})


class TestSelectSearchApi(unittest.TestCase):
    def test_serp_selected_when_url_and_token_present(self) -> None:
        cfg = {"search_engine": "serp", "serp_url": "https://serp.example", "serp_api_token": "tok"}
        self.assertEqual(config._select_search_api(cfg), "serp")

    def test_serper_selected_when_url_and_token_present(self) -> None:
        cfg = {"search_engine": "serper", "serper_url": "https://serper.example", "serper_api_token": "tok"}
        self.assertEqual(config._select_search_api(cfg), "serper")

    def test_serp_without_token_falls_back_to_ddgs(self) -> None:
        cfg = {"search_engine": "serp", "serp_url": "https://serp.example", "serp_api_token": None}
        self.assertEqual(config._select_search_api(cfg), "ddgs")

    def test_serper_without_url_falls_back_to_ddgs(self) -> None:
        cfg = {"search_engine": "serper", "serper_api_token": "tok"}
        self.assertEqual(config._select_search_api(cfg), "ddgs")

    def test_unknown_engine_falls_back_to_ddgs(self) -> None:
        cfg = {"search_engine": "ddgs", "serp_api_token": "tok", "serp_url": "https://serp.example"}
        self.assertEqual(config._select_search_api(cfg), "ddgs")

    def test_missing_engine_falls_back_to_ddgs(self) -> None:
        self.assertEqual(config._select_search_api({}), "ddgs")

    def test_token_for_other_engine_does_not_leak_selection(self) -> None:
        cfg = {"search_engine": "serp", "serper_url": "https://serper.example", "serper_api_token": "tok"}
        self.assertEqual(config._select_search_api(cfg), "ddgs")


class TestRedactSecrets(unittest.TestCase):
    def test_set_tokens_are_masked(self) -> None:
        redacted = config._redact_secrets({"serp_api_token": "secret", "serper_api_token": "secret2"})
        self.assertEqual(redacted, {"serp_api_token": "***", "serper_api_token": "***"})

    def test_unset_tokens_stay_none(self) -> None:
        redacted = config._redact_secrets({"serp_api_token": None, "serper_api_token": None})
        self.assertEqual(redacted, {"serp_api_token": None, "serper_api_token": None})

    def test_non_token_keys_are_untouched(self) -> None:
        cfg = {"model_name": "m", "out_dir": "./results", "serp_url": "https://serp.example"}
        self.assertEqual(config._redact_secrets(cfg), cfg)

    def test_original_config_is_not_mutated(self) -> None:
        cfg = {"serp_api_token": "secret"}
        config._redact_secrets(cfg)
        self.assertEqual(cfg["serp_api_token"], "secret")

    def test_no_secret_value_appears_in_rendered_log_line(self) -> None:
        cfg = {"serp_api_token": "sk-serp-secret", "serper_api_token": "sk-serper-secret", "model_name": "m"}
        rendered = f"⚙️ config: {config._redact_secrets(cfg)}"
        self.assertNotIn("sk-serp-secret", rendered)
        self.assertNotIn("sk-serper-secret", rendered)
        self.assertIn("model_name", rendered)


if __name__ == "__main__":
    unittest.main()
