"""Unit tests for common.run_configs and scripts/open_ecommerce/local/prepare_run_configs.py (full-data run)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

from common.run_configs import rewrite_paths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "open_ecommerce" / "local" / "prepare_run_configs.py"

spec = importlib.util.spec_from_file_location("prepare_full_run_configs", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load {SCRIPT}")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


class TestRewritePaths(unittest.TestCase):
    PREFIX_MAP = {"./results/x/": "outputs", "./logs/x/": "logs"}

    def test_prefixes_moved_and_keep_prefixes_untouched(self) -> None:
        config = {
            "out_dir": "./results/x/user/v4",
            "log_dir": "./logs/x/user",
            "fixed_input": "./results/x/fixed/patterns.json",
            "nested": [{"out_dir": "./results/x/a"}, {"n": 1}],
            "prompt": "./configs/prompt/p.txt",
            "empty": "",
        }
        out = rewrite_paths(config, "./results/x/run/", self.PREFIX_MAP, keep_prefixes=("./results/x/fixed/",))
        self.assertEqual(out["out_dir"], "./results/x/run/outputs/user/v4")
        self.assertEqual(out["log_dir"], "./results/x/run/logs/user")
        self.assertEqual(out["fixed_input"], "./results/x/fixed/patterns.json")
        self.assertEqual(out["nested"][0]["out_dir"], "./results/x/run/outputs/a")
        self.assertEqual(out["nested"][1], {"n": 1})
        self.assertEqual(out["prompt"], "./configs/prompt/p.txt")
        self.assertEqual(out["empty"], "")
        self.assertEqual(config["out_dir"], "./results/x/user/v4")  # input is not mutated


class TestPrepareFullRunConfigs(unittest.TestCase):
    def _configs(self, run_dir: str, model_name: str | None = None) -> dict[str, dict]:
        configs = {}
        for p in full.prepare_run_configs(run_dir, model_name=model_name):
            with open(p) as f:
                configs[p.stem] = yaml.safe_load(f)
        return configs

    def test_stages_chain_inside_the_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = f"{tmp}/run"
            configs = self._configs(run_dir)
            self.assertEqual(sorted(configs), sorted(full.STAGES))
            for stage, config in configs.items():
                self.assertTrue(config["out_dir"].startswith(f"{run_dir}/outputs/"), stage)
                self.assertTrue(config["log_dir"].startswith(f"{run_dir}/logs/"), stage)
            self.assertEqual(configs["search"]["validation_path"], configs["scan"]["out_dir"])
            self.assertEqual(configs["predict_transaction"]["validation_path"], configs["search"]["out_dir"])
            self.assertEqual(configs["predict_user"]["validation_path"], configs["predict_transaction"]["out_dir"])
            self.assertEqual(configs["predict_user"]["scan_output_path"], configs["scan"]["out_dir"])
            self.assertEqual(configs["cluster_attribute"]["query_path"], configs["predict_user"]["out_dir"])
            self.assertEqual(configs["tag_cluster"]["query_dir"], configs["cluster_attribute"]["out_dir"])
            for branch in configs["judge_user_attribute"]["attribute_branches"]:
                self.assertEqual(branch["tag_db_dir"], configs["tag_cluster"]["out_dir"])
            self.assertEqual(
                configs["judge_demographic_attribute"]["demographic_db_dir"], configs["tag_cluster"]["out_dir"]
            )
            self.assertEqual(
                configs["investigate_confidence_feasibility"]["tag_db_dir"], configs["tag_cluster"]["out_dir"]
            )

    def test_fixed_inputs_and_shared_paths_stay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            configs = self._configs(f"{tmp}/run")
            self.assertEqual(
                configs["predict_user"]["frequent_patterns_path"],
                "./results/open_ecommerce/frequent_pattern_mining/frequent_patterns.json",
            )
            self.assertEqual(
                configs["prepare_titles"]["purchases_csv"], "./data/public/open-ecommerce/amazon-purchases.csv"
            )
            self.assertEqual(configs["scan"]["query_path"], "./data/public/open-ecommerce/titles_ge3_buyers.csv")
            self.assertEqual(configs["search"]["search_cache_dir"], "./search_cache/open-ecommerce")
            for branch in configs["cluster_attribute"]["attribute_branches"]:
                self.assertTrue(branch["cache_embedding_path"].startswith("./cache/embedding/"))

    def test_model_name_override_applies_to_llm_stages_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            configs = self._configs(f"{tmp}/run", model_name="./models/Other-LLM")
            for stage in full.LLM_STAGES:
                self.assertEqual(configs[stage]["model_name"], "./models/Other-LLM", stage)
            self.assertEqual(configs["cluster_attribute"]["model_name"], "./models/Qwen3-Embedding-0.6B")
            for stage in ("prepare_titles", "search", "investigate_confidence_feasibility"):
                self.assertNotIn("model_name", configs[stage], stage)


if __name__ == "__main__":
    unittest.main()
