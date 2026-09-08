"""Unit tests for scripts/open_ecommerce/local/e2e_sample/prepare_run_configs.py."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "open_ecommerce" / "local" / "e2e_sample" / "prepare_run_configs.py"

spec = importlib.util.spec_from_file_location("prepare_run_configs", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load {SCRIPT}")
prepare_run_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_run_configs)


class TestRewritePaths(unittest.TestCase):
    def test_results_and_logs_prefixes_are_moved_into_run_dir(self) -> None:
        config = {
            "out_dir": "./results/e2e_sample/scan",
            "log_dir": "./logs/e2e_sample/scan",
            "query_path": "./results/e2e_sample/prepare_titles/titles_sample10.csv",
            "prompt_path": "./configs/prompt/open-ecommerce/scan_diagnostic.txt",
            "search_cache_dir": "./search_cache/e2e_sample",
            "batch_size": 10,
            "validation_path": "",
        }
        out = prepare_run_configs.rewrite_paths(config, "./results/e2e_sample/2026-01-01_00-00-00")
        self.assertEqual(out["out_dir"], "./results/e2e_sample/2026-01-01_00-00-00/outputs/scan")
        self.assertEqual(out["log_dir"], "./results/e2e_sample/2026-01-01_00-00-00/logs/scan")
        self.assertEqual(
            out["query_path"], "./results/e2e_sample/2026-01-01_00-00-00/outputs/prepare_titles/titles_sample10.csv"
        )
        # Untouched: prompts, shared cache, non-string values, empty strings.
        for key in ("prompt_path", "search_cache_dir", "batch_size", "validation_path"):
            self.assertEqual(out[key], config[key], key)


class TestPrepareRunConfigs(unittest.TestCase):
    def test_writes_one_config_per_stage_with_chained_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = f"{tmp}/run"
            paths = prepare_run_configs.prepare_run_configs(run_dir)
            self.assertEqual([p.name for p in paths], [f"{s}.yaml" for s in prepare_run_configs.STAGES])
            configs = {}
            for p in paths:
                with open(p) as f:
                    configs[p.stem] = yaml.safe_load(f)
            # Every stage writes into the run dir ...
            for stage, config in configs.items():
                self.assertTrue(config["out_dir"].startswith(f"{run_dir}/outputs/"), stage)
                self.assertTrue(config["log_dir"].startswith(f"{run_dir}/logs/"), stage)
            # ... and reads the previous stage's output from the run dir.
            self.assertEqual(configs["scan"]["query_path"], configs["prepare_titles"]["out_csv"])
            self.assertEqual(configs["search"]["validation_path"], configs["scan"]["out_dir"])
            self.assertEqual(configs["predict_transaction"]["validation_path"], configs["search"]["out_dir"])
            self.assertEqual(configs["predict_user"]["validation_path"], configs["predict_transaction"]["out_dir"])
            self.assertEqual(configs["predict_user"]["scan_output_path"], configs["scan"]["out_dir"])
            self.assertEqual(configs["cluster_attribute"]["query_path"], configs["predict_user"]["out_dir"])
            self.assertEqual(configs["tag_cluster"]["query_dir"], configs["cluster_attribute"]["out_dir"])
            # nested branch paths are rewritten as well
            for branch in configs["cluster_attribute"]["attribute_branches"]:
                self.assertTrue(branch["out_dir"].startswith(f"{run_dir}/outputs/clustering/"), branch["name"])
                self.assertTrue(branch["cache_embedding_path"].startswith(f"{run_dir}/outputs/embedding_cache/"))
            self.assertTrue(configs["tag_cluster"]["cache_embed_dir"].startswith(f"{run_dir}/outputs/embedding_cache"))
            # Nothing outside the run dir is redirected.
            self.assertEqual(configs["search"]["search_cache_dir"], "./search_cache/e2e_sample")
            self.assertEqual(configs["prepare_titles"]["purchases_csv"], "./tests/e2e/fixtures/sample_purchases_10.csv")

    def test_model_name_override_applies_to_llm_stages_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_run_configs.prepare_run_configs(f"{tmp}/run", model_name="./models/Other-LLM")
            configs = {}
            for p in paths:
                with open(p) as f:
                    configs[p.stem] = yaml.safe_load(f)
            for stage in ("scan", "predict_transaction", "predict_user", "tag_cluster"):
                self.assertEqual(configs[stage]["model_name"], "./models/Other-LLM", stage)
            for stage in ("prepare_titles", "search"):
                self.assertNotIn("model_name", configs[stage], stage)
            # the embedding model of cluster_attribute is not an LLM and must stay untouched
            self.assertEqual(configs["cluster_attribute"]["model_name"], "./models/Qwen3-Embedding-0.6B")


if __name__ == "__main__":
    unittest.main()
