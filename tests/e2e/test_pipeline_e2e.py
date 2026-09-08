"""Offline end-to-end test: 10 sample purchases through tagged attribute clusters.

Runs the real ``main()`` of the first 7 pipeline stages (prepare_titles → scan →
search → predict_transaction → predict_user → cluster_attribute → tag_cluster)
in sequence, chained through file I/O exactly like a production run. The LLM is
replaced by canned responses (tests/e2e/fixtures/llm_stub_responses.json: per
product for scan / transaction / user, per attribute for the cluster tags), the
search stage is served entirely from a pre-seeded diskcache
(search_stub_results.json), and the embedding model is replaced by a keyword
based stub that maps every attribute to one of a few themes, so the test is
deterministic and needs no GPU, model weights, or API keys.

Run with:
    uv run python -m unittest tests.e2e.test_pipeline_e2e -v

Every run records itself under ``results/e2e_offline_test/<timestamp>/``
(override the parent directory with the E2E_OFFLINE_TEST_DIR environment
variable):
    configs/   the resolved task_config.yaml of each stage
    logs/      the loguru log of each stage
    outputs/   the output files of each stage (titles CSV, output-*.json)
    logs/, outputs/ also hold the clustering / tag stages (and outputs/embedding_cache)
    summary.json / summary.txt   per-test outcome (ok / FAIL / ERROR / skipped)
"""

import csv
import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import traceback
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from diskcache import Cache
from loguru import logger

import base_agent.model as base_agent_model
from base_agent.model import BaseAgent
from cluster_attribute.model import ClusterSubAgent
from search.model import WebRetriever

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
E2E_CONFIG_DIR = REPO_ROOT / "configs" / "open_ecommerce" / "e2e_sample"
RUN_ROOT = Path(os.environ.get("E2E_OFFLINE_TEST_DIR", REPO_ROOT / "results" / "e2e_offline_test"))
STAGES = ("prepare_titles", "scan", "search", "transaction", "user", "clustering", "tag")
TIMEZONE = ZoneInfo("Asia/Tokyo")  # same timezone as base_agent.config.activate_logging

STAGE_TO_FIXTURE_KEY = {
    "ScanSubAgent": "scan",
    "TransactionSubAgent": "transaction",
    "UserSubAgent": "user",
    "TaggerSubAgent": "tag",
}

# Keyword -> theme of the embedding stub. Attributes sharing a theme get (almost)
# identical vectors, so k-means with the fixed n_clusters of the e2e config
# separates exactly these themes.
EMBEDDING_THEMES: list[tuple[str, tuple[str, ...]]] = [
    ("grooming", ("hair", "barber")),
    ("pet", ("dog",)),
    ("kitchen", ("cooking", "coffee")),
    ("family", ("conceive", "pregnant", "newborn")),
    ("moving", ("moving",)),
    ("gender", ("woman", "man ")),
    ("age", ("presbyopia", "older")),
]
EMBEDDING_DIM = len(EMBEDDING_THEMES) + 1


def theme_of(attribute: str) -> str:
    lowered = attribute.lower()
    for theme, keywords in EMBEDDING_THEMES:
        if any(keyword in lowered for keyword in keywords):
            return theme
    return "other"


def stub_compute_embeddings(agent: ClusterSubAgent, attribute_texts: list[str]) -> np.ndarray:
    """ClusterSubAgent._compute_embeddings replacement: one-hot theme + tiny deterministic jitter."""
    theme_index = {theme: i for i, (theme, _) in enumerate(EMBEDDING_THEMES)}
    vectors = np.zeros((len(attribute_texts), EMBEDDING_DIM), dtype=float)
    for row, text in enumerate(attribute_texts):
        vectors[row, theme_index.get(theme_of(text), EMBEDDING_DIM - 1)] = 1.0
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors[row] += np.array([b / 255.0 for b in digest[:EMBEDDING_DIM]]) * 1e-3
    return vectors


def _load_fixture(name: str) -> Any:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


def make_stub_generate(responses: dict[str, dict[str, Any]]) -> Callable[..., list[dict]]:
    """Build a BaseAgent.generate replacement: returns the canned JSON whose
    product title appears in the prompt (raw for scan/transaction prompts,
    JSON-escaped for the predict_user items context)."""

    def stub_generate(agent: BaseAgent, prompt: str | list[str], **kwargs: Any) -> list[dict]:
        stage = STAGE_TO_FIXTURE_KEY[type(agent).__name__]
        prompts = [prompt] if isinstance(prompt, str) else prompt
        outputs = []
        for p in prompts:
            content = None
            for title, response in responses[stage].items():
                if title in p or json.dumps(title)[1:-1] in p:
                    if stage == "tag":
                        # tag fixtures are keyed by an attribute of the cluster -> tag string
                        response = {"reasoning": f"personas share the theme of '{title}'", "tag": response}
                    content = json.dumps(response, ensure_ascii=False)
                    break
            if content is None:
                raise AssertionError(f"no stub response matches this {stage} prompt: {p[:200]}")
            outputs.append({"thinking": "", "content": content, "input_tokens": 1, "output_tokens": 1})
        return outputs

    return stub_generate


class TestPipelineEndToEnd(unittest.TestCase):
    """10 sample products through all 5 stages, asserting the stage contracts
    and the approved per-product attribute expectations."""

    tmp_dir: ClassVar[Path]
    run_dir: ClassVar[Path]
    out: ClassVar[dict[str, Path]]
    titles_csv: ClassVar[Path]
    cache_dir: ClassVar[Path]
    titles: ClassVar[list[str]]
    llm_stub: ClassVar[dict[str, Any]]
    search_stub: ClassVar[dict[str, Any]]
    started_at: ClassVar[datetime.datetime]
    outcomes: ClassVar[dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.started_at = datetime.datetime.now(TIMEZONE)
        cls.run_dir = RUN_ROOT / cls.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        cls.run_dir.mkdir(parents=True, exist_ok=False)
        cls.outcomes = {}

        # Scratch space only for the search cache; everything worth keeping goes to run_dir.
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="e2e_sample_"))
        cls.out = {stage: cls.run_dir / "outputs" / stage for stage in STAGES}
        cls.titles_csv = cls.out["prepare_titles"] / "titles_sample10.csv"
        cls.cache_dir = cls.tmp_dir / "search_cache"

        with open(FIXTURE_DIR / "sample_purchases_10.csv") as f:
            cls.titles = [row["Title"] for row in csv.DictReader(f)]
        cls.llm_stub = _load_fixture("llm_stub_responses.json")
        cls.search_stub = _load_fixture("search_stub_results.json")

        try:
            cls._run_pipeline()
        except BaseException:
            # tearDownClass is skipped when setUpClass fails, so record the crash here.
            cls._write_summary(pipeline_error=traceback.format_exc())
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)
        cls._write_summary()

    def tearDown(self) -> None:
        self.outcomes[self._testMethodName] = self._current_outcome()

    # ------------------------------------------------------------ run record
    def _current_outcome(self) -> str:
        """Outcome of the running test: ok / FAIL / ERROR / skipped (best effort;
        relies on unittest internals that differ slightly across Python versions)."""
        outcome = getattr(self, "_outcome", None)
        if outcome is None:
            return "unknown"

        def _is_me(test: Any) -> bool:
            return test is self or getattr(test, "test_case", None) is self

        result = getattr(outcome, "result", None)
        if result is not None:
            if any(_is_me(t) for t, _ in result.errors):
                return "ERROR"
            if any(_is_me(t) for t, _ in result.failures):
                return "FAIL"
            if any(_is_me(t) for t, _ in result.skipped):
                return "skipped"
        # Python <= 3.10 buffers (test, exc_info) pairs on the outcome until run() ends.
        for test, exc_info in getattr(outcome, "errors", []):
            if _is_me(test) and exc_info is not None:
                return "FAIL" if issubclass(exc_info[0], self.failureException) else "ERROR"
        return "ok" if getattr(outcome, "success", True) else "FAIL"

    @classmethod
    def _write_summary(cls, pipeline_error: str | None = None) -> None:
        finished_at = datetime.datetime.now(TIMEZONE)
        statuses = list(cls.outcomes.values())
        counts = {status: statuses.count(status) for status in ("ok", "FAIL", "ERROR", "skipped")}
        summary = {
            "command": "uv run python -m unittest tests.e2e.test_pipeline_e2e -v",
            "started_at": cls.started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tests": cls.outcomes,
            "counts": counts,
            "passed": pipeline_error is None and counts["FAIL"] == 0 and counts["ERROR"] == 0 and bool(cls.outcomes),
            "pipeline_error": pipeline_error,
        }
        with open(cls.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        lines = [
            f"$ {summary['command']}",
            f"started_at:  {summary['started_at']}",
            f"finished_at: {summary['finished_at']}",
            "",
        ]
        lines += [f"{name} ... {status}" for name, status in cls.outcomes.items()]
        if pipeline_error:
            lines += ["", "pipeline setup failed:", pipeline_error.rstrip()]
        lines += ["", f"Ran {len(cls.outcomes)} tests", "", "OK" if summary["passed"] else "FAILED"]
        with open(cls.run_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        cls._sanitize_recorded_files()

    @classmethod
    def _sanitize_recorded_files(cls) -> None:
        """Replace the absolute repo path with "." in the recorded configs / logs / summaries.

        The stages need absolute paths while they run, but the recorded run is committed,
        so it must not carry the developer's machine-specific directory. (Private IPs never
        appear here: this offline test starts no vLLM engine, and the GPU run scripts mask
        them in the console stream before it is written — see _drop_private_ips in
        scripts/open_ecommerce/local/run_db.sh and e2e_sample/run.sh.)
        """
        repo = str(REPO_ROOT)
        for pattern in ("configs/*.yaml", "logs/**/*.log", "summary.txt", "summary.json"):
            for path in cls.run_dir.glob(pattern):
                text = path.read_text(encoding="utf-8")
                if repo in text:
                    path.write_text(text.replace(repo, "."), encoding="utf-8")

    # ------------------------------------------------------------ pipeline run
    @classmethod
    def _write_config(
        cls,
        stage: str,
        overrides: dict[str, Any],
        branch_overrides: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> str:
        with open(E2E_CONFIG_DIR / stage / "task_config.yaml") as f:
            config = yaml.safe_load(f)
        config.update(overrides)
        if branch_overrides is not None:
            for branch in config.get("attribute_branches") or []:
                branch.update(branch_overrides(branch))
        config["log_dir"] = str(cls.run_dir / "logs" / stage)
        # Prompt paths in the committed configs are repo-relative.
        for key in ("prompt_path", "prompt_recovery_transaction_path", "prompt_user_attribute_path"):
            if config.get(key):
                config[key] = str(REPO_ROOT / config[key])
        config_dir = cls.run_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f"{stage}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        return str(config_path)

    @classmethod
    def _run_stage(cls, module_name: str, config_path: str) -> None:
        # Each stage adds its own file sink; reset so a stage's log holds only that stage.
        logger.remove()
        logger.add(sys.stderr)
        module = __import__(f"{module_name}.__main__", fromlist=["main"])
        with mock.patch.object(sys, "argv", [module_name, "--config", config_path]):
            module.main()

    @classmethod
    def _seed_search_cache(cls, query_suffix: str) -> None:
        with Cache(directory=str(cls.cache_dir)) as cache:
            for title, results in cls.search_stub.items():
                cache[title + query_suffix] = results

    @classmethod
    def _run_pipeline(cls) -> None:
        def fail_if_searched(self: WebRetriever, query: str) -> list[dict[str, str]]:
            raise AssertionError(f"search API called for '{query}' — cache seeding is incomplete")

        embedding_cache = cls.out["clustering"].parent / "embedding_cache"
        with (
            mock.patch.object(BaseAgent, "load_model_and_tokenizer", lambda self, **kwargs: (None, None)),
            mock.patch.object(BaseAgent, "generate", make_stub_generate(cls.llm_stub)),
            mock.patch.object(WebRetriever, "ddgs_search", fail_if_searched),
            mock.patch.object(WebRetriever, "serper_search", fail_if_searched),
            mock.patch.object(WebRetriever, "serp_search", fail_if_searched),
            # cluster_attribute loads Qwen3-Embedding through AutoModel/AutoTokenizer directly.
            mock.patch.object(BaseAgent, "find_config_dir", lambda self, root: Path(root)),
            mock.patch.object(base_agent_model.AutoModel, "from_pretrained", lambda *a, **k: mock.MagicMock()),
            mock.patch.object(
                base_agent_model.AutoTokenizer,
                "from_pretrained",
                lambda *a, **k: types.SimpleNamespace(eos_token="<|endoftext|>"),  # noqa: S106
            ),
            mock.patch.object(ClusterSubAgent, "_compute_embeddings", stub_compute_embeddings),
        ):
            cls._run_stage(
                "prepare_titles",
                cls._write_config(
                    "prepare_titles",
                    {
                        "out_dir": str(cls.out["prepare_titles"]),
                        "purchases_csv": str(FIXTURE_DIR / "sample_purchases_10.csv"),
                        "out_csv": str(cls.titles_csv),
                    },
                ),
            )
            cls._run_stage(
                "scan",
                cls._write_config(
                    "scan",
                    {"out_dir": str(cls.out["scan"]), "query_path": str(cls.titles_csv)},
                ),
            )
            cls._seed_search_cache(query_suffix=" Amazon product")
            cls._run_stage(
                "search",
                cls._write_config(
                    "search",
                    {
                        "out_dir": str(cls.out["search"]),
                        "validation_path": str(cls.out["scan"]),
                        "search_cache_dir": str(cls.cache_dir),
                    },
                ),
            )
            cls._run_stage(
                "predict_transaction",
                cls._write_config(
                    "predict_transaction",
                    {
                        "out_dir": str(cls.out["transaction"]),
                        "validation_path": str(cls.out["search"]),
                        "query_path": str(cls.out["search"]),
                    },
                ),
            )
            cls._run_stage(
                "predict_user",
                cls._write_config(
                    "predict_user",
                    {
                        "out_dir": str(cls.out["user"]),
                        "validation_path": str(cls.out["transaction"]),
                        "query_path": str(cls.out["transaction"]),
                        "scan_output_path": str(cls.out["scan"]),
                    },
                ),
            )
            cls._run_stage(
                "cluster_attribute",
                cls._write_config(
                    "cluster_attribute",
                    {"out_dir": str(cls.out["clustering"]), "query_path": str(cls.out["user"])},
                    branch_overrides=lambda branch: {
                        "out_dir": str(cls.out["clustering"] / branch["name"]),
                        "cache_embedding_path": str(embedding_cache / branch["name"]),
                    },
                ),
            )
            cls._run_stage(
                "tag_cluster",
                cls._write_config(
                    "tag_cluster",
                    {
                        "out_dir": str(cls.out["tag"]),
                        "query_dir": str(cls.out["clustering"]),
                        "cache_embed_dir": str(embedding_cache),
                    },
                ),
            )

    # -------------------------------------------------------------- helpers
    @classmethod
    def _stage_output(cls, stage: str) -> list[dict]:
        json_paths = sorted(cls.out[stage].glob("output-*.json"))
        if len(json_paths) != 1:
            raise AssertionError(f"expected 1 output file in {cls.out[stage]}, got {json_paths}")
        with open(json_paths[0]) as f:
            return json.load(f)

    @staticmethod
    def _by_query(records: list[dict]) -> dict[str, dict]:
        return {r["resolved_query"]: r for r in records}

    @staticmethod
    def _attributes(record: dict, group: str) -> dict[str, Any]:
        entries = record["user"][group]["consumer_attributes"]
        return {e["category"]: e["attribute"] for e in entries}

    def _title(self, index: int) -> str:
        """1-based index into the fixture CSV rows."""
        return self.titles[index - 1]

    @classmethod
    def _expected_free_attributes(cls, group: str) -> dict[str, float]:
        """{attribute: confidence} of a group over all 10 fixture products."""
        expected: dict[str, float] = {}
        for response in cls.llm_stub["user"].values():
            expected.update(response[group]["consumer_attributes"][-1].get("confidence") or {})
        return expected

    # ------------------------------------------------------------ assertions
    def test_prepare_titles_extracts_all_sample_titles(self) -> None:
        with open(self.titles_csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sorted(r["Title"] for r in rows), sorted(self.titles))
        for row in rows:
            self.assertEqual(row["freq"], "1")
            self.assertEqual(row["n_unique_buyers"], "1")

    def test_scan_covers_all_titles_with_expected_flags(self) -> None:
        records = self._by_query(self._stage_output("scan"))
        self.assertEqual(sorted(records), sorted(self.titles))
        for title, record in records.items():
            expected = self.llm_stub["scan"][title]
            self.assertTrue(record["scan"]["is_json"], title)
            self.assertEqual(record["scan"]["need_search"], expected["need_search"], title)
            self.assertEqual(record["scan"]["is_diagnostic"], expected["is_diagnostic"], title)

    def test_search_serves_gated_titles_from_cache(self) -> None:
        records = self._by_query(self._stage_output("search"))
        searchable = [t for t in self.titles if self.llm_stub["scan"][t]["need_search"]]
        self.assertEqual(sorted(records), sorted(searchable))
        # The generic USB-C cable (need_search=False) must be gated out here.
        self.assertNotIn(self._title(10), records)
        for title, record in records.items():
            self.assertTrue(record["search"]["searched"], title)
            self.assertTrue(record["search"]["success_search"], title)
            self.assertEqual(record["search"]["search_result"], self.search_stub[title], title)

    def test_transaction_recovers_canonical_names(self) -> None:
        records = self._by_query(self._stage_output("transaction"))
        self.assertEqual(sorted(records), sorted(self.llm_stub["transaction"]))
        for title, record in records.items():
            self.assertTrue(record["transaction"]["success_recovery"], title)
            self.assertEqual(
                record["transaction"]["prediction"]["canonical_name"],
                self.llm_stub["transaction"][title]["canonical_name"],
                title,
            )

    def test_user_output_covers_all_10_products_with_valid_groups(self) -> None:
        records = self._by_query(self._stage_output("user"))
        self.assertEqual(sorted(records), sorted(self.titles))
        group_sizes = {
            "user_attribute_demographic": 8,  # 7 signals + free_description
            "user_attribute_psycho_behavioral": 5,  # 4 signals + free_description
            "user_attribute_life_event": 9,  # 8 signals + free_description
        }
        for title, record in records.items():
            self.assertTrue(record["user"]["run_predict"], title)
            self.assertTrue(record["user"]["success_predict"], title)
            for group, size in group_sizes.items():
                self.assertTrue(record["user"][group]["is_json"], f"{title} / {group}")
                self.assertEqual(len(record["user"][group]["consumer_attributes"]), size, f"{title} / {group}")

    def test_gated_out_product_is_recovered_via_scan_only_merge(self) -> None:
        cable = self._by_query(self._stage_output("user"))[self._title(10)]
        self.assertTrue(cable["transaction"]["from_scan_only"])
        self.assertEqual(cable["items"], [{"product_name": self._title(10), "canonical_name": self._title(10)}])

    def test_expected_user_attributes_per_product(self) -> None:
        records = self._by_query(self._stage_output("user"))
        expectations: list[tuple[int, str, str, Any]] = [
            (1, "user_attribute_life_event", "signal:recent-childbirth", "positive"),
            (1, "user_attribute_demographic", "signal:gender", "female"),
            (2, "user_attribute_life_event", "signal:recent-childbirth", "positive"),
            (2, "user_attribute_psycho_behavioral", "signal:lifestyle", ["family-oriented"]),
            (3, "user_attribute_life_event", "signal:recent-relocation", "positive"),
            (4, "user_attribute_demographic", "signal:gender", "female"),
            (4, "user_attribute_life_event", "signal:recent-childbirth", "positive"),
            (5, "user_attribute_demographic", "signal:gender", "male"),
            (6, "user_attribute_demographic", "signal:age-bin", ["45-54", "55-64"]),
            (
                7,
                "user_attribute_psycho_behavioral",
                "free_description_attributes",
                ["user owns a senior small-breed dog"],
            ),
            (8, "user_attribute_psycho_behavioral", "signal:lifestyle", ["luxury-oriented"]),
            (9, "user_attribute_psycho_behavioral", "signal:lifestyle", ["gourmet-foodie"]),
        ]
        for index, group, category, expected in expectations:
            with self.subTest(product=index, category=category):
                attributes = self._attributes(records[self._title(index)], group)
                self.assertEqual(attributes[category], expected)

    def test_no_signal_product_yields_all_unknown(self) -> None:
        cable = self._by_query(self._stage_output("user"))[self._title(10)]
        for group in ("user_attribute_demographic", "user_attribute_psycho_behavioral", "user_attribute_life_event"):
            attributes = self._attributes(cable, group)
            free = attributes.pop("free_description_attributes")
            self.assertEqual(free, [], group)
            self.assertTrue(all(v == "unknown" for v in attributes.values()), f"{group}: {attributes}")

    # ---------------------------------------------- clustering / tagging
    GROUP_TO_CLUSTERS = {
        "user_attribute_demographic": 2,
        "user_attribute_psycho_behavioral": 3,
        "user_attribute_life_event": 2,
    }

    def test_clustering_assigns_every_free_attribute_to_a_cluster(self) -> None:
        records = self._by_query(self._stage_output("clustering"))
        self.assertEqual(sorted(records), sorted(self.titles))
        for group, n_clusters in self.GROUP_TO_CLUSTERS.items():
            expected = self._expected_free_attributes(group)
            seen: dict[str, dict[str, Any]] = {}
            for record in records.values():
                seen.update(record["attribute2cluster_id"].get(group, {}))
            self.assertEqual(sorted(seen), sorted(expected), group)
            for attribute, info in seen.items():
                self.assertIsInstance(info["cluster_id"], int, attribute)
                self.assertEqual(info["pseudo_confidence"], expected[attribute], attribute)
            # k-means with the fixed n_clusters of the e2e config, capped by the attribute count.
            self.assertEqual(len({info["cluster_id"] for info in seen.values()}), min(n_clusters, len(expected)), group)

    def test_clusters_follow_attribute_themes(self) -> None:
        records = self._by_query(self._stage_output("clustering"))
        for group in self.GROUP_TO_CLUSTERS:
            theme_to_clusters: dict[str, set[int]] = {}
            for record in records.values():
                for attribute, info in record["attribute2cluster_id"].get(group, {}).items():
                    theme_to_clusters.setdefault(theme_of(attribute), set()).add(info["cluster_id"])
            # every theme ends up in exactly one cluster, and different themes in different clusters
            for theme, clusters in theme_to_clusters.items():
                self.assertEqual(len(clusters), 1, f"{group}: {theme} -> {clusters}")
            all_clusters = [next(iter(c)) for c in theme_to_clusters.values()]
            self.assertEqual(len(all_clusters), len(set(all_clusters)), group)

    def test_tags_cover_every_cluster_and_match_the_canned_responses(self) -> None:
        records = self._by_query(self._stage_output("tag"))
        self.assertEqual(sorted(records), sorted(self.titles))
        for group in self.GROUP_TO_CLUSTERS:
            suffix = group.replace("user_attribute_", "")
            for title, record in records.items():
                attributes = record["attribute2cluster_id"].get(group, {})
                tags = record[f"cluster_id2tag_{suffix}"]
                confidences = record[f"tag2pseudo_confidence_{suffix}"]
                n_clusters = len({info["cluster_id"] for info in attributes.values()})
                self.assertEqual(len(tags), n_clusters, f"{title}/{suffix}")
                for attribute, info in attributes.items():
                    tag = tags[str(info["cluster_id"])]
                    self.assertEqual(tag, self.llm_stub["tag"][attribute], f"{title}/{attribute}")
                    self.assertGreaterEqual(confidences[tag], info["pseudo_confidence"], f"{title}/{attribute}")

    def test_no_signal_product_has_no_tags(self) -> None:
        cable = self._by_query(self._stage_output("tag"))[self._title(10)]
        for suffix in ("demographic", "psycho_behavioral", "life_event"):
            self.assertEqual(cable[f"cluster_id2tag_{suffix}"], {}, suffix)
            self.assertEqual(cable[f"tag2pseudo_confidence_{suffix}"], {}, suffix)


if __name__ == "__main__":
    unittest.main()
