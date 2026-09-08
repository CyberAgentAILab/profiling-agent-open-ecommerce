import ast
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

from base_agent.config import save_as_json, validate_json
from base_agent.model import BaseAgent


def select_representative_indices(
    n_samples: int,
    n_select: int,
    method: str = "first",
    seed: int | None = 42,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """
    Select representative sample indices from a cluster.

    Args:
        n_samples: Total number of samples in the cluster
        n_select: Number of samples to select
        method: Selection method ("first", "random", "kmedoids")
        seed: Random seed (for "random" and "kmedoids" methods)
        x: Embedding vectors (n_samples, n_dim). Required for "kmedoids" method.

    Returns:
        np.ndarray of selected indices
    """
    n_select = min(n_select, n_samples)

    if method == "first":
        return np.arange(n_select)
    elif method == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(n_samples, size=n_select, replace=False)
    elif method == "kmedoids":
        if x is None:
            raise ValueError("x (embeddings) is required for method='kmedoids'")
        if n_select >= n_samples:
            return np.arange(n_samples)
        # Approximate k-medoids: cluster with KMeans, then pick the real point
        # nearest to each centroid as that cluster's medoid.
        km = KMeans(n_clusters=n_select, random_state=seed, n_init=10).fit(x)
        labels = km.labels_
        centers = km.cluster_centers_
        medoid_indices = []
        for c in range(n_select):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            dists = np.linalg.norm(x[members] - centers[c], axis=1)
            medoid_indices.append(int(members[np.argmin(dists)]))
        return np.array(medoid_indices)
    else:
        raise ValueError(f"Unknown method: {method}")


class TaggerSubAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        out_dir: str,
        timestamp: str,
        max_new_tokens: int,
        use_vllm: bool,
        tensor_parallel_size: int,
        trust_remote_code: bool,
        seed: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        json_schema_tag: list,
        enable_cache: bool = True,
        sampling_method: str = "kmedoids",
        cache_embed_dir: str = "",
        cache_embed_model: str = "",
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        enable_thinking: bool | None = None,
        max_model_len: int | None = None,
    ):
        super().__init__(
            model_name=model_name,
            torch_dtype=torch_dtype,
            path_prompt_template=path_prompt_template,
            out_dir=out_dir,
            timestamp=timestamp,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_cache=enable_cache,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )
        self.json_schema_tag = json_schema_tag
        self.sampling_method = sampling_method
        self.cache_embed_dir = cache_embed_dir
        # Embedding caches are namespaced per model (cluster_attribute writes
        # <category>/<model_slug>/*.csv). Filter to this model so different
        # models' vectors (e.g. plamo 2048-dim vs qwen3 1024-dim) never mix.
        self.cache_embed_model_slug = self._model_slug(cache_embed_model) if cache_embed_model else ""
        self.embeddings: np.ndarray | None = None
        self.attribute_to_index: dict[str, int] | None = None
        self.seed = seed

        # Load the embedding cache only when sampling with kmedoids.
        if self.sampling_method == "kmedoids":
            if self.cache_embed_dir:
                self._load_embeddings()
            else:
                logger.warning("sampling_method='kmedoids' but cache_embed_dir is empty; falling back to 'head'")
                self.sampling_method = "head"

    @staticmethod
    def _model_slug(model_name: str) -> str:
        """Filesystem-safe leaf name identifying the embedding model.

        Matches cluster_attribute's namespacing so the same model resolves to the
        same directory name. E.g. "./models/plamo-embedding-1b" -> "plamo-embedding-1b".
        """
        slug = Path(model_name).name or model_name
        return re.sub(r"[^0-9A-Za-z._-]+", "_", slug)

    def _resolve_embed_csv_paths(self) -> list[str]:
        """Return the CSV paths under cache_embed_dir.

        cache_embed_dir may point to a directory (all *.csv under it, recursively,
        are loaded — e.g. cluster_attribute writes one CSV per attribute category)
        or to a single CSV file (bank pipeline, backward compat).

        When cache_embed_model is set, only CSVs under the matching <model_slug>/
        directory are kept, so different embedding models' caches never mix.
        """
        p = Path(self.cache_embed_dir)
        if p.is_dir():
            paths = sorted(str(f) for f in p.glob("**/*.csv"))
            if self.cache_embed_model_slug:
                paths = [f for f in paths if Path(f).parent.name == self.cache_embed_model_slug]
            return paths
        if p.is_file():
            return [str(p)]
        return []

    def _load_embeddings(self) -> None:
        """Load the embedding CSV(s) under cache_embed_dir and build the attribute -> reduced-embedding map.

        CSVs from multiple categories are concatenated; when the same attribute_text appears more
        than once, the last (most recent) row wins.
        """
        csv_paths = self._resolve_embed_csv_paths()
        if not csv_paths:
            logger.warning(
                f"No embedding CSV found under cache_embed_dir={self.cache_embed_dir}; falling back to 'head'"
            )
            self.sampling_method = "head"
            return

        try:
            df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

            # Detect the CSV format: use the attribute_text and embedding columns
            if "attribute_text" in df.columns and "embedding" in df.columns:
                df = df.drop_duplicates(subset=["attribute_text"], keep="last").reset_index(drop=True)
                self.attribute_to_index = {attr: idx for idx, attr in enumerate(df["attribute_text"])}
                self.embeddings = np.array([ast.literal_eval(s) for s in df["embedding"]], dtype=np.float32)
            else:
                # Legacy format: attribute, embed_0, embed_1, ..., embed_n
                attribute_col = "attribute" if "attribute" in df.columns else df.columns[0]
                embedding_cols = [col for col in df.columns if col.startswith("embed_")]
                df = df.drop_duplicates(subset=[attribute_col], keep="last").reset_index(drop=True)
                self.attribute_to_index = {attr: idx for idx, attr in enumerate(df[attribute_col])}
                self.embeddings = df[embedding_cols].values.astype(np.float32)

            logger.info(
                f"📂 Loaded {len(self.attribute_to_index)} attribute embeddings "
                f"from {len(csv_paths)} CSV(s) under {self.cache_embed_dir}"
            )

        except Exception as e:
            logger.warning(f"Warning: Failed to load embeddings from {self.cache_embed_dir}: {e}")
            self.sampling_method = "head"

    def _build_context(self, cluster: dict[str, Any]) -> str:
        """Build the tag-generation context from a cluster's attributes.

        Args:
            cluster: Cluster info (contains cluster_id and cluster_attributes)

        Returns:
            Context string for tag generation
        """
        cluster_attributes = cluster["cluster_attributes"]

        # When sampling_method=kmedoids and embeddings are loaded, pick representative samples via k-medoids
        if self.sampling_method == "kmedoids" and self.embeddings is not None and self.attribute_to_index is not None:
            # Collect the embeddings for the attributes in this cluster
            valid_indices = []
            valid_attributes = []

            for attr in cluster_attributes:
                if attr in self.attribute_to_index:
                    valid_indices.append(self.attribute_to_index[attr])
                    valid_attributes.append(attr)

            if valid_indices:
                # Fetch the matching embedding vectors
                cluster_embeddings = self.embeddings[valid_indices]

                # Decide the number of representatives (at most 100, capped by cluster size)
                n_select = min(100, len(valid_attributes))

                try:
                    # Select representative indices with k-medoids
                    logger.debug("build_context with k-medoids sampling")
                    representative_indices = select_representative_indices(
                        n_samples=len(valid_attributes),
                        n_select=n_select,
                        method="kmedoids",
                        seed=self.seed,
                        x=cluster_embeddings,
                    )

                    # Pick the representative attributes
                    selected_attributes = [valid_attributes[i] for i in representative_indices]
                    return "\n".join(selected_attributes)

                except Exception as e:
                    logger.warning(f"Warning: k-medoids failed for cluster {cluster.get('cluster_id', 'unknown')}: {e}")
                    # Fall back to the first 100 attributes
                    logger.info("Fallback: build_context with the first 100 attributes")
                    pass

        # Fallback: use the first 100 attributes as before
        return "\n".join(cluster_attributes[:100])

    def add_tag(
        self,
        batch_cluster: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        batch_context = list(map(self._build_context, batch_cluster))
        # Use the actual batch length so the final (smaller) batch is not misreported.
        time_scale = 1 / max(len(batch_context), 1)
        begin = time.perf_counter()
        batch_tag = self.generate(prompt=batch_context)
        duration_per_sample = time_scale * (time.perf_counter() - begin)

        results = []
        for tag, cluster in zip(batch_tag, batch_cluster, strict=True):
            attribute = validate_json(tag["content"], required_keys=self.json_schema_tag)
            run_tagger = True
            # validate_json sets is_json only when the output matches the tag
            # schema, so invalid JSON is never reported as a success.
            success_tagger = bool(attribute.get("is_json"))

            cluster.update(
                {
                    "tagger": attribute,
                    "run_tagger": run_tagger,
                    "success_tagger": success_tagger,
                    "duration_tagger": float(f"{duration_per_sample:.04f}"),
                    "input_tokens": tag["input_tokens"],
                    "output_tokens": tag["output_tokens"],
                }
            )
            results.append(cluster)
        return results

    def _process_single_pattern(
        self,
        cluster_id2attributes: dict[int, list[str]],
        batch_size: int,
        debug_num_sample: int | None,
        pattern_name: str,
    ) -> dict[int, str]:
        """Run tagging for a single pattern (attribute category).

        Args:
            cluster_id2attributes: Mapping of cluster_id -> attribute list
            batch_size: Batch size
            debug_num_sample: Sample-count cap for debugging
            pattern_name: Pattern name (for display)

        Returns:
            Mapping of cluster_id -> tag
        """
        # Build the cluster list
        clusters = []
        for cluster_id, cluster_attributes in cluster_id2attributes.items():
            clusters.append({"cluster_id": cluster_id, "cluster_attributes": cluster_attributes})

        if debug_num_sample is not None:
            clusters = clusters[:debug_num_sample]

        # Run LLM inference
        tags = []
        num_total_batch = (len(clusters) + batch_size - 1) // batch_size
        pbar = tqdm(
            range(0, len(clusters), batch_size),
            total=num_total_batch,
            desc=f"Tagging {pattern_name}",
        )
        for step in pbar:
            batch_cluster = clusters[step : step + batch_size]
            batch_tag = self.add_tag(
                batch_cluster=batch_cluster,
            )
            tags.extend(batch_tag)

        # Return the cluster_id -> tag mapping
        result_mapping = {}
        for tag in tags:
            cluster_id = tag["cluster_id"]
            if cluster_id == -1:
                # HDBSCAN's outlier cluster always gets the fixed "outlier" tag
                result_mapping[cluster_id] = "outlier"
            else:
                result_mapping[cluster_id] = tag["tagger"].get("tag", "")
        return result_mapping

    @staticmethod
    def _apply_pattern_tags(
        profile: dict[str, Any],
        pattern_name: str,
        cluster_id2tag: dict[int, str],
        attribute2cluster_id: dict[str, Any],
        attribute2pseudo_confidence: dict[str, float],
    ) -> None:
        """Write one pattern's tags and the tag -> pseudo_confidence aggregation into the profile."""
        # user_attribute_<pattern> -> cluster_id2tag_<pattern>
        suffix = pattern_name.replace("user_attribute_", "")

        # Per-pattern mapping of tag -> pseudo_confidence
        tag2pseudo_confidence: dict[str, float] = {}

        # Handle both dict cluster_info (new format) and plain values (old format)
        cluster_ids = []
        for attribute, cluster_info in attribute2cluster_id.items():
            if isinstance(cluster_info, dict):
                cluster_id = cluster_info.get("cluster_id")
                pseudo_conf = cluster_info.get("pseudo_confidence")
            else:
                cluster_id = cluster_info
                pseudo_conf = None

            # cluster_attribute's normal output always carries pseudo_confidence, so this
            # fallback only kicks in for missing records / the old format (backfilled
            # from the globally aggregated attribute2pseudo_confidence).
            if pseudo_conf is None:
                pseudo_conf = attribute2pseudo_confidence.get(attribute)

            if cluster_id is None:
                continue
            cluster_ids.append(cluster_id)

            # Aggregate pseudo_confidence per tag (keep the maximum)
            tag = cluster_id2tag.get(cluster_id)
            if tag and pseudo_conf is not None:
                # Convert to float when given as a string
                if isinstance(pseudo_conf, str):
                    try:
                        pseudo_conf = float(pseudo_conf)
                    except ValueError:
                        continue
                if tag not in tag2pseudo_confidence:
                    tag2pseudo_confidence[tag] = pseudo_conf
                else:
                    tag2pseudo_confidence[tag] = max(tag2pseudo_confidence[tag], pseudo_conf)

        profile[f"cluster_id2tag_{suffix}"] = {
            str(cluster_id): cluster_id2tag.get(cluster_id) for cluster_id in cluster_ids
        }
        # Add the per-pattern tag -> pseudo_confidence mapping
        profile[f"tag2pseudo_confidence_{suffix}"] = tag2pseudo_confidence

    def run_tagging(
        self,
        profiles: list[dict[str, Any]],
        cluster_id2attributes: dict[str, dict[int, list[str]]],
        attribute2pseudo_confidence: dict[str, float],
        batch_size: int,
        debug_num_sample: int | None,
        job_id: str | None = None,
        cluster_id_namespace: str = "shared",
    ) -> str:
        begin = time.perf_counter()

        # Switch the tagging strategy based on the cluster_id namespace.
        #   shared      : all patterns share one cluster_id space
        #                 (clustered jointly, so same id = same cluster)
        #   per_pattern : each pattern has an independent cluster_id space
        #                 (clustered per category, so ids may collide)
        if cluster_id_namespace == "per_pattern":
            pattern2cluster_id2tag = {
                pattern_name: self._process_single_pattern(
                    cluster_id2attributes=pattern_cluster_id2attributes,
                    batch_size=batch_size,
                    debug_num_sample=debug_num_sample,
                    pattern_name=pattern_name,
                )
                for pattern_name, pattern_cluster_id2attributes in cluster_id2attributes.items()
            }
        else:
            # Merge every pattern's cluster_ids into one shared space and tag once
            merged_cluster_id2attributes: dict[int, list[str]] = {}
            for pattern_cluster_id2attributes in cluster_id2attributes.values():
                for cluster_id, attributes in pattern_cluster_id2attributes.items():
                    merged_cluster_id2attributes.setdefault(cluster_id, []).extend(attributes)
            shared_cluster_id2tag = self._process_single_pattern(
                cluster_id2attributes=merged_cluster_id2attributes,
                batch_size=batch_size,
                debug_num_sample=debug_num_sample,
                pattern_name="merged_patterns",
            )
            pattern2cluster_id2tag = dict.fromkeys(cluster_id2attributes, shared_cluster_id2tag)

        # Attach the results to each profile
        for profile in profiles:
            attribute2cluster_id_all = profile.get("attribute2cluster_id")
            if not isinstance(attribute2cluster_id_all, dict):
                continue

            # For each pattern, attach results using that pattern's cluster_id2tag
            for pattern_name in cluster_id2attributes.keys():
                self._apply_pattern_tags(
                    profile=profile,
                    pattern_name=pattern_name,
                    cluster_id2tag=pattern2cluster_id2tag[pattern_name],
                    attribute2cluster_id=attribute2cluster_id_all.get(pattern_name, {}),
                    attribute2pseudo_confidence=attribute2pseudo_confidence,
                )

        duration_tag = time.perf_counter() - begin
        # Fixed: this used to be applied only to the last profile
        for profile in profiles:
            profile["duration_tag"] = float(f"{duration_tag:.04f}")

        save_path = save_as_json(
            responses=profiles,
            out_dir=self.out_dir,
            timestamp=self.timestamp,
            job_id=job_id,
        )

        logger.info(f"💾 tagging results: {len(profiles)} profiles -> {save_path}")
        return save_path
