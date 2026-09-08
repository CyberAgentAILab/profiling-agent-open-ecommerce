import ast
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from umap import UMAP

from base_agent.model import BaseAgent
from cluster_attribute.dataset import extract_pseudo_confidence

_VALID_EMBEDDING_BACKENDS = {"sentence_transformer", "plamo", "qwen3_embedding"}


class ClusterSubAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        path_prompt_template: str,
        out_dir: str,
        timestamp: str,
        embedding_backend: str,
        n_clusters: int,
        algorithm: str,
        seed: int,
        batch_size: int,
        min_cluster_ratio: float,
        min_samples: int | None,
        metric: str,
        cluster_selection_method: str,
        dimension_reduction: str,
        pca_n_components: int,
        pca_random_state: int,
        umap_n_components: int,
        umap_min_dist: float,
        umap_n_neighbors: int,
        umap_metric: str,
        minibatch_kmeans_batch_size: int,
        minibatch_kmeans_max_iter: int,
        minibatch_kmeans_n_init: int,
        minibatch_kmeans_max_no_improvement: int,
        minibatch_kmeans_reassignment_ratio: float,
        cache_embedding_path: str,
        hdbscan_n_jobs: int | None = None,
        enable_cache: bool = True,
        qwen3_max_length: int = 8192,
    ):
        if embedding_backend not in _VALID_EMBEDDING_BACKENDS:
            raise ValueError(
                f"embedding_backend must be one of {sorted(_VALID_EMBEDDING_BACKENDS)}, got {embedding_backend!r}"
            )
        use_sentence_transformer = embedding_backend == "sentence_transformer"
        use_qwen3_embedding = embedding_backend == "qwen3_embedding"

        super().__init__(
            model_name=model_name,
            torch_dtype=torch_dtype,
            path_prompt_template=path_prompt_template,
            use_sentence_transformer=use_sentence_transformer,
            use_qwen3_embedding=use_qwen3_embedding,
            out_dir=out_dir,
            timestamp=timestamp,
            enable_cache=enable_cache,
        )
        self.embedding_backend = embedding_backend
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.seed = seed
        self.batch_size = batch_size
        self.min_cluster_ratio = min_cluster_ratio
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.dimension_reduction = dimension_reduction
        self.pca_n_components = pca_n_components
        self.pca_random_state = pca_random_state
        self.umap_n_components = umap_n_components
        self.umap_min_dist = umap_min_dist
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_metric = umap_metric
        self.minibatch_kmeans_batch_size = minibatch_kmeans_batch_size
        self.minibatch_kmeans_max_iter = minibatch_kmeans_max_iter
        self.minibatch_kmeans_n_init = minibatch_kmeans_n_init
        self.minibatch_kmeans_max_no_improvement = minibatch_kmeans_max_no_improvement
        self.minibatch_kmeans_reassignment_ratio = minibatch_kmeans_reassignment_ratio
        self.cache_embedding_path = cache_embedding_path
        # Namespace the cache by model so different embedding models never read
        # each other's CSVs (e.g. plamo's 2048-dim vs qwen3's 1024-dim vectors).
        self.model_slug = self._model_slug(model_name)
        self.model_cache_dir = str(Path(cache_embedding_path) / self.model_slug)
        self.hdbscan_n_jobs = hdbscan_n_jobs
        self.qwen3_max_length = qwen3_max_length
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")
        if not 0.0 < self.min_cluster_ratio <= 1.0:
            raise ValueError(f"min_cluster_ratio must be in (0.0, 1.0], got {self.min_cluster_ratio}")
        return

    # ── embedding helpers ────────────────────────────────────────────────────

    @staticmethod
    def _model_slug(model_name: str) -> str:
        """Filesystem-safe leaf name identifying the embedding model.

        Used to namespace the embedding cache per model so caches from different
        models (which may have different dimensionality) are never mixed up. E.g.
        "./models/Qwen3-Embedding-0.6B" -> "Qwen3-Embedding-0.6B".
        """
        slug = Path(model_name).name or model_name
        return re.sub(r"[^0-9A-Za-z._-]+", "_", slug)

    def _load_embeddings_from_cache(self, attribute_texts: list[str]) -> np.ndarray | None:
        """Return embeddings from the latest cache CSV, or None if absent/mismatched.

        The cached rows must match `attribute_texts` exactly (same texts, same
        order) — otherwise embeddings would silently misalign to attributes, so
        a mismatched cache is ignored and embeddings are recomputed.
        """
        cache_dir = Path(self.model_cache_dir)
        if not cache_dir.exists():
            logger.info(f"🔍 Cache directory does not exist yet: {self.model_cache_dir}")
            return None
        files = sorted(cache_dir.glob("attribute_embeddings_*.csv"))
        if not files:
            logger.info(f"🔍 No existing embedding files found in cache directory: {self.model_cache_dir}")
            return None
        latest = files[-1]
        logger.info(f"🔍 Found existing embedding file for model '{self.model_slug}' in cache: {latest}")
        loaded_df = pd.read_csv(str(latest))
        if "attribute_text" not in loaded_df.columns:
            logger.warning(f"⚠️ Cache file has no attribute_text column, recomputing embeddings: {latest}")
            return None
        cached_texts = loaded_df["attribute_text"].astype(str).tolist()
        if cached_texts != attribute_texts:
            logger.warning(
                f"⚠️ Cached embeddings do not match the current attribute set "
                f"({len(cached_texts)} cached vs {len(attribute_texts)} current), recomputing: {latest}"
            )
            return None
        logger.info("⚡ Skipping embedding calculation, loading from file...")
        logger.info(f"📂 Loaded {len(loaded_df)} embeddings from {latest}")
        embeddings = np.array([ast.literal_eval(emb) for emb in loaded_df["embedding"]])
        logger.info(f"✅ Restored embeddings shape: {embeddings.shape}")
        return embeddings

    def _compute_embeddings(self, attribute_texts: list[str]) -> np.ndarray:
        """Run batched embedding inference and return an (N, D) float ndarray."""
        n_attrs = len(attribute_texts)
        n_batches = (n_attrs + self.batch_size - 1) // self.batch_size
        logger.info(f"📦 Batch inference started: {n_attrs} items at {self.batch_size} per batch ({n_batches} batches)")

        begin = time.perf_counter()
        embeddings_list = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            batch_texts = attribute_texts[start_idx : start_idx + self.batch_size]
            if "plamo-embedding-1b" in self.model_name:
                batch_emb = self.model.encode_document(batch_texts, self.tokenizer)
            elif self.embedding_backend == "qwen3_embedding":
                batch_emb = self._encode_qwen3(batch_texts)
            else:
                batch_emb = self.model.encode(batch_texts, batch_size=self.batch_size)
            embeddings_list.append(self._to_numpy(batch_emb))
            logger.debug(f"✅ Batch {batch_idx + 1}/{n_batches}: processed {len(batch_texts)} items")

        embeddings = np.vstack(embeddings_list)
        logger.info(f"⏱️ Embedding calculation completed in {time.perf_counter() - begin:.2f} seconds")
        return embeddings

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the hidden state of the last real token of each sequence.

        Mirrors the reference implementation from the Qwen3-Embedding model card
        (https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). With left padding the
        last real token is always at index -1; otherwise it is located via the
        attention mask.
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _encode_qwen3(self, texts: list[str]) -> torch.Tensor:
        """Embed a batch of texts with Qwen3-Embedding (last-token pool + L2 norm).

        Attributes are encoded as retrieval *documents*, which per the model card
        require no instruction prefix (only queries are wrapped with an Instruct:
        template). The returned tensor is L2-normalized so cosine geometry holds.
        """
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.qwen3_max_length,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        # Cast to float32: bfloat16 tensors cannot be converted by _to_numpy().
        return F.normalize(embeddings.float(), p=2, dim=1)

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        if hasattr(tensor, "cpu"):
            return tensor.cpu().numpy()
        if not isinstance(tensor, np.ndarray):
            return np.array(tensor)
        return tensor

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA or UMAP dimension reduction and return the transformed array."""
        if self.dimension_reduction == "pca":
            logger.info(
                f"🔬 Applying PCA dimension reduction: {embeddings.shape} -> n_components={self.pca_n_components}"
            )
            pca = PCA(n_components=self.pca_n_components, random_state=self.pca_random_state)
            result = pca.fit_transform(embeddings)
            logger.info(
                f"✅ PCA completed: shape={result.shape}, "
                f"explained_variance_ratio={pca.explained_variance_ratio_.sum():.4f}"
            )
            return result
        if self.dimension_reduction == "umap":
            logger.info(
                f"🔬 Applying UMAP dimension reduction: {embeddings.shape} -> n_components={self.umap_n_components}"
            )
            result = UMAP(
                n_components=self.umap_n_components,
                min_dist=self.umap_min_dist,
                n_neighbors=self.umap_n_neighbors,
                metric=self.umap_metric,
            ).fit_transform(embeddings)
            logger.info(f"✅ UMAP completed: shape={result.shape}")
            return result
        logger.info(f"⏭️ Skipping dimension reduction (dimension_reduction={self.dimension_reduction})")
        return embeddings

    def _save_embeddings_to_cache(
        self,
        embeddings: np.ndarray,
        attribute_texts: list[str],
        raw_queries: list[str],
        patterns: list[str],
    ) -> None:
        """Persist embeddings CSV to the model-namespaced cache directory."""
        os.makedirs(self.model_cache_dir, exist_ok=True)
        filename = os.path.join(self.model_cache_dir, f"attribute_embeddings_{self.timestamp}.csv")
        pd.DataFrame(
            {
                "resolved_query": raw_queries,
                "pattern": patterns,
                "attribute_text": attribute_texts,
                "embedding": embeddings.tolist(),
            }
        ).to_csv(filename, index=False)
        logger.info(f"💾 Saved reduced attribute embeddings to cache: {filename}")

    def vectorize_attribute(self, df_user_attributes: pd.DataFrame) -> pd.DataFrame:
        attribute_texts = df_user_attributes["attribute"].dropna().unique().tolist()
        filtered_df = df_user_attributes[df_user_attributes["attribute"].isin(attribute_texts)]
        result_df = filtered_df.drop_duplicates(subset=["attribute"])[["attribute", "resolved_query", "pattern"]]
        raw_queries = result_df["resolved_query"].tolist()
        patterns = result_df["pattern"].tolist()

        cached = self._load_embeddings_from_cache(attribute_texts)
        if cached is not None:
            embeddings = cached
        else:
            embeddings = self._compute_embeddings(attribute_texts)
            embeddings = self._reduce_dimensions(embeddings)
            self._save_embeddings_to_cache(embeddings, attribute_texts, raw_queries, patterns)

        attribute_emb_df = pd.DataFrame(
            {
                "resolved_query": raw_queries,
                "pattern": patterns,
                "attribute_text": attribute_texts,
                "embedding": embeddings.tolist(),
                "duration_embedding_per_attribute": 0.0,
            }
        )
        logger.info(f"✅ num_attributes: {len(attribute_texts)}")
        return attribute_emb_df

    # ── clustering helpers ───────────────────────────────────────────────────

    def _fit_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the configured clustering algorithm and return cluster labels."""
        logger.info(f"🧮 apply {self.algorithm}")
        n_attributes = embeddings.shape[0]
        if n_attributes < 2:
            # Nothing to separate (e.g. a small slice yielded a single free-description
            # attribute): every clustering backend needs >= 2 samples, so put the
            # attribute(s) into one cluster instead of failing the branch.
            logger.warning(f"⚠️ only {n_attributes} attribute(s) to cluster; assigning a single cluster (id 0)")
            return np.zeros(n_attributes, dtype=int)
        if self.algorithm == "hdbscan":
            # Derive min_cluster_size as a ratio of the number of attributes being
            # clustered (HDBSCAN requires an int >= 2), so the threshold scales with
            # the dataset instead of being a fixed magic number.
            min_cluster_size = max(2, int(n_attributes * self.min_cluster_ratio))
            logger.info(
                f"📐 min_cluster_size={min_cluster_size} "
                f"(min_cluster_ratio={self.min_cluster_ratio} × {n_attributes} attributes)"
            )
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method=self.cluster_selection_method,
                n_jobs=self.hdbscan_n_jobs,
            ).fit_predict(normalize(embeddings, norm="l2", axis=1))
            return labels
        if self.algorithm == "minibatch_kmeans":
            logger.info(
                f"📊 MiniBatch KMeans parameters: "
                f"batch_size={self.minibatch_kmeans_batch_size}, "
                f"max_iter={self.minibatch_kmeans_max_iter}, "
                f"n_init={self.minibatch_kmeans_n_init}"
            )
            model = MiniBatchKMeans(
                n_clusters=min(self.n_clusters, n_attributes),
                random_state=self.seed,
                batch_size=self.minibatch_kmeans_batch_size,
                max_iter=self.minibatch_kmeans_max_iter,
                n_init=self.minibatch_kmeans_n_init,
                max_no_improvement=self.minibatch_kmeans_max_no_improvement,
                reassignment_ratio=self.minibatch_kmeans_reassignment_ratio,
                verbose=0,
            ).fit(embeddings)
            logger.info(f"✅ MiniBatch KMeans completed: inertia={model.inertia_:.2f}")
            return model.labels_
        if self.algorithm == "kmeans":
            model = KMeans(n_clusters=min(self.n_clusters, n_attributes), random_state=self.seed).fit(embeddings)
            logger.info(f"✅ KMeans completed: inertia={model.inertia_:.2f}")
            return model.labels_
        raise ValueError(f"Unknown algorithm: {self.algorithm}. Choose from 'kmeans', 'minibatch_kmeans', 'hdbscan'")

    def _extract_pattern_attributes(
        self,
        user_profile: dict,
        target_patterns: list[str] | None,
    ) -> dict[str, dict[str, float]]:
        """Return {pattern: pseudo_confidence_dict} filtered to target_patterns."""
        result: dict[str, dict[str, float]] = {}
        for pattern, attr in user_profile.items():
            if target_patterns and pattern not in target_patterns:
                continue
            if not isinstance(attr, dict):
                continue
            attr_list = attr.get("consumer_attributes")
            if not isinstance(attr_list, list) or not attr_list:
                continue
            pseudo_confidence = extract_pseudo_confidence(attr_list[-1])
            if pseudo_confidence is None:
                continue
            result[pattern] = pseudo_confidence
        return result

    def run_clustering(
        self,
        profiles: list[dict[str, Any]],
        attribute_df: pd.DataFrame,
        target_patterns: list[str] | None = None,
    ) -> None:
        """Cluster this branch's attributes and merge cluster ids into profiles in place.

        Each profile's "attribute2cluster_id" is updated (not overwritten) per pattern,
        so multiple branches can accumulate their results into the same profile list.
        """
        if attribute_df.empty:
            logger.warning(f"⚠️ no attributes to cluster for target_patterns={target_patterns}; skipping this branch")
            return
        attribute_emb_df = self.vectorize_attribute(df_user_attributes=attribute_df)
        attribute_emb_df["cluster_id"] = self._fit_clusters(np.vstack(attribute_emb_df["embedding"]))
        attribute2cluster = dict(zip(attribute_emb_df["attribute_text"], attribute_emb_df["cluster_id"], strict=True))

        for profile in profiles:
            user_profile = profile.get("user")
            if user_profile is None:
                continue
            pattern_pseudo_confidence = self._extract_pattern_attributes(user_profile, target_patterns)
            merged = profile.setdefault("attribute2cluster_id", {})
            for pattern, pseudo_confidence_dict in pattern_pseudo_confidence.items():
                merged[pattern] = {
                    attribute: {
                        "cluster_id": attribute2cluster.get(attribute),
                        "pseudo_confidence": pseudo_conf,
                    }
                    for attribute, pseudo_conf in pseudo_confidence_dict.items()
                }
