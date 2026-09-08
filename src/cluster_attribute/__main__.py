from loguru import logger

from base_agent.config import load_config, save_as_json

from .dataset import build_attribute_df, load_profiles
from .model import ClusterSubAgent


def _run_branch(config: dict, profiles: list[dict]) -> bool:
    """Cluster one attribute branch. Returns False when the branch had no attributes to cluster."""
    df_attributes = build_attribute_df(profiles, config.get("target_patterns"))
    if df_attributes.empty:
        logger.warning(f"⚠️ no attributes for target_patterns={config.get('target_patterns')}; skipping this branch")
        return False

    cluster_subagent = ClusterSubAgent(
        model_name=config["model_name"],
        torch_dtype=config["torch_dtype"],
        path_prompt_template="",
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
        embedding_backend=config["embedding_backend"],
        n_clusters=config["n_clusters"],
        algorithm=config["algorithm"],
        seed=config["seed"],
        batch_size=config["batch_size"],
        min_cluster_ratio=config["min_cluster_ratio"],
        min_samples=config["min_samples"],
        metric=config["metric"],
        cluster_selection_method=config["cluster_selection_method"],
        dimension_reduction=config["dimension_reduction"],
        pca_n_components=config["pca_n_components"],
        pca_random_state=config["pca_random_state"],
        umap_n_components=config["umap_n_components"],
        umap_min_dist=config["umap_min_dist"],
        umap_n_neighbors=config["umap_n_neighbors"],
        umap_metric=config["umap_metric"],
        minibatch_kmeans_batch_size=config["minibatch_kmeans_batch_size"],
        minibatch_kmeans_max_iter=config["minibatch_kmeans_max_iter"],
        minibatch_kmeans_n_init=config["minibatch_kmeans_n_init"],
        minibatch_kmeans_max_no_improvement=config["minibatch_kmeans_max_no_improvement"],
        minibatch_kmeans_reassignment_ratio=config["minibatch_kmeans_reassignment_ratio"],
        cache_embedding_path=config["cache_embedding_path"],
        hdbscan_n_jobs=config.get("hdbscan_n_jobs"),
        enable_cache=config["enable_cache"],
        qwen3_max_length=int(config.get("qwen3_max_length", 8192)),
    )

    cluster_subagent.run_clustering(
        profiles=profiles,
        attribute_df=df_attributes,
        target_patterns=config.get("target_patterns"),
    )
    return True


def main() -> None:
    config = load_config()
    config.setdefault("embedding_backend", "plamo")

    profiles = load_profiles(config["query_path"])

    branches = config.get("attribute_branches")
    if branches:
        failed, empty = [], []
        for branch in branches:
            name = branch.get("name", "<unnamed>")
            try:
                if not _run_branch({**config, **branch}, profiles):
                    empty.append(name)
            except Exception:
                logger.exception(f"❌ branch failed, skipping: {name}")
                failed.append(name)
        if empty:
            logger.warning(f"⚠️ {len(empty)}/{len(branches)} branches had no attributes to cluster: {empty}")
        if failed:
            logger.warning(f"⚠️ {len(failed)}/{len(branches)} branches failed: {failed}")
    else:
        _run_branch(config, profiles)

    save_path = save_as_json(
        responses=profiles,
        out_dir=config["out_dir"],
        timestamp=config["timestamp"],
    )
    logger.info(f"💾 clustering results: {len(profiles)} profiles -> {save_path}")


if __name__ == "__main__":
    main()
