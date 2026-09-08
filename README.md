# profiling agent for open-ecommerce

[![ICDM 2026](https://img.shields.io/badge/ICDM%202026-Applied%20Research-blue)](#citation)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Dataset: Open E-Commerce 1.0](https://img.shields.io/badge/Dataset-Open%20E--Commerce%201.0-orange)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY)

Official code for the paper **"From "Who Is This User?" to "What Does This Purchase Mean?": A Deployed Pipeline for Semantic User Profiling at Bank Scale"**, accepted at the **ICDM 2026** Applied Research track.

![Rethinking purchase history as recurring transaction patterns: resolve each product to a canonical name, profile it into demographic / psycho-behavioral / life-event attributes, cluster the attributes, and build a semantic-tag-oriented per-transaction pattern database](teaser/ICDM2026_teaser.png)

This repository reproduces a pipeline on the public [Open E-Commerce 1.0 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY) (Amazon purchase histories paired with survey responses), an end-to-end pipeline that infers user attributes from purchase histories with an LLM and evaluates their validity against self-reported survey answers. The deployed system described in the paper runs on confidential bank transaction data and thus cannot be released due to privacy concern.

## Release Notes

- **2026-09-09** Released the repository

## Pipeline

Each stage provides `src/<module>` and loads its settings from `configs/open_ecommerce/<module>/task_config.yaml`. Stages communicate only through file I/O and never import one another.

| # | Stage | Role |
|---|-------|------|
| 1 | `prepare_titles` | Extract and normalize product titles from purchase histories |
| 2 | `scan` | Gate that decides whether a title needs a web search |
| 3 | `search` | Enrich product information via web search (with caching) |
| 4 | `predict_transaction` | Normalize product names (canonical product recovery) |
| 5 | `predict_user` | Infer user attributes per transaction (demographic / psycho_behavioral / life_event); also takes in the pre-computed frequent product combinations (see below) and profiles each combination as one multi-item purchase |
| 6 | `cluster_attribute` | Embed and cluster the inferred attributes |
| 7 | `tag_cluster` | Assign LLM labels to clusters |
| 8 | `judge_user_attribute` | Evaluate the validity of the inferred attributes |

### How to integrate the results of frequent pattern mining

Frequent pattern mining (`frequent_pattern_mining`) runs outside this pipeline: it mines co-purchased title combinations (FP-Growth, CPU only) directly from the raw purchases, independently of stages 1-4. Its output [results/open_ecommerce/frequent_pattern_mining/frequent_patterns.json](results/open_ecommerce/frequent_pattern_mining/frequent_patterns.json) (541 combinations of 2-4 titles) is committed, and stage 5 only reads it through `frequent_patterns_path` in [configs/open_ecommerce/predict_user/task_config.yaml](configs/open_ecommerce/predict_user/task_config.yaml); set it to `""` to profile single products only. Re-mining is not needed to run the pipeline, but `bash scripts/open_ecommerce/local/frequent_pattern_mining/run.sh` reproduces the file from the downloaded dataset in about 15 seconds.

Two standalone evaluation modules are also provided: `judge_demographic_attribute` (demographic baseline) and `investigate_confidence_feasibility` (pseudo-confidence validation). Both analyze the `tag_cluster` output of a full-data run and are not part of the commands below; see their `task_config.yaml` under `configs/open_ecommerce/` for the inputs they expect.

## Setup

> [!IMPORTANT]
> **Supported OS: Ubuntu (Linux x86_64) only.** This repository was developed and verified on Ubuntu; macOS (and Windows) have not been tested and are not supported. The pinned [vLLM](https://github.com/vllm-project/vllm) has no macOS build, so `uv sync` and every command below, including the offline test, are expected to work only on Linux. If you want to try the pipeline on a Mac, a macOS port of vLLM such as [vllm-metal](https://github.com/vllm-project/vllm-metal) would be required, but such setups are outside the scope of this repository.

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # optional (see the note below)
```

No API key is required. The search stage uses [ddgs](https://github.com/deedy5/ddgs) (DuckDuckGo) by default. To reproduce the paper's results, set `SERPER_API_TOKEN` ([Serper](https://serper.dev)) in `.env`; otherwise ddgs provides the search functionality instead.

Running the pipeline with a real model requires a machine with an NVIDIA GPU (the offline test below runs without one). Use an environment with at least the following GPU for the model you want to run:

| Model | GPU | VRAM | # GPUs |
|---|---|---|---|
| Qwen3.5-27B (paper) | NVIDIA A100 | 80GB | x1 |
| Qwen3.5-4B (default) | NVIDIA A100 | 40GB | x1 |
| Qwen3.5-4B (default) | NVIDIA L4 | 24GB | x1 |

> [!IMPORTANT]
> Below shell command produces roughly **9GB** model files in `./models/` directory.
> You can change the model size from 4B to 27B which proposed in our paper, but model size increase up to **60GB**.
> If you just try whether the code works properly, see "Offline test (without GPUs)" described in the next section.

The pipeline layouts two models, loaded from a local directory (`model_name` in each stage's `task_config.yaml`): **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** for every generation stage by default and **[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** for embeddings. Download both into `./models/` with:

```bash
bash scripts/open_ecommerce/local/download_models.sh
```

The paper used **[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)** (~52GB, needs an 80GB GPU). To use it, or any other local model, download it and pass its path with `MODEL_NAME` to the run scripts below; the configs themselves need no edit:

```bash
uv run hf download Qwen/Qwen3.5-27B --local-dir ./models/Qwen3.5-27B
MODEL_NAME=./models/Qwen3.5-27B bash scripts/open_ecommerce/local/e2e_sample/run.sh
MODEL_NAME=./models/Qwen3.5-27B bash scripts/open_ecommerce/local/run_db.sh
```

## Quick start: 10-sample end-to-end test

The repository provides 10-purchase samples ([tests/e2e/fixtures/sample_purchases_10.csv](tests/e2e/fixtures/sample_purchases_10.csv)) that run 7 pipeline stages, from raw purchases through user-attribute inference to tagged attribute clusters (`prepare_titles` → `scan` → `search` → `predict_transaction` → `predict_user` → `cluster_attribute` → `tag_cluster`; the pre-computed frequent product combinations are not used for this tiny sample). Because the sample is tiny, `cluster_attribute` uses k-means with a fixed number of clusters per attribute group (demographic 2, psycho-behavioral 3, life-event 2) instead of the HDBSCAN setup of the full run.

### Offline test (without GPU)
Runs the real `main()` of the 7 stages chained through file I/O, with LLM responses, search results and attribute embeddings replaced by committed fixtures / deterministic stubs:

```bash
uv run python -m unittest tests.e2e.test_pipeline_e2e -v
```

The results are saved on [results/e2e_offline_test/](results/e2e_offline_test/) in a `<timestamp>/` folder (resolved configs, per-stage logs, the intermediate outputs of all 7 stages, and a per-test summary).

### Real-model run (with GPU)
Runs the same 10 samples through the actual local model (`./models/Qwen3.5-4B` by default; set `MODEL_NAME=<local model dir>` to use another one). The search stage uses ddgs unless `SERPER_API_TOKEN` is set in `.env`:

```bash
bash scripts/open_ecommerce/local/e2e_sample/run.sh
# MODEL_NAME=./models/Qwen3.5-27B bash scripts/open_ecommerce/local/e2e_sample/run.sh
```

The results are saved on [results/e2e_sample/](results/e2e_sample/) in a `<timestamp>/` folder (resolved configs, per-stage logs, and the outputs of all 7 stages); `outputs/user/` holds one record per product with the three inferred attribute groups, and `outputs/tag/` the same records with their clustered and LLM-tagged free-description attributes.

## Verified environments

All runs were made on **Ubuntu 24.04 LTS (Linux x86_64)**; no other OS has been tested (see the note in [Setup](#setup)). The 10-sample real-model run has been verified with the following LLM / GPU combinations (the recorded runs are under [results/e2e_sample/](results/e2e_sample/); Qwen3-Embedding-0.6B is used for embeddings in every case):

| Model | GPU | VRAM | # GPUs |
|---|---|---|---|
| Qwen3.5-27B | NVIDIA A100 | 80GB | x1 |
| Qwen3.5-4B | NVIDIA A100 | 80GB | x1 |
| Qwen3.5-4B | NVIDIA A100 | 40GB | x1 |
| Qwen3.5-4B | NVIDIA L4 | 24GB | x1 |

## Reproduction of Open e-commerce data

Download the Open E-Commerce 1.0 dataset into `data/public/open-ecommerce/`:

> [!IMPORTANT]
> Below script produces **300MB** dataset files in `./data/public/open-ecommerce/`.

```bash
uv run scripts/open_ecommerce/local/download_dataset.py
```

You can run all stages in order by following shell command:

```bash
bash scripts/open_ecommerce/local/run_db.sh
```

Every invocation is recorded in its own timestamped run directory `results/open_ecommerce/<timestamp>/`: `configs/` holds the task configs actually passed to each stage (the committed `task_config.yaml` files with their result / log paths redirected into the run directory), `outputs/<stage>/` the stage outputs (`scan/`, `search/`, `transaction/`, `user/v4/`, `clustering/v4/`, `tag/v4/`, `judge_user_attribute/`), `logs/<stage>/` the pipeline and console logs, and `summary.txt` the per-stage status and duration. The folder's [README](results/open_ecommerce/README.md) describes the layout, and [results/open_ecommerce/sample/](results/open_ecommerce/sample/) is a recorded example run on a 100-title slice (`NUM_SAMPLES=100 NUM_USERS=10`, with the frequent product combinations included in `predict_user`).

You can also run each stage individually by pointing out the stages like below.
It is usefull for debugging and per-module improving.

```bash
# stage can be selected invididually via 'STAGES' variable
STAGES="scan" bash scripts/open_ecommerce/local/run_db.sh

# If you want to run multiple stages, set following STAGES variable
STAGES="prepare_titles scan search predict_transaction" bash scripts/open_ecommerce/local/run_db.sh

# Later stages read the earlier stages' outputs from the run directory, so point RUN_DIR
# at an existing run to continue (or re-run) stages on it
RUN_DIR=./results/open_ecommerce/<timestamp> STAGES="predict_user cluster_attribute tag_cluster judge_user_attribute" bash scripts/open_ecommerce/local/run_db.sh
```

The two standalone evaluation modules (`judge_demographic_attribute`, `investigate_confidence_feasibility`) are run the same way, by naming them in `STAGES` together with the `RUN_DIR` whose `tag_cluster` output they should evaluate.

To try the full-data pipeline on a small slice first, set `NUM_SAMPLES`.
The `scan` stage then gates only the first N titles (written as `output-<timestamp>-0-N.json`) and the later stages shrink accordingly; without `NUM_SAMPLES` every stage processes all data:
Likewise, `NUM_USERS` limits the final `judge_user_attribute` stage to the first N evaluable users (all users without it):

```bash
NUM_SAMPLES=100 NUM_USERS=10 bash scripts/open_ecommerce/local/run_db.sh
```

`MODEL_NAME` switches the LLM of every generation stage (`scan`, `predict_transaction`, `predict_user`, `tag_cluster`, `judge_user_attribute`) to another local model directory; `cluster_attribute` keeps its embedding model. The variables can be combined:

```bash
MODEL_NAME=./models/Qwen3.5-27B NUM_SAMPLES=100 NUM_USERS=10 bash scripts/open_ecommerce/local/run_db.sh
```

## License

This repository is licensed under Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

Note that installing the GPU dependencies (`uv sync` on a CUDA machine) pulls in NVIDIA CUDA runtime libraries (`nvidia-cuda-*`, `nvidia-cublas`, `nvidia-cudnn-*`, ...), which are distributed under NVIDIA's own proprietary EULA, not Apache-2.0. They are downloaded from PyPI at install time and are not redistributed by this repository.

## Citation

```bibtex
@inproceedings{mitsuhashi2026who,
  title     = {From ``Who Is This User?'' to ``What Does This Purchase Mean?'': A Deployed Pipeline for Semantic User Profiling at Bank Scale},
  author    = {Mitsuhashi, Ryota and Morimura, Tetsuro and Ito, Hirotake},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```
