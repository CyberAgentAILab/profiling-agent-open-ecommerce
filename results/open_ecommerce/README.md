# Full-data run outputs

`bash scripts/open_ecommerce/local/run_db.sh` records every invocation in its own run
directory `<timestamp>/` inside this folder (`RUN_DIR` selects it; the default is a new
`YYYY-MM-DD_HH-MM-SS` folder in Asia/Tokyo time). Run directories are not committed.
Two folders are committed:

- `frequent_pattern_mining/frequent_patterns.json` — pre-computed co-purchased title
  combinations (size 2-4), the `frequent_patterns_path` input of `predict_user`. It is
  produced outside the pipeline by `scripts/open_ecommerce/local/frequent_pattern_mining/run.sh`
  and lives at this fixed path, so the per-run config rewriting leaves it untouched.
- `sample/` — a recorded example run on a 100-title slice (`NUM_SAMPLES=100 NUM_USERS=10`,
  Qwen3.5-4B, with the 30 frequent combinations whose titles fall in the slice); see its
  `summary.txt`.

## Layout of a run directory

```
<timestamp>/
  summary.txt          command, GPU, LLM, per-stage status and duration
  configs/<stage>.yaml the task configs actually used (committed task_config.yaml with
                       ./results/open_ecommerce/... -> <run>/outputs/... and
                       ./logs/open_ecommerce/...    -> <run>/logs/...)
  logs/<stage>/        agent-<timestamp>.log (pipeline log) and console.log (stdout/stderr)
  outputs/<stage>/     output-<timestamp>.json (or output-<timestamp>-<begin>-<end>.json
                       for a NUM_SAMPLES slice)
```

| Stage | Output folder (under `outputs/`) | Content |
|---|---|---|
| prepare_titles | `../../data/public/open-ecommerce/titles_ge3_buyers.csv` (shared by all runs) | product titles with `freq` and `n_unique_buyers` |
| scan | `scan/` | per title: `need_search`, `is_diagnostic`, `diagnostic_for` |
| search | `search/` | scan records that passed the gate, with web search results |
| predict_transaction | `transaction/` | canonical product names recovered from the search results |
| predict_user | `user/v4/` | per product and per frequent combination: demographic / psycho_behavioral / life_event attributes |
| cluster_attribute | `clustering/v4/` | the user records plus `attribute2cluster_id` per attribute group |
| tag_cluster | `tag/v4/` | the clustered records plus `cluster_id2tag_*` and `tag2pseudo_confidence_*` |
| judge_user_attribute | `judge_user_attribute/` | per-user predictions vs. survey answers and the AUC metrics |
| judge_demographic_attribute (standalone, name it in `STAGES`) | `judge_demographic_attribute/` | demographic baseline metrics and confusion matrices |
| investigate_confidence_feasibility (standalone, name it in `STAGES`) | `investigate_confidence_feasibility/v1/` | pseudo-confidence validation and plots |
