#!/usr/bin/env python3
"""Offline analysis that recomputes per-attribute AUC with 95% confidence intervals
(stratified bootstrap) from judge_user_attribute's saved prediction JSONs.

No LLM re-run is needed: only the per-user x per-attribute judgment scores
(``predicted_proba[attr]``) and ground truths (``ground_truth[attr]``) kept in each
prediction record are resampled. Method: per-attribute stratified bootstrap for the
CI, plus a cross-attribute Wilcoxon signed-rank test for method comparison.

Usage:
  .venv/bin/python scripts/judge_user_attribute_auc_ci.py
  .venv/bin/python scripts/judge_user_attribute_auc_ci.py baseline=path1.json hybrid=path2.json

By default the three methods baseline / tags_only / hybrid are compared.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

# Default comparison targets (label -> result JSON path). Override via argv as label=path.
RESULTS_DIR = Path("results/open_ecommerce/judge_user_attribute")
DEFAULT_FILES: dict[str, str] = {
    "baseline": str(RESULTS_DIR / "min3" / "output-2026-05-25_20-10-59-baseline-min3.json"),
    "tags_only": str(RESULTS_DIR / "output-2026-06-01_20-46-25-db_3.5_27B_top500_min3_adaptive_clustering.json"),
    "hybrid": str(RESULTS_DIR / "output-2026-06-03_13-57-16-db_3.5_27B_top500_min3_adaptive_clustering_hybrid.json"),
}

B = 1000  # bootstrap iterations (percentile CIs estimate the tails, so use >= 1000)
ALPHA = 0.05  # 95% CI
SEED = 0  # fixed for reproducibility


def stratified_bootstrap_auc(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = B, alpha: float = ALPHA, seed: int = SEED
) -> dict:
    """Resample positives and negatives separately (keeping the original counts) to build
    the AUC distribution; return the point estimate, 95% percentile CI, and SE.
    Structurally prevents zero-positive resamples.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),  # n+ positives drawn from positives
                rng.choice(neg, size=len(neg), replace=True),  # n- negatives drawn from negatives
            ]
        )
        aucs[b] = roc_auc_score(y_true[idx], y_score[idx])
    point = float(roc_auc_score(y_true, y_score))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc": point,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "se": float(aucs.std(ddof=1)),  # SD across bootstrap samples = SE estimate of the AUC; do not divide by sqrt(B)
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
    }


def attr_arrays(predictions: list[dict], attr: str) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild (y_true, y_score) for one attribute from the saved predictions.
    Users whose ground_truth is None are excluded (out of evaluation scope); missing
    scores become 0.0. Same convention as evaluate_predictions / _score_block.
    """
    y_true: list[int] = []
    y_score: list[float] = []
    for p in predictions:
        gt = (p.get("ground_truth") or {}).get(attr)
        if gt is None:
            continue
        proba = (p.get("predicted_proba") or {}).get(attr)
        y_true.append(int(bool(gt)))
        y_score.append(float(proba) if proba is not None else 0.0)
    return np.asarray(y_true), np.asarray(y_score, dtype=float)


def attribute_order(metrics: dict) -> list[str]:
    """Extract just the aggregation attribute keys from the insertion order of
    metrics (= the BINARY_TARGETS + LIFE_CHANGE_LABELS order)."""
    return [k for k in metrics.keys() if not k.startswith("__")]


def load_method(label: str, path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    preds = data["predictions"]
    attrs = attribute_order(data["metrics"])
    per_attr: dict[str, dict] = {}
    for a in attrs:
        yt, ys = attr_arrays(preds, a)
        if yt.size == 0 or len(set(yt.tolist())) < 2:
            # AUC is undefined for single-class attributes; record and skip.
            per_attr[a] = {"auc": None, "n": int(yt.size), "n_pos": int(yt.sum())}
            continue
        per_attr[a] = stratified_bootstrap_auc(yt, ys)
    aucs = [v["auc"] for v in per_attr.values() if v.get("auc") is not None]
    macro_auc = float(np.mean(aucs)) if aucs else None
    return {
        "label": label,
        "path": path,
        "mode": data.get("mode"),
        "n_predictions": len(preds),
        "attributes": attrs,
        "per_attr": per_attr,
        "macro_auc": macro_auc,
    }


def compare_methods(a: dict, b: dict) -> dict:
    """Per-attribute AUC differences (b - a) + Wilcoxon signed-rank test + macro-AUC diff.
    Only attributes with a computable AUC under both methods are compared."""
    common = [
        k
        for k in a["attributes"]
        if a["per_attr"].get(k, {}).get("auc") is not None and b["per_attr"].get(k, {}).get("auc") is not None
    ]
    diffs = [b["per_attr"][k]["auc"] - a["per_attr"][k]["auc"] for k in common]
    wins = sum(1 for d in diffs if d > 0)
    result = {
        "from": a["label"],
        "to": b["label"],
        "n_attrs": len(common),
        "per_attr_diff": dict(zip(common, [float(d) for d in diffs], strict=True)),
        "macro_auc_diff": (
            float(b["macro_auc"] - a["macro_auc"])
            if a["macro_auc"] is not None and b["macro_auc"] is not None
            else None
        ),
        "wins": wins,
        "losses": len(common) - wins,
    }
    if len(diffs) >= 1 and any(d != 0 for d in diffs):
        stat, p = wilcoxon(diffs)  # two-sided
        result["wilcoxon_stat"] = float(stat)
        result["wilcoxon_p"] = float(p)
    else:
        result["wilcoxon_stat"] = None
        result["wilcoxon_p"] = None
    return result


def parse_args(argv: list[str]) -> dict[str, str]:
    files = dict(DEFAULT_FILES)
    overrides = [a for a in argv if "=" in a]
    if overrides:
        files = {}
        for a in overrides:
            label, path = a.split("=", 1)
            files[label] = path
    return files


def print_method_table(m: dict) -> None:
    print(f"\n=== {m['label']}  (mode={m['mode']}, n_pred={m['n_predictions']}) ===")
    print(f"{'attribute':<18}{'n':>6}{'n_pos':>7}{'AUC':>8}{'95% CI':>20}{'SE':>8}")
    for a in m["attributes"]:
        v = m["per_attr"][a]
        if v.get("auc") is None:
            print(f"{a:<18}{v['n']:>6}{v['n_pos']:>7}{'  n/a (single class)':>28}")
            continue
        ci = f"[{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}]"
        print(f"{a:<18}{v['n']:>6}{v['n_pos']:>7}{v['auc']:>8.3f}{ci:>20}{v['se']:>8.3f}")
    if m["macro_auc"] is not None:
        print(f"{'macro_auc':<18}{'':>6}{'':>7}{m['macro_auc']:>8.3f}")


def print_comparison(c: dict) -> None:
    print(f"\n--- {c['to']} vs {c['from']}  (Δ = {c['to']} - {c['from']}) ---")
    print(f"  common attributes: {c['n_attrs']}   direction: {c['wins']} wins / {c['losses']} losses")
    if c["macro_auc_diff"] is not None:
        print(f"  macro-AUC diff: {c['macro_auc_diff']:+.4f}")
    if c["wilcoxon_p"] is not None:
        print(f"  Wilcoxon signed-rank test (two-sided): stat={c['wilcoxon_stat']:.1f}, p={c['wilcoxon_p']:.4f}")
    for a, d in c["per_attr_diff"].items():
        print(f"    {a:<18}{d:+.4f}")


def main() -> None:
    files = parse_args(sys.argv[1:])
    print(f"📊 stratified bootstrap AUC CI (B={B}, alpha={ALPHA}, seed={SEED})")

    methods: dict[str, dict] = {}
    for label, path in files.items():
        if not Path(path).exists():
            print(f"⚠️ skip {label}: file not found {path}")
            continue
        methods[label] = load_method(label, path)
        print_method_table(methods[label])

    # Compare consecutive method pairs + baseline vs the last method (when present)
    labels = list(methods.keys())
    comparisons: list[dict] = []
    pairs: list[tuple[str, str]] = []
    for i in range(len(labels) - 1):
        pairs.append((labels[i], labels[i + 1]))
    if len(labels) >= 3 and (labels[0], labels[-1]) not in pairs:
        pairs.append((labels[0], labels[-1]))
    for frm, to in pairs:
        c = compare_methods(methods[frm], methods[to])
        comparisons.append(c)
        print_comparison(c)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"auc_ci-{ts}.json"
    payload = {
        "config": {"B": B, "alpha": ALPHA, "seed": SEED, "files": files},
        "methods": methods,
        "comparisons": comparisons,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 saved → {out_path}")


if __name__ == "__main__":
    main()
