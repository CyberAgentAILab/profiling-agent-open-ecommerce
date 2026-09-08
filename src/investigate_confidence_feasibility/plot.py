"""Visualization for the pseudo_confidence feasibility investigation.

Per tag we overlay the positive-label and negative-label pseudo_confidence
distributions. They are drawn *density-normalized* (each histogram integrates
to 1), so the typically very different group sizes (n_pos << n_neg) do not
distort the visual comparison — the picture answers the same question as the
distribution-separation tests in model.py, just by eye.
"""

from typing import Any

import numpy as np
from loguru import logger

# Colorblind-safe (Wong) palette, consistent with the rest of the repo.
_POS_COLOR = "#D55E00"  # orange — survey positives
_NEG_COLOR = "#0072B2"  # blue   — survey negatives


def plot_confidence_distributions(
    pos_conf: list[float],
    neg_conf: list[float],
    attribute: str,
    auc: float,
    separation: dict[str, Any],
    out_path: str,
    bins: int = 20,
) -> str | None:
    """Overlay density-normalized pseudo_confidence histograms for the positive
    vs negative cohort of one tag, annotated with the separation stats.

    Returns the written path, or None if matplotlib is unavailable or there is
    nothing to plot.
    """
    if not pos_conf or not neg_conf:
        logger.warning(f"{attribute}: no positives or no negatives, skipping distribution plot")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import japanize_matplotlib  # noqa: F401  (registers JP fonts on import)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning(f"matplotlib unavailable, skipping distribution plot: {e}")
        return None

    pos = np.asarray(pos_conf, dtype=float)
    neg = np.asarray(neg_conf, dtype=float)

    # Shared bin edges over the full observed range so the two histograms align.
    lo = float(min(pos.min(), neg.min()))
    hi = float(max(pos.max(), neg.max()))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(neg, bins=edges, density=True, alpha=0.45, color=_NEG_COLOR, label=f"negative (n={neg.size})")
    ax.hist(pos, bins=edges, density=True, alpha=0.45, color=_POS_COLOR, label=f"positive (n={pos.size})")

    # Mean markers mirror calibration.mean_conf_positive / _negative.
    ax.axvline(neg.mean(), color=_NEG_COLOR, linestyle="--", linewidth=1.5, label=f"mean neg = {neg.mean():.3f}")
    ax.axvline(pos.mean(), color=_POS_COLOR, linestyle="--", linewidth=1.5, label=f"mean pos = {pos.mean():.3f}")

    mw = separation.get("mann_whitney_u", {})
    ks = separation.get("ks_2samp", {})
    stats_txt = (
        f"AUC = {auc:.3f}\n"
        f"Cliff's δ = {separation.get('cliffs_delta', float('nan')):.3f}\n"
        f"MWU p = {mw.get('p_value', float('nan')):.2e}\n"
        f"KS p = {ks.get('p_value', float('nan')):.2e}"
    )
    ax.text(
        0.02,
        0.97,
        stats_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    ax.set_title(f"pseudo_confidence distribution by survey label — {attribute}")
    ax.set_xlabel("pseudo_confidence")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"🖼️ wrote confidence distribution plot: {out_path}")
    return out_path
