import csv
from collections import defaultdict
from pathlib import Path

from loguru import logger


def filter_titles_by_buyers(
    purchases_csv: str,
    min_unique_buyers: int,
) -> list[dict]:
    """Aggregate purchases by Title and keep titles bought by >= min_unique_buyers.

    Returns list of dicts with columns: Title, freq, n_unique_buyers.
    Title-empty rows are skipped (cannot be a query).
    """
    title_buyers: dict[str, set[str]] = defaultdict(set)
    title_freq: dict[str, int] = defaultdict(int)

    with open(purchases_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "") or ""
            user = row.get("Survey ResponseID", "") or ""
            if not title or not user:
                continue
            title_buyers[title].add(user)
            title_freq[title] += 1

    qualifying: list[dict] = [
        {"Title": title, "freq": title_freq[title], "n_unique_buyers": len(buyers)}
        for title, buyers in title_buyers.items()
        if len(buyers) >= min_unique_buyers
    ]
    qualifying.sort(key=lambda r: r["n_unique_buyers"], reverse=True)

    logger.info(
        f"📊 unique titles total: {len(title_buyers):,} → kept: {len(qualifying):,} "
        f"(min_unique_buyers={min_unique_buyers})"
    )
    return qualifying


def save_titles_csv(records: list[dict], out_path: str) -> str:
    out_path_abs = str(Path(out_path).resolve())
    Path(out_path_abs).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_abs, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Title", "freq", "n_unique_buyers"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    logger.info(f"💾 saved {len(records):,} titles to {out_path_abs}")
    return out_path_abs
