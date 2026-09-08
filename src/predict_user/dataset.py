import csv
import json
from pathlib import Path

from loguru import logger


def _load_allowed_titles(allowed_titles_csv: str) -> set[str]:
    """Read the `Title` column from a prepare_titles CSV into a set."""
    allowed: set[str] = set()
    with open(allowed_titles_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "") or ""
            if title:
                allowed.add(title)
    logger.info(f"🎯 loaded {len(allowed):,} allowed titles from {allowed_titles_csv}")
    return allowed


def _collect_json_paths(path: str) -> list[str]:
    path_obj = Path(path)
    if path_obj.is_file():
        return [path]
    if path_obj.is_dir():
        return sorted(str(p) for p in path_obj.glob("*.json"))
    raise ValueError(f"Path is not a file or directory: {path}")


def _attach_items_field(transactions: list[dict]) -> None:
    """Populate transaction['items'] for ecommerce singletons / scan-only records.

    Each record becomes an items-list of length 1 so that downstream
    `_build_context` can emit the same JSON schema for both singletons and
    multi-item frequent-pattern records.
    """
    for t in transactions:
        rq = t.get("resolved_query", "")
        canonical = (t.get("transaction") or {}).get("prediction", {}).get("canonical_name", "")
        t["items"] = [{"product_name": rq, "canonical_name": canonical}]


def _load_frequent_pattern_records(patterns_path: str, name_to_canonical: dict[str, str]) -> list[dict]:
    """Load frequent-pattern combinations and synthesize transaction records.

    `frequent_patterns.json` contains a list of item combinations of size 2-4,
    where each item only carries `product_name`. We resolve each name to its
    canonical_name via the singleton-derived `name_to_canonical` map. If any
    name in a pattern cannot be resolved, the whole pattern is skipped.
    """
    with open(patterns_path) as f:
        patterns = json.load(f)

    records: list[dict] = []
    skipped = 0
    for pattern in patterns:
        product_names = [p["product_name"] for p in pattern]
        missing = [n for n in product_names if n not in name_to_canonical]
        if missing:
            logger.warning(f"🛒 pattern skipped, missing canonical_name for: {missing}")
            skipped += 1
            continue

        items = [{"product_name": n, "canonical_name": name_to_canonical[n]} for n in product_names]
        resolved_query = " + ".join(product_names)
        records.append(
            {
                "resolved_query": resolved_query,
                "items": items,
                "transaction": {
                    "prediction": {"canonical_name": resolved_query},
                    "from_pattern": True,
                },
            }
        )

    logger.info(
        f"🛒 loaded {len(records)} pattern records, skipped {skipped} (out of {len(patterns)}) from {patterns_path}"
    )
    return records


def _load_scan_only_records(
    scan_output_path: str,
    seen: set[str],
    allowed_titles: set[str] | None = None,
) -> list[dict]:
    """Load scan output and synthesize transaction-shaped records for every scan
    item not already present in `seen` (set of resolved_query already covered by
    the predict_transaction output).

    This deliberately ignores the scan flags: it picks up items dropped by the
    need_search / is_diagnostic gate AND items that entered search but were lost
    to search/recovery failures, so that every scanned product ends up with a
    user-attribute prediction (this matches the pipeline runs behind the paper).
    When `allowed_titles` is provided, only records whose resolved_query is in
    the set are kept.
    """
    json_paths = _collect_json_paths(scan_output_path)
    if not json_paths:
        logger.warning(f"🛒 scan_output_path has no .json files: {scan_output_path}")
        return []

    synthesized: list[dict] = []
    for jp in json_paths:
        with open(jp) as f:
            scan_records = json.load(f)
        scan_filtered_out = 0
        for r in scan_records:
            rq = r.get("resolved_query")
            if not isinstance(rq, str) or not rq.strip() or rq in seen:
                continue
            if allowed_titles is not None and rq not in allowed_titles:
                scan_filtered_out += 1
                continue
            seen.add(rq)
            synthesized.append(
                {
                    "resolved_query": rq,
                    "raw_query_mapping": r.get("raw_query_mapping"),
                    "freq_mapping": r.get("freq_mapping"),
                    "query_mapping": r.get("query_mapping"),
                    "direction": r.get("direction"),
                    "scan": r.get("scan"),
                    "search": {"search_result": []},
                    "transaction": {
                        "prediction": {
                            "canonical_name": rq,
                            "category_inferred": "",
                            "household_signals": [],
                            "use_case": [],
                        },
                        "from_scan_only": True,
                    },
                }
            )
        if allowed_titles is not None:
            logger.info(
                f"🛒 scanned {len(scan_records)} records from {jp} "
                f"(filtered out {scan_filtered_out} not in allowed_titles)"
            )
        else:
            logger.info(f"🛒 scanned {len(scan_records)} records from {jp}")
    return synthesized


def _filter_recovered(
    transactions: list[dict],
    allowed_titles: set[str] | None,
) -> tuple[list[dict], int, int]:
    """Keep transactions with a non-blank canonical_name (and allowed title).

    Returns (kept, num_recovery_failures, num_filtered_out_by_title).
    """
    kept: list[dict] = []
    num_recovery_failure = 0
    num_filtered_out = 0
    for transaction in transactions:
        attribute = transaction["transaction"].get("prediction", {})
        canonical_name = attribute.get("canonical_name") if attribute else None
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            num_recovery_failure += 1
            continue
        if allowed_titles is not None:
            rq = transaction.get("resolved_query")
            if not isinstance(rq, str) or rq not in allowed_titles:
                num_filtered_out += 1
                continue
        kept.append(transaction)
    return kept, num_recovery_failure, num_filtered_out


def load_recovered_transaction(
    path: str,
    is_reverse: bool = False,
    scan_output_path: str | None = None,
    frequent_patterns_path: str | None = None,
    allowed_titles_csv: str | None = None,
) -> list[dict]:
    json_paths = _collect_json_paths(path)
    if not json_paths:
        logger.warning(f"No .json files found in {path}")
        return []

    allowed_titles: set[str] | None = None
    if allowed_titles_csv:
        allowed_titles = _load_allowed_titles(allowed_titles_csv)

    all_transactions = []
    total_samples = 0
    total_failures = 0
    total_filtered_out = 0

    for json_path in json_paths:
        with open(str(json_path)) as f:
            transactions = json.load(f)

        filtered_transactions, num_recovery_failure, num_filtered_out = _filter_recovered(transactions, allowed_titles)

        all_transactions.extend(filtered_transactions)
        total_samples += len(transactions)
        total_failures += num_recovery_failure
        total_filtered_out += num_filtered_out

        logger.info(f"📚 loaded {len(transactions)} samples from {str(json_path)}")
        logger.info(f"Omitted {num_recovery_failure} samples due to recovery failure")
        if allowed_titles is not None:
            logger.info(f"🎯 Filtered out {num_filtered_out} samples not in allowed_titles")

    predict_transaction_count = len(all_transactions)
    scan_only_count = 0
    if scan_output_path:
        seen_queries = {rq for t in all_transactions if isinstance(rq := t.get("resolved_query"), str)}
        scan_only = _load_scan_only_records(scan_output_path, seen_queries, allowed_titles=allowed_titles)
        all_transactions.extend(scan_only)
        scan_only_count = len(scan_only)
        logger.info(
            f"🛒 appended {scan_only_count} scan-only records (no search/transaction context) from {scan_output_path}"
        )

    pattern_count = 0
    _attach_items_field(all_transactions)
    if frequent_patterns_path:
        name_to_canonical = {
            t["resolved_query"]: t["items"][0]["canonical_name"]
            for t in all_transactions
            if isinstance(t.get("resolved_query"), str) and t.get("items")
        }
        pattern_records = _load_frequent_pattern_records(frequent_patterns_path, name_to_canonical)
        all_transactions.extend(pattern_records)
        pattern_count = len(pattern_records)

    if is_reverse:
        all_transactions.reverse()

    logger.info(f"📊 Total: {total_samples} samples, {total_failures} failures")
    if allowed_titles is not None:
        logger.info(f"🎯 Filtered out {total_filtered_out} transaction samples not in allowed_titles")
    logger.info(
        f"📦 returning {len(all_transactions)} transactions "
        f"(predict_transaction={predict_transaction_count}, "
        f"scan_only={scan_only_count}, pattern={pattern_count})"
    )
    return all_transactions
