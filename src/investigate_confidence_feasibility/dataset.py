"""Data loading for the pseudo_confidence feasibility investigation.

Pipeline (per target tag):

  tag DB (product-level JSON)        amazon-purchases.csv          survey.csv
  ───────────────────────────       ──────────────────────        ──────────────
  tag → {title: max pseudo_conf}  →  user → representative_conf  ↔  user → label

The tag DB is large (~800MB) and pretty-printed as a JSON *array of product
objects*, so it is parsed with a string-aware streaming scanner rather than
`json.load`. The join key from a product back to a purchase row is
`raw_query_mapping` (verified to match `amazon-purchases.csv::Title` 1:1),
NOT `resolved_query` (which is canonicalized and only partially overlaps).
"""

import json
from collections import abc, defaultdict

import pandas as pd
from loguru import logger


class _ArrayScanner:
    """String-aware scanner for a JSON *array of objects*, fed one char at a
    time. Quote/escape state is tracked so braces inside string values (e.g.
    search-result snippets) never corrupt object boundaries. ``feed`` returns
    the source text of a completed top-level object, or None."""

    def __init__(self) -> None:
        self.depth = 0
        self.buf: list[str] = []
        self.in_str = False
        self.esc = False
        self.started = False

    def feed(self, ch: str) -> str | None:
        if self.started:
            self.buf.append(ch)
        if self.in_str:
            self._scan_in_string(ch)
            return None
        return self._scan_structural(ch)

    def _scan_in_string(self, ch: str) -> None:
        if self.esc:
            self.esc = False
        elif ch == "\\":
            self.esc = True
        elif ch == '"':
            self.in_str = False

    def _scan_structural(self, ch: str) -> str | None:
        if ch == '"':
            self._ensure_started(ch)
            self.in_str = True
        elif ch == "{":
            self._ensure_started(ch)
            self.depth += 1
        elif ch == "}":
            self.depth -= 1
            if self.depth == 0 and self.started:
                obj = "".join(self.buf)
                self.buf = []
                self.started = False
                return obj
        return None

    def _ensure_started(self, ch: str) -> None:
        if not self.started:
            self.started = True
            self.buf.append(ch)


def iter_product_objects(path: str, buf_size: int = 1 << 20) -> abc.Iterator[dict]:
    """Yield each top-level product object from a (large) JSON array file,
    keeping memory flat regardless of file size."""
    scanner = _ArrayScanner()
    with open(path) as f:
        f.read(1)  # skip the leading '['
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            for ch in chunk:
                obj = scanner.feed(ch)
                if obj is not None:
                    yield json.loads(obj)


def load_tag_confidences(
    tag_db_path: str,
    tag_to_category: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Scan the tag DB once and collect, for each target tag, the per-title
    pseudo_confidence.

    For a product object the tag-level confidence lives in
    ``tag2pseudo_confidence_{category}[tag]``. A product maps to one or more
    raw purchase titles via ``raw_query_mapping``; each such title inherits
    that product's confidence. When the same title appears across multiple
    products carrying the tag, the maximum confidence is kept.

    Returns ``{tag: {title: max_pseudo_confidence}}``.
    """
    tag_title_conf: dict[str, dict[str, float]] = {t: defaultdict(float) for t in tag_to_category}
    n_obj = 0
    for obj in iter_product_objects(tag_db_path):
        n_obj += 1
        raw_titles = obj.get("raw_query_mapping") or []
        if not raw_titles:
            continue
        for tag, category in tag_to_category.items():
            conf = obj.get(f"tag2pseudo_confidence_{category}", {}).get(tag)
            if conf is None:
                continue
            conf = float(conf)
            slot = tag_title_conf[tag]
            for title in raw_titles:
                if conf > slot[title]:
                    slot[title] = conf
    logger.info(f"🏷️ scanned {n_obj} product objects from tag DB")
    for tag, slot in tag_title_conf.items():
        logger.info(f"   tag={tag!r}: {len(slot)} titles carry it")
    return {t: dict(v) for t, v in tag_title_conf.items()}


def build_user_tag_confidence(
    purchases_csv: str,
    tag_title_conf: dict[str, dict[str, float]],
    agg: str = "max",
) -> dict[str, dict[str, float]]:
    """For each tag, aggregate per-title confidences into one representative
    value per user over the user's purchased titles that carry the tag.

    ``agg='max'`` (default, per the spec): a user's representative confidence
    is the highest pseudo_confidence among the tag-carrying products they
    bought. ``agg='mean'`` averages them instead. Users who bought no
    tag-carrying title are absent from the returned mapping for that tag.

    Returns ``{tag: {user_id: representative_confidence}}``.
    """
    df = pd.read_csv(purchases_csv, usecols=["Title", "Survey ResponseID"])
    df = df.dropna(subset=["Survey ResponseID", "Title"])

    # Accumulate per (tag, user): max -> running max; mean -> (sum, count).
    # Keys are normalized to str so they always match the survey's string index
    # (consistent with the other modules' str(user_id) keying), even when pandas
    # infers numeric-looking IDs as int/float.
    acc: dict[str, dict[str, list[float]]] = {t: defaultdict(lambda: [0.0, 0.0]) for t in tag_title_conf}
    for user, title in zip(df["Survey ResponseID"].to_numpy(), df["Title"].to_numpy(), strict=True):
        for tag, title_conf in tag_title_conf.items():
            conf = title_conf.get(title)
            if conf is None:
                continue
            cell = acc[tag][str(user)]
            if agg == "mean":
                cell[0] += conf
                cell[1] += 1.0
            else:  # max
                if conf > cell[0]:
                    cell[0] = conf

    out: dict[str, dict[str, float]] = {}
    for tag, users in acc.items():
        if agg == "mean":
            out[tag] = {u: (s / c if c else 0.0) for u, (s, c) in users.items()}
        else:
            out[tag] = {u: v[0] for u, v in users.items()}
        logger.info(f"👥 tag={tag!r}: {len(out[tag])} users hold the tag (agg={agg})")
    return out


def load_survey(survey_csv: str) -> pd.DataFrame:
    df = pd.read_csv(survey_csv)
    logger.info(f"📋 loaded survey: {len(df)} respondents")
    return df


def label_for(value: object, match: str, positive_value: str) -> int | None:
    """Map one survey cell to a binary label.

    - ``match='yes_no'``  : 'Yes' -> 1, 'No' -> 0, anything else
      (e.g. 'Prefer not to say', NaN) -> None (excluded).
    - ``match='substring'``: case-insensitive substring test of
      ``positive_value`` (e.g. 'Became pregnant' within the multi-select
      Q-life-changes free text). Absence is a definite 0, never None.
    """
    if match == "substring":
        text = value if isinstance(value, str) else ""
        return 1 if positive_value.lower() in text.lower() else 0
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == positive_value.strip().lower():
        return 1
    if v == "no":
        return 0
    return None
