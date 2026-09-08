"""Download the Open E-Commerce 1.0 dataset from Harvard Dataverse.

Fetches the whole dataset as a single zip (original file formats, so
survey.csv arrives as CSV rather than Dataverse's ingested TSV) and extracts
it into data/public/open-ecommerce/, the layout every task_config.yaml
expects. The dataset is ~300MB; files already present are kept unless
--force is given.

Usage
-----
uv run scripts/open_ecommerce/local/download_dataset.py
uv run scripts/open_ecommerce/local/download_dataset.py --force
uv run scripts/open_ecommerce/local/download_dataset.py --out-dir /path/to/dir
"""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "data/public/open-ecommerce"

DATASET_DOI = "doi:10.7910/DVN/YGLYDY"
ZIP_URL = (
    "https://dataverse.harvard.edu/api/access/dataset/:persistentId/"
    f"?persistentId={DATASET_DOI}&format=original"
)

EXPECTED_FILES = [
    "amazon-purchases.csv",
    "survey.csv",
    "fields.csv",
    "survey-instrument.pdf",
    "prescreen-survey-instrument.pdf",
]


def download_zip(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "profiling-agent-open-ecommerce"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_mb = f"/{int(total) / 1e6:.0f}MB" if total else ""
        done = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r⬇️  {done / 1e6:.0f}MB{total_mb}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)


def extract_zip(zip_path: Path, out_dir: Path, force: bool) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name == "MANIFEST.TXT":
                continue
            target = out_dir / name
            if target.exists() and not force:
                print(f"⏭️  {name} already exists — skipping (use --force to overwrite)")
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"✔️  {name} ({target.stat().st_size / 1e6:.1f}MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in EXPECTED_FILES if not (out_dir / name).exists()]
    if not missing and not args.force:
        print(f"✔️  all dataset files already present in {out_dir} — nothing to do")
        return

    print(f"⬇️  downloading Open E-Commerce 1.0 ({DATASET_DOI}) to {out_dir}")
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".zip", delete=False) as tmp:
        tmp_zip = Path(tmp.name)
    try:
        download_zip(ZIP_URL, tmp_zip)
        extract_zip(tmp_zip, out_dir, force=args.force)
    finally:
        tmp_zip.unlink(missing_ok=True)

    still_missing = [name for name in EXPECTED_FILES if not (out_dir / name).exists()]
    if still_missing:
        raise SystemExit(f"❌ expected files missing after extraction: {still_missing}")
    print("✔️  dataset ready")


if __name__ == "__main__":
    main()
