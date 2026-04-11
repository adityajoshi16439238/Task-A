#!/usr/bin/env python3
"""
Download monthly NYC TLC trip Parquet files from CloudFront.

Source index: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Only yellow and green taxis have files for every month in the catalog; FHV and
HVFHV appear for subsets of months/years. This script probes each URL and skips
missing objects (CloudFront often returns 403 for absent keys).

Example:
  python scripts/download_nyc_tlc_parquet.py --start-year 2023 --end-year 2025
  python scripts/download_nyc_tlc_parquet.py --start-year 2009 --end-year 2026 --types yellow green
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

TYPE_PREFIX = {
    "yellow": "yellow",
    "green": "green",
    "fhv": "fhv",
    "fhvhv": "fhvhv",
}

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "ANN_example_1-nyc-tlc-downloader/1.0 (+local script)",
    }
)


def month_paths(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def url_for(dataset: str, ym: str) -> str:
    prefix = TYPE_PREFIX[dataset]
    return f"{BASE}/{prefix}_tripdata_{ym}.parquet"


def head_ok(url: str) -> tuple[bool, int | None]:
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=60)
    except requests.RequestException:
        return False, None
    if r.status_code == 200:
        cl = r.headers.get("Content-Length")
        return True, int(cl) if cl and cl.isdigit() else None
    return False, None


def download_file(url: str, dest: Path, expected_size: int | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return True
    try:
        with SESSION.get(url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
    except requests.RequestException:
        if dest.with_suffix(dest.suffix + ".part").exists():
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Download NYC TLC trip Parquet files.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("dataset_nyc_parquet"),
        help="Root folder for per-type subdirectories (default: dataset_nyc_parquet)",
    )
    p.add_argument("--start-year", type=int, default=2024)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument(
        "--types",
        nargs="+",
        choices=list(TYPE_PREFIX),
        default=list(TYPE_PREFIX),
        help="Which datasets to try (default: all four)",
    )
    p.set_defaults(skip_existing=True)
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-download even when a local file matches remote Content-Length",
    )
    p.add_argument(
        "-q",
        "--quiet-missing",
        action="store_true",
        help="Do not print a line for each unavailable month (no tqdm only)",
    )
    args = p.parse_args()

    if args.start_year > args.end_year:
        print("start-year must be <= end-year", file=sys.stderr)
        return 1

    tasks: list[tuple[str, str, Path, str]] = []
    for y in range(args.start_year, args.end_year + 1):
        for m in range(1, 13):
            ym = month_paths(y, m)
            for ds in args.types:
                u = url_for(ds, ym)
                dest = args.out / ds / f"{TYPE_PREFIX[ds]}_tripdata_{ym}.parquet"
                tasks.append((ds, ym, dest, u))

    iterator = tasks
    if tqdm is not None:
        iterator = tqdm(tasks, desc="Probing / downloading", unit="file")

    ok = skipped_missing = skipped_local = failed = 0
    for ds, ym, dest, u in iterator:
        exists, remote_size = head_ok(u)
        if not exists:
            skipped_missing += 1
            if tqdm is None and not args.quiet_missing:
                print(f"skip (unavailable) {ds} {ym}")
            continue
        if args.skip_existing and dest.exists() and remote_size is not None:
            if dest.stat().st_size == remote_size:
                skipped_local += 1
                ok += 1
                continue
        if tqdm is not None:
            iterator.set_postfix_str(f"{ds} {ym}")
        if download_file(u, dest, remote_size):
            ok += 1
        else:
            failed += 1
            print(f"FAILED {u}", file=sys.stderr)

    print(
        f"Done. downloaded_or_present={ok}, unavailable={skipped_missing}, "
        f"skipped_same_size={skipped_local}, failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
