#!/usr/bin/env python3
"""
Concatenate all monthly Parquet files per TLC dataset type into one file each.

Yellow, green, FHV, and HVFHV use different schemas; they are not row-joinable.
This script produces one combined Parquet per type under dataset_nyc_parquet/combined/
for easier cleaning within each fleet.

Cross-year schemas can differ; pandas aligns columns with NaNs for missing fields.
Very large histories may need more memory than one machine has; narrow years or use
DuckDB/Spark for out-of-core workflows.

Example:
  python scripts/combine_nyc_tlc_parquet.py --root dataset_nyc_parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

TYPES = ("yellow", "green", "fhv", "fhvhv")


def combine_type(root: Path, dataset: str, out_dir: Path) -> Path | None:
    folder = root / dataset
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(f"{dataset}_tripdata_*.parquet"))
    if not files:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_all.parquet"

    dfs = []
    it = files
    if tqdm is not None:
        it = tqdm(files, desc=f"{dataset} read")
    for fp in it:
        dfs.append(pd.read_parquet(fp, engine="pyarrow"))
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Combine monthly NYC TLC Parquet files per type.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("dataset_nyc_parquet"),
        help="Root folder with yellow/, green/, fhv/, fhvhv/ (default: dataset_nyc_parquet)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/combined)",
    )
    p.add_argument(
        "--types",
        nargs="+",
        choices=TYPES,
        default=list(TYPES),
        help="Which types to combine (default: all)",
    )
    args = p.parse_args()

    out_dir = args.out_dir or (args.root / "combined")
    produced: list[Path] = []

    for ds in args.types:
        path = combine_type(args.root, ds, out_dir)
        if path:
            produced.append(path)
            print(f"Wrote {path} ({path.stat().st_size // (1024 * 1024)} MiB)")
        else:
            print(f"No parquet files under {args.root / ds}", file=sys.stderr)

    if not produced:
        print("Nothing combined. Run download script first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
