"""CLI: collect 20-year data from 中彩网 and cross-check against 福彩官网."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .data import DEFAULT_DATA_DIR, collect_20y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    manifest = collect_20y(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        data_dir=Path(args.data_dir),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
