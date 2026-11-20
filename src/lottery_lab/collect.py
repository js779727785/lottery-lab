"""CLI: collect and validate official 双色球 history."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .data import DEFAULT_DATA_DIR, collect_and_validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01", help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=date.today().isoformat(), help="end date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--skip-crosscheck", action="store_true")
    args = parser.parse_args()
    manifest = collect_and_validate(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        data_dir=Path(args.data_dir),
        crosscheck=not args.skip_crosscheck,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
