"""Print the latest strict and experimental candidate pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DEFAULT_WEB_DATA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", default=str(DEFAULT_WEB_DATA))
    args = parser.parse_args()
    data = json.loads(Path(args.dashboard).read_text(encoding="utf-8"))
    print(json.dumps(
        {
            "next_issue": data["meta"]["next_issue"],
            "strict": data["champion"]["latest_prediction"],
            "experimental": data["experimental"]["latest_prediction"],
            "note": data["disclaimer"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
