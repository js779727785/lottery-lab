"""Serve the self-contained dashboard and game from web/."""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(PROJECT_ROOT / "web"))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving http://{args.bind}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
