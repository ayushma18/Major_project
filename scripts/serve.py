"""Serve the project interface locally.

    python scripts/serve.py            # http://127.0.0.1:8765/web/
    python scripts/serve.py --port 9000

Serves the repo root read-only on localhost only, so the interface can read
literature/catalog.json, results/*.json, project/status.json, the local paper PDFs in
literature/pdfs/ and, if present, private/ (the proposal and slides, never committed).
Directory listings are off and outputs/ (raw generations) is never served.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKED_PREFIXES = ("/outputs", "/.git", "/.venv", "/data/activations", "/logs")


class Handler(http.server.SimpleHTTPRequestHandler):
    def list_directory(self, path):  # noqa: D401 - stdlib hook
        self.send_error(403, "Directory listing disabled")
        return None

    def do_GET(self):  # noqa: N802 - stdlib hook
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/web/")
            self.end_headers()
            return
        if any(self.path.startswith(p) for p in BLOCKED_PREFIXES):
            self.send_error(403, "Not served")
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    handler = partial(Handler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {ROOT} at http://127.0.0.1:{args.port}/web/  (Ctrl+C to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
