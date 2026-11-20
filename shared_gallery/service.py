"""Shared gallery HTTP service: receives track-update batches and answers with global person IDs.

    POST /track_updates   body = one batch (one line of track_updates.jsonl)  -> {"assignments": [...]}
    GET  /mapping         -> {"mapping": [(camera_id, person_id) -> global_person_id rows], "aliases": [merges],
                              "stats": {...}}
    POST /reset           -> clears the gallery (do this when a camera's pipeline restarts: person IDs restart at 1)
    GET  /health

Each batch is checked with the interface's own validator (integrate/person_pipeline/track_update_contract.py, the
same one the exporter uses; standard library only), so a batch that breaks the interface is rejected with 400 and
the reasons. Needs only shared_gallery/requirements.txt (no torch / YOLO / Re-ID).

    python service.py                     # http://127.0.0.1:8765, Postgres from docker-compose.yml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integrate"))

from gallery_store import DEFAULT_DSN, GalleryParams, SharedGalleryStore  # noqa: E402
from person_pipeline.track_update_contract import SITE_TIMEZONE, validate_batch  # noqa: E402  (stdlib only)


def make_handler(store: SharedGalleryStore, tz: ZoneInfo, verbose: bool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive

        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/mapping":
                self._reply(200, {"mapping": store.mapping(), "aliases": store.aliases(), "stats": store.stats()})
            elif self.path == "/health":
                self._reply(200, {"ok": True})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/reset":
                store.reset()
                self._reply(200, {"reset": True})
                return
            if self.path != "/track_updates":
                self._reply(404, {"error": "not found"})
                return
            try:
                batch = json.loads(body)
            except json.JSONDecodeError as e:
                self._reply(400, {"error": f"not JSON: {e}"})
                return
            problems = validate_batch(batch, tz) if isinstance(batch, dict) else ["batch is not an object"]
            if problems:
                self._reply(400, {"error": "invalid batch", "problems": problems[:20]})
                return
            t = time.perf_counter()
            res = store.submit(batch)
            res["gallery_ms"] = round((time.perf_counter() - t) * 1000, 1)
            if verbose:
                new = [f"{a['camera_id']}/P{a['person_id']}->G{a['global_person_id']}"
                       for a in res["assignments"] if a["status"] == "new"]
                print(f"[service] {batch['tick_ts']} {len(batch['records'])} records, {res['gallery_ms']} ms"
                      + (f", new {new}" if new else "") + (f", merges {res['merges']}" if res["merges"] else ""))
            self._reply(200, res)

        def log_message(self, *args):  # quiet: one line per batch is printed above when --verbose
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--timezone", default=SITE_TIMEZONE, help="time zone of the batch times")
    ap.add_argument("--merge-threshold", type=float, default=GalleryParams.merge_threshold,
                    help="cosine of the two banks' mean embeddings needed to merge two global persons")
    ap.add_argument("--min-embeddings", type=int, default=GalleryParams.min_embeddings,
                    help="embeddings each bank needs before it can be merged")
    ap.add_argument("--bank-size", type=int, default=GalleryParams.bank_size, help="per global person per camera")
    ap.add_argument("--verbose", action="store_true", help="print every batch")
    a = ap.parse_args()

    params = GalleryParams(a.merge_threshold, a.min_embeddings, a.bank_size, a.timezone)
    store = SharedGalleryStore(a.dsn, params)
    server = HTTPServer((a.host, a.port), make_handler(store, ZoneInfo(params.timezone), a.verbose))
    print(f"[service] http://{a.host}:{a.port}  db {a.dsn.rsplit('@', 1)[-1]}  {params}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
