"""Download a Hugging Face model repo into a plain local folder, fast and verified.

Why this exists: on some networks the HF client (Xet or single-stream CDN) stalls
at a few hundred KB/s. This script lists the repo, downloads small files directly,
and fetches every large file in parallel 32 MB byte ranges, then checks each large
file against its LFS sha256. Re-running resumes: files whose hash already matches
are skipped.

Usage:
    python scripts/fetch_model.py unsloth/Llama-3.1-8B-Instruct --dest ~/models
    python scripts/fetch_model.py cais/HarmBench-Llama-2-13b-cls --dest ~/models --threads 12

The model lands in <dest>/<org>--<name>/ and can be loaded with
`from_pretrained("<dest>/<org>--<name>")`. Set HF_TOKEN for gated repos.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "affect-as-attack-fetch/1.0"}
CHUNK = 32 << 20
DEFAULT_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.py"]


def _headers() -> dict:
    h = dict(UA)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def list_files(repo: str, revision: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=60) as r:
        return [x for x in json.load(r) if x.get("type") == "file"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # noqa: D401 - urllib hook
        return None


def resolve_cdn(repo: str, revision: str, path: str) -> str:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    try:
        urllib.request.build_opener(_NoRedirect).open(
            urllib.request.Request(url, method="HEAD", headers=_headers()), timeout=60
        )
        return url
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers["Location"]
            return loc if loc.startswith("http") else "https://huggingface.co" + loc
        raise


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_small(url: str, dest: Path) -> None:
    for attempt in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            return
        except Exception as exc:  # noqa: BLE001 - retry anything network-shaped
            print(f"  retry {attempt + 1} {dest.name}: {exc}", flush=True)
            time.sleep(2 + attempt)
    raise RuntimeError(f"failed to fetch {url}")


def fetch_large(url: str, dest: Path, size: int, threads: int) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    with open(part, "ab"):
        pass
    os.truncate(part, size)
    fd = os.open(part, os.O_RDWR)
    q: queue.Queue = queue.Queue()
    for s in range(0, size, CHUNK):
        q.put((s, min(size, s + CHUNK) - 1))
    total = q.qsize()
    done = [0]
    failed: list = []
    lock = threading.Lock()
    t0 = time.time()

    def worker() -> None:
        while True:
            try:
                s, e = q.get_nowait()
            except queue.Empty:
                return
            for attempt in range(40):
                try:
                    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={s}-{e}"})
                    with urllib.request.urlopen(req, timeout=90) as r:
                        b = r.read()
                    if len(b) != e - s + 1:
                        raise OSError(f"short read {len(b)} != {e - s + 1}")
                    os.pwrite(fd, b, s)
                    with lock:
                        done[0] += 1
                        if done[0] % 10 == 0 or done[0] == total:
                            mb = done[0] * CHUNK / 1e6
                            print(f"  {dest.name}: {done[0]}/{total} chunks, {mb / max(1e-6, time.time() - t0):.1f} MB/s", flush=True)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(2 + attempt)
            else:
                with lock:
                    failed.append((s, e))

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    os.close(fd)
    if failed:
        raise RuntimeError(f"{dest.name}: {len(failed)} chunks failed; re-run to resume")
    os.replace(part, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--dest", default=os.environ.get("AFFECT_MODELS_DIR", str(Path.home() / "models")))
    ap.add_argument("--revision", default="main")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--include", nargs="*", default=DEFAULT_PATTERNS)
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.dest)) / args.repo.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_files(args.repo, args.revision) if any(fnmatch.fnmatch(f["path"].split("/")[-1], p) for p in args.include)]
    print(f"{args.repo}: {len(files)} files -> {out}", flush=True)
    for f in files:
        dest = out / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        lfs = f.get("lfs")
        if lfs:
            if dest.exists() and dest.stat().st_size == lfs["size"] and sha256_of(dest) == lfs["oid"]:
                print(f"  ok (cached) {f['path']}", flush=True)
                continue
            url = resolve_cdn(args.repo, args.revision, f["path"])
            print(f"  fetching {f['path']} ({lfs['size'] / 1e9:.2f} GB)", flush=True)
            fetch_large(url, dest, lfs["size"], args.threads)
            got = sha256_of(dest)
            if got != lfs["oid"]:
                dest.unlink()
                raise RuntimeError(f"sha256 mismatch for {f['path']}: {got} != {lfs['oid']}")
            print(f"  verified {f['path']}", flush=True)
        else:
            fetch_small(f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{f['path']}", dest)
    (out / ".fetch_complete").write_text(json.dumps({"repo": args.repo, "revision": args.revision, "files": len(files), "at": time.time()}))
    print(f"DONE {args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
