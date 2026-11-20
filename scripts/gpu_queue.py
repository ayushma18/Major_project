"""A tiny persistent GPU job queue, so long experiment chains can be reordered while they run.

    python scripts/gpu_queue.py run              # process logs/queue.txt until it is empty
    python scripts/gpu_queue.py add "<command>"  # append a job
    python scripts/gpu_queue.py show

logs/queue.txt holds one shell command per line, run from the repo root in order. A job is
moved to logs/queue_done.txt with its exit code and duration when it finishes; edit
queue.txt at any time to reorder or remove waiting jobs. One job at a time: the GPU is the
bottleneck, so running two model processes at once only slows both.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "logs" / "queue.txt"
DONE = ROOT / "logs" / "queue_done.txt"


def pop() -> str | None:
    if not Q.exists():
        return None
    lines = [l for l in Q.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    job = lines[0]
    rest = Q.read_text().splitlines()
    rest.remove(job)
    Q.write_text("\n".join(rest) + ("\n" if rest else ""))
    return job


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "add":
        with open(Q, "a") as fh:
            fh.write(sys.argv[2] + "\n")
        return
    if cmd == "show":
        print(Q.read_text() if Q.exists() else "(empty)")
        return
    while True:
        job = pop()
        if job is None:
            print("QUEUE EMPTY", flush=True)
            time.sleep(30)
            continue
        t0 = time.time()
        print(f"=== START {time.strftime('%H:%M:%S')} {job}", flush=True)
        rc = subprocess.call(job, shell=True, cwd=ROOT)
        dt = time.time() - t0
        print(f"=== END rc={rc} {dt:.0f}s {job}", flush=True)
        with open(DONE, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\trc={rc}\t{dt:.0f}s\t{job}\n")


if __name__ == "__main__":
    main()
