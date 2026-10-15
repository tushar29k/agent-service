#!/usr/bin/env python3
"""Concurrency test: N parallel agent threads, throughput and failures.

Spins up a ThreadPoolExecutor of N workers on the offline mock backends
(react + native), one ReActAgent per worker (fresh checkpoint dir each, so
no shared mutable state), runs one eval task per worker round-robin from
evals/tasks.yaml, and reports tasks/sec, per-run latency p50/p99, and
failures (exceptions + eval misses).

The table is the deliverable: it prints to stdout and appends a dated row
per backend to evals/concurrency-results.md (committed).

    python3 evals/concurrency_test.py --n 16
    python3 evals/concurrency_test.py --n 8 --backends react
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

sys.path.insert(0, ".")
sys.path.insert(0, "evals")

import yaml
from agent import MockBackend, MockReActBackend, ReActAgent
from run_eval import run_task  # same pass/fail bar as the eval suite

RESULTS_MD = "evals/concurrency-results.md"


def percentile(xs, p):
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def make_backend(name):
    # fresh brain per worker — the mock is a stateless rules engine,
    # so the only thing shared across threads is immutable data
    return MockReActBackend() if name == "react" else MockBackend()


def worker(worker_id, backend_name, task, ckpt_base):
    # one agent per thread: checkpoints, guardrail logs and cost trackers
    # live under the worker's own dir, never shared
    agent = ReActAgent(
        backend=make_backend(backend_name),
        checkpoint_dir=os.path.join(ckpt_base, backend_name,
                                    f"worker-{worker_id}"))
    t0 = time.perf_counter()
    try:
        _ok_tools, _no_misuse, _ok_answer, ok, _used, final = run_task(
            agent, task)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"worker": worker_id, "task": task["id"], "error": None,
                "passed": bool(ok), "steps": final["steps"],
                "latency_ms": round(latency_ms, 1)}
    except Exception as e:  # a crashed thread is a failure, not a miss
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"worker": worker_id, "task": task["id"],
                "error": f"{type(e).__name__}: {e}", "passed": False,
                "steps": 0, "latency_ms": round(latency_ms, 1)}


def run_backend(backend_name, n, tasks, ckpt_base):
    # tasks round-robin: N workers each run exactly one task, so a high
    # N exercises thread-safety rather than just cycling the suite
    jobs = [(i, tasks[i % len(tasks)]) for i in range(n)]
    results = []
    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(worker, i, backend_name, task, ckpt_base): i
                for i, task in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    wall_s = time.perf_counter() - wall_t0
    lat = [r["latency_ms"] for r in results]
    passed = sum(1 for r in results if r["passed"])
    errors = sum(1 for r in results if r["error"] is not None)
    misses = sum(1 for r in results if r["error"] is None and not r["passed"])
    return {
        "backend": backend_name, "threads": n, "runs": n,
        "wall_s": round(wall_s, 2),
        "tasks_per_s": round(len(results) / wall_s, 2) if wall_s else 0,
        "p50_ms": round(percentile(lat, 50), 1),
        "p99_ms": round(percentile(lat, 99), 1),
        "passed": passed, "eval_miss": misses, "errors": errors,
        "worst_errors": [r for r in results if r["error"]][:3],
    }


HEADER = """# Concurrency test results

N parallel ReAct agent runs (one agent instance per thread, fresh
checkpoint dir each) on the offline mock backends, each run consuming
one eval task round-robin. `wall_s` is the pool window (first submit →
last finish); `tasks/s` = runs / wall_s. The mock brain answers
near-instantly, so these numbers measure agent-loop overhead and
thread-safety — not model latency. `eval_miss` = run finished but failed
the suite's pass bar; `errors` = worker threads that raised.

| date | backend | threads | runs | wall_s | tasks/s | p50_ms | p99_ms | passed | eval_miss | errors |
|---|---|---|---|---|---|---|---|---|---|---|
"""


def append_table(rows, path):
    today = date.today().isoformat()
    need_header = not os.path.exists(path)
    with open(path, "a") as f:
        if need_header:
            f.write(HEADER)
        for r in rows:
            f.write(
                f"| {today} | {r['backend']} | {r['threads']} | {r['runs']} "
                f"| {r['wall_s']} | {r['tasks_per_s']} | {r['p50_ms']} "
                f"| {r['p99_ms']} | {r['passed']} | {r['eval_miss']} "
                f"| {r['errors']} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8,
                    help="parallel threads per backend (default 8)")
    ap.add_argument("--backends", default="react,native",
                    help="comma-separated: react, native (default both)")
    ap.add_argument("--ckpt-dir", default="/tmp/agent_conc_ckpt",
                    help="base dir for per-worker checkpoints")
    ap.add_argument("--out", default=RESULTS_MD,
                    help="results table path (default %(default)s)")
    args = ap.parse_args()

    # mock-only run: no tracing or cost-log writes from N workers,
    # keeps the shared logs/ files clean and timing uncontaminated
    os.environ["AGENT_TRACING"] = "off"
    os.environ["AGENT_COST_LOG"] = ""

    tasks = yaml.safe_load(open("evals/tasks.yaml"))["tasks"]
    backends = [b.strip() for b in args.backends.split(",")]
    rows = [run_backend(b, args.n, tasks, args.ckpt_dir) for b in backends]
    append_table(rows, args.out)

    print(f"{'backend':8s} {'n':>3s} {'wall_s':>7s} {'tasks/s':>8s} "
          f"{'p50_ms':>8s} {'p99_ms':>8s} {'pass':>6s} {'miss':>4s} {'err':>3s}")
    for r in rows:
        print(f"{r['backend']:8s} {r['threads']:>3d} {r['wall_s']:>7.2f} "
              f"{r['tasks_per_s']:>8.2f} {r['p50_ms']:>8.1f} "
              f"{r['p99_ms']:>8.1f} {r['passed']:>3d}/{r['runs']:<3d} "
              f"{r['eval_miss']:>4d} {r['errors']:>3d}")
        for e in r["worst_errors"]:
            print(f"   worker {e['worker']} ({e['task']}): {e['error']}")
    print(f"\ntable appended to {args.out}")
    if any(r["errors"] or r["eval_miss"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
