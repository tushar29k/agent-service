"""Slowest-step finder: read the trace log, find where the time goes.

  python3 slowest_step.py [n]

Reads AGENT_TRACE_LOG (default logs/traces.jsonl), prints the latest
run's per-step timings, the n slowest individual steps overall, and
average timings per step kind and per tool — the "where is it slow?"
answer in one glance.
"""
import json
import os
import sys
from collections import defaultdict

from tracing import trace_log_path


def load(path):
    steps, runs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # never let one bad line kill the report
            (steps if rec.get("type") == "step" else runs).append(rec)
    return steps, runs


def main():
    path = trace_log_path()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if not path or not os.path.exists(path):
        print(f"no traces yet at {path or '(logging disabled)'} — "
              f"run something first (python3 tracing.py seeds one)")
        return 1

    steps, runs = load(path)
    steps = [s for s in steps if s.get("duration_ms") is not None]
    if not steps:
        print(f"{path} has no step records yet")
        return 1

    latest = runs[-1]["run_id"] if runs else steps[-1]["run_id"]
    mine = [s for s in steps if s["run_id"] == latest]

    print(f"traces: {len(steps)} steps across {len({r['run_id'] for r in runs}) or 1} "
          f"runs ({path})")
    print(f"\nlatest run {latest} — per-step timings:")
    for s in mine:
        what = s["tool"] or "think"
        print(f"  step {s.get('step')}  {s['kind']:5} {what:12} "
              f"{s['duration_ms']:8.1f} ms")

    print(f"\nslowest {n} steps overall:")
    for s in sorted(steps, key=lambda r: r["duration_ms"],
                    reverse=True)[:n]:
        what = s["tool"] or "think"
        print(f"  {s['duration_ms']:8.1f} ms  {s['kind']:5} {what:12} "
              f"(run {s['run_id']}, step {s.get('step')})")

    # where the average millisecond goes: by kind, then by tool
    kind = defaultdict(list)
    tool = defaultdict(list)
    for s in steps:
        kind[s["kind"]].append(s["duration_ms"])
        tool[s["tool"] or s["kind"]].append(s["duration_ms"])
    print("\naverage per step kind:")
    for k, ds in sorted(kind.items(),
                        key=lambda kv: sum(kv[1]) / len(kv[1]),
                        reverse=True):
        print(f"  {k:5} {sum(ds) / len(ds):8.1f} ms avg over {len(ds)} steps")
    print("\naverage per tool:")
    for t, ds in sorted(tool.items(),
                        key=lambda kv: sum(kv[1]) / len(kv[1]),
                        reverse=True):
        print(f"  {t:12} {sum(ds) / len(ds):8.1f} ms avg over "
              f"{len(ds)} steps")

    slow = max(steps, key=lambda r: r["duration_ms"])
    print(f"\nbottleneck: {slow['tool'] or slow['kind']} "
          f"({slow['duration_ms']:.1f} ms, step {slow.get('step')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
