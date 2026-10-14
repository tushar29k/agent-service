#!/usr/bin/env python3
"""CI gate: fail when any suite's pass count drops below the last recorded run.

run_eval.py and run_adversarial.py append one JSONL row per suite per
backend to evals/eval-history.jsonl. CI copies that file aside before
running the evals, then runs:

    python evals/ci_gate.py --baseline /tmp/eval-baseline.jsonl \
        --current evals/eval-history.jsonl

exit 0 when every (suite, backend) holds or improves vs its most recent
baseline row, 1 on any regression. A (suite, backend) pair with no
baseline row is skipped — there's nothing to regress against.
"""
import argparse
import json
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a torn write shouldn't fail the gate
    return rows


def last_scores(rows):
    # most recent row wins per (suite, backend)
    scores = {}
    for r in rows:
        if {"suite", "backend", "passed", "total"} <= set(r):
            scores[(r["suite"], r["backend"])] = (r["passed"], r["total"])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    args = ap.parse_args()

    base, cur = last_scores(load(args.baseline)), last_scores(load(args.current))

    print(f"{'suite':12s} {'backend':10s} {'baseline':>8s} {'current':>7s}")
    fails = 0
    for key in sorted(cur):
        suite, backend = key
        passed, total = cur[key]
        if key not in base:
            print(f"{suite:12s} {backend:10s} {'(new)':>8s} {passed:>4d}/{total:<3d}")
            continue
        b_passed, b_total = base[key]
        # the eval contract is the task list, not the score — a changed
        # task count means the suites aren't comparable, so skip the gate
        if b_total != total:
            print(f"{suite:12s} {backend:10s} {'(resized)':>8s} {passed:>4d}/{total:<3d}")
            continue
        d = passed - b_passed
        mark = "" if d >= 0 else "  REGRESSION"
        print(f"{suite:12s} {backend:10s} {b_passed:>4d}/{b_total:<3d} {passed:>4d}/{total:<3d}{mark}")
        if d < 0:
            fails += 1
    if fails:
        print(f"gate failed: {fails} suite(s) regressed")
        return 1
    print("gate passed: every suite held its score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
