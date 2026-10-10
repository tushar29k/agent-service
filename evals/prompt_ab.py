"""Prompt A/B: score two prompt versions on the eval set, declare a winner.

Versions live in prompts/ (v1 = the original, frozen; v2 = the tighter
discipline variant). Each version gets the full eval set through run_eval's
run_task scoring; the runner prints a table and declares a winner.

  --backend mock (default): MockReActBackend. The mock's rules never read
    the prompt text, so equal pass rates are expected — the A/B then scores
    what the mock CAN see: mean prompt size (chars at first think — real
    cost per model call) plus a format-contract lint of the prompt text.
    Pass rate first, cheaper prompt wins ties.
  --backend free: the true A/B — FreeBackend with each prompt version on
    a real free-tier model (needs LLM_API_KEY). Slower, real.

    python3 evals/prompt_ab.py
    python3 evals/prompt_ab.py --backend free    # true model A/B
"""
import argparse
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals")

import yaml

from agent import _react_prompt, FreeBackend, MockReActBackend, ReActAgent
from prompts import list_versions, load_prompt
from run_eval import run_task
from tools import TOOLS


def lint_prompt(tmpl):
    """Format-contract checks: every shipped prompt has to carry these,
    whatever its discipline variant says."""
    checks = [
        ("tool-list slot", "{tool_list}" in tmpl),
        ("transcript slot", "{transcript}" in tmpl),
        ("one-step constraint", "one step" in tmpl),
        ("action format", "Action:" in tmpl),
        ("answer format", "Answer:" in tmpl),
        ("approval-gate note", "approval gate" in tmpl),
    ]
    return checks, sum(1 for _, ok in checks if ok)


def mean_prompt_chars(version, tasks):
    # the prompt the brain sees on the first think of each task — the
    # instruction text is the only thing that varies by version
    total = sum(len(_react_prompt([{"role": "user",
                                    "content": t["question"]}],
                                  TOOLS, version=version))
                for t in tasks)
    return total / len(tasks)


def make_backend(kind, version):
    if kind == "free":
        if not os.environ.get("LLM_API_KEY"):
            print("no LLM_API_KEY — falling back to the mock backend "
                  "(set it and rerun for a true model A/B)")
            return MockReActBackend()
        return FreeBackend(prompt_version=version)
    return MockReActBackend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default="v1,v2",
                    help="comma-separated prompt versions to compare")
    ap.add_argument("--backend", default="mock", choices=["mock", "free"])
    args = ap.parse_args()
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    known = set(list_versions())
    for v in versions:
        if v not in known:
            sys.exit(f"unknown prompt version {v!r} — have "
                     f"{sorted(known)}")

    tasks = yaml.safe_load(open("evals/tasks.yaml"))["tasks"]
    rows = []
    for v in versions:
        backend = make_backend(args.backend, v)
        passed = 0
        for task in tasks:
            agent = ReActAgent(backend=backend,
                               checkpoint_dir=f"/tmp/agent_ab_ckpt_{v}")
            ok = run_task(agent, task)[3]
            passed += ok
        chars = mean_prompt_chars(v, tasks)
        _, lint = lint_prompt(load_prompt(v))
        rows.append({"version": v, "passes": passed, "total": len(tasks),
                     "chars": chars, "lint": lint})

    print(f"{'version':8s} {'pass':8s} {'avg prompt chars':16s} lint")
    for r in rows:
        print(f"{r['version']:8s} {r['passes']}/{r['total']:<7d} "
              f"{r['chars']:<16.0f} {r['lint']}/6")

    # winner: pass rate first; ties go to the cheaper prompt; a near-tie
    # on cost keeps the incumbent (the first version listed)
    ranked = sorted(rows, key=lambda r: (-r["passes"], r["chars"]))
    winner = ranked[0]
    incumbent = rows[0]
    for r in ranked[1:]:
        if (r["passes"] == winner["passes"] and winner["chars"] and
                abs(r["chars"] - winner["chars"]) / winner["chars"] < 0.02):
            winner = incumbent
            break

    if args.backend == "mock":
        print("\nnote: the mock backend is prompt-blind (deterministic rules), "
              "so equal pass rates are expected — scoring prompt efficiency "
              "and the format lint. Set LLM_API_KEY and use --backend free "
              "for a true model A/B.")
    w = winner
    others = [r for r in rows if r is not w]
    if len({r["passes"] for r in rows}) == 1 and others:
        o = others[0]
        pct = abs(o["chars"] - w["chars"]) / w["chars"] * 100
        cheaper = "fewer" if w["chars"] < o["chars"] else "more"
        reason = (f"tied on pass rate ({w['passes']}/{w['total']}), "
                  f"{pct:.0f}% {cheaper} prompt chars per think "
                  f"than {o['version']}")
    else:
        reason = f"highest pass rate ({w['passes']}/{w['total']})"
    print(f"\nWINNER: {w['version']} — {reason}")


if __name__ == "__main__":
    main()
