"""Cost vs quality across backends: runs the eval task set on every
backend, captures the per-run JSONL cost records, and writes a
cost-vs-quality markdown table to evals/cost-vs-quality.md.

Offline by design: the three backends are the mock-driven variants —
react (MockReActBackend, ReAct text path), native (MockBackend, native
function-calling path), free (MockFreeBackend, free-tier ReAct path).
Each is priced at its production counterpart's model rate, so the cost
numbers sit on the same scale as live runs. With real keys set, re-run
with the real backends for usage-based (non-estimate) numbers.

    python3 evals/cost_vs_quality.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "evals")
from agent import MockBackend, MockFreeBackend, MockReActBackend, ReActAgent
from run_eval import run_task


def read_records(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    tasks = yaml.safe_load(open("evals/tasks.yaml"))["tasks"]
    tmp = tempfile.mkdtemp(prefix="agent_cost_eval_")
    log_path = os.path.join(tmp, "agent_cost.jsonl")
    # point the per-run logger at a scratch file — the repo's
    # logs/agent_cost.jsonl stays out of eval traffic
    os.environ["AGENT_COST_LOG"] = log_path

    backends = {
        "react": MockReActBackend(),   # ReAct text prompting path
        "native": MockBackend(),       # native function-calling path
        "free": MockFreeBackend(),     # free-tier ReAct path
    }
    # per backend: task id -> (passed, cost record)
    results = {name: {} for name in backends}
    for name, backend in backends.items():
        seen = len(read_records(log_path))
        print(f"--- backend: {name} ({type(backend).__name__}) ---")
        for task in tasks:
            agent = ReActAgent(backend=backend,
                               checkpoint_dir=f"/tmp/agent_cost_eval_{name}")
            passed = run_task(agent, task)[3]
            rec = read_records(log_path)[seen]
            seen += 1
            assert rec["task"].startswith(task["question"][:40]), \
                f"cost record out of sync on {task['id']}"
            results[name][task["id"]] = (passed, rec)
            print(f"  {task['id']:28s} {'PASS' if passed else 'FAIL'} "
                  f"${rec['cost_usd']:.7f} "
                  f"({rec['prompt_tokens']}+{rec['completion_tokens']} tok, "
                  f"{rec['token_basis']})")
        n = len(tasks)
        npass = sum(1 for t in tasks if results[name][t["id"]][0])
        print(f"{name}: {npass}/{n} passed\n")

    # the table: one row per backend, cost per task included
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Cost vs quality — backends (mock-driven, offline)",
        "",
        f"Generated {now}. Each backend ran the {len(tasks)} eval tasks in "
        "evals/tasks.yaml; cost is per-run USD from the JSONL records, "
        "quality is the eval pass rate. Token counts are chars/4 estimates "
        "priced at each backend's production model rate (no API keys in "
        "the sandbox, so no real resp.usage).",
        "",
        "## Per backend",
        "",
        "| backend | class | model | tasks | pass rate | "
        "total cost (USD) | cost / task (USD) | tokens / task | token basis |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in backends:
        recs = [results[name][t["id"]][1] for t in tasks]
        passed = sum(1 for t in tasks if results[name][t["id"]][0])
        total = sum(r["cost_usd"] for r in recs)
        toks = sum(r["prompt_tokens"] + r["completion_tokens"] for r in recs)
        basis = {r["token_basis"] for r in recs}
        lines.append(
            f"| {name} | {recs[0]['backend']} | {recs[0]['model']} | "
            f"{len(tasks)} | {passed}/{len(tasks)} | "
            f"${total:.6f} | ${total / len(tasks):.7f} | "
            f"{toks / len(tasks):.0f} | {','.join(sorted(basis))} |")
    lines += [
        "",
        "## Per task (cost per run, USD)",
        "",
        "| task | react | native | free | pass (react / native / free) |",
        "|---|---|---|---|---|",
    ]
    for t in tasks:
        cols = []
        marks = []
        for name in backends:
            passed, rec = results[name][t["id"]]
            cols.append(f"${rec['cost_usd']:.7f}")
            marks.append("\u2713" if passed else "\u2717")
        lines.append(f"| {t['id']} | {' | '.join(cols)} | "
                     f"{' / '.join(marks)} |")
    lines += [
        "",
        "## Reading it",
        "",
        "- Same deterministic mock brain everywhere, so quality differences "
        "between backends come from the plumbing (prompt path vs native "
        "function calling), not model cleverness.",
        "- Cost differences come from each backend's model pricing: "
        "gpt-4o-mini ($0.15/$0.60 per 1M in/out) vs gemini-2.0-flash "
        "($0.10/$0.40). The free-tier path is ~a third cheaper per task "
        "at identical quality (mock-driven).",
    ]
    out = "evals/cost-vs-quality.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
