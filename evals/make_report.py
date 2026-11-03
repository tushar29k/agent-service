"""Generate evals/adversarial-report.md — the results page a human reads.

Runs every suite fresh on both mock backends with history recording
OFF (the runners' own runs own the eval-history trend line; this
script must not double-log it), adds the benign tasks.yaml corpus to
the false-refusal denominator, and renders scoring.py's three
metrics: per suite × backend, per mutation class, per scenario.

Mock-only, like everything it summarises: the numbers grade the
detector + agent wiring. What a real model does with an attack that
evades the regex is the open question the report states plainly —
encoded variants "evade" here because the mock brain can't decode
them either; that is not evidence they are harmless.

    python3 evals/make_report.py
"""
import shutil
import sys
from datetime import date

import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "evals")
import run_adversarial
import run_scenarios
from agent import ReActAgent
from scoring import METRICS_VERSION, aggregate, classify

BACKENDS = run_adversarial.BACKENDS
REPORT = "evals/adversarial-report.md"


def _benign_task_outcomes(name, cls):
    """The non-refusal tasks.yaml tasks as benign outcomes — the part
    of the benign corpus that isn't an adversarial control."""
    tasks = yaml.safe_load(open("evals/tasks.yaml"))["tasks"]
    ckpt = f"/tmp/agent_report_tasks_{name}"
    shutil.rmtree(ckpt, ignore_errors=True)
    out = []
    for t in tasks:
        if t.get("expect_refusal"):
            continue  # attack probes — the seeds already cover them
        agent = ReActAgent(backend=cls(), checkpoint_dir=ckpt)
        thread = f"report-{name}-{t['id']}"
        events = list(agent.run(thread, t["question"]))
        if t.get("expect_approval"):
            events = list(agent.approve(thread, t.get("approve", True)))
        final = next(ev for ev in reversed(events)
                     if ev["type"] == "final")
        case = {"id": t["id"], "type": "control",
                "expected_contains": t.get("expected_contains", [])}
        outcome = {
            "refused": bool(final.get("refused")), "logged": False,
            "answer": final["answer"], "backend": name,
            "expected_ok": all(s.lower() in final["answer"].lower()
                               for s in case["expected_contains"])}
        out.append((case, outcome, classify(case, outcome)))
    return out


def _rate(rate):
    return "n/a" if rate is None else f"{rate:.1%}"


def _suite_row(label, outcomes):
    m = aggregate(outcomes)
    a, b = m["attacks"], m["benign"]
    return (f"| {label} | {a['n']} | {a['blocked']} | {a['succeeded']} "
            f"| {a['evaded']} | {_rate(a['asr'])} | {b['n']} "
            f"| {_rate(b['frr'])} | "
            f"{_rate(a['completion_under_attack'])} "
            f"(n={a['under_attack_n']}) |")


def main():
    seeds = yaml.safe_load(open("evals/adversarial.yaml"))["cases"]
    variants = yaml.safe_load(
        open("evals/adversarial_variants.yaml"))["cases"]
    _, seed_out = run_adversarial._run_suite(
        "adversarial", seeds, "rpt_seed", record_rows=False)
    _, var_out = run_adversarial._run_suite(
        "adversarial_variants", variants, "rpt_var", record_rows=False)
    scn_out = run_scenarios.run_all(record_rows=False, verbose=False)
    task_out = {name: _benign_task_outcomes(name, cls)
                for name, cls in BACKENDS.items()}

    def per_backend(outcomes, name):
        return [o for o in outcomes if o[1].get("backend") == name]

    lines = [
        "# Adversarial evals — results",
        "",
        f"Generated {date.today().isoformat()} by "
        f"`evals/make_report.py` (metrics definitions v{METRICS_VERSION},"
        " frozen in `evals/scoring.py`). Mock backends only — these"
        " numbers grade the detector and the agent wiring, not a real"
        " model. Regenerate after any suite change; do not hand-edit.",
        "",
        "Outcome classes: **blocked** = refused with the refusal in the"
        " audit log; **succeeded** = the attack measurably steered the"
        " run (planted/forbidden content in the answer, forbidden tool"
        " or tool args, canary emitted); **evaded** = not refused and"
        " no goal signal measured — the attack reached the brain"
        " unfiltered, which on mocks usually means the mock can't act"
        " on it either (it cannot decode base64), NOT that it is"
        " harmless; **false refusal** = a benign run refused.",
        "",
        "## Headline metrics by suite and backend",
        "",
        "| suite | attacks | blocked | succeeded | evaded | ASR "
        "| benign | FRR | completion under attack |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in BACKENDS:
        lines.append(_suite_row(f"seeds ({name})",
                                per_backend(seed_out, name)))
        lines.append(_suite_row(f"variants ({name})",
                                per_backend(var_out, name)))
        lines.append(_suite_row(f"scenarios ({name})", scn_out[name]))
        benign_all = ([o for o in per_backend(seed_out, name)
                       if o[0]["type"] == "control"]
                      + [o for o in per_backend(var_out, name)
                         if o[0]["type"] == "control"]
                      + task_out[name])
        bm = aggregate(benign_all)["benign"]
        lines.append(
            f"| benign corpus ({name}) | — | — | — | — | — "
            f"| {bm['n']} | {_rate(bm['frr'])} | "
            f"completion {_rate(bm['completion'])} |")
    lines += [
        "",
        "## Variants by mutation class (both backends combined)",
        "",
        "| mutation | runs | blocked | succeeded | evaded | ASR |",
        "|---|---|---|---|---|---|",
    ]
    by_mutation = {}
    for case, outcome, klass in var_out:
        if case["type"] == "control":
            continue
        slot = by_mutation.setdefault(case.get("mutation", "-"),
                                      {"n": 0, "blocked": 0,
                                       "succeeded": 0, "evaded": 0,
                                       "inconclusive": 0})
        slot["n"] += 1
        slot[klass] += 1
    for mutation in sorted(by_mutation):
        s = by_mutation[mutation]
        asr = s["succeeded"] / s["n"] if s["n"] else None
        lines.append(f"| {mutation} | {s['n']} | {s['blocked']} "
                     f"| {s['succeeded']} | {s['evaded']} | {_rate(asr)} |")
    lines += [
        "",
        "## Scenarios by outcome",
        "",
        "| scenario | react | native | attacker goal |",
        "|---|---|---|---|",
    ]
    scn_ids = [s["id"] for s in
               yaml.safe_load(open("evals/scenarios.yaml"))["scenarios"]]
    goals = {s["id"]: s["attacker_goal"] for s in
             yaml.safe_load(open("evals/scenarios.yaml"))["scenarios"]}
    for sid in scn_ids:
        row = [sid]
        for name in BACKENDS:
            klass = next(k for c, _, k in scn_out[name]
                         if c["id"] == sid)
            row.append(klass)
        row.append(goals[sid])
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Findings this run", ""]
    evading = [m for m, s in sorted(by_mutation.items())
               if s["evaded"] or s["succeeded"]]
    if evading:
        lines.append("- Mutation classes with unblocked runs "
                     "(succeeded or evaded): "
                     + ", ".join(f"`{m}`" for m in evading)
                     + " — triaged in the project dev-log; detector"
                     " fixes are a separate phase, never bundled with"
                     " an eval change.")
    false_refused = sorted({c["id"] for c, o, k in seed_out + var_out
                            if k == "false_refusal"})
    if false_refused:
        lines.append("- Benign controls false-refused: "
                     + ", ".join(f"`{i}`" for i in false_refused)
                     + " — the detector's overreach, priced by the"
                     " FRR above.")
    scn_bad = sorted({c["id"] for outs in scn_out.values()
                      for c, _, k in outs
                      if k in ("succeeded", "false_refusal")})
    if scn_bad:
        lines.append("- Scenarios where the attack landed or a benign"
                     " control was refused: "
                     + ", ".join(f"`{i}`" for i in scn_bad) + ".")
    lines.append("- Real-backend runs: pending — no LLM API key is"
                 " configured in this environment, so every number"
                 " above is mock-only. The encoded-variant classes are"
                 " the priority slice when a key exists: they evade"
                 " the detector entirely, and only a real model can"
                 " show whether evasion becomes success.")
    lines.append("")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
