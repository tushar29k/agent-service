#!/usr/bin/env bash
# demo: the approval gate (approve + deny) and checkpoint recovery, fully
# scripted and offline. every answer is piped in — no human, no key, no net.
set -euo pipefail
cd "$(dirname "$0")"

# mock brain for determinism — the real model paths are keyed off env and
# must never be required by a demo
export AGENT_BACKEND=mock

hdr() { echo; echo "=== $1 ==="; }
fail() { echo "DEMO FAILED: $1" >&2; exit 1; }

APPR_LOG="logs/approvals.jsonl"
before=0
[ -f "$APPR_LOG" ] && before=$(wc -l < "$APPR_LOG")

hdr "act 1 — approve path: refund order 12345 through the gate"
out1=$(printf 'approve\nlegit refund, ticket #881\n' | python3 approval.py)
echo "$out1"
echo "$out1" | grep -q "approved — 'issue_refund'" || fail "act 1: tool was not approved"
echo "$out1" | grep -q "final -> Refund issued for order 12345" || fail "act 1: refund did not complete"

hdr "act 2 — deny path: empty input denies by default"
out2=$(printf '\n' | python3 approval.py)
echo "$out2"
echo "$out2" | grep -q "denied — 'issue_refund'" || fail "act 2: empty input did not deny"
echo "$out2" | grep -q "final -> Cancelled by human" || fail "act 2: run did not cancel"
echo "-> the deny path lands on the approval_denied run record (see act 4)"

hdr "act 3 — checkpoint recovery: the process 'dies' at the gate, a fresh one resumes"
rm -f checkpoints/demo-sh-resume.json
echo "-- process 1: start the refund run, pause at the gate, then die --"
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from agent import ReActAgent, make_backend
agent = ReActAgent(backend=make_backend())
for ev in agent.run("demo-sh-resume", "Issue a refund for order 777."):
    if ev["type"] == "approval_required":
        print(f"  paused at gate: {ev['action']['name']} {ev['action']['args']}")
        print("  checkpoint saved to disk — the process dies here")
        break
PY
echo "-- the checkpoint on disk (pending action persisted, not lost) --"
python3 -c "import json; d=json.load(open('checkpoints/demo-sh-resume.json')); print('  pending_action:', d['pending_action']['name'], d['pending_action']['args']); print('  steps so far:', d['steps'])"
echo "-- process 2: a brand-new agent instance with empty memory resumes the thread --"
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from agent import ReActAgent, make_backend
agent = ReActAgent(backend=make_backend())  # nothing in memory — all state from disk
events = agent.approve("demo-sh-resume", True, reason="recovered from checkpoint")
first = None
for ev in events:
    t = ev["type"]
    if first is None:
        first = t
    if t == "tool_result":
        print(f"  resumed mid-run, tool executed: {ev['result']}")
    elif t == "final":
        print(f"  final: {ev['answer']} (steps {ev['steps']})")
# if it had restarted from scratch it would have hit the gate again —
# the first event would be 'thought', then 'approval_required'
assert first == "tool_result", f"expected to resume mid-run, first event was {first}"
print("  -> resumed from the checkpoint: no rethink, no second gate")
PY

hdr "act 4 — the audit trail: approvals.jsonl grew by this run"
after=$(wc -l < "$APPR_LOG")
echo "  approvals.jsonl: $before -> $after entries"
python3 - "$APPR_LOG" <<'PY'
import json, sys
for line in open(sys.argv[1]).read().splitlines()[-2:]:
    r = json.loads(line)
    print(f"  {r['thread_id']} {r['tool']}: {r['decision']} (outcome={r['run_outcome']}, reason={r['reason']!r})")
PY
[ "$after" -gt "$before" ] || fail "act 4: approvals.jsonl did not grow"

echo
echo "demo complete — approval gate (approve + deny-by-default) and"
echo "checkpoint recovery (pause at gate, die, resume from disk) all green."
