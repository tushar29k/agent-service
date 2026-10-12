"""CLI approver for destructive tools.

The ReAct loop pauses before a destructive tool (approval_required
event) — this module is the human on the other end of that pause for
terminal runs: it prints the tool + args, reads approve/deny + a reason
from stdin, logs the decision, and resumes the loop through
ReActAgent.approve (whose approval_denied path closes the run record).

Run the refund demo through it:

    printf 'approve\nlooks legit\n' | python3 approval.py   # approve path
    printf 'deny\nno ticket yet\n'   | python3 approval.py   # deny path
    printf '\n'                      | python3 approval.py   # empty = deny
"""
import json
import os
import sys
from datetime import datetime, timezone


def approval_log_path():
    """Where approval decisions go: env override, else logs/ next to the
    repo. empty string disables logging entirely."""
    return os.environ.get("AGENT_APPROVAL_LOG", "logs/approvals.jsonl")


def log_approval_decision(thread_id, action, approved, reason, outcome=None):
    """One JSONL record per approval decision: tool, args, decision,
    reason, and the run's eventual outcome (answered / approval_denied /
    refused). The run log (logs/agent_cost.jsonl) carries the outcome too;
    this is the audit trail of the human's decision itself."""
    path = approval_log_path()
    if not path:
        return None  # logging disabled via empty AGENT_APPROVAL_LOG
    record = {"ts": datetime.now(timezone.utc).isoformat(),
              "thread_id": thread_id,
              "tool": action["name"],
              "args": action["args"],
              "decision": "approved" if approved else "denied",
              "reason": reason,
              "run_outcome": outcome}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


class CLIApprover:
    """Ask a human on stdin before a destructive tool runs.

    The decision line accepts 'approve' (or y/yes); anything else —
    including an empty line or EOF — denies (deny-by-default, so a
    script that runs dry never lets a destructive tool through). Then a
    free-text reason line for the audit log.
    """

    def __init__(self, stdin=sys.stdin, stdout=sys.stdout):
        self.stdin = stdin
        self.stdout = stdout

    def _prompt(self, text):
        self.stdout.write(text)
        self.stdout.flush()

    def decide(self, thread_id, action):
        """Print the pending tool, read the decision + reason, return
        (approved: bool, reason: str)."""
        args = json.dumps(action["args"], ensure_ascii=False)
        self._prompt(
            f"\napproval required — tool '{action['name']}' is destructive\n"
            f"  thread: {thread_id}\n"
            f"  args:   {args}\n"
            "approve / deny (empty = deny): ")
        decision = (self.stdin.readline() or "").strip().lower()
        approved = decision in ("approve", "y", "yes")
        self._prompt("reason: ")
        reason = (self.stdin.readline() or "").strip()
        self._prompt(f"{'approved' if approved else 'denied'} "
                     f"— '{action['name']}'\n")
        return approved, reason


def run_with_approver(agent, thread_id, message, approver=None):
    """Run a task, pausing at each destructive tool for the CLI approver.

    Generator — yields every event (stream them the way you'd stream
    run()), drives agent.approve() with the human's decision, and writes
    the approval decision record once the run lands on its final event.
    """
    approver = approver or CLIApprover()
    pending = []  # (action, approved, reason) — flushed to the log at final
    events = agent.run(thread_id, message)
    while True:
        for ev in events:
            yield ev
            if ev["type"] == "approval_required":
                approved, reason = approver.decide(thread_id, ev["action"])
                pending.append((ev["action"], approved, reason))
                events = agent.approve(thread_id, approved, reason=reason)
                break  # new event stream — keep driving it
            if ev["type"] == "final":
                answer = str(ev.get("answer", ""))
                outcome = ("refused" if ev.get("refused")
                           else "approval_denied"
                           if answer.startswith("Cancelled")
                           else "answered")
                for action, ok, why in pending:
                    log_approval_decision(thread_id, action, ok, why,
                                          outcome)
                return


if __name__ == "__main__":
    # the refund demo through the CLI approver — scripted above so the
    # approve and deny paths are both reproducible from a pipe
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agent import ReActAgent, make_backend

    agent = ReActAgent(backend=make_backend())
    print("--- task: refund through the CLI approver ---")
    for ev in run_with_approver(agent, "refund-cli-demo",
                                "Issue a refund for order 12345."):
        t = ev["type"]
        if t == "thought":
            print("thought ->", str(ev["thought"])[:100])
        elif t == "approval_required":
            print("approval_required ->", ev["action"]["name"],
                  ev["action"]["args"])
        elif t == "tool_result":
            print("tool_result ->", str(ev["result"])[:80])
        elif t == "final":
            print("final ->", ev["answer"],
                  f"(steps {ev['steps']}, cost ${ev['est_cost_usd']})")
