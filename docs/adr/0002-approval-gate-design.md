# ADR 0002 — approval gates: destructive tools pause, persist, and wait for a human

date: 2026-10-16
status: accepted

## context

The agent can run destructive tools (the refund tool is the canonical
one). Letting the model fire them unattended is a non-starter, but a gate
that blocks the process or loses the run state would make the service
unusable over HTTP — the approval decision arrives on a later request.

## decision

- Tools carry a `destructive: bool` flag in `tools.py` (`issue_refund`
  sets it). The linter (tools.py `lint_tool_descriptions`) additionally
  requires destructive tools to name the danger in their description.
- When the loop reaches a destructive action it does not run it: it saves
  the checkpoint, stores `pending_action` in the thread state, and yields
  an `approval_required` event. `approve(thread_id, approved, reason=…)`
  resumes the same thread — the run survives process restarts between the
  gate and the decision.
- `approval.py` ships the CLI approver: prompts on stdin, empty input
  means **deny** (deny-by-default), and every decision plus the reason is
  appended to `logs/approvals.jsonl` — the audit trail for who allowed
  what, when.
- PII redaction runs on the tool args before the gate, so the approver
  only ever sees scrubbed values.
- Sub-agent delegation deliberately gives researchers a read-only tool
  subset, so destructive tools can't even reach a fan-out — no gates
  mid-delegation by construction, not by convention.

## consequences

- Destructive runs are two-request flows (`/run` → gate → `/approve`) —
  the UI and docs present them that way.
- Deny-by-default means a human who walks away cancels the action; the
  transcript records the denial with its reason.
- Denials are first-class run records (`approval_denied` in the cost log),
  so evals can assert the gate fired instead of the tool.
