# ADR 0001 — framework vs from-scratch: hand-rolled ReAct loop is the primary, LangGraph the variant

date: 2026-10-16
status: accepted

## context

The agent needed a think → act loop with tools, approval gates, streaming
events, and JSON checkpoints. LangGraph offers all of that as framework
primitives — but the project also needed deterministic mock brains that run
fully offline (evals must be green in CI with no keys) and an exact event
stream the demo UI consumes.

## decision

The from-scratch ReAct loop in `agent.py` is the primary implementation.
`langgraph_agent.py` exists as a real, working variant of the same
think → route → act loop on LangGraph, kept as the framework comparison and
the escape hatch for teams that prefer a supported framework.

Why from-scratch leads:

- The loop owns its event stream (`thought` / `approval_required` /
  `approval_denied` / `final` …) and its checkpoint format, so the UI and
  the `/approve` resume path have a stable contract no framework upgrade
  can shift.
- Mock brains are deterministic end to end — evals run 12/12 on both mock
  backends offline, which the approval-gate, cost-cap, and adversarial
  suites all depend on.
- The approval gate (ADR 0002) pauses mid-loop, persists state, and resumes
  across HTTP requests — hand-rolling keeps that pause/resume explicit
  instead of fighting a framework's checkpoint model.

Why the LangGraph variant stays: it proves the design ports cleanly,
it gets visualisation + checkpointing + streaming from the framework
instead of hand-rolling them, and it answers "what would this look like
in our stack?" for teams already on LangChain.

## consequences

- Two implementations to keep aligned — new tools go in `tools.py`
  (shared), not in either loop, so they can't drift.
- The graph variant lags the primary by design (it doesn't carry the
  latest loop features); README marks which is canonical.
