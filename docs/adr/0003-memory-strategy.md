# ADR 0003 — memory strategy: working set + rolling summary, opt-in per agent

date: 2026-10-16
status: accepted

## context

Long agent threads eat the context window — and the cost cap with it. The
options were truncation (lose history, break multi-step tasks),
per-step summarisation (a model call per turn, expensive and lossy), or
retrieval over past steps (a second index to maintain).

## decision

`memory.py` does the middle path: keep the newest N steps verbatim (the
working set — that's what the next decision actually needs) and roll
everything older into one running summary. The summary grows
incrementally: each turn only folds in the steps that just aged out of
the working set, the rest carries over verbatim — so one turn costs one
small summarisation, not a full re-read.

- A token budget backstops it (`Memory(total_tokens=…)`); the demo keeps a
  30-step thread under 1500 tokens (3791 → 1479).
- The summariser is a deterministic stand-in with a `# SWAP:` marker for
  a real model call — offline evals stay honest about the mechanics.
- Memory is opt-in per agent (`ReActAgent(memory=Memory())`), default off:
  short threads shouldn't pay compaction cost, and the loop never reads
  anything memory didn't write — `memory.context()` only ever shrinks
  what the model sees.

## consequences

- The brain's prompt contract is unchanged; memory is a pre-filter on the
  transcript, not a new input channel the model must learn.
- Summaries are lossy by design — factual anchors (amounts, ids, decisions)
  are the summariser's job to keep, and the evals' multi-step tasks are the
  regression test for that.
- `python memory.py` demos the budget behaviour standalone, so the memory
  layer is testable without running an agent.
