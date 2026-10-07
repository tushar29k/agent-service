"""Per-run cost cap: a kill-switch for runaway tasks.

max_steps stops long runs, but a run can still burn real money in few
steps (long contexts, big completions). CostTracker accumulates the
estimated USD cost of every model call in a run and the ReAct loop aborts
with a distinct `cost_exceeded` event once the budget is gone — stopped
by cost, not by steps.

Token source, best-effort to honest:
  - real backends (OpenAI / Anthropic) report resp.usage after each call,
    which we price with a small per-model table;
  - anything without usage data (mock, free-tier text API) falls back to
    a per-call estimate: chars/4 for the prompt, and the decision text
    length for the completion. Mock runs are pennies anyway, so the
    estimate only has to catch pathological runs.

Config: AGENT_MAX_COST (USD per run, default 0.10), or pass
max_cost=... to ReActAgent. AGENT_COST_IN_PER_1K /
AGENT_COST_OUT_PER_1K override the default per-1k-token prices for
unlisted models.

Per-run logging: log_run_record() appends one JSONL record per finished
run (task, backend, tokens in/out + how they were counted, cost, steps,
outcome) to logs/agent_cost.jsonl (AGENT_COST_LOG overrides the path).
the log file is runtime data — git-ignored, never committed — the
feature is the logging itself, and evals/cost_vs_quality.py builds the
cost-vs-quality table from it.

python3 cost.py runs the runaway demo: a looping mock brain is killed by
the budget while max_steps still has room.
"""
import json
import os
from datetime import datetime, timezone

# USD per 1M tokens — close enough for a budget fuse, not an invoice
_PRICES_PER_M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "gemini-2.0-flash": (0.10, 0.40),
}
_DEFAULT_MODEL = "gpt-4o-mini"


def _model_prices(model):
    """Price table lookup by model name; env override, then prefix match,
    then the default — a price is always returned."""
    env_in = os.environ.get("AGENT_COST_IN_PER_1K")
    env_out = os.environ.get("AGENT_COST_OUT_PER_1K")
    if env_in or env_out:
        return (float(env_in or 0.15) / 1000, float(env_out or 0.60) / 1000)
    name = (model or "").lower()
    for key, (pi, po) in _PRICES_PER_M.items():
        if key in name:
            return pi / 1e6, po / 1e6
    pi, po = _PRICES_PER_M[_DEFAULT_MODEL]
    return pi / 1e6, po / 1e6


class CostTracker:
    """USD meter for one run: record each think() call, ask spent/exceeded.

    Also accumulates prompt/completion token totals (real resp.usage when
    the backend reports it, chars/4 estimate otherwise) so a finished run
    can be logged as one record by log_run_record()."""

    def __init__(self, cap_usd=None):
        if cap_usd is None:
            cap_usd = float(os.environ.get("AGENT_MAX_COST", "0.10"))
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_calls = 0  # calls priced from real resp.usage
        self.estimate_calls = 0  # calls priced from the chars/4 fallback

    def reset(self):
        """New run, fresh budget."""
        self.spent_usd = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_calls = 0
        self.estimate_calls = 0

    @property
    def token_basis(self):
        """How the run's token totals were counted — honest about it."""
        if self.usage_calls and self.estimate_calls:
            return "mixed"
        if self.usage_calls:
            return "usage"
        if self.estimate_calls:
            return "estimate"
        return "none"

    def record_think(self, backend, messages, decision):
        """Price one model call: real usage when the backend reported it,
        character-based estimate otherwise (good enough for a fuse)."""
        usage = getattr(backend, "last_usage", None)
        if usage:
            backend.last_usage = None  # consumed — never double-count
            prompt_tok, completion_tok = usage
            self.usage_calls += 1
        else:
            # the estimate: prompts ~chars/4, completions from what the
            # brain actually decided (thought + action + answer text)
            prompt_tok = sum(len(str(m.get("content", "")))
                             for m in messages) // 4
            d = decision or {}
            completion_tok = max(
                len(str(d.get("thought", "")) +
                    json.dumps(d.get("action") or {}) +
                    str(d.get("answer") or "")) // 4, 1)
            self.estimate_calls += 1
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        price_in, price_out = _model_prices(getattr(backend, "model", None))
        self.spent_usd += prompt_tok * price_in + completion_tok * price_out
        self.calls += 1
        return self.spent_usd

    def exceeded(self):
        """The fuse: True once the budget is gone."""
        return self.spent_usd > self.cap_usd


def cost_log_path():
    """Where per-run records go: env override, else logs/ next to the repo.
    empty string disables logging entirely."""
    return os.environ.get("AGENT_COST_LOG", "logs/agent_cost.jsonl")


def log_run_record(task, backend, tracker, steps, outcome):
    """One JSONL record for a finished run. Called by ReActAgent._final —
    every termination path (answered, refused, max_steps, cost_exceeded,
    loop_detected, approval_denied) lands exactly one record."""
    path = cost_log_path()
    if not path:
        return None  # logging disabled via empty AGENT_COST_LOG
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task": str(task)[:200],  # the question, not the whole transcript
        "backend": type(backend).__name__,
        "model": getattr(backend, "model", None),
        "prompt_tokens": tracker.prompt_tokens,
        "completion_tokens": tracker.completion_tokens,
        "token_basis": tracker.token_basis,
        "cost_usd": round(tracker.spent_usd, 6),
        "steps": steps,
        "outcome": outcome,
    }
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


if __name__ == "__main__":
    # runaway demo: a brain that never answers, only acts. the loop
    # detector can't catch it (args differ each step) and max_steps is
    # generous — the budget has to be what stops it.
    import sys
    sys.path.insert(0, ".")
    from agent import MockBackend, ReActAgent

    class Runaway(MockBackend):
        def __init__(self):
            self.n = 0

        def think(self, messages, tools):
            self.n += 1
            return {"thought": f"keep going, step {self.n}",
                    "action": {"name": "calculator",
                               "args": {"expression": f"{self.n}+{self.n}"}},
                    "answer": None}

    agent = ReActAgent(backend=Runaway(), max_steps=50, max_cost=1e-9)
    saw = [ev for ev in agent.run("runaway-demo",
                                  "keep calculating forever")]
    killed = next(ev for ev in saw if ev["type"] == "cost_exceeded")
    final = next(ev for ev in reversed(saw) if ev["type"] == "final")
    print(f"killed by cost: {final['answer']}")
    print(f"steps used {final['steps']} of max_steps 50, "
          f"spent ${final['est_cost_usd']:.6f} of ${killed['cap_usd']:.9f}")
    assert final["steps"] < 50, "max_steps stopped it, not the budget"
    print("cost cap works — stopped by budget, not steps")
