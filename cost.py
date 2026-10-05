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

python3 cost.py runs the runaway demo: a looping mock brain is killed by
the budget while max_steps still has room.
"""
import json
import os

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
    """USD meter for one run: record each think() call, ask spent/exceeded."""

    def __init__(self, cap_usd=None):
        if cap_usd is None:
            cap_usd = float(os.environ.get("AGENT_MAX_COST", "0.10"))
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0

    def reset(self):
        """New run, fresh budget."""
        self.spent_usd = 0.0
        self.calls = 0

    def record_think(self, backend, messages, decision):
        """Price one model call: real usage when the backend reported it,
        character-based estimate otherwise (good enough for a fuse)."""
        usage = getattr(backend, "last_usage", None)
        if usage:
            backend.last_usage = None  # consumed — never double-count
            prompt_tok, completion_tok = usage
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
        price_in, price_out = _model_prices(getattr(backend, "model", None))
        self.spent_usd += prompt_tok * price_in + completion_tok * price_out
        self.calls += 1
        return self.spent_usd

    def exceeded(self):
        """The fuse: True once the budget is gone."""
        return self.spent_usd > self.cap_usd


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
