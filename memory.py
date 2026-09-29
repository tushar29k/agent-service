"""Conversation summarisation memory for the ReAct agent.

Long threads eat the context window, so: keep the newest N steps raw
(the working set — that's what the next decision actually needs) and
roll everything older into one running summary instead of dropping it.
The summary grows incrementally: each turn only folds in the steps that
just aged out of the working set, the rest is carried over verbatim.

The summarizer here is a deterministic stand-in so the whole thing runs
with no API key. `# SWAP:` marks where a real model call goes.

Usage:
    mem = Memory(recent_steps=6, token_budget=4000)
    # inside the loop, after each step:
    mem.update(state["messages"])
    # and what the backend actually sees:
    compacted = mem.context(state["messages"])
"""


def estimate_tokens(text):
    """~4 chars per token — the usual back-of-envelope number."""
    return max(1, len(str(text)) // 4)


def total_tokens(messages):
    return sum(
        estimate_tokens(m.get("content") or m.get("action") or "")
        for m in messages
    )


def _render_steps(messages):
    """One tight line per action/observation pair — that's the unit of
    compression."""
    lines = []
    for m in messages:
        if m["role"] == "action":
            a = m["action"]
            lines.append(f"called {a['name']} with {a['args']}")
        elif m["role"] == "observation":
            lines.append(f"  -> observed: {str(m['content'])[:60]}")
    return lines


# SWAP: call llm_client.FreeLLMClient().generate() here with a prompt like
# "compress these agent steps into 2-3 sentences, keep facts and numbers"
def mock_summarize(prev_summary, new_step_lines):
    """Deterministic stand-in: keeps tool calls + observation heads so
    nothing a test depends on gets lost."""
    chunk = " | ".join(new_step_lines)
    if prev_summary:
        return f"{prev_summary} Also: {chunk}."
    return f"Earlier steps: {chunk}."


class Memory:
    def __init__(self, recent_steps=6, token_budget=4000, summarizer=None):
        # recent_steps: how many action/observation steps stay raw;
        # token_budget: hard cap on what context() returns
        self.recent_steps = recent_steps
        self.token_budget = token_budget
        self.summarize = summarizer or mock_summarize
        self.summary = None  # rolling summary of steps older than the working set
        self._folded = 0     # middle messages already folded into the summary

    def _middle(self, messages):
        # the task (message 0) is never summarized; the newest
        # recent_steps*2 messages are the raw working set; the rest is
        # summary territory
        keep = self.recent_steps * 2
        return messages[1:-keep] if len(messages) > 1 + keep else []

    def update(self, messages):
        """Fold steps that just aged out of the working set into the summary."""
        middle = self._middle(messages)
        new = middle[self._folded:]
        if new:
            self.summary = self.summarize(self.summary, _render_steps(new))
        self._folded = len(middle)

    def working_set(self, messages):
        keep = self.recent_steps * 2
        return [messages[0]] + (messages[-keep:] if len(messages) > 1 else [])

    def context(self, messages):
        """The transcript the backend sees: task + summary + working set,
        always under the token budget."""
        out = [messages[0]]
        if self.summary:
            out.append({"role": "assistant",
                        "content": "Summary of earlier steps: " + self.summary})
        out += self.working_set(messages)[1:]
        # backstop: drop oldest working-set messages (never task/summary)
        while total_tokens(out) > self.token_budget and len(out) > 2:
            out.pop(2)
        return out


if __name__ == "__main__":
    # the roadmap's done-when: a 30-step thread stays under the token budget
    messages = [{"role": "user",
                 "content": "Research topic X and summarize the findings."}]
    for i in range(30):
        messages.append({"role": "action", "action": {"name": "search_docs",
                         "args": {"query": f"topic X part {i}"}}})
        messages.append({"role": "observation", "content":
                         f"Document {i} about topic X. " + "filler detail " * 30})
    raw = total_tokens(messages)

    mem = Memory(recent_steps=5, token_budget=1500)
    mem.update(messages)
    compacted = mem.context(messages)
    kept = total_tokens(compacted)
    print(f"raw thread:  {len(messages)} messages, ~{raw} tokens")
    print(f"compacted:   {len(compacted)} messages, ~{kept} tokens "
          f"(summary: {'yes' if mem.summary else 'no'}, budget 1500)")
    assert kept <= mem.token_budget, "budget blown"
    assert compacted[0]["role"] == "user", "task anchor lost"
    assert any(m["role"] == "action" for m in compacted), "no raw steps left"

    # incremental: 6 more steps, summary absorbs the aged-out ones
    for i in range(30, 36):
        messages.append({"role": "action", "action": {"name": "search_docs",
                         "args": {"query": f"topic X part {i}"}}})
        messages.append({"role": "observation", "content":
                         f"Document {i} about topic X. " + "filler detail " * 30})
    mem.update(messages)
    compacted = mem.context(messages)
    kept = total_tokens(compacted)
    print(f"after +6:    {len(compacted)} messages, ~{kept} tokens "
          f"(still under 1500: {kept <= 1500})")
    assert kept <= mem.token_budget, "budget blown after incremental update"
    print("30-step thread fits under the token budget — done")
