"""ReAct agent, production-shaped: explicit state, think/act nodes, a router,
loop detection, JSON checkpointing, streaming events, and human-in-the-loop
approval gates for destructive tools.

The "model" is a swappable backend:
  MockBackend   — deterministic rules for demos/tests (NOT intelligent)
  OpenAIBackend — real LLM calls (needs OPENAI_API_KEY; same interface)

# SWAP (production): pass OpenAIBackend() instead of MockBackend().
"""
import json
import os
import re

from tools import TOOLS, ToolNode


# ---------------------------------------------------------------- backends
class ModelBackend:
    def think(self, messages, tools):
        """Return {'thought': str, 'action': {'name', 'args'} | None,
        'answer': str | None}."""
        raise NotImplementedError


class MockBackend(ModelBackend):
    """Deterministic stand-in so the whole loop runs without an API key.

    It classifies intent with rules — enough for the bundled eval tasks.
    A real backend (below) lets the LLM do this thinking.
    """

    def think(self, messages, tools):
        question = next(m["content"] for m in messages
                        if m["role"] == "user")
        q = question.lower()
        obs = [m for m in messages if m["role"] == "observation"]
        acts = [m["action"]["name"] for m in messages if m["role"] == "action"]
        last_obs = obs[-1]["content"] if obs else None

        if not obs:
            if ("issue" in q or "process" in q) and "refund" in q:
                m = re.search(r"order\s*#?(\w+)", q)
                oid = m.group(1) if m else "unknown"
                return {"thought": "User wants a refund issued — destructive, "
                                   "will need approval.",
                        "action": {"name": "issue_refund",
                                   "args": {"order_id": oid}}, "answer": None}
            if "month" in q and "year" in q:
                return {"thought": "Need the warranty period first, then convert.",
                        "action": {"name": "search_docs",
                                   "args": {"query": "warranty"}}, "answer": None}
            if re.search(r"\d+\s*[+\-*/]\s*\d+", q):
                expr = re.search(r"[\d\s+\-*/().]+", q).group(0).strip()
                return {"thought": "Arithmetic in the question — calculate.",
                        "action": {"name": "calculator",
                                   "args": {"expression": expr}}, "answer": None}
            return {"thought": "Factual question — search the knowledge base.",
                    "action": {"name": "search_docs", "args": {"query": question}},
                    "answer": None}

        if acts and acts[-1] == "search_docs":
            if last_obs == "NO_RESULTS":
                return {"thought": "Nothing relevant found — say so honestly.",
                        "action": None,
                        "answer": "I couldn't find anything about that in the "
                                  "knowledge base."}
            if "month" in q and "year" in last_obs.lower():
                m = re.search(r"(\d+)\s*year", last_obs.lower())
                n = m.group(1) if m else "2"
                return {"thought": f"Found {n} years — convert to months.",
                        "action": {"name": "calculator",
                                   "args": {"expression": f"{n}*12"}},
                        "answer": None}
            return {"thought": "Have the fact — answer from the observation.",
                    "action": None,
                    "answer": f"Based on the knowledge base: {last_obs}"}

        if acts and acts[-1] == "calculator":
            return {"thought": "Calculation done.",
                    "action": None, "answer": f"The result is {last_obs}."}

        if acts and acts[-1] == "issue_refund":
            return {"thought": "Refund executed — report the outcome.",
                    "action": None, "answer": last_obs}

        return {"thought": "No further action needed.", "action": None,
                "answer": "Done."}


class OpenAIBackend(ModelBackend):
    """Real backend. pip install openai; export OPENAI_API_KEY."""

    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def think(self, messages, tools):
        sys = ("You are a ReAct agent. Reply with ONLY a JSON object: "
               '{"thought": "...", "action": {"name": "...", "args": {...}} } '
               'or {"thought": "...", "answer": "..."}. '
               "Tools: " + ", ".join(f"{t.name}: {t.description}" for t in tools))
        flat = "\n".join(f"{m['role']}: {m.get('content', m.get('action'))}"
                         for m in messages)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": flat}],
            response_format={"type": "json_object"},
            temperature=0)
        return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------- agent
class ReActAgent:
    def __init__(self, backend=None, tools=TOOLS, max_steps=10,
                 checkpoint_dir="checkpoints"):
        self.backend = backend or MockBackend()
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.destructive = {t.name for t in tools if t.destructive}
        self.max_steps = max_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    # -- persistence: conversations are just thread_ids ------------------
    def _path(self, thread_id):
        safe = re.sub(r"[^\w-]", "_", thread_id)
        return os.path.join(self.checkpoint_dir, f"{safe}.json")

    def _save(self, thread_id, state):
        json.dump(state, open(self._path(thread_id), "w"))

    def _load(self, thread_id):
        p = self._path(thread_id)
        return json.load(open(p)) if os.path.exists(p) else None

    # -- the loop: think -> route -> act -> observe ----------------------
    def _loop(self, thread_id, state):
        while True:
            state["steps"] += 1
            if state["steps"] > self.max_steps:
                yield self._final(state, "Stopped: max steps exceeded.")
                return
            d = self.backend.think(state["messages"], self.tools)
            yield {"type": "thought", "thought": d["thought"]}
            if d.get("answer"):
                state["messages"].append({"role": "assistant",
                                          "content": d["answer"]})
                self._save(thread_id, state)
                yield self._final(state, d["answer"])
                return
            action = d["action"]
            # loop detection: same action 3x in a row -> stuck
            recent = [m["action"] for m in state["messages"]
                      if m["role"] == "action"][-2:]
            if len(recent) == 2 and all(r == action for r in recent + [action]):
                yield self._final(state, "Stopped: repeating the same action "
                                         "(loop detected).")
                return
            # approval gate: destructive tools pause for a human
            if action["name"] in self.destructive:
                state["pending_action"] = action
                self._save(thread_id, state)
                yield {"type": "approval_required", "action": action,
                       "reason": f"'{action['name']}' is destructive"}
                return
            result = self.tool_node.run(action["name"], action["args"])
            state["messages"].append({"role": "action", "action": action})
            state["messages"].append({"role": "observation", "content": result})
            self._save(thread_id, state)
            yield {"type": "tool_result", "tool": action["name"],
                   "args": action["args"], "result": result}

    def _final(self, state, answer):
        est_tokens = sum(len(str(m.get("content", ""))) for m in
                         state["messages"]) // 4
        return {"type": "final", "answer": answer, "steps": state["steps"],
                "est_tokens": est_tokens}

    def run(self, thread_id, message):
        """Start a task. Yields events (stream these to the UI)."""
        state = {"messages": [{"role": "user", "content": message}],
                 "steps": 0}
        yield from self._loop(thread_id, state)

    def approve(self, thread_id, approved):
        """Resume after an approval gate. approved=True executes the action."""
        state = self._load(thread_id)
        action = state.pop("pending_action", None)
        if action is None:
            yield self._final(state, "Nothing awaiting approval.")
            return
        if not approved:
            state["messages"].append(
                {"role": "assistant",
                 "content": "Cancelled: human did not approve."})
            self._save(thread_id, state)
            yield self._final(state, "Cancelled by human.")
            return
        result = self.tool_node.run(action["name"], action["args"])
        state["messages"].append({"role": "action", "action": action})
        state["messages"].append({"role": "observation", "content": result})
        yield {"type": "tool_result", "tool": action["name"],
               "args": action["args"], "result": result}
        yield from self._loop(thread_id, state)


if __name__ == "__main__":
    agent = ReActAgent()
    print("--- task 1: faq ---")
    for ev in agent.run("demo-1", "What is the refund window?"):
        print(ev["type"], "->", str(ev.get("answer") or ev.get("thought") or ev.get("result"))[:80])
    print("--- task 2: approval gate ---")
    for ev in agent.run("demo-2", "Issue a refund for order 12345."):
        print(ev["type"], "->", str(ev.get("action") or ev.get("answer"))[:80])
    print("--- approve it ---")
    for ev in agent.approve("demo-2", True):
        print(ev["type"], "->", str(ev.get("answer") or ev.get("result"))[:80])
