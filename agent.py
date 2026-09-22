"""ReAct agent, built the way you'd actually ship one: explicit state, a
think/act loop with a router, loop detection, JSON checkpoints per thread,
streaming events, and human approval gates before destructive tools.

The "brain" is swappable:
  MockBackend   — deterministic rules for demos/tests (NOT intelligent)
  OpenAIBackend — real LLM via native function calling (needs OPENAI_API_KEY)

# Run the FAQ demo on the real model: AGENT_BACKEND=openai python3 agent.py
# (needs: pip install openai, export OPENAI_API_KEY)
"""
import inspect
import json
import os
import re

from tools import TOOLS, ToolNode


# --- the brain (swappable) ---
class ModelBackend:
    def think(self, messages, tools):
        """One job: look at the conversation + tools and decide what to do.
        Return {'thought': str, 'action': {'name', 'args'} | None,
        'answer': str | None}."""
        raise NotImplementedError


class MockBackend(ModelBackend):
    """Deterministic stand-in so the whole loop runs with no API key.

    Just rules and regexes — tuned for the bundled eval tasks, definitely
    not intelligent. The real backend (below) lets the LLM do the thinking
    instead.
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
    """Native function calling: the Tool dataclasses become function
    schemas, the model picks tools, and tool_calls parse back into the
    same {'name', 'args'} actions the loop already understands."""

    def __init__(self, model=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit(
                "pip install openai  (AGENT_BACKEND=openai needs the package)"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "AGENT_BACKEND=openai but no OPENAI_API_KEY in the environment")
        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def think(self, messages, tools):
        sys = ("You are a ReAct agent: answer the user, using the provided "
               "tools whenever a step needs one. If a request is destructive "
               "(like issuing a refund), still pick the tool — a human "
               "approval gate runs afterwards, that's not your call.")
        chat, n = [], 0
        for m in messages:
            r = m["role"]
            if r == "action":
                n += 1
                a = m["action"]
                chat.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": f"call_{n}", "type": "function",
                                             "function": {"name": a["name"],
                                                          "arguments": json.dumps(a["args"])}}]})
            elif r == "observation":
                # each observation answers the tool call right before it
                chat.append({"role": "tool", "tool_call_id": f"call_{n}",
                             "content": m["content"]})
            else:  # user / assistant
                chat.append({"role": r, "content": m["content"]})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys}] + chat,
            tools=_tool_schemas(tools),
            tool_choice="auto",
            temperature=0)
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]  # one action per think — the loop asks again
            return {"thought": msg.content or f"Calling {tc.function.name}.",
                    "action": {"name": tc.function.name,
                               "args": json.loads(tc.function.arguments or "{}")},
                    "answer": None}
        return {"thought": msg.content or "Answering directly.",
                "action": None,
                "answer": msg.content or "No answer returned by the model."}


def _tool_schemas(tools):
    """Turn the Tool dataclasses into OpenAI function schemas.

    Arg names/types come from each tool function's signature, so new
    tools get wired in with zero extra code."""
    schemas = []
    for t in tools:
        props, required = {}, []
        for p in inspect.signature(t.func).parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            props[p.name] = {"type": {str: "string", int: "integer",
                                      float: "number",
                                      bool: "boolean"}.get(p.annotation,
                                                            "string")}
            if p.default is p.empty:
                required.append(p.name)
        schemas.append({"type": "function",
                        "function": {"name": t.name,
                                     "description": t.description,
                                     "parameters": {"type": "object",
                                                    "properties": props,
                                                    "required": required}}})
    return schemas


# --- pick the brain: mock is the default, env flips it to openai ---
def make_backend(name=None):
    """'mock' (default) or 'openai'. AGENT_BACKEND env var picks for you."""
    name = name or os.environ.get("AGENT_BACKEND", "mock")
    if name == "mock":
        return MockBackend()
    if name == "openai":
        return OpenAIBackend()
    raise ValueError(f"unknown backend '{name}' — want 'mock' or 'openai'")


# --- the agent itself ---
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

    # persistence is deliberately boring: one JSON file per thread_id,
    # that's the whole conversation store
    def _path(self, thread_id):
        safe = re.sub(r"[^\w-]", "_", thread_id)
        return os.path.join(self.checkpoint_dir, f"{safe}.json")

    def _save(self, thread_id, state):
        json.dump(state, open(self._path(thread_id), "w"))

    def _load(self, thread_id):
        p = self._path(thread_id)
        return json.load(open(p)) if os.path.exists(p) else None

    # the loop: think -> route -> act -> observe, until done
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
            # cheap stuck-detector: same action 3 times in a row means
            # we're going in circles
            recent = [m["action"] for m in state["messages"]
                      if m["role"] == "action"][-2:]
            if len(recent) == 2 and all(r == action for r in recent + [action]):
                yield self._final(state, "Stopped: repeating the same action "
                                         "(loop detected).")
                return
            # destructive tools don't just run — stop here and wait for a human
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
        """Resume after an approval gate. approved=True actually runs the action."""
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
    # quick smoke test: an FAQ, then a refund that hits the approval gate
    # (AGENT_BACKEND=openai runs this same demo on the real model)
    agent = ReActAgent(backend=make_backend())
    print("--- task 1: faq ---")
    for ev in agent.run("demo-1", "What is the refund window?"):
        print(ev["type"], "->", str(ev.get("answer") or ev.get("thought") or ev.get("result"))[:80])
    print("--- task 2: approval gate ---")
    for ev in agent.run("demo-2", "Issue a refund for order 12345."):
        print(ev["type"], "->", str(ev.get("action") or ev.get("answer"))[:80])
    print("--- approve it ---")
    for ev in agent.approve("demo-2", True):
        print(ev["type"], "->", str(ev.get("answer") or ev.get("result"))[:80])
