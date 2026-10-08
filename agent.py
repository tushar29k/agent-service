"""ReAct agent, built the way you'd actually ship one: explicit state, a
think/act loop with a router, loop detection, JSON checkpoints per thread,
streaming events, and human approval gates before destructive tools.

The "brain" is swappable:
  MockBackend        — deterministic rules for demos/tests (NOT intelligent)
  MockReActBackend   — same rules, but answers in ReAct text so the
                       Action-line parser gets exercised with no API key
  MockFreeBackend    — FreeBackend's ReAct-text loop with the mock's rules
                       standing in for the model (no LLM_API_KEY needed)
  OpenAIBackend      — real LLM via native function calling (needs OPENAI_API_KEY)
  ReActPromptBackend — real LLM via a text Thought/Action/Observation loop,
                       parsed back into the same actions (needs OPENAI_API_KEY)
  AnthropicBackend   — real LLM via Anthropic native tool use
                       (needs ANTHROPIC_API_KEY)
  FreeBackend        — real LLM over a free API (Gemini / OpenRouter),
                       ReAct text loop (needs LLM_API_KEY); falls back to
                       the mock's rules if the model call fails

# Run the FAQ demo on the real model: AGENT_BACKEND=openai python3 agent.py
# AGENT_BACKEND=react runs the same demo through the ReAct-prompt backend.
# AGENT_BACKEND=anthropic runs it through Anthropic's tool use instead.
# (needs: pip install openai|anthropic, export OPENAI_API_KEY|ANTHROPIC_API_KEY)

# Long threads: memory.py compacts them (summary + working-set) before
# they reach the brain — pass memory=Memory() to ReActAgent to turn it on.
# python3 memory.py runs the 30-step budget demo.
#
# Sub-agent delegation: agent.delegate(thread_id, topic, subtasks) fans a
# research task out to one researcher per subtask (read-only tools, no
# approval gates by construction) and a writer merges the findings into
# one coherent answer — task 3 of the `python3 agent.py` demo.
#
# Guardrails: guardrails.py redacts PII from tool args and refuses
# prompt-injection attempts (user task or tool observation), logging both
# to guardrails.jsonl. On by default; AGENT_GUARDRAILS=off disables.
# python3 test_guardrails.py proves the refusal + redaction paths.
#
# Cost cap: cost.py tracks estimated USD per run and the loop aborts with
# a `cost_exceeded` event once the budget is gone — the kill-switch for
# runaway tasks. AGENT_MAX_COST sets the budget (default $0.10/run),
# or pass max_cost=... to ReActAgent. python3 cost.py proves the kill.
# Every finished run also appends one JSONL record (tokens, cost, steps,
# outcome) to logs/agent_cost.jsonl (AGENT_COST_LOG overrides) — the raw
# data behind the evals/cost-vs-quality.md table.
#
# Tracing: tracing.py times every think() and every tool call and logs
# one JSONL record per step to logs/traces.jsonl — per-step timings,
# always local. Set LANGSMITH_API_KEY and it also pushes the run to
# LangSmith (best-effort; a failed push never breaks a run).
# python3 slowest_step.py reads the log and names the slowest step.
"""
import inspect
import json
import os
import re
import sys

from cost import CostTracker, log_run_record
from memory import Memory
from guardrails import Guardrails
from tracing import Tracer
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

    # the model this mock stands in for — the cost meter prices it like
    # the real OpenAIBackend default, so mock cost-vs-quality numbers are
    # on the same scale as live ones
    model = "gpt-4o-mini"

    def think(self, messages, tools):
        question = next(m["content"] for m in messages
                        if m["role"] == "user")
        q = question.lower()
        obs = [m for m in messages if m["role"] == "observation"]
        acts = [m["action"]["name"] for m in messages if m["role"] == "action"]
        last_obs = obs[-1]["content"] if obs else None

        if q.startswith("merge:"):
            # the writer's merge step in delegate(): the researchers'
            # findings are already in the prompt, so composing them into
            # one answer is the merge — no tool needed
            facts = [ln[2:] for ln in question.splitlines()
                     if ln.startswith("- ")]
            return {"thought": "Merging the researchers' findings into one "
                               "answer.",
                    "action": None,
                    "answer": "Research summary:\n" +
                              "\n".join(f"- {f}" for f in facts)}

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
            if "fibonacci" in q:
                return {"thought": "Needs a loop — the calculator can't "
                                   "iterate, python_exec can.",
                        "action": {"name": "python_exec",
                                   "args": {"code": "a, b = 0, 1\n"
                                                    "for _ in range(20):\n"
                                                    "    a, b = b, a + b\na"}},
                        "answer": None}
            if any(w in q for w in ("latest", "news", "current events")):
                return {"thought": "Wants current/external info — the web, "
                                   "not the knowledge base.",
                        "action": {"name": "web_search",
                                   "args": {"query": question}},
                        "answer": None}
            if "factorial" in q:
                # calculator only handles arithmetic expressions — factorial
                # needs math.factorial inside the sandbox
                m = re.search(r"factorial of (\d+)", q)
                n = m.group(1) if m else "5"
                return {"thought": "Factorial — the calculator can't do "
                                   "that, python_exec can.",
                        "action": {"name": "python_exec",
                                   "args": {"code": f"math.factorial({n})"}},
                        "answer": None}
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
            if "minute" in q and "hour" in last_obs.lower():
                # the hours figure came from the docs — convert with code
                m = re.search(r"(\d+)\s*hour", last_obs.lower())
                n = m.group(1) if m else "8"
                return {"thought": f"Docs say {n} hours — convert to minutes.",
                        "action": {"name": "python_exec",
                                   "args": {"code": f"{n}*60"}},
                        "answer": None}
            return {"thought": "Have the fact — answer from the observation.",
                    "action": None,
                    "answer": f"Based on the knowledge base: {last_obs}"}

        if acts and acts[-1] == "web_search":
            if last_obs == "NO_RESULTS":
                return {"thought": "Nothing on the web — say so honestly.",
                        "action": None,
                        "answer": "I couldn't find anything about that online."}
            if "factorial" in q:
                # chained computation on the web result: factorial of the
                # major version number the snippet reported
                m = re.search(r"python (\d+)\.", last_obs.lower())
                n = m.group(1) if m else "3"
                return {"thought": "Got the version — now compute its "
                                   "factorial.",
                        "action": {"name": "python_exec",
                                   "args": {"code": f"math.factorial({n})"}},
                        "answer": None}
            return {"thought": "Have the web result — answer from it.",
                    "action": None,
                    "answer": f"Based on web search: {last_obs}"}

        if acts and acts[-1] in ("calculator", "python_exec"):
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
        self.last_usage = None  # (prompt_tok, completion_tok) — the cost meter reads this

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
        if getattr(resp, "usage", None):
            # real token counts for the cost meter (chars/4 otherwise)
            self.last_usage = (resp.usage.prompt_tokens,
                               resp.usage.completion_tokens)
        if msg.tool_calls:
            tc = msg.tool_calls[0]  # one action per think — the loop asks again
            return {"thought": msg.content or f"Calling {tc.function.name}.",
                    "action": {"name": tc.function.name,
                               "args": json.loads(tc.function.arguments or "{}")},
                    "answer": None}
        return {"thought": msg.content or "Answering directly.",
                "action": None,
                "answer": msg.content or "No answer returned by the model."}


def _arg_spec(tool):
    """Arg names/types/required from a tool function's signature, so new
    tools get wired in with zero extra code."""
    props, required = {}, []
    for p in inspect.signature(tool.func).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[p.name] = {"type": {str: "string", int: "integer",
                                  float: "number",
                                  bool: "boolean"}.get(p.annotation,
                                                        "string")}
        if p.default is p.empty:
            required.append(p.name)
    return props, required


def _tool_schemas(tools):
    """Turn the Tool dataclasses into OpenAI function schemas."""
    schemas = []
    for t in tools:
        props, required = _arg_spec(t)
        schemas.append({"type": "function",
                        "function": {"name": t.name,
                                     "description": t.description,
                                     "parameters": {"type": "object",
                                                    "properties": props,
                                                    "required": required}}})
    return schemas


def _anthropic_tool_schemas(tools):
    """Same tools, but in the shape Anthropic's native tool use wants."""
    schemas = []
    for t in tools:
        props, required = _arg_spec(t)
        schemas.append({"name": t.name,
                        "description": t.description,
                        "input_schema": {"type": "object",
                                         "properties": props,
                                         "required": required}})
    return schemas


_RE_ACTION = re.compile(r"^Action:\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$",
                        re.M | re.S)
# \Z (not $ with re.M): the model may write a multi-line answer and $ would
# stop at the first line end — the whole answer has to survive
_RE_ANSWER = re.compile(r"^Answer:\s*(.*?)\s*\Z", re.M | re.S)
_RE_THOUGHT = re.compile(r"^Thought:\s*(.*?)\s*$", re.M | re.S)


def _parse_react_step(text, tools):
    """Turn one ReAct text block back into the think() dict the loop wants.

    Text like 'Thought: ...\\nAction: search_docs({"query": "refund"})'
    becomes {'thought', 'action': {'name', 'args'}} — the exact same shape
    native function calling produces, so the loop can't tell them apart.
    """
    tool_names = {t.name for t in tools}
    thought = _RE_THOUGHT.search(text)
    thought = thought.group(1) if thought else text.strip()[:120]
    m = _RE_ACTION.search(text)
    if m:
        name, raw = m.group(1), m.group(2).strip()
        if name not in tool_names:
            # never let a hallucinated tool reach the runner — end the run
            return {"thought": thought, "action": None,
                    "answer": f"I can't use '{name}': not one of my tools."}
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            # garbled args are worse than no action — answer instead
            return {"thought": thought, "action": None, "answer": text.strip()}
        return {"thought": thought,
                "action": {"name": name, "args": args}, "answer": None}
    m = _RE_ANSWER.search(text)
    if m:
        return {"thought": thought, "action": None, "answer": m.group(1)}
    # model went off-format — safest is to answer with what it said
    return {"thought": thought, "action": None, "answer": text.strip()}


def _react_prompt(messages, tools):
    """One prompt: tool list + instructions + the whole transcript so far."""
    tool_list = "\n".join(
        f"- {t.name}: {t.description.split('.')[0]}. "
        f"Args: {', '.join(inspect.signature(t.func).parameters)}"
        for t in tools)
    lines = []
    for m in messages:
        r = m["role"]
        if r == "user":
            lines.append(f"User: {m['content']}")
        elif r == "action":
            a = m["action"]
            lines.append(f"Action: {a['name']}({json.dumps(a['args'])})")
        elif r == "observation":
            lines.append(f"Observation: {m['content']}")
        elif r == "assistant":
            lines.append(f"Assistant: {m['content']}")
    transcript = "\n".join(lines)
    return (
        "You are a ReAct agent: answer the user, using tools when a step "
        "needs one. Tools:\n" + tool_list +
        "\n\nReply with exactly one step, in this format and nothing else:\n"
        "Thought: <one line of reasoning>\n"
        "Action: tool_name({\"arg\": value})\n"
        "or:\n"
        "Thought: <one line of reasoning>\n"
        "Answer: <final answer to the user>\n"
        "Only use tools from the list above. If a request is destructive "
        "(like issuing a refund), still pick the tool — a human approval "
        "gate runs afterwards, that's not your call.\n\n" + transcript +
        "\n\nYour next step:")


class ReActPromptBackend(ModelBackend):
    """ReAct as pure text: the model writes Thought/Action lines, we parse
    the Action line back into {'name', 'args'} — the same shape as native
    function calling, so the loop runs identically either way."""

    def __init__(self, model=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit(
                "pip install openai  (AGENT_BACKEND=react needs the package)"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "AGENT_BACKEND=react but no OPENAI_API_KEY in the environment")
        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def think(self, messages, tools):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": _react_prompt(messages, tools)}],
            temperature=0)
        return _parse_react_step(resp.choices[0].message.content or "", tools)


class AnthropicBackend(ModelBackend):
    """Native tool use via the Anthropic Messages API: same tools, same
    action dicts — the loop can't tell it apart from the OpenAI backend.

    Actions become tool_use blocks in an assistant turn, observations
    become tool_result blocks in the following user turn (that's the
    pairing Anthropic expects)."""

    def __init__(self, model=None):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise SystemExit(
                "pip install anthropic  (AGENT_BACKEND=anthropic needs the package)"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "AGENT_BACKEND=anthropic but no ANTHROPIC_API_KEY in the environment")
        self.client = Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL",
                                             "claude-haiku-4-5")
        self.last_usage = None  # (prompt_tok, completion_tok) — the cost meter reads this

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
                chat.append({"role": "assistant",
                             "content": [{"type": "tool_use", "id": f"toolu_{n}",
                                          "name": a["name"],
                                          "input": a["args"]}]})
            elif r == "observation":
                # each observation answers the tool_use right before it
                chat.append({"role": "user",
                             "content": [{"type": "tool_result",
                                          "tool_use_id": f"toolu_{n}",
                                          "content": m["content"]}]})
            elif r == "user":
                chat.append({"role": "user", "content": m["content"]})
            else:  # assistant
                chat.append({"role": "assistant", "content": m["content"]})
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=sys,
            messages=chat, tools=_anthropic_tool_schemas(tools))
        if getattr(resp, "usage", None):
            # real token counts for the cost meter (chars/4 otherwise)
            self.last_usage = (resp.usage.input_tokens,
                               resp.usage.output_tokens)
        tool_use = next((b for b in resp.content
                         if getattr(b, "type", None) == "tool_use"), None)
        text = " ".join(b.text for b in resp.content
                        if getattr(b, "type", None) == "text")
        if tool_use:
            return {"thought": text or f"Calling {tool_use.name}.",
                    "action": {"name": tool_use.name,
                               "args": tool_use.input or {}},
                    "answer": None}
        return {"thought": text or "Answering directly.",
                "action": None,
                "answer": text or "No answer returned by the model."}


class FreeBackend(ModelBackend):
    """Real model over a free API (Gemini / OpenRouter), ReAct as pure text:
    the model writes Thought/Action lines via _react_prompt, _parse_react_step
    turns them into the same action dicts native function calling produces.

    Needs LLM_API_KEY in the environment. If the model call fails, falls
    back to the mock's deterministic rules so the loop (and the approval
    gates) never die mid-demo."""

    def __init__(self):
        from llm_client import FreeLLMClient, FreeLLMError
        client = FreeLLMClient.from_env()
        if client is None:
            raise RuntimeError(
                "AGENT_BACKEND=free but no LLM_API_KEY in the environment")
        self.client = client
        self._FreeLLMError = FreeLLMError
        self._mock = MockBackend()
        self.last_error = None  # last api failure, if any — on /info

    def think(self, messages, tools):
        try:
            text = self.client.generate(_react_prompt(messages, tools),
                                        max_tokens=512, temperature=0)
            self.last_error = None  # recovered
            return _parse_react_step(text or "", tools)
        except self._FreeLLMError as e:
            # model unreachable — the mock's rules keep the demo running;
            # the failure lands in stderr (and Render logs) rather than
            # the user-facing event stream
            self.last_error = str(e)  # key-free — safe for /info
            print(f"free backend: model call failed ({e}) — mock rules "
                  f"instead", file=sys.stderr)
            return self._mock.think(messages, tools)


class MockFreeBackend(FreeBackend):
    """FreeBackend's ReAct-text loop, model swapped for the mock's rules:
    runs the REAL FreeBackend.think (same _react_prompt construction, same
    _parse_react_step parse) against a stub client whose generate() formats
    the mock's deterministic decision as one ReAct text step — so the
    free-tier path gets eval'd offline with no API key."""

    # priced like the production free-tier default, not the OpenAI one
    model = "gemini-2.0-flash"

    def __init__(self):
        from llm_client import FreeLLMError  # lazy — same as FreeBackend
        self._FreeLLMError = FreeLLMError
        self.last_error = None
        self._mock = MockBackend()
        backend = self

        class _StubClient:
            """The mock's brain behind FreeLLMClient.generate's signature:
            same args, a ReAct text step instead of a network call."""

            def generate(self, prompt, max_tokens=512, temperature=0):
                d = backend._mock.think(backend._messages, backend._tools)
                if d.get("answer"):
                    return (f"Thought: {d['thought']}\n"
                            f"Answer: {d['answer']}")
                a = d["action"]
                return (f"Thought: {d['thought']}\n"
                        f"Action: {a['name']}({json.dumps(a['args'])})")

        self.client = _StubClient()

    def think(self, messages, tools):
        # the stub needs what generate() doesn't receive — held per call
        # so backends stay re-entrant across interleaved runs
        self._messages, self._tools = messages, tools
        try:
            return super().think(messages, tools)
        finally:
            self._messages = self._tools = None


class MockReActBackend(MockBackend):
    """The mock's deterministic decisions, reformatted as ReAct text, then
    run through the real Action-line parser — so the prompt-path plumbing
    (and its tool choices) gets eval'd with no API key."""

    def think(self, messages, tools):
        d = super().think(messages, tools)
        if d.get("answer"):
            text = f"Thought: {d['thought']}\nAnswer: {d['answer']}"
        else:
            a = d["action"]
            text = (f"Thought: {d['thought']}\n"
                    f"Action: {a['name']}({json.dumps(a['args'])})")
        return _parse_react_step(text, tools)


# --- pick the brain: mock is the default, env flips to a real model ---
def make_backend(name=None):
    """'mock' (default), 'react' (ReAct text prompt), 'openai' (native
    function calling), 'anthropic' (native tool use), or 'free' (free-tier
    API via llm_client, ReAct text prompt). AGENT_BACKEND env var picks
    for you; a set LLM_API_KEY auto-selects 'free' when AGENT_BACKEND is
    unset."""
    name = name or os.environ.get("AGENT_BACKEND")
    if name is None:
        name = "free" if os.environ.get("LLM_API_KEY") else "mock"
    if name == "mock":
        return MockBackend()
    if name == "react":
        return ReActPromptBackend()
    if name == "openai":
        return OpenAIBackend()
    if name == "anthropic":
        return AnthropicBackend()
    if name == "free":
        return FreeBackend()
    raise ValueError(f"unknown backend '{name}' — want 'mock', 'react', "
                     f"'openai', 'anthropic', 'free'")


# --- the agent itself ---
class ReActAgent:
    def __init__(self, backend=None, tools=TOOLS, max_steps=10,
                 checkpoint_dir="checkpoints", memory=None, guardrails=None,
                 max_cost=None):
        self.backend = backend or MockBackend()
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.destructive = {t.name for t in tools if t.destructive}
        self.max_steps = max_steps
        self.checkpoint_dir = checkpoint_dir
        # cost cap: per-run USD budget, the kill-switch for runaway tasks.
        # max_cost=... overrides AGENT_MAX_COST (default $0.10/run)
        self.cost = CostTracker(max_cost)
        # tracing: per-step timings (think/tool) to logs/traces.jsonl —
        # local by default, langsmith when LANGSMITH_API_KEY is set,
        # AGENT_TRACING=off disables entirely
        self.tracer = Tracer()
        # opt-in: when set, the backend sees the compacted transcript
        # (task + summary + working set) while checkpoints keep the full
        # thread — memory only ever shrinks what the model reads
        self.memory = memory
        # guardrails on by default: PII redaction on tool args + injection
        # refusal. pass Guardrails(enabled=False) — or set
        # AGENT_GUARDRAILS=off — to run raw
        if guardrails is None:
            guardrails = Guardrails()
        elif guardrails is False:
            guardrails = Guardrails(enabled=False)
        if os.environ.get("AGENT_GUARDRAILS", "").lower() in (
                "off", "0", "false", "no"):
            guardrails.enabled = False
        self.guardrails = guardrails
        if not self.guardrails.log_path:
            # the audit log lives next to the thread checkpoints
            self.guardrails.log_path = os.path.join(checkpoint_dir,
                                                    "guardrails.jsonl")
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
                yield self._final(state, "Stopped: max steps exceeded.",
                                   outcome="max_steps")
                return
            # the kill-switch: a runaway run dies on budget before it
            # dies on steps — distinct event, distinct final line
            if self.cost.exceeded():
                yield {"type": "cost_exceeded",
                       "spent_usd": self.cost.spent_usd,
                       "cap_usd": self.cost.cap_usd,
                       "calls": self.cost.calls}
                yield self._final(
                    state,
                    f"Stopped: cost cap exceeded "
                    f"(${self.cost.spent_usd:.4f} of "
                    f"${self.cost.cap_usd:.4f} budget).",
                    outcome="cost_exceeded")
                return
            # backend reads the compacted transcript when memory is on;
            # state itself always keeps the full thread
            msgs = (self.memory.context(state["messages"]) if self.memory
                    else state["messages"])
            # timed: the trace shows per-step think() durations
            with self.tracer.step("think", step=state["steps"]):
                d = self.backend.think(msgs, self.tools)
            self.cost.record_think(self.backend, msgs, d)
            yield {"type": "thought", "thought": d["thought"]}
            if d.get("answer"):
                state["messages"].append({"role": "assistant",
                                          "content": d["answer"]})
                self._save(thread_id, state)
                yield self._final(state, d["answer"])
                return
            action = d["action"]
            # PII redaction before anything sees the args — the approval
            # gate and the tool only ever get the scrubbed version
            red = self.guardrails.redact_tool_args(action["name"],
                                                    action["args"],
                                                    thread_id=thread_id)
            action["args"] = red["args"]
            if red["redacted"]:
                yield {"type": "guardrail", "tool": action["name"],
                       "redacted": red["redacted"]}
            # cheap stuck-detector: same action 3 times in a row means
            # we're going in circles
            recent = [m["action"] for m in state["messages"]
                      if m["role"] == "action"][-2:]
            if len(recent) == 2 and all(r == action for r in recent + [action]):
                yield self._final(state, "Stopped: repeating the same action "
                                         "(loop detected).",
                                 outcome="loop_detected")
                return
            # destructive tools don't just run — stop here and wait for a human
            if action["name"] in self.destructive:
                state["pending_action"] = action
                self._save(thread_id, state)
                yield {"type": "approval_required", "action": action,
                       "reason": f"'{action['name']}' is destructive"}
                return
            # timed too — the trace shows per-tool durations, so the
            # slowest-step finder can blame search_docs vs calculator
            with self.tracer.step("tool", tool=action["name"],
                                  step=state["steps"]):
                result = self.tool_node.run(action["name"], action["args"])
            # observations can carry injections too (poisoned tool output) —
            # a hit stops the run instead of steering the next thought
            oref = self.guardrails.check_observation(
                result, thread_id=thread_id, tool=action["name"])
            if oref:
                self._save(thread_id, state)
                yield {"type": "refusal", "source": oref["source"],
                       "patterns": oref["patterns"], "tool": action["name"]}
                yield self._final(state, oref["answer"], refused=True)
                return
            state["messages"].append({"role": "action", "action": action})
            state["messages"].append({"role": "observation", "content": result})
            self._save(thread_id, state)
            if self.memory:
                self.memory.update(state["messages"])
            yield {"type": "tool_result", "tool": action["name"],
                   "args": action["args"], "result": result}

    def _final(self, state, answer, refused=False, outcome=None):
        est_tokens = sum(len(str(m.get("content", ""))) for m in
                         state["messages"]) // 4
        # one JSONL record per finished run — the raw data for per-run
        # cost tracking (every termination path lands exactly one record)
        task = state["messages"][0]["content"] if state["messages"] else ""
        log_run_record(task=task, backend=self.backend, tracker=self.cost,
                       steps=state["steps"],
                       outcome=outcome or
                       ("refused" if refused else "answered"))
        # the trace closes on the same path the cost record takes —
        # every termination gets one summary record
        self.tracer.finish_run(outcome or
                               ("refused" if refused else "answered"))
        return {"type": "final", "answer": answer, "steps": state["steps"],
                "est_tokens": est_tokens, "refused": refused,
                "est_cost_usd": round(self.cost.spent_usd, 6)}

    def run(self, thread_id, message):
        """Start a task. Yields events (stream these to the UI)."""
        self.cost.reset()  # the budget is per run, not per agent
        self.tracer.start_run(thread_id, message,
                              type(self.backend).__name__)  # timings start here
        state = {"messages": [{"role": "user", "content": message}],
                 "steps": 0}
        # injection check first: a hostile task never reaches the brain
        refusal = self.guardrails.check_user_task(message,
                                                  thread_id=thread_id)
        if refusal:
            self._save(thread_id, state)
            yield {"type": "refusal", "source": refusal["source"],
                   "patterns": refusal["patterns"]}
            yield self._final(state, refusal["answer"], refused=True)
            return
        yield from self._loop(thread_id, state)

    def approve(self, thread_id, approved):
        """Resume after an approval gate. approved=True actually runs the action."""
        state = self._load(thread_id)
        action = state.pop("pending_action", None)
        if action is None:
            yield self._final(state, "Nothing awaiting approval.",
                               outcome="nothing_pending")
            return
        if not approved:
            state["messages"].append(
                {"role": "assistant",
                 "content": "Cancelled: human did not approve."})
            self._save(thread_id, state)
            yield self._final(state, "Cancelled by human.",
                               outcome="approval_denied")
            return
        # the post-approval tool run traces too (reopens after the gate)
        with self.tracer.step("tool", tool=action["name"],
                              step=state["steps"]):
            result = self.tool_node.run(action["name"], action["args"])
        state["messages"].append({"role": "action", "action": action})
        state["messages"].append({"role": "observation", "content": result})
        self._save(thread_id, state)
        if self.memory:
            self.memory.update(state["messages"])
        yield {"type": "tool_result", "tool": action["name"],
               "args": action["args"], "result": result}
        yield from self._loop(thread_id, state)

    def delegate(self, thread_id, topic, subtasks, max_steps=6):
        """Fan out + merge. One researcher per subtask (own thread, own
        loop, read-only tools), then a writer merges their findings into
        one coherent answer. Yields events — stream these to the UI the
        same way you stream run()."""
        findings = []
        for i, sub in enumerate(subtasks):
            f = Researcher(backend=self.backend, max_steps=max_steps,
                           memory=self.memory).research(
                               f"{thread_id}-research-{i}", sub)
            findings.append(f)
            yield {"type": "researcher_result", "subtopic": sub,
                   "answer": f["answer"], "tools": f["tools"],
                   "steps": f["steps"]}
        # the writer: one more ReAct pass over the collected findings, so
        # the merge itself goes through a backend instead of string concat
        prompt = ("merge: combine these research findings on '" + topic +
                  "' into one coherent answer, keeping each fact:\n" +
                  "\n".join(f"- {f['answer']}" for f in findings))
        writer = ReActAgent(backend=self.backend, tools=READ_ONLY_TOOLS,
                            max_steps=4, memory=self.memory)
        yield from writer.run(f"{thread_id}-writer", prompt)


# --- sub-agent delegation: researcher -> writer ---
# a research task fans out: one researcher per subtask, each with its own
# ReAct loop, its own thread, and a read-only tool subset — destructive
# tools can't even reach a researcher, so there are no approval gates
# mid-fan-out. the writer then merges the findings into one answer.
READ_ONLY_TOOLS = [t for t in TOOLS
                   if not t.destructive and t.name != "python_exec"]
# python_exec stays out on purpose: researchers read facts, they don't run
# code — keeps their blast radius obvious


class Researcher:
    """One sub-agent: its own ReActAgent, read-only tools, fresh thread."""

    def __init__(self, backend=None, max_steps=6, memory=None):
        self.agent = ReActAgent(backend=backend or MockBackend(),
                                tools=READ_ONLY_TOOLS, max_steps=max_steps,
                                memory=memory)

    def research(self, thread_id, subtopic):
        """Run the subtopic through a fresh ReAct loop; return the findings
        dict the writer merges (subtopic, answer, tools used, steps)."""
        used, final = [], None
        for ev in self.agent.run(thread_id, subtopic):
            if ev["type"] == "tool_result":
                used.append(ev["tool"])
            if ev["type"] == "final":
                final = ev
        return {"subtopic": subtopic, "answer": final["answer"],
                "tools": used, "steps": final["steps"]}


if __name__ == "__main__":
    # quick smoke test: an FAQ, then a refund that hits the approval gate
    # (AGENT_BACKEND=openai or =anthropic runs this same demo on a real model)
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
    print("--- task 3: delegation fan-out + merge ---")
    for ev in agent.delegate(
            "demo-3", "refund window, warranty period, and t20 news",
            ["What is the refund window?",
             "How many months is the warranty?",
             "What is the latest news on the 2026 T20 World Cup final?"]):
        print(ev["type"], "->",
              str(ev.get("answer") or ev.get("thought")
                  or ev.get("subtopic"))[:110])
    print("--- task 4: kill-switch (runaway brain, tiny budget) ---")
    class RunawayBrain(MockBackend):
        """Never answers, always acts with fresh args — the loop detector
        can't catch it, max_steps is generous, so only the cost cap can
        stop it."""

        def __init__(self):
            self.n = 0

        def think(self, messages, tools):
            self.n += 1
            return {"thought": f"still working, step {self.n}",
                    "action": {"name": "calculator",
                               "args": {"expression": f"{self.n}+{self.n}"}},
                    "answer": None}

    poor = ReActAgent(backend=RunawayBrain(), max_steps=50, max_cost=1e-9)
    for ev in poor.run("demo-4", "keep calculating forever"):
        if ev["type"] == "cost_exceeded":
            print(f"cost_exceeded -> ${ev['spent_usd']:.6f} of "
                  f"${ev['cap_usd']:.9f} after {ev['calls']} calls")
        if ev["type"] == "final":
            print(ev["type"], "->", ev["answer"],
                  f"(steps {ev['steps']} of max 50)")
