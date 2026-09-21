# agent-service

A ReAct-style agent you can actually run, break, and fix — no LLM API key needed.

## Live demo

**[https://tushar29k-agent-service.onrender.com](https://tushar29k-agent-service.onrender.com)** — Chat with the ReAct agent: watch thoughts and tool calls stream, and approve or deny the destructive actions yourself.
> Hosted on Render's free tier — the first visit after a while can take ~30s while the instance wakes up.

## The idea

Every "AI agent" tutorial stops right where the interesting part starts: the loop. Think, pick a tool, run it, read the result, repeat — plus all the production plumbing around it. What happens when the agent gets stuck calling the same tool forever? Who approves the scary actions? How do you resume after a crash? How do you know it still works after you change something?

This project is that loop, built the way you'd build it in production, with one deliberate cheat: instead of a real LLM, the "brain" is a deterministic mock — a handful of rules that decides which tool to call. So you can run every demo, every eval, and the whole API today, on your own machine, for free. The mock speaks the same interface a real model backend would, which means swapping in OpenAI later is a one-line change.

## How it works

`agent.py` holds the `ReActAgent`. The heart of it is `_loop`: it asks the backend what to do next, and the backend replies with either a final answer or an action to take. From there:

- **Route and act** — the action goes to `ToolNode` in `tools.py`, which runs the tool with a timeout and hands back either a result or a machine-readable error string. A tool can never crash the agent.
- **Loop detection** — if the agent picks the exact same action three times in a row, it's going in circles, so the loop stops with a "loop detected" message instead of burning through its step budget.
- **Approval gates** — tools flagged `destructive` (like `issue_refund`, which moves money) don't run immediately. The loop pauses, saves a checkpoint, and yields an `approval_required` event. A human then calls `/approve` with yes or no, and the loop resumes or cancels.
- **Checkpoints** — every step is saved as JSON under `checkpoints/<thread_id>.json`. Kill the process mid-run and the conversation picks up where it left off — state is explicit and serialisable, which is the whole point.
- **Streaming** — `run()` and `approve()` are generators yielding events (`thought`, `tool_result`, `approval_required`, `final`), so a UI can stream progress live.

The tools live in `tools.py`: a safe calculator (walks the AST, never `eval`s), a `search_docs` over a tiny fake knowledge base, and `issue_refund`. Each tool is a dataclass carrying a docstring — which doubles as what the model sees when choosing tools — plus a timeout and a destructive flag.

`langgraph_agent.py` is the same idea rebuilt on LangGraph: the loop becomes an explicit graph with nodes and edges, and you get the framework's checkpointing, streaming, and visualisation instead of hand-rolling them. Reading it side-by-side with `agent.py`'s `_loop` is the fastest way to learn what a framework actually buys you.

`service.py` wraps the agent in FastAPI: `POST /run` streams events as JSON lines, `POST /approve` resumes after an approval gate.

## How to run

Prerequisites: Python 3.10+. Nothing else for the core — no API keys, no accounts.

```bash
cd agent-service
pip install -r requirements.txt   # just pyyaml, for the evals

python3 tools.py            # sanity-check the tools (timeouts, errors, destructive flag)
python3 agent.py            # the demo: an FAQ task, then a refund that hits the approval gate
python3 evals/run_eval.py   # the eval suite
```

What you'll see:

- `tools.py` prints `tools OK` — calculator, search, and refund all behave.
- `agent.py` runs two demos. Task 1 ("What is the refund window?") thinks, calls `search_docs`, and answers "Based on the knowledge base: Refund policy: full refunds are available within 30 days…". Task 2 ("Issue a refund for order 12345") thinks, then stops with `approval_required` for `issue_refund` — and the "approve it" step runs the refund and reports "Refund issued for order 12345."
- `evals/run_eval.py` prints a table of the 5 tasks with the tools each used, ending in `5/5 tasks passed`.

The API needs `pip install fastapi uvicorn`:

```bash
uvicorn service:app --reload
```

```bash
curl -X POST localhost:8000/run -H 'Content-Type: application/json' \
  -d '{"thread_id":"u1","message":"What is the refund window?"}'
# streams JSON-lines events: thought -> tool_result -> final

curl -X POST localhost:8000/approve -H 'Content-Type: application/json' \
  -d '{"thread_id":"u2","approved":true}'
# resumes a refund that paused at the approval gate
```

The LangGraph variant needs `pip install langgraph langchain-core`:

```bash
python3 langgraph_agent.py   # builds the graph and runs the refund FAQ through it
```

## Project layout

```
agent.py            ReActAgent: the think -> route -> act loop, loop detection,
                    JSON checkpoints per thread_id, streaming events,
                    approval gates, swappable MockBackend / OpenAIBackend
tools.py            Tool dataclass (timeout + destructive flag), ToolNode with
                    timeouts and machine-readable errors, safe calculator,
                    doc search, refund tool
langgraph_agent.py  the same loop as a LangGraph StateGraph — compare with agent.py
service.py          FastAPI: /run streams NDJSON events, /approve resumes
evals/tasks.yaml    the 5 eval tasks: questions, required tools, expected answers
evals/run_eval.py   runs each task, asserts right tools + answer content + step budget
checkpoints/        example saved conversation states
```

## Evals

Five tasks, each checking something the loop has to get right:

1. **refund_lookup** — "What is the refund window?" Must call `search_docs` and answer with "30 days".
2. **warranty_months** — "The warranty is 2 years. How many months is that?" Must chain `search_docs` then `calculator` and land on "24".
3. **refund_approval** — "Issue a refund for order 12345." Must pause at the approval gate, then complete the refund once approved.
4. **refund_approval_denied** — same ask, but the human says no. Must cancel cleanly with "Cancelled by human".
5. **unknown_topic** — "What is the CEO's favourite colour?" Must search, find nothing, and say so honestly instead of inventing an answer.

Every task also carries a step budget (`max_steps`). Current score: **5/5 tasks passed**.

## Honest notes

- The "brain" is a deterministic mock — rules and regexes tuned for these eval tasks. It is not intelligent and doesn't pretend to be. The point is that everything *around* the brain (the loop, tools, gates, checkpoints, evals) is real and testable without spending anything on API calls.
- To plug in a real model: `pip install openai`, `export OPENAI_API_KEY`, and pass `OpenAIBackend()` to `ReActAgent` instead of the default `MockBackend()`. Same interface, no other changes. Fair warning, though: the mock was tuned for these exact tasks, so a real LLM has to *earn* its 5/5 — watch which tasks get better and which get worse. That's a genuinely interesting experiment.
