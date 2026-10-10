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

The tools live in `tools.py`: a safe calculator (walks the AST, never `eval`s),
a sandboxed `python_exec` for calculations that need loops or math functions,
a `search_docs` over a tiny fake knowledge base, a mock `web_search`, and
`issue_refund`. Each tool is a dataclass carrying a docstring — which doubles
as what the model sees when choosing tools — plus a timeout and a destructive
flag.

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
- `evals/run_eval.py` prints a table of the 12 tasks (10 capability + 2 injection-refusal guardrail tasks) with the tools each used, ending in `12/12 tasks passed`.

### Using a real model

```bash
pip install openai          # (already in requirements.txt under "production swaps")
export OPENAI_API_KEY=...   # your key
AGENT_BACKEND=openai python3 agent.py   # same FAQ demo, native function calling
```

The tools become OpenAI function schemas automatically (arg names/types come
from the function signatures), the model picks tools, and its `tool_calls`
parse back into the same actions the loop already runs. `OPENAI_MODEL`
overrides the default `gpt-4o-mini`. The mock stays the default — nothing
changes unless you set `AGENT_BACKEND`.

### Using a free model (no card, no bill)

```bash
export LLM_API_KEY=<your key>   # Google AI Studio (free tier) or OpenRouter
AGENT_BACKEND=free python3 agent.py   # same demo, ReAct text loop
```

`LLM_PROVIDER` picks `gemini` (default, model `gemini-3.8-flash`) or
`openrouter` (default model `openai/gpt-oss-20b:free`); `LLM_MODEL`
overrides either. With a key set and no `AGENT_BACKEND`, the service
auto-selects `free` — the live Render demo just needs the env var. If a
model call fails (20s timeout, one retry on 429/5xx), the loop falls back
to the mock's deterministic rules and keeps the approval gates intact,
so the demo never breaks.

### Use your own key

The live demo above runs on the author's key. To point your own copy at a
real model:

1. **Get a free key.** Go to `aistudio.google.com/api-keys` and click
   **Create API key** — pick "Create API key in new project" (no Cloud
   project and no credit card needed). Alternative: an OpenRouter key
   (`openrouter.ai`) used with a `:free` model slug.
2. **Local run:** `export LLM_API_KEY=your-key-here` before starting the
   server — or put it in a `.env` file you never commit.
3. **Render deploy:** dashboard → your service → Environment → add
   `LLM_API_KEY` → Save. Render redeploys automatically and the fresh
   build reads the key at startup (the backend is chosen once at import,
   so a restart is required — there is no hot-swap).
4. **Confirm it's live:** the header badge turns green
   (`LLM: LIVE · gemini-3.8-flash`), or `GET /info` returns
   `"real_llm": true`.
5. **Keep the key safe:** keys live in environment variables or a secret
   manager only — never in code, never in a commit.

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
llm_client.py       free-tier LLM client (gemini | openrouter), stdlib only
agent.py            ReActAgent: the think -> route -> act loop, loop detection,
                    JSON checkpoints per thread_id, streaming events,
                    approval gates, swappable MockBackend / OpenAIBackend /
                    AnthropicBackend / FreeBackend (free API)
tools.py            Tool dataclass (timeout + destructive flag), ToolNode with
                    timeouts and machine-readable errors, safe calculator,
                    sandboxed python_exec, web search, doc search, refund tool
guardrails.py       PII redaction on tool args (email/phone/Aadhaar/PAN/SSN/
                    card) + prompt-injection detector; refusals and redactions
                    logged to guardrails.jsonl; on by default in the loop
test_guardrails.py  proves injection refusal, poisoned-observation refusal,
                    and pre-execution PII redaction (python3 test_guardrails.py)
langgraph_agent.py  the same loop as a LangGraph StateGraph — compare with agent.py
tracing.py          per-step timing traces: think()/tool durations to
                    logs/traces.jsonl (env-key optional LangSmith push);
                    off with AGENT_TRACING=off
slowest_step.py     reads the trace log, prints per-step timings and the
                    slowest steps per tool (python3 slowest_step.py)
failure_taxonomy.py classifies failed runs into fix-oriented buckets
                    (bad tool choice / bad tool result / bad prompt /
                    bad retrieval / budget / refused / denied) — 20 seeded
                    failures, all classified (python3 failure_taxonomy.py)
prompts/            versioned ReAct prompt text: v1 is the original (frozen),
                    v2 the tighter discipline variant (no-repeat rule,
                    least-powerful tool, quote-the-observation). The
                    prompt-path backends take prompt_version=... or
                    AGENT_PROMPT_VERSION
evals/prompt_ab.py  prompt A/B runner: scores versions on the eval set,
                    prints a pass/efficiency/lint table and declares a
                    winner (python3 evals/prompt_ab.py; --backend free
                    for a true model A/B with LLM_API_KEY)
service.py          FastAPI: /run streams NDJSON events, /approve resumes
evals/tasks.yaml    the 12 eval tasks: questions, required tools, expected answers
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

Every task also carries a step budget (`max_steps`). Current score: **12/12 tasks passed**.

Two of the twelve are guardrail tasks: `injection_ignore_instructions` and
`injection_role_override` must be *refused* — no tools run, the answer starts
with "Refused", and the refusal lands in `guardrails.jsonl` (the eval harness
checks the log, not just the answer).

## Guardrails

On by default, wired into `ReActAgent` — no config needed:

- **PII redaction** — every tool call's args are scanned before the tool (or
  the approval gate) sees them. Emails, phone numbers, Aadhaar/PAN/SSN-shaped
  IDs and Luhn-valid card numbers become `[REDACTED:<class>]`.
- **Injection detector** — a heuristic pattern set scans the user task before
  it reaches the brain, and every tool observation before it steers the next
  thought. A hit stops the run with a `refusal` event and a "Refused: ..."
  final answer.
- **Audit log** — `guardrails.jsonl` next to the checkpoints records
  `injection_refused` (source, patterns) and `pii_redacted` (tool, arg →
  classes). Raw PII values never touch the log.

To run raw: `Guardrails(enabled=False)`, or `AGENT_GUARDRAILS=off`.
`python3 test_guardrails.py` proves all four paths: task refusal, poisoned
observation refusal, pre-execution redaction, and disabled mode.

## Cost cap (kill-switch)

`max_steps` stops long runs, but a run can still burn real money in a few
steps (long contexts, big completions). So every run also gets a USD
budget: `CostTracker` prices each model call — real `resp.usage` token
counts from the OpenAI/Anthropic backends, character-based estimate
otherwise — and the loop aborts with a `cost_exceeded` event (and a
"Stopped: cost cap exceeded" final line) once the budget is gone. Stopped
by cost, not by steps.

- Default cap: **$0.10/run**, via `AGENT_MAX_COST`. `ReActAgent(max_cost=...)`
  overrides per instance.
- Price table is per-model; `AGENT_COST_IN_PER_1K` / `AGENT_COST_OUT_PER_1K`
  override the per-1k-token input/output prices for unlisted models.
- Every `final` event carries `est_cost_usd`; the demo UI shows the running
  cost and a red banner when the fuse trips.

`python3 cost.py` proves the kill: a runaway brain that never answers is
stopped after 2 steps with max_steps still at 50.

## Tracing (slowest-step finder)

Guessing where a run is slow loses to measuring it. Every run records one
JSON line per step — how long each `think()` and each tool call took —
to `logs/traces.jsonl` (`AGENT_TRACE_LOG` overrides; `AGENT_TRACING=off`
disables). Set `LANGSMITH_API_KEY` and the run also ships to LangSmith
on finish, best-effort: a failed push never breaks the run, and the local
file stays the source of truth. `python3 slowest_step.py` reads the log
and names the bottleneck: the latest run's per-step timings, the slowest
steps overall, and average timings per tool.

## Honest notes

- The "brain" is a deterministic mock — rules and regexes tuned for these eval tasks. It is not intelligent and doesn't pretend to be. The point is that everything *around* the brain (the loop, tools, gates, checkpoints, evals) is real and testable without spending anything on API calls.
- To plug in a real model: `pip install openai`, `export OPENAI_API_KEY`, and pass `OpenAIBackend()` to `ReActAgent` instead of the default `MockBackend()`. Same interface, no other changes. Fair warning, though: the mock was tuned for these exact tasks, so a real LLM has to *earn* its 10/10 — watch which tasks get better and which get worse. That's a genuinely interesting experiment.
