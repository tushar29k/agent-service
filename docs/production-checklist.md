# Production checklist

Everything on this list names something concrete in this repo. Work through it
top to bottom before pointing anyone's real workload at this service.

## 1. Secrets

- [ ] `LLM_API_KEY` (free-tier brain via `LLM_PROVIDER`/`LLM_MODEL`) comes from
      the environment only. Render dashboard → Environment. It never appears in
      code, logs, or error messages — `FreeLLMError` messages deliberately strip it.
- [ ] Paid-backend keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
      `LANGSMITH_API_KEY`) are optional extras: when absent the backends fall
      back to the mock path (`agent.py` backend factory). Same rule — env only.
- [ ] `.env` is git-ignored (`ls` it, you'll see it's in `.gitignore`) — never
      commit it, never paste a real key into a test. `test_llm_client.py` uses
      placeholder values (`"k"`) for exactly this reason.
- [ ] No key in `checkpoints/` or JSONL logs: tool args are persisted in
      checkpoints and audit trails, so `guardrails.py` redacts PII to
      `[REDACTED:<class>]` before anything hits a tool. Keys must never ride
      inside a tool arg — keep them out of prompts and tool inputs.
- [ ] Verify: `grep -r "sk-" --include="*.py" . ; grep -ri "api_key.*=.*['\"][A-Za-z]" --include="*.py" .` comes back empty. `git log --all -S"sk-"` too.

## 2. Timeouts

- [ ] Model calls: `llm_client.py` `_TIMEOUT = 20` (s) on every `urlopen`. One hung
      provider call can't pin the ReAct loop — the exception surfaces as
      `FreeLLMError` and the run ends.
- [ ] Tool calls: every `Tool` carries a `timeout` (`tools.py` dataclass,
      default 10.0s); `python_exec` gets `timeout=20.0` headroom because the
      sandboxed subprocess itself caps code at 5s. `ToolNode` runs the tool in a
      thread and returns `ERROR: tool '<name>' timed out after Xs` on expiry —
      a tool can never crash the agent.
- [ ] LangSmith batch push (`tracing.py`): the push runs behind a
      `urlopen(..., timeout=5)` and best-effort try/except — telemetry failures
      never fail a run. Verify no trace call can block shutdown.
- [ ] Render/Gunicorn: set a worker timeout ≥ 2× the longest expected single
      run (step budget × per-step cost). A `/run` stream that outlives the
      worker gets cut — know that number, don't guess it.

## 3. Retries

- [ ] `FreeLLMClient._send`: exactly **one** retry with `_BACKOFF = 1.5`s, on
      429/5xx and on connection errors only. Anything else (400s, bad JSON
      shape, unknown provider) is final — retrying those is burning money.
- [ ] Tool retries: none by default, and that's deliberate — `issue_refund` is
      a real destructive op, so a blind retry could double-charge. If a
      retrying wrapper is added later, it must be per-tool opt-in and
      idempotency-keyed, never blanket.
- [ ] OpenAI/Anthropic backends (`agent.py`): use the SDKs' own retry defaults.
      Note the defaults down in the deploy runbook — if a provider changes its
      retry policy, your p99 step latency moves with it.
- [ ] Verify: `python3 test_llm_client.py` covers the retry path; the retry
      count is a constant in one file, not scattered.

## 4. Rate limits

- [ ] Know your budget: Gemini free tier is ~15 req/min, 1500 req/day; a run
      burns one model call per step plus tool calls. Size `AGENT_MAX_COST`
      (default $0.10/run in `cost.py`) and `max_steps` (default 10) to the
      tier you're actually on.
- [ ] Kill switches live in `agent.py` `_loop`: cost cap (`cost_exceeded` event
      + `Stopped: cost cap exceeded`), step budget (`max_steps` → `max_steps`
      outcome), loop detection (same action 3× → `loop_detected`). Every
      termination path writes exactly one record to `logs/agent_cost.jsonl`.
- [ ] Load shape is measured, not assumed: `evals/concurrency_test.py` runs N
      parallel agents and appends wall time, tasks/s, p50/p99 latency to
      `evals/concurrency-results.md` (16 threads: react 47.6 tasks/s, p50 65ms,
      p99 309ms, 0 errors — mock backends; re-run against the real brain before
      quoting these). Don't put the service in front of traffic it wasn't
      measured against.
- [ ] If this ever goes multi-tenant: rate-limit by API key before the
      `POST /run` handler in `service.py`, and put a token bucket in front of
      the model client. Today it's absent — the kill switches bound one run,
      not N concurrent ones.

## 5. Audit log

- [ ] Run records: `logs/agent_cost.jsonl` (or `AGENT_COST_LOG`) — one JSON
      record per finished run: task, backend, model, tokens in/out, token
      basis, cost_usd, steps, outcome (`answered`/`refused`/`max_steps`/
      `cost_exceeded`/`loop_detected`/`approval_denied`). This is the
      money-and-quality ledger; `evals/cost-vs-quality.md` is built from it.
- [ ] Step traces: `logs/traces.jsonl` (or `AGENT_TRACE_LOG`) — per think/tool
      timing per run, written by `tracing.py`. `slowest_step.py` finds the
      bottleneck; opt-out with `AGENT_TRACING=off`, LangSmith batch push only
      when `LANGSMITH_API_KEY` is set.
- [ ] Approval decisions: `logs/approvals.jsonl` (or `AGENT_APPROVAL_LOG`) —
      every human gate: tool, args, approved/denied, free-text reason, run
      outcome. `approval.py` logs here on both CLI and API paths; empty
      `AGENT_APPROVAL_LOG` disables it — know whether your deploy does.
- [ ] Guardrail events: `guardrails.jsonl` (repo root) — injection refusals and
      PII-redaction classes (never raw values). Verify refusals actually land:
      `evals/run_adversarial.py` asserts 14/14 each run.
- [ ] `logs/` and `guardrails.jsonl` are git-ignored: audit data is runtime
      state, not source. `checkpoints/` **is** committed (demos), so keep
      sensitive data out of demo tasks — checkpoints persist full user
      messages and tool args.
- [ ] Retention: decide and document it (e.g. 30 days for traces, 90 for run
      records, 1 year for approvals). Add a cron or a `logs/` rotation before
      disk fills — JSONL grows at roughly one line per step per run.
- [ ] Verify: run one end-to-end task and confirm all four files (or their
      env-overridden paths) got their line, and `evals/run_eval.py` still
      reports 12/12.

## 6. Before first traffic

- [ ] `.gitignore` keeps `logs/`, `guardrails.jsonl`, `.env`, `__pycache__/` out
      of the repo. Confirm before the first deploy commit.
- [ ] `GET /health` returns 200 from a cold start (Render free tier sleeps —
      first request ~30s). Smoke `POST /run` on the live URL, not localhost.
- [ ] Set `AGENT_MAX_COST` for the tier you're on, confirm evals stay green
      (`evals/run_eval.py`, `evals/run_adversarial.py`), and re-run
      `evals/concurrency_test.py` against the configured backend.
- [ ] Read `docs/adr/` first — the approval gate (0002) and memory strategy
      (0003) constrain what "safe to change" means in this codebase.
