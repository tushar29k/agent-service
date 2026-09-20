# agent-service — next versions plan

Current state (v1): ReAct loop with a deterministic mock brain, tools
(calculator, doc search, refund), approval gates for destructive tools, JSON
checkpoints, streaming events, LangGraph variant, evals at 5/5. Every item
below is one day's commit: implement it, keep evals green, commit with a
human-style message.

## v2.1 — a real brain

- [ ] Wire a real OpenAI backend (function calling) behind the existing backend interface; key from env, mock stays default — done when `backend: openai` runs the FAQ demo with a key set
- [ ] ReAct-prompt vs native function calling: implement both, compare tool-choice accuracy in evals — done when evals report both numbers
- [ ] Anthropic backend variant behind the same interface — done when `backend: anthropic` passes the smoke demo
- [ ] New tool: web search (mock results with a `# SWAP:` to a real API) — done when an eval task requires it and passes
- [ ] New tool: sandboxed Python exec (AST-safe, timeout, no network/imports) — done when a calculation task uses it instead of the calculator
- [ ] Tool description linter: scores tool docstrings for model-misuse risk, warns on vague descriptions — done when `python tools.py` prints lint scores
- [ ] Evals: expand tasks.yaml to 10 tasks including multi-step and tool-misuse traps — done when run_eval.py reports 10/10

## v2.2 — memory, multi-agent & guardrails

- [ ] Conversation summarisation memory: compress steps older than N, keep a working set — done when a 30-step thread stays under the token budget
- [ ] Sub-agent delegation: researcher → writer pattern in agent.py — done when a research task fans out and merges
- [ ] Supervisor graph variant in langgraph_agent.py (two agents, one supervisor) — done when it passes the research task
- [ ] MCP client: connect one MCP server, expose its tools to the ReAct loop — done when an MCP-served tool is callable
- [ ] Guardrails: PII redaction on tool inputs + prompt-injection detector heuristic — done when injection eval tasks are refused and logged
- [ ] Kill-switch + max-cost cap (not just max-iteration): abort a run that exceeds budget — done when a runaway task is stopped by cost, not steps
- [ ] Evals: adversarial tasks — injection attempts must be refused or flagged — done when adversarial set passes

## v2.3 — observability & cost

- [ ] Per-run token/cost logging to JSONL; cost-vs-quality table across backends — done when 3 backends show cost per task
- [ ] Tracing integration (LangSmith or Langfuse, env-key optional): slowest-step finder script — done when a trace shows per-step timings
- [ ] Failure taxonomy script: categorise failed runs (bad tool? bad prompt? bad retrieval?) — done when 20 seeded failures are classified
- [ ] Prompt versioning: prompts/ directory + A/B runner scoring two versions on the eval set — done when the runner declares a winner
- [ ] LLM-as-judge eval: rubric prompt judging task success; measure judge-human agreement on 10 cases — done when agreement % is recorded
- [ ] Approval UX: CLI approver for destructive tools (approve/deny with reason) — done when the refund demo runs through it
- [ ] Eval history dashboard: tiny HTML report of scores over time from the JSONL logs — done when report.html renders the trend

## v2.4 — ship it

- [ ] Dockerfile + compose — done when the API serves from a container
- [ ] CI: GitHub Action running evals on push, failing on regression — done when the workflow is green
- [ ] Concurrency test: N parallel threads, report throughput and failures — done when the table is committed
- [ ] ADRs: framework vs from-scratch, approval-gate design, memory strategy — done when docs/adr/ has 3 files
- [ ] Production checklist doc: secrets, timeouts, retries, rate limits, audit log — done when docs/production-checklist.md exists
- [ ] Demo script: scripted run showing approval gate + checkpoint recovery — done when demo.sh reproduces it
- [ ] Release notes v2.0 in README with real numbers (eval score, cost/task, p99 step latency) — done when README leads with them
