# agent-service — a production-shaped ReAct agent you can actually run

The full agent loop with **no API key**: the model backend is a deterministic
mock so every mechanism (routing, tools, approvals, checkpoints, evals) is
exercisable today. `OpenAIBackend` in `agent.py` is the one-line swap.

## Run it

```bash
cd agent-service
pip install -r requirements.txt

python3 tools.py            # unit-check tools (timeouts, errors, destructive flag)
python3 agent.py            # demo: FAQ task + approval gate + human approval
python3 evals/run_eval.py   # 5 tasks: tools, approvals, budgets
python3 langgraph_agent.py  # the same loop as a LangGraph graph (needs langgraph)
uvicorn service:app --reload
#   POST /run      {"thread_id":"u1","message":"What is the refund window?"}
#   POST /approve  {"thread_id":"u1","approved":true}
```

## The shape

```
agent.py            ReActAgent: state, think->route->act loop, loop detection,
                    JSON checkpoints per thread_id, streaming events,
                    approval gates, MockBackend / OpenAIBackend
tools.py            Tool dataclass (timeout, destructive flag), ToolNode with
                    timeouts + machine-readable errors, safe calculator
langgraph_agent.py  the same loop as a LangGraph StateGraph (compare the two!)
service.py          FastAPI: /run streams NDJSON events, /approve resumes
evals/tasks.yaml    tasks: right tools, approval allow/deny, budgets
evals/run_eval.py   asserts tools used + answer contains expected + max steps
```

## Exercises (do these — this is the production learning)

1. **Add a tool.** Write `get_order_status(order_id)` backed by a dict.
   Give it a proper docstring, timeout, and non-destructive flag. Add an eval
   task that must use it.
2. **Break the loop detection.** Write a MockBackend variant that always
   retries the same failing tool. Watch the agent stop at 3 repeats — then
   remove the detector and watch it hit max_steps instead.
3. **Crash recovery.** Start a task, kill the process mid-run, restart, and
   call `/approve` or re-`run` — the JSON checkpoint resumes the conversation.
   (This is why state is explicit and serialisable.)
4. **Wire the real model.** Set `OPENAI_API_KEY`, pass `OpenAIBackend()` to
   `ReActAgent`, re-run evals. Which tasks get better? Which get *worse*
   (hint: the mock was tuned for these tasks — the LLM must earn it)?
5. **Compare with LangGraph.** Read `langgraph_agent.py` next to `agent.py`'s
   `_loop`. Map each line of `_loop` to a graph node/edge. When would you
   choose the framework over hand-rolled?
6. **Cost accounting.** The `final` event carries `est_tokens`. Add per-task
   cost in $ for a real model price and a budget alarm in `run_eval.py`.
