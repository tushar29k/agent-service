"""The same agent, rebuilt on LangGraph (pip install langgraph langchain-core).

agent.py's think -> route -> act loop, but as an explicit graph — you get
visualisation, checkpointing, and streaming from the framework instead of
hand-rolling them. Run this file to check the graph builds and executes.
"""
try:
    from langgraph.graph import StateGraph, START, END
    try:  # langgraph >= 1.0 moved checkpointing to langgraph-checkpoint
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        from langgraph.checkpointing.memory import MemorySaver
    from langchain_core.tools import tool as lc_tool
    HAS_LG = True
except ImportError:
    HAS_LG = False


if HAS_LG:
    from typing import TypedDict, Annotated
    import operator

    from tools import (calculator as calc_fn, search_docs as search_fn,
                        web_search as web_fn)

    # thin wrappers — the docstrings are what the model sees when picking tools
    @lc_tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '12*13'."""
        return calc_fn(expression)

    @lc_tool
    def search_docs(query: str) -> str:
        """Search the company knowledge base for policies and FAQs."""
        return search_fn(query)

    @lc_tool
    def web_search(query: str) -> str:
        """Search the live web for current or external info."""
        return web_fn(query)

    LG_TOOLS = [calculator, search_docs, web_search]

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]   # append-only history

    def call_model(state: AgentState):
        # In prod this would be: llm.bind_tools(LG_TOOLS).invoke(state["messages"])
        # Here it's a hardcoded stand-in so this runs with no API key.
        last = state["messages"][-1]
        text = last["content"] if isinstance(last, dict) else str(last)
        if "refund" in text.lower() and "window" in text.lower():
            obs = search_fn("refund window")
            return {"messages": [{"role": "assistant",
                                   "content": f"Based on the knowledge base: {obs}"}]}
        return {"messages": [{"role": "assistant",
                               "content": "I couldn't find anything about that."}]}

    def should_continue(state: AgentState):
        # With bind_tools you'd check tool_calls on the last message here.
        # Our demo model just answers directly, so we're always done.
        return END

    def build_app():
        wf = StateGraph(AgentState)
        wf.add_node("agent", call_model)
        wf.add_edge(START, "agent")
        wf.add_conditional_edges("agent", should_continue, [END])
        return wf.compile(checkpointer=MemorySaver())

    # --- supervisor graph: two worker agents, one supervisor ---
    # the classic multi-agent shape: the supervisor breaks the research
    # task into subtasks, routes each to the right specialist, and a
    # writer merges the findings. with a real model the supervisor would
    # decompose the question and route each subtask itself; here a mock
    # supervisor does the same job with rules, so the graph runs with
    # no API key (same stand-in pattern as call_model above).
    class SupervisorState(TypedDict):
        messages: Annotated[list, operator.add]   # append-only history
        question: str                             # the original research task
        planned: bool                           # True once the task is split
        pending: list                             # subtasks not yet assigned
        current: str                              # subtask a worker is on
        next_worker: str                          # worker the supervisor chose
        findings: Annotated[list, operator.add]   # one dict per worker

    def _mock_splitter(question):
        # stand-in for "supervisor reads the question and plans the work":
        # one subtask per sentence, 'Research:' prefix stripped
        q = question.strip()
        if q.lower().startswith("research:"):
            q = q[len("research:"):]
        return [s.strip() + "?" for s in q.split("?") if s.strip()]

    _WEB_WORDS = ("news", "latest", "current", "today", "recently")

    def _mock_route(subtask):
        # stand-in for the supervisor's routing decision: web-flavoured
        # subtasks go to the web researcher, everything else to the docs one
        return ("worker_web" if any(w in subtask.lower()
                                    for w in _WEB_WORDS)
                else "worker_docs")

    def supervisor(state: SupervisorState):
        # first visit: split the task; every visit: assign one subtask
        # (planned flag stops an empty pending list from re-triggering
        # the split — an exhausted queue is not a fresh question)
        if state.get("planned"):
            pending = state["pending"]
        else:
            pending = _mock_splitter(state["question"])
        if not pending:
            return {"messages": [{"role": "supervisor",
                                   "content": "All subtasks done — sending "
                                              "the findings to the writer."}],
                    "pending": pending, "planned": True, "next_worker": None}
        current = pending[0]
        worker = _mock_route(current)
        return {"messages": [{"role": "supervisor",
                               "content": f"Routing '{current}' to {worker}."}],
                "planned": True, "pending": pending[1:], "current": current,
                "next_worker": worker}

    def _route_next(state: SupervisorState):
        # supervisor picks the next worker, workers only research
        return state.get("next_worker") or "writer"

    def worker_docs(state: SupervisorState):
        sub = state["current"]
        finding = search_fn(sub)  # the real docs tool (offline mock data)
        return {"messages": [{"role": "worker_docs",
                               "content": f"Researched '{sub}': {finding}"}],
                "findings": [{"worker": "worker_docs", "subtopic": sub,
                              "finding": finding}]}

    def worker_web(state: SupervisorState):
        sub = state["current"]
        finding = web_fn(sub)  # the real web tool (offline mock data)
        return {"messages": [{"role": "worker_web",
                               "content": f"Researched '{sub}': {finding}"}],
                "findings": [{"worker": "worker_web", "subtopic": sub,
                              "finding": finding}]}

    def writer(state: SupervisorState):
        # merge the workers' findings into one answer, no string concat
        # in the supervisor — the merge is its own node
        lines = [f"- {f['subtopic']}: {f['finding']}"
                 for f in state["findings"]]
        return {"messages": [{"role": "assistant",
                               "content": "Research summary:\n" +
                                          "\n".join(lines)}]}

    def build_supervisor_app():
        wf = StateGraph(SupervisorState)
        wf.add_node("supervisor", supervisor)
        wf.add_node("worker_docs", worker_docs)
        wf.add_node("worker_web", worker_web)
        wf.add_node("writer", writer)
        wf.add_edge(START, "supervisor")
        wf.add_conditional_edges("supervisor", _route_next,
                                 ["worker_docs", "worker_web", "writer"])
        wf.add_edge("worker_docs", "supervisor")
        wf.add_edge("worker_web", "supervisor")
        wf.add_edge("writer", END)
        return wf.compile(checkpointer=MemorySaver())

    if __name__ == "__main__":
        app = build_app()
        cfg = {"configurable": {"thread_id": "demo"}}
        for chunk in app.stream({"messages": [
                {"role": "user", "content": "What is the refund window?"}]},
                cfg, stream_mode="updates"):
            print(chunk)
        # graph visualisation: app.get_graph().draw_mermaid_png() in a notebook
        print("--- supervisor graph: the research task ---")
        task = ("Research: what is the refund window? "
                "What is the latest news on the 2026 T20 World Cup final?")
        sup = build_supervisor_app()
        final = None
        for chunk in sup.stream(
                {"messages": [{"role": "user", "content": task}],
                 "question": task, "planned": False, "pending": [],
                 "current": "", "next_worker": None, "findings": []},
                {"configurable": {"thread_id": "demo-supervisor"}},
                stream_mode="updates"):
            node, update = next(iter(chunk.items()))
            msgs = update.get("messages", [])
            if msgs:
                m = msgs[-1]
                print(f"[{node}] {m['role']}: {m['content'][:100]}")
            if node == "writer" and msgs:
                final = msgs[-1]["content"]
        assert final and "30 days" in final and "Wankhede" in final, \
            f"supervisor graph failed the research task: {final!r}"
        print("research task passed: both workers' findings in the answer")
else:
    if __name__ == "__main__":
        raise SystemExit("pip install langgraph langchain-core  (then re-run)")
