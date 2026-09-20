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

    from tools import calculator as calc_fn, search_docs as search_fn

    # thin wrappers — the docstrings are what the model sees when picking tools
    @lc_tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '12*13'."""
        return calc_fn(expression)

    @lc_tool
    def search_docs(query: str) -> str:
        """Search the company knowledge base for policies and FAQs."""
        return search_fn(query)

    LG_TOOLS = [calculator, search_docs]

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

    if __name__ == "__main__":
        app = build_app()
        cfg = {"configurable": {"thread_id": "demo"}}
        for chunk in app.stream({"messages": [
                {"role": "user", "content": "What is the refund window?"}]},
                cfg, stream_mode="updates"):
            print(chunk)
        # graph visualisation: app.get_graph().draw_mermaid_png() in a notebook
else:
    if __name__ == "__main__":
        raise SystemExit("pip install langgraph langchain-core  (then re-run)")
