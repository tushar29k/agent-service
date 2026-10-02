"""Proof that an MCP-served tool is callable through the ReAct loop.

Spawns examples/mcp_server.py over stdio, wraps its tools, and runs a
ReAct task that can only be answered by calling the MCP tool. Asserts:
the handshake surfaced the tool, the loop actually called it, and the
observation flowed into the final answer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ModelBackend, ReActAgent
from mcp_client import connect_mcp
from tools import TOOLS, lint_tool_descriptions, LINT_THRESHOLD

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "examples", "mcp_server.py")


class TimeBackend(ModelBackend):
    """Demo brain: the question needs the clock, the clock is the MCP tool."""

    def think(self, messages, tools):
        names = {t.name for t in tools}
        obs = [m for m in messages if m["role"] == "observation"]
        if not obs:
            assert "get_utc_time" in names, "MCP tool missing from tool list"
            return {"thought": "Need the current time — the MCP clock "
                               "tool has it.",
                    "action": {"name": "get_utc_time", "args": {}},
                    "answer": None}
        return {"thought": "Got the time from the MCP tool.",
                "action": None,
                "answer": f"The current UTC time is {obs[-1]['content']}"}


def main():
    with connect_mcp(sys.executable, [SERVER]) as conn:
        # 1. handshake surfaced the tool with a usable description
        names = [t.name for t in conn.tools]
        assert names == ["get_utc_time"], f"unexpected tools: {names}"
        for name, score, warnings in lint_tool_descriptions(conn.tools):
            assert score >= LINT_THRESHOLD, \
                f"{name} lint {score} < {LINT_THRESHOLD}: {warnings}"
        print(f"connected: tools={names}, "
              f"destructive={[t.destructive for t in conn.tools]}")

        # 2. run a ReAct task that requires the MCP-served tool
        agent = ReActAgent(backend=TimeBackend(), tools=TOOLS + conn.tools)
        events = list(agent.run("mcp-demo",
                                "What time is it in UTC right now?"))
        calls = [e for e in events
                 if e["type"] == "tool_result" and e["tool"] == "get_utc_time"]
        assert calls, "MCP tool was never called through the loop"
        result = calls[0]["result"]
        assert result.startswith("20"), f"bad clock result: {result!r}"
        print(f"MCP tool called through the loop -> {result}")

        # 3. the observation made it into the final answer
        final = events[-1]
        assert final["type"] == "final", f"run didn't finish: {final}"
        assert result in final["answer"], "observation lost before the answer"
        print(f"final: {final['answer']}")
    print("mcp demo OK")


if __name__ == "__main__":
    main()
