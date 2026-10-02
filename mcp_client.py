"""MCP client: connect one MCP server, expose its tools to the ReAct loop.

One server, stdio transport, the official SDK doing the handshake
(initialize -> tools/list -> tools/call). The SDK is async, so each
connection owns a background event loop thread — the wrapped tools stay
plain sync callables, which is what ToolNode expects.

  conn = connect_mcp(sys.executable, ["examples/mcp_server.py"])
  agent = ReActAgent(backend=..., tools=TOOLS + conn.tools)
  ... run tasks ...
  conn.close()          # or: with connect_mcp(...) as conn: ...

The MCP tool's annotations decide the approval flag: destructiveHint means
the tool goes through the same human approval gate as issue_refund.
"""
import asyncio
import inspect
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tools import Tool

# server spawn + handshake budget — stdio servers start fast, this is slack
_CONNECT_TIMEOUT = 30.0
# per-call budget — a tool call shouldn't outlive the agent's own timeouts
_CALL_TIMEOUT = 30.0
# JSON-schema types the MCP server may declare, mapped for inspect signatures
# (so _arg_spec in agent.py builds real function schemas for these tools)
_TYPE_MAP = {"string": str, "integer": int, "number": float,
             "boolean": bool}


def _content_to_text(result):
    """MCP returns a list of content blocks — the model just wants text."""
    parts = [c.text for c in (result.content or [])
             if getattr(c, "type", None) == "text" and c.text]
    if parts:
        return "\n".join(parts)
    if getattr(result, "isError", False):
        return "ERROR: MCP tool returned an error with no text"
    return str(result.content or "")


def _signature_from_schema(schema):
    """Give the wrapped func a real signature from the MCP inputSchema,
    so native function-calling backends see actual args, not **kwargs."""
    schema = schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    # required first — inspect refuses a defaulted param before a bare one
    for name in sorted(props, key=lambda n: n not in required):
        ptype = _TYPE_MAP.get(props[name].get("type"), str)
        default = (inspect.Parameter.empty if name in required else None)
        params.append(inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default, annotation=ptype))
    return inspect.Signature(params)


def _wrap_tool(conn, mcp_tool):
    """One MCP tool -> one Tool the ReAct loop already understands."""
    ann = mcp_tool.annotations
    destructive = bool(getattr(ann, "destructiveHint", False))
    # read-only tools are the safe default; only an explicit destructive
    # hint opts a tool into the approval gate
    desc = (mcp_tool.description or "").strip() or \
        f"MCP tool '{mcp_tool.name}' (no description from server)"

    def func(**kwargs):
        return conn.call_tool(mcp_tool.name, kwargs)

    func.__name__ = mcp_tool.name
    func.__doc__ = desc
    func.__signature__ = _signature_from_schema(mcp_tool.input_schema)
    return Tool(name=mcp_tool.name, description=desc, func=func,
                timeout=_CALL_TIMEOUT, destructive=destructive)


class MCPConnection:
    """A live MCP server connection. Owns the subprocess, the session,
    and the background loop; .tools is the wrapped list for the agent."""

    def __init__(self, command, args=None):
        self.command, self.args = command, list(args or [])
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        daemon=True,
                                        name="mcp-loop")
        self._thread.start()
        try:
            fut = asyncio.run_coroutine_threadsafe(self._open(), self._loop)
            fut.result(timeout=_CONNECT_TIMEOUT)
        except Exception:
            self.close()
            raise

    async def _open(self):
        params = StdioServerParameters(command=self.command,
                                       args=self.args)
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write))
        await self._session.initialize()
        # v2 session: list_tools()/call_tool() live on the session itself
        listed = await self._session.list_tools()
        self.tools = [_wrap_tool(self, t) for t in listed.tools]

    def call_tool(self, name, args):
        """Sync call — runs the async tools/call on the background loop."""
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, args or {}), self._loop)
        result = fut.result(timeout=_CALL_TIMEOUT)
        text = _content_to_text(result)
        if getattr(result, "isError", False) and not text.startswith("ERROR"):
            return f"ERROR: MCP tool '{name}' failed: {text}"
        return text

    def close(self):
        stack = getattr(self, "_stack", None)
        if stack is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(stack.aclose(),
                                                       self._loop)
                fut.result(timeout=10)
            except Exception:
                pass  # teardown is best-effort; the thread is a daemon
            self._stack = None
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def connect_mcp(command, args=None):
    """Connect one MCP server over stdio; returns an MCPConnection whose
    .tools plug straight into ReActAgent(tools=TOOLS + conn.tools)."""
    return MCPConnection(command, args)


if __name__ == "__main__":
    # quick manual check: list the sample server's tools, call the clock
    import os
    import sys

    server = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "examples", "mcp_server.py")
    with connect_mcp(sys.executable, [server]) as conn:
        for t in conn.tools:
            print(f"tool: {t.name}  destructive={t.destructive}")
            print(f"  {t.description}")
        print("call:", conn.tools[0].func())
