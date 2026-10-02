"""Sample MCP server for the offline demo: one real tool over stdio.

The client (mcp_client.py) spawns this itself — `python
mcp_client.py` runs the whole handshake end to end with no network.
Needs: pip install mcp
"""
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer  # mcp 2.x renamed FastMCP

server = MCPServer("demo-clock")


@server.tool()
def get_utc_time() -> str:
    """Get the current UTC time from this server's clock. Args: none —
    takes no arguments. Returns an ISO-8601 UTC timestamp string, e.g.
    '2026-10-02T03:00:00+00:00'. Never fails."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    server.run()  # stdio transport — the client talks to stdin/stdout
