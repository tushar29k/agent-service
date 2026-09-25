"""The agent's hands.

Each tool has a name, a docstring (which doubles as what the model sees when
picking tools — so write it like you'd explain it to a new hire), a timeout,
and a destructive flag. Destructive tools pause for human approval first.

Rules of thumb (from guide 08, part 3):
- docstring = when to use + args + return shape + failure modes
- keep outputs small, errors machine-readable, and never let a tool run
  without a timeout
"""
import ast
import operator
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass

# tiny fake knowledge base for the search_docs demo
KNOWLEDGE = {
    "refund": ("Refund policy: full refunds are available within 30 days of "
               "purchase for unused products in original packaging."),
    "warranty": ("Warranty: all electronics carry a 2 year manufacturer "
                 "warranty covering manufacturing defects."),
    "support": ("Support: email support@example.com with your order ID. "
                "Response within 1 business day."),
}


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '12*13'. Args: expression.
    Returns the result as a string, or ERROR: ... on bad input.
    Safe by construction — it walks the AST, never evals."""
    allowed = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.Pow: operator.pow, ast.Mod: operator.mod}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return _eval(node.operand) if isinstance(node.op, ast.UAdd) else -_eval(node.operand)
        raise ValueError("only numbers and + - * / ** % allowed")

    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as e:
        return f"ERROR: bad expression ({e})"


def search_docs(query: str) -> str:
    """Search the company knowledge base. Args: query — just keywords.
    Returns the best matching entry, or NO_RESULTS — never invent content."""
    toks = set(query.lower().split())
    best, best_score = "NO_RESULTS", 0
    for key, text in KNOWLEDGE.items():
        score = len(toks & set(text.lower().split()))
        if key in query.lower():
            score += 3                      # an exact topic keyword counts triple
        if score > best_score:
            best, best_score = text, score
    return best


# canned web index for the demo — nothing here touches the network.
# SWAP: when SEARCH_API_KEY is set, call a real search API here
# (Tavily / Brave / Serper) and only fall back to this index in tests.
WEB_INDEX = {
    "t20": ("2026 T20 World Cup final",
            "India beat South Africa by 7 runs in a last-over thriller "
            "at the Wankhede."),
    "weather": ("Bengaluru weather today",
                "28°C, partly cloudy — evening showers likely after 6pm."),
    "python": ("Python 3.14 released",
               "Faster startup, deferred annotation evaluation, "
               "and new template strings."),
}


def web_search(query: str) -> str:
    """Search the live web for current or external info the knowledge base
    can't cover. Args: query — keywords or a plain question.
    Returns the top result as 'title — snippet', or NO_RESULTS.
    Mock data for now — the real API hooks in where the SWAP comment says."""
    toks = set(query.lower().split())
    best, best_score = "NO_RESULTS", 0
    for key, (title, snippet) in WEB_INDEX.items():
        score = len(toks & set(snippet.lower().split()))
        if key in query.lower():
            score += 3                      # exact topic keyword counts triple
        if score > best_score:
            best, best_score = f"{title} — {snippet}", score
    return best


def issue_refund(order_id: str) -> str:
    """Issue a refund for an order. Args: order_id. DESTRUCTIVE — this one
    moves money, so the agent stops for human approval first."""
    if not re.fullmatch(r"\w+", order_id):
        return "ERROR: invalid order id"
    return f"Refund issued for order {order_id}."


@dataclass
class Tool:
    name: str
    description: str
    func: callable
    timeout: float = 10.0
    destructive: bool = False


TOOLS = [
    Tool("calculator", calculator.__doc__, calculator),
    Tool("search_docs", search_docs.__doc__, search_docs),
    Tool("web_search", web_search.__doc__, web_search),
    Tool("issue_refund", issue_refund.__doc__, issue_refund,
         destructive=True),                 # the approval gate itself lives in agent.py
]


class ToolNode:
    """Runs tool calls with a timeout. If a tool blows up, the agent gets an
    error string back — never a crashed process."""

    def __init__(self, tools):
        self.by_name = {t.name: t for t in tools}

    def run(self, name, args):
        tool = self.by_name[name]
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(tool.func, **args)
            try:
                return fut.result(timeout=tool.timeout)
            except FuturesTimeout:
                return f"ERROR: tool '{name}' timed out after {tool.timeout}s"
            except Exception as e:          # never let a tool crash the agent
                return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    node = ToolNode(TOOLS)
    assert node.run("calculator", {"expression": "12*13"}) == "156"
    assert "30 days" in node.run("search_docs", {"query": "refund window"})
    assert node.run("search_docs", {"query": "ceo favourite colour"}) == "NO_RESULTS"
    assert "12345" in node.run("issue_refund", {"order_id": "12345"})
    assert "7 runs" in node.run("web_search",
                               {"query": "latest news t20 world cup 2026 final"})
    assert node.run("web_search",
                    {"query": "quantum teleportation futures"}) == "NO_RESULTS"
    assert node.run("calculator", {"expression": "__import__('os')" }).startswith("ERROR")
    print("tools OK")
