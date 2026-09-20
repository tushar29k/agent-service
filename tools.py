"""Tools: the agent's hands. Each tool has a name, a docstring-as-UX, a timeout,
and a destructive flag (destructive tools pause for human approval).

Tool design rules (see guide 08, Part 3):
- docstring = when to use + args + return shape + failure modes
- bounded outputs, machine-readable errors, every tool gets a timeout
"""
import ast
import operator
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass

# ---------------------------------------------------------------- knowledge
KNOWLEDGE = {
    "refund": ("Refund policy: full refunds are available within 30 days of "
               "purchase for unused products in original packaging."),
    "warranty": ("Warranty: all electronics carry a 2 year manufacturer "
                 "warranty covering manufacturing defects."),
    "support": ("Support: email support@example.com with your order ID. "
                "Response within 1 business day."),
}


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Args: expression, e.g. '12*13'.
    Returns the result as a string, or ERROR: ... on bad input."""
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
    """Search the company knowledge base. Args: query — keywords.
    Returns the best matching entry, or NO_RESULTS — never invent content."""
    toks = set(query.lower().split())
    best, best_score = "NO_RESULTS", 0
    for key, text in KNOWLEDGE.items():
        score = len(toks & set(text.lower().split()))
        if key in query.lower():
            score += 3                      # topic keyword = strong signal
        if score > best_score:
            best, best_score = text, score
    return best


def issue_refund(order_id: str) -> str:
    """Issue a refund for an order. Args: order_id. DESTRUCTIVE — moves money."""
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
    Tool("issue_refund", issue_refund.__doc__, issue_refund,
         destructive=True),                 # approval gate in agent.py
]


class ToolNode:
    """Executes tool calls with timeouts and machine-readable errors."""

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
    assert node.run("calculator", {"expression": "__import__('os')" }).startswith("ERROR")
    print("tools OK")
