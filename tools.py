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
import base64
import marshal
import operator
import re
import subprocess
import sys
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


# math.* functions/constants the sandbox may touch — no files, no network,
# no processes reachable through any of these
_MATH_FUNCS = frozenset((
    "sqrt", "cbrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "log", "log2", "log10", "exp", "pow", "floor", "ceil",
    "factorial", "gcd", "comb", "perm", "isclose", "hypot",
    "degrees", "radians", "trunc"))
_MATH_CONSTS = frozenset(("pi", "e", "tau", "inf", "nan"))
# plain helpers: arithmetic iteration and printing, nothing privileged
_SAFE_HELPERS = frozenset((
    "len", "sum", "min", "max", "abs", "round", "range",
    "enumerate", "zip", "int", "float", "str", "bool",
    "list", "dict", "tuple", "set", "print"))
# every AST node type allowed through the validator below — anything else
# (imports, defs, lambdas, comprehensions, try/with, ...) is rejected
_SAFE_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.If, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.keyword, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice,
    ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)


def _check_sandbox(tree):
    """Walk the parsed code, reject anything that could escape arithmetic."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder/private names are not allowed")
        if isinstance(node, ast.Attribute):
            # math.sqrt is fine; anything else (x.y = z too) is not
            if not (isinstance(node.value, ast.Name)
                    and node.value.id == "math"
                    and node.attr in _MATH_FUNCS | _MATH_CONSTS
                    and isinstance(node.ctx, ast.Load)):
                raise ValueError(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.Call):
            f = node.func
            math_call = (isinstance(f, ast.Attribute)
                         and isinstance(f.value, ast.Name)
                         and f.value.id == "math" and f.attr in _MATH_FUNCS)
            if not (math_call or (isinstance(f, ast.Name)
                                  and f.id in _SAFE_HELPERS)):
                raise ValueError("only math.* and safe helpers are callable")
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"{type(node).__name__} is not allowed")


# the child runs this: no site imports, a bare-bones __builtins__, and the
# code itself arrives as marshalled bytecode so it can't smuggle new source
_SANDBOX_CHILD = r"""
import base64, marshal, math, sys
code = marshal.loads(base64.b64decode(sys.argv[1]))
safe_builtins = {"len": len, "sum": sum, "min": min, "max": max, "abs": abs,
                 "round": round, "range": range, "enumerate": enumerate,
                 "zip": zip, "int": int, "float": float, "str": str,
                 "bool": bool, "list": list, "dict": dict, "tuple": tuple,
                 "set": set, "print": print}
ns = {"__builtins__": safe_builtins, "math": math}
exec(code, ns)
if "_result" in ns:                 # set when the code ends in an expression
    print(repr(ns["_result"]))
"""


def python_exec(code: str) -> str:
    """Run Python for calculations the calculator can't express: loops,
    math functions (sqrt, sin, log, factorial, ...), multi-step numeric
    work. Args: code — plain statements; the value of the LAST expression
    statement is returned as text (loops and assignments above it are
    fine). Blocked: imports, file/network access, defs, lambdas,
    comprehensions, attribute access outside math.*. Runs with a 5s
    timeout in a locked-down subprocess — returns ERROR: ... on bad code,
    sandbox violations, or timeout, never raises."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"ERROR: syntax error ({e})"
    try:
        _check_sandbox(tree)
    except ValueError as e:
        return f"ERROR: sandbox violation ({e})"
    # a trailing bare expression becomes the return value: `a` -> `_result = a`
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_result", ctx=ast.Store())],
            value=tree.body[-1].value)
        ast.fix_missing_locations(tree)
    payload = base64.b64encode(
        marshal.dumps(compile(tree, "<sandbox>", "exec"))).decode()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _SANDBOX_CHILD, payload],
            capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "ERROR: python_exec timed out after 5s"
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        return f"ERROR: {err[-1] if err else 'sandbox crashed'}"
    out = proc.stdout.strip()
    return out if out else "(no output)"


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
    Tool("python_exec", python_exec.__doc__, python_exec,
         timeout=20.0),              # subprocess + 5s code timeout need headroom
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
    assert node.run("python_exec",
                    {"code": "a, b = 0, 1\nfor _ in range(20):\n"
                             "    a, b = b, a + b\na"}) == "6765"
    assert node.run("python_exec",
                    {"code": "math.sqrt(16) + sum(range(5))"}) == "14.0"
    assert node.run("python_exec",
                    {"code": "import os"}).startswith("ERROR")
    assert node.run("python_exec",
                    {"code": "(1).__class__"}).startswith("ERROR")
    assert node.run("python_exec",
                    {"code": "1/0"}).startswith("ERROR")
    assert node.run("python_exec",
                    {"code": "while True:\n    pass"}).startswith("ERROR")
    print("tools OK")
