"""Per-step tracing: where does the time actually go?

Every run records one record per step — how long each think() and each
tool call took — so the slowest step is measured, not guessed. The sink
is env-key optional:

  no key (default)      traces land in logs/traces.jsonl, one JSON line
                        per step (AGENT_TRACE_LOG overrides the path;
                        empty string disables file logging)
  LANGSMITH_API_KEY set also pushes the run to LangSmith on finish —
                        best-effort: a failed push logs to stderr and the
                        local file stays the source of truth

Wire-in is three lines in the loop: start_run() per task, the step()
context manager around think() and tool_node.run(), finish_run() in
_final. python3 slowest_step.py reads the log and names the bottleneck.

python3 tracing.py runs the self-demo: a 3-step mock run + the finder
output over the fresh traces.
"""
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


def trace_log_path():
    """Where step records go: env override, else logs/ next to the repo.
    empty string disables file logging entirely."""
    return os.environ.get("AGENT_TRACE_LOG", "logs/traces.jsonl")


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


class Tracer:
    """Per-run step timer. One Tracer per agent instance; a run opens on
    start_run(), each think()/tool call is timed with step(), and
    finish_run() closes it with a summary record. record_step() lazily
    reopens after an approval-gate resume, so approve() needs no code."""

    def __init__(self):
        # tracing never breaks a run: a flag off, a bad path, or a dead
        # langsmith endpoint all degrade to "no traces", never an error
        self.enabled = os.environ.get("AGENT_TRACING", "").lower() not in (
            "off", "0", "false", "no")
        self.log_path = trace_log_path()
        self.langsmith_key = os.environ.get("LANGSMITH_API_KEY")
        self._run = None
        self._run_t0 = 0.0
        self._steps = 0
        self._total_ms = 0.0

    def start_run(self, thread_id, task, backend_name):
        self._run = {"run_id": uuid.uuid4().hex[:8],
                     "thread_id": str(thread_id),
                     "task": str(task)[:200],
                     "backend": str(backend_name),
                     "started": _utcnow()}
        self._run_t0 = time.perf_counter()
        self._steps = 0
        self._total_ms = 0.0
        return self._run["run_id"]

    @contextmanager
    def step(self, kind, tool=None, step=None, extra=None):
        """Time one think() or tool call; records on exit, re-raises."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record_step(kind, (time.perf_counter() - t0) * 1000,
                             tool=tool, step=step, extra=extra)

    def record_step(self, kind, duration_ms, tool=None, step=None,
                    extra=None):
        if not self.enabled:
            return None
        if self._run is None:
            # approval-gate resume: finish_run() closed the run when the
            # gate hit; reopen lazily so the resumed steps still trace
            self.start_run("resumed", "", "unknown")
        rec = {"ts": _utcnow(), "type": "step",
               "run_id": self._run["run_id"],
               "thread_id": self._run["thread_id"],
               "backend": self._run["backend"],
               "step": step, "kind": kind,
               "tool": tool, "duration_ms": round(duration_ms, 3)}
        if extra:
            rec["extra"] = extra
        self._steps += 1
        self._total_ms += duration_ms
        self._write(rec)
        return rec

    def finish_run(self, outcome):
        if not self.enabled or self._run is None:
            return None
        rec = {"ts": _utcnow(), "type": "run",
               "run_id": self._run["run_id"],
               "thread_id": self._run["thread_id"],
               "backend": self._run["backend"],
               "task": self._run["task"],
               "started": self._run["started"],
               "outcome": outcome, "steps": self._steps,
               "total_ms": round(self._total_ms, 3)}
        self._write(rec)
        self._push_langsmith(rec)
        self._run = None
        return rec

    def _write(self, rec):
        if not self.log_path:
            return
        d = os.path.dirname(self.log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _push_langsmith(self, run_rec):
        """Best-effort: also ship the run to LangSmith when a key is set.
        Any failure (network, key, API) is a stderr note, never an error."""
        if not self.langsmith_key:
            return
        try:
            import urllib.request
            run = {
                "id": str(uuid.uuid4()),
                "name": f"agent-service/{run_rec['thread_id']}",
                "run_type": "chain",
                "start_time": run_rec["started"],
                "end_time": _utcnow(),
                "inputs": {"task": run_rec["task"]},
                "outputs": {"outcome": run_rec["outcome"],
                            "steps": run_rec["steps"]},
                "extra": {"total_ms": run_rec["total_ms"],
                          "backend": run_rec["backend"]},
            }
            body = json.dumps({"post": [run]}).encode()
            req = urllib.request.Request(
                "https://api.smith.langchain.com/runs/batch", data=body,
                headers={"x-api-key": self.langsmith_key,
                         "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            # the local file is the trace; langsmith is a bonus
            print(f"tracing: langsmith push failed ({e}) — local file "
                  f"only", file=sys.stderr)


if __name__ == "__main__":
    # self-demo: a 3-step mock run traces per-step timings, then the
    # finder names the slowest step — the "done when" in one command
    import sys
    sys.path.insert(0, ".")
    from agent import MockBackend, ReActAgent

    tr = Tracer()
    tr.start_run("trace-demo", "What is the warranty in months?", "mock")
    for kind, tool, pause in (("think", None, 0.01),
                             ("tool", "search_docs", 0.02),
                             ("think", None, 0.01),
                             ("tool", "calculator", 0.03),
                             ("think", None, 0.01)):
        t0 = time.perf_counter()
        time.sleep(pause)  # stand-in for the real think()/tool call
        tr.record_step(kind, (time.perf_counter() - t0) * 1000,
                       tool=tool, step=1)
    tr.finish_run("answered")
    print(f"traced a run -> {trace_log_path()}")
    print("finder output:")
    from slowest_step import main as slowest_main
    raise SystemExit(slowest_main())
