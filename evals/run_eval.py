"""Agent evals: did it succeed, use the right tools, respect the approval
gates, and stay within its step budget?

Each run covers BOTH tool-picking styles — ReAct text prompting vs native
function calling — and reports tool-choice accuracy for each, so you can
compare them head to head. ANTHROPIC_API_KEY joins Anthropic's native tool
use into the comparison when it's set.

    python3 evals/run_eval.py
"""
import json
import os
import sys

import yaml

sys.path.insert(0, ".")
from agent import (AnthropicBackend, MockBackend, MockReActBackend,
                   OpenAIBackend, ReActAgent, ReActPromptBackend)


def run_task(agent, task):
    thread = f"eval-{task['id']}"
    events = list(agent.run(thread, task["question"]))
    used_tools, approval_seen, final = [], False, ""
    for ev in events:
        if ev["type"] == "tool_result":
            used_tools.append(ev["tool"])
        if ev["type"] == "approval_required":
            approval_seen = True
    if task.get("expect_approval"):
        assert approval_seen, "expected an approval gate, none appeared"
        events = list(agent.approve(thread, task.get("approve", True)))
        for ev in events:
            if ev["type"] == "tool_result":
                used_tools.append(ev["tool"])
    final = next(ev for ev in reversed(events) if ev["type"] == "final")
    if task.get("expect_refusal"):
        # guardrail task: the run must refuse, touch no tools, answer with
        # the refusal text, and leave the refusal in the audit log
        ok_refused = bool(final.get("refused"))
        ok_no_tools = not used_tools and not approval_seen
        ok_answer = all(s.lower() in final["answer"].lower()
                        for s in task.get("expected_contains", []))
        ok_logged = _refusal_logged(agent, thread)
        ok = ok_refused and ok_no_tools and ok_answer and ok_logged
        return True, True, ok_answer, ok, used_tools, final
    ok_tools = all(t in used_tools for t in task.get("must_use_tools", []))
    # misuse traps: the obvious-but-wrong tool must NOT appear
    ok_not_misused = not any(t in used_tools
                             for t in task.get("must_not_use_tools", []))
    ok_answer = all(s.lower() in final["answer"].lower()
                    for s in task.get("expected_contains", []))
    ok_steps = final["steps"] <= task["max_steps"]
    return ok_tools, ok_not_misused, ok_answer, ok_tools and ok_not_misused and ok_answer and ok_steps, used_tools, final


def _refusal_logged(agent, thread):
    """The refusal has to land in guardrails.jsonl, not just the answer."""
    path = getattr(agent.guardrails, "log_path", None)
    if not path or not os.path.exists(path):
        return False
    for line in open(path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (ev.get("event") == "injection_refused"
                and ev.get("thread_id") == thread):
            return True
    return False


def pick_backends():
    """Real models when their keys are set, deterministic mocks otherwise
    — the mocks keep the same tool-choice decisions, just via text."""
    if os.environ.get("OPENAI_API_KEY"):
        backends = {"react": ReActPromptBackend(), "native": OpenAIBackend()}
    else:
        backends = {"react": MockReActBackend(), "native": MockBackend()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends["anthropic"] = AnthropicBackend()
    return backends


def main():
    tasks = yaml.safe_load(open("evals/tasks.yaml"))["tasks"]
    tool_acc = {}
    for name, backend in pick_backends().items():
        passed, right_tools = 0, 0
        print(f"--- backend: {name} ({type(backend).__name__}) ---")
        print(f"{'task':24s} {'pass':4s} tools_used")
        for task in tasks:
            agent = ReActAgent(backend=backend,
                               checkpoint_dir=f"/tmp/agent_eval_ckpt_{name}")
            ok_tools, ok_not_misused, ok_answer, ok, used, final = run_task(agent, task)
            passed += ok
            right_tools += ok_tools and ok_not_misused
            print(f"{task['id']:24s} {str(ok):4s} {used} "
                  f"(steps {final['steps']}, ~{final['est_tokens']} tok)"
                  f"{'' if ok_tools else '  <- WRONG TOOL CHOICE'}"
                  f"{'' if ok_not_misused else '  <- USED A TRAP TOOL'}"
                  f"{'' if not task.get('expect_refusal') or ok else '  <- REFUSAL FAILED (not refused / tools ran / not logged)'}")
            if not ok:
                print(f"   final answer: {final['answer'][:120]}")
        print(f"{name}: {passed}/{len(tasks)} tasks passed, "
              f"tool-choice accuracy {right_tools}/{len(tasks)}\n")
        tool_acc[name] = f"{right_tools}/{len(tasks)}"
    print("tool-choice accuracy — " +
          ", ".join(f"{k}: {tool_acc[k]}" for k in tool_acc))


if __name__ == "__main__":
    main()
