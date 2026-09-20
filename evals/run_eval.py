"""Agent eval runner: task success + right tools + approval gates + budgets.

    python3 evals/run_eval.py
"""
import sys

import yaml

sys.path.insert(0, ".")
from agent import ReActAgent


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
    ok_tools = all(t in used_tools for t in task.get("must_use_tools", []))
    ok_answer = all(s.lower() in final["answer"].lower()
                    for s in task.get("expected_contains", []))
    ok_steps = final["steps"] <= task["max_steps"]
    return ok_tools and ok_answer and ok_steps, used_tools, final


def main():
    tasks = yaml.safe_load(open("evals/tasks.yaml"))["tasks"]
    passed = 0
    print(f"{'task':24s} {'pass':4s} tools_used")
    for task in tasks:
        agent = ReActAgent(checkpoint_dir="/tmp/agent_eval_ckpt")
        ok, used, final = run_task(agent, task)
        passed += ok
        print(f"{task['id']:24s} {str(ok):4s} {used} "
              f"(steps {final['steps']}, ~{final['est_tokens']} tok)")
        if not ok:
            print(f"   final answer: {final['answer'][:120]}")
    print(f"\n{passed}/{len(tasks)} tasks passed")


if __name__ == "__main__":
    main()
