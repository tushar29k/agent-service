"""Cost cap proof: a runaway task is stopped by budget, not by steps.

A brain that never answers and always acts (fresh args each time, so the
loop detector can't save it) must trip the cost fuse with max_steps
untouched — a `cost_exceeded` event fires, and the final line says
"cost cap", not "max steps".  python3 test_cost_cap.py
"""
import os
import sys

sys.path.insert(0, ".")
from agent import MockBackend, ReActAgent


class RunawayBrain(MockBackend):
    def __init__(self):
        self.n = 0

    def think(self, messages, tools):
        self.n += 1
        return {"thought": f"still working, step {self.n}",
                "action": {"name": "calculator",
                           "args": {"expression": f"{self.n}+{self.n}"}},
                "answer": None}


def main():
    # 1. runaway run: budget 1e-9, max_steps generous
    agent = ReActAgent(backend=RunawayBrain(), max_steps=50, max_cost=1e-9,
                       checkpoint_dir="/tmp/agent_cost_eval")
    events = list(agent.run("cost-demo", "keep calculating forever"))
    killed = [ev for ev in events if ev["type"] == "cost_exceeded"]
    final = next(ev for ev in reversed(events) if ev["type"] == "final")
    assert len(killed) == 1, f"expected one cost_exceeded event, got {len(killed)}"
    assert final["steps"] < 50, "max_steps stopped it — not a cost kill"
    assert "cost cap" in final["answer"], f"wrong stop line: {final['answer']}"
    assert final["est_cost_usd"] > 0, "final must report the spend"
    print(f"1. runaway killed by cost after {final['steps']} steps "
          f"(max_steps 50 untouched): {final['answer']} "
          f"spent=${final['est_cost_usd']:.6f} cap=${killed[0]['cap_usd']}")

    # 2. normal run under a sane cap: no kill-switch, budget reported
    agent2 = ReActAgent(backend=MockBackend(), max_cost=0.10,
                        checkpoint_dir="/tmp/agent_cost_eval")
    events2 = list(agent2.run("cost-normal", "What is 12*30?"))
    final2 = next(ev for ev in reversed(events2) if ev["type"] == "final")
    assert not any(ev["type"] == "cost_exceeded" for ev in events2), \
        "normal run must not trip the fuse"
    assert final2["est_cost_usd"] < 0.10, "normal run must stay in budget"
    print(f"2. normal run unharmed: {final2['answer'][:40]} "
          f"spent=${final2['est_cost_usd']:.6f}")

    # 3. env config honored
    os.environ["AGENT_MAX_COST"] = "0.02"
    agent3 = ReActAgent(backend=MockBackend(),
                        checkpoint_dir="/tmp/agent_cost_eval")
    assert agent3.cost.cap_usd == 0.02, "AGENT_MAX_COST not picked up"
    del os.environ["AGENT_MAX_COST"]
    print("3. AGENT_MAX_COST env honored: cap $0.02")
    print("cost cap ok — kill-switch proven, normal runs untouched")


if __name__ == "__main__":
    main()
