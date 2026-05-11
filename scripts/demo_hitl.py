"""Bonus 4 -- Real HITL (Human-in-the-Loop) Demo.

Demonstrates LangGraph interrupt() / resume flow without Streamlit:
  1. Graph runs until approval_node calls interrupt() and pauses
  2. Human decision is injected via graph.invoke(Command(resume=...))
  3. Graph resumes from the interrupted state and completes

Usage:
    python scripts/demo_hitl.py
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import initial_state

DB_PATH = "outputs/hitl.db"
SCENARIO_ID = "S04_risky"


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()

    # Enable real interrupt() in approval_node
    os.environ["LANGGRAPH_INTERRUPT"] = "true"

    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    scenario = next(s for s in scenarios if s.id == SCENARIO_ID)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    graph = build_graph(checkpointer=SqliteSaver(conn=conn))

    state = initial_state(scenario)
    thread_id = state["thread_id"]
    run_cfg = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== HITL Demo — scenario: {SCENARIO_ID} ===")
    print(f"  query: {scenario.query}")

    # --- Phase 1: run until interrupt ---
    print("\n[Phase 1] Running graph until interrupt at approval_node...")
    result = graph.invoke(state, config=run_cfg)

    # When interrupt() fires, invoke() returns with an Interrupt in the result
    pending = graph.get_state(config=run_cfg)
    interrupts = pending.tasks[0].interrupts if pending.tasks else []

    if interrupts:
        interrupt_value = interrupts[0].value
        print(f"  [PAUSED] Graph interrupted at approval_node")
        print(f"  proposed_action : {interrupt_value.get('proposed_action', '')[:80]}")
        print(f"  risk_level      : {interrupt_value.get('risk_level')}")

        # --- Phase 2: human provides decision ---
        human_decision = {"approved": True, "reviewer": "demo-human", "comment": "Approved via HITL demo"}
        print(f"\n[Phase 2] Human reviews and approves...")
        print(f"  decision: {human_decision}")

        # Resume graph by injecting the human decision
        final = graph.invoke(Command(resume=human_decision), config=run_cfg)
        print(f"\n[Phase 3] Graph resumed and completed")
        print(f"  route       : {final.get('route')}")
        print(f"  final_answer: {final.get('final_answer')}")
        approval = final.get("approval", {})
        print(f"  approval    : reviewer={approval.get('reviewer')}  approved={approval.get('approved')}")
        print("\n  hitl_success=True -- interrupt/resume completed!")
    else:
        print("  [INFO] No interrupt fired (mock mode active?). Check LANGGRAPH_INTERRUPT env var.")
        print(f"  final_answer: {result.get('final_answer')}")

    conn.close()
    # Reset env so other scripts are not affected
    os.environ.pop("LANGGRAPH_INTERRUPT", None)


if __name__ == "__main__":
    main()
