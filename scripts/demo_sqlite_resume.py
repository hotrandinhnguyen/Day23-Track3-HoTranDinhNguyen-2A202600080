"""Bonus 2 — SQLite Crash-Resume Demo.

Demonstrates that LangGraph state survives process interruption:
  Run 1: invoke graph with SQLite checkpointer, capture thread_id
  Run 2: create a NEW graph instance with the SAME SQLite file + thread_id
          and call get_state() to prove the checkpoint was persisted.

Usage:
    python scripts/demo_sqlite_resume.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import initial_state

DB_PATH = "outputs/crash_resume.db"
SCENARIO_ID = "S01_simple"


def build_sqlite_graph(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    checkpointer = SqliteSaver(conn=conn)
    return build_graph(checkpointer=checkpointer), conn


def run1_simulate_first_run() -> str:
    """Simulate first run — invoke graph and save checkpoint to SQLite."""
    print("\n=== RUN 1: First execution (simulating normal run) ===")
    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    scenario = next(s for s in scenarios if s.id == SCENARIO_ID)

    graph, conn = build_sqlite_graph(DB_PATH)
    state = initial_state(scenario)
    thread_id = state["thread_id"]
    run_cfg = {"configurable": {"thread_id": thread_id}}

    final = graph.invoke(state, config=run_cfg)
    print(f"  thread_id   : {thread_id}")
    print(f"  route       : {final.get('route')}")
    print(f"  final_answer: {final.get('final_answer')}")
    print(f"  nodes visited: {len(final.get('events', []))}")
    print(f"  [Checkpoint written to {DB_PATH}]")
    conn.close()
    return thread_id


def run2_simulate_resume(thread_id: str) -> None:
    """Simulate resume — create brand-new graph instance, read state from SQLite."""
    print("\n=== RUN 2: New process instance — reading checkpoint from SQLite ===")
    print(f"  [Opening existing DB: {DB_PATH}]")

    graph, conn = build_sqlite_graph(DB_PATH)
    run_cfg = {"configurable": {"thread_id": thread_id}}

    # get_state() loads the latest checkpoint for this thread_id
    saved = graph.get_state(config=run_cfg)

    if saved and saved.values:
        values = saved.values
        print(f"  thread_id   : {thread_id}")
        print(f"  route       : {values.get('route')}")
        print(f"  final_answer: {values.get('final_answer')}")
        print(f"  events count: {len(values.get('events', []))}")
        print(f"\n  resume_success=True — state survived simulated process kill!")
    else:
        print("  ERROR: No checkpoint found for this thread_id")

    conn.close()


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    # Remove old DB so demo is clean
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()

    thread_id = run1_simulate_first_run()

    print("\n  ~~~ Simulating process kill (new Python instance) ~~~")

    run2_simulate_resume(thread_id)
    print()


if __name__ == "__main__":
    main()
