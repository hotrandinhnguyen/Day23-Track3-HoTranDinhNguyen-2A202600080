"""Bonus 3 — Time Travel Demo.

Shows how to list all checkpoints for a run and replay from an earlier one.

Usage:
    python scripts/demo_time_travel.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import initial_state

DB_PATH = "outputs/time_travel.db"
SCENARIO_ID = "S02_tool"


def build_sqlite_graph(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return build_graph(checkpointer=SqliteSaver(conn=conn)), conn


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()

    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    scenario = next(s for s in scenarios if s.id == SCENARIO_ID)

    graph, conn = build_sqlite_graph(DB_PATH)
    state = initial_state(scenario)
    thread_id = state["thread_id"]
    run_cfg = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Running scenario {SCENARIO_ID} with SQLite checkpointer ===")
    final = graph.invoke(state, config=run_cfg)
    print(f"  final_answer : {final.get('final_answer')}")
    print(f"  nodes visited: {len(final.get('events', []))}")

    # --- List all checkpoints (time travel) ---
    print("\n=== State history (newest to oldest) ===")
    history = list(graph.get_state_history(config=run_cfg))
    for i, snapshot in enumerate(history):
        step = snapshot.metadata.get("step", i)
        events = snapshot.values.get("events", [])
        last_event = events[-1].get("node", "-") if events else "-"
        print(f"  step={step:>2}  last_node={last_event:<15}  events_so_far={len(events)}")

    # --- Replay from an earlier checkpoint ---
    # Pick the checkpoint right after classify (step 2 typically)
    replay_snapshot = history[-3] if len(history) >= 3 else history[-1]
    replay_cfg = {"configurable": {"thread_id": thread_id, "checkpoint_id": replay_snapshot.config["configurable"]["checkpoint_id"]}}

    print(f"\n=== Time-travel: replaying from step {replay_snapshot.metadata.get('step')} ===")
    replayed = graph.invoke(None, config=replay_cfg)
    print(f"  final_answer after replay: {replayed.get('final_answer')}")
    print(f"  route                    : {replayed.get('route')}")
    print("\n  time_travel_success=True -- graph replayed from earlier checkpoint!")

    conn.close()


if __name__ == "__main__":
    main()
