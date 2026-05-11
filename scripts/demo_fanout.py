"""Bonus 5 -- Parallel Fan-out Demo using Send() API.

Builds a separate mini-graph that fans out to two mock tools in parallel,
then merges their results via the append (add) reducer.

Graph structure:
    START -> dispatch --(Send)--> tool_a \
                      --(Send)--> tool_b  --> merge -> END

Usage:
    python scripts/demo_fanout.py
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# --- State ---

class FanoutState(TypedDict, total=False):
    query: str
    tool_results: Annotated[list[str], add]
    final_answer: str


# --- Routing: returns list[Send] to trigger parallel execution ---

def fan_out(state: FanoutState) -> list[Send]:
    """Dispatch the same query to tool_a and tool_b in parallel."""
    query = state.get("query", "")
    return [
        Send("tool_a", {"query": query}),
        Send("tool_b", {"query": query}),
    ]


# --- Worker nodes (each receives its own state from Send) ---

def tool_a_node(state: dict) -> dict:
    """Mock tool A: order database lookup."""
    return {"tool_results": [f"[Tool A] Order DB result for: '{state.get('query', '')}'"]}


def tool_b_node(state: dict) -> dict:
    """Mock tool B: inventory system lookup."""
    return {"tool_results": [f"[Tool B] Inventory result for: '{state.get('query', '')}'"]}


# --- Merge node: receives combined state after both tools finish ---

def merge_node(state: FanoutState) -> dict:
    results = state.get("tool_results", [])
    answer = "Combined results:\n" + "\n".join(f"  - {r}" for r in results)
    return {"final_answer": answer}


# --- Build graph ---

def build_fanout_graph():
    graph = StateGraph(FanoutState)
    graph.add_node("tool_a", tool_a_node)
    graph.add_node("tool_b", tool_b_node)
    graph.add_node("merge", merge_node)

    # fan_out returns list[Send] -> triggers parallel execution
    graph.add_conditional_edges(START, fan_out, ["tool_a", "tool_b"])
    graph.add_edge("tool_a", "merge")
    graph.add_edge("tool_b", "merge")
    graph.add_edge("merge", END)
    return graph.compile()


def main() -> None:
    graph = build_fanout_graph()
    query = "lookup order 12345"

    print("\n=== Parallel Fan-out Demo ===")
    print(f"  query: {query}")
    print("  [Dispatching to tool_a and tool_b in parallel via Send() ...]")

    result = graph.invoke({"query": query, "tool_results": [], "final_answer": ""})

    print(f"\n  tool_results ({len(result['tool_results'])} items, merged via add reducer):")
    for r in result["tool_results"]:
        print(f"    {r}")
    print(f"\n  final_answer:\n    {result['final_answer']}")
    print("\n  fanout_success=True -- two tools ran in parallel and results merged!")


if __name__ == "__main__":
    main()
