"""Run hidden grading scenarios and output metrics."""
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import initial_state
from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics, write_metrics

SCENARIOS_PATH = "data/sample/scenarios_hidden.jsonl"
OUTPUT_PATH    = "outputs/metrics.json"

scenarios = load_scenarios(SCENARIOS_PATH)
graph     = build_graph(build_checkpointer("memory"))
metrics   = []

for s in scenarios:
    state  = initial_state(s)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    m      = metric_from_state(result, s.expected_route.value, s.requires_approval)
    metrics.append(m)
    status = "PASS" if m.success else "FAIL"
    print(f"[{status}] {s.id:<20} expected={s.expected_route.value:<12} actual={result.get('route')}")

report = summarize_metrics(metrics)
write_metrics(report, OUTPUT_PATH)
print(f"\nsuccess_rate = {report.success_rate:.0%}  ({sum(m.success for m in metrics)}/{len(metrics)})")
