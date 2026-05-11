import json

data = json.load(open("outputs/metrics.json"))
fail = 0
for s in data["scenario_metrics"]:
    status = "PASS" if s["success"] else "FAIL"
    if not s["success"]:
        fail += 1
    print(f'[{status}] {s["scenario_id"]:<25} expected={s["expected_route"]:<12} actual={s["actual_route"]}')

print(f"\n{len(data['scenario_metrics']) - fail}/{len(data['scenario_metrics'])} pass")
