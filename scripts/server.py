"""Real-time LangGraph Agent Monitor — FastAPI backend.

Run from project root:
    uvicorn scripts.server:app --host 0.0.0.0 --port 8000 --reload
Or:
    python scripts/server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import threading
import uuid
from pathlib import Path
from queue import Empty, Queue

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["LANGGRAPH_INTERRUPT"] = "true"
os.chdir(ROOT)  # ensure relative paths (data/, outputs/) resolve correctly

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.scenarios import load_scenarios  # noqa: E402
from langgraph_agent_lab.state import initial_state  # noqa: E402

# ── Global graph + scenarios ───────────────────────────────────────────────────
DB_PATH = "outputs/monitor.db"

def _make_graph():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return build_graph(checkpointer=SqliteSaver(conn=conn))

GRAPH = _make_graph()
SCENARIOS = {s.id: s for s in load_scenarios("data/sample/scenarios.jsonl")}

# ── Run state ──────────────────────────────────────────────────────────────────

class RunState:
    def __init__(self, run_id: str, scenario_id: str):
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.queue: Queue = Queue()
        self.decision_event = threading.Event()
        self.decision: dict | None = None
        self.done = False
        self.node_outputs: dict = {}

RUNS: dict[str, RunState] = {}

# ── Graph execution thread ─────────────────────────────────────────────────────

def _serialize(obj) -> object:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)

def _run_graph(rs: RunState) -> None:
    scenario = SCENARIOS[rs.scenario_id]
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": f"monitor-{rs.run_id}"}}

    def _stream(input_) -> None:
        for chunk in GRAPH.stream(input_, config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "__interrupt__":
                    continue
                rs.node_outputs[node_name] = node_output
                rs.queue.put({
                    "type": "node_done",
                    "node": node_name,
                    "output": _serialize(node_output),
                })

    try:
        _stream(state)

        # Check for HITL interrupt
        graph_state = GRAPH.get_state(config)
        interrupts = graph_state.tasks[0].interrupts if graph_state.tasks else []

        if interrupts:
            payload = interrupts[0].value
            rs.queue.put({
                "type": "interrupt",
                "node": "approval",
                "payload": _serialize(payload),
            })
            rs.decision_event.wait(timeout=300)
            if rs.decision:
                _stream(Command(resume=rs.decision))

        # Final state
        final = GRAPH.get_state(config).values
        rs.queue.put({
            "type": "done",
            "route": final.get("route", ""),
            "final_answer": final.get("final_answer") or final.get("pending_question") or "",
            "events": _serialize(final.get("events", [])),
        })
    except Exception as exc:
        rs.queue.put({"type": "error", "message": str(exc)})
    finally:
        rs.done = True

# ── Time Travel & Crash Recover helpers ───────────────────────────────────────

def _get_history(run_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": f"monitor-{run_id}"}}
    history = list(GRAPH.get_state_history(config))
    result = []
    for snap in reversed(history):
        step = snap.metadata.get("step", -1)
        if step < 0:
            continue
        events = snap.values.get("events") or []
        last_node = events[-1].get("node", "-") if events else "-"
        nodes_done = list(dict.fromkeys(
            e.get("node") for e in events if e.get("node")
        ))
        result.append({
            "step": step,
            "checkpoint_id": snap.config["configurable"].get("checkpoint_id", ""),
            "last_node": last_node,
            "events_count": len(events),
            "route": snap.values.get("route", ""),
            "nodes_done": nodes_done,
        })
    return result

def _replay_thread(rs: RunState, run_id: str, checkpoint_id: str) -> None:
    checkpoint_config = {"configurable": {
        "thread_id": f"monitor-{run_id}",
        "checkpoint_id": checkpoint_id,
    }}
    latest_config = {"configurable": {"thread_id": f"monitor-{run_id}"}}
    try:
        for chunk in GRAPH.stream(None, checkpoint_config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "__interrupt__":
                    continue
                rs.node_outputs[node_name] = node_output
                rs.queue.put({
                    "type": "node_done",
                    "node": node_name,
                    "output": _serialize(node_output),
                })
        # Get the LATEST state (after replay), not the checkpoint state
        final = GRAPH.get_state(latest_config).values
        rs.queue.put({
            "type": "done",
            "route": final.get("route", ""),
            "final_answer": final.get("final_answer") or final.get("pending_question") or "",
            "events": _serialize(final.get("events", [])),
        })
    except Exception as exc:
        rs.queue.put({"type": "error", "message": str(exc)})
    finally:
        rs.done = True

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="LangGraph Monitor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/scenarios")
def get_scenarios():
    return [
        {
            "id": s.id,
            "query": s.query,
            "expected_route": s.expected_route.value,
            "requires_approval": s.requires_approval,
            "tags": s.tags,
        }
        for s in SCENARIOS.values()
    ]

@app.post("/api/run/{scenario_id}")
def start_run(scenario_id: str):
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, "Scenario not found")
    run_id = uuid.uuid4().hex[:8]
    rs = RunState(run_id, scenario_id)
    RUNS[run_id] = rs
    threading.Thread(target=_run_graph, args=(rs,), daemon=True).start()
    return {"run_id": run_id}

@app.get("/api/stream/{run_id}")
async def stream_events(run_id: str):
    rs = RUNS.get(run_id)
    if not rs:
        raise HTTPException(404, "Run not found")

    async def generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: rs.queue.get(timeout=0.3))
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            except Empty:
                if rs.done:
                    break
                yield 'data: {"type":"ping"}\n\n'
            await asyncio.sleep(0)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/history/{run_id}")
def get_history(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    return _get_history(run_id)

@app.post("/api/replay/{run_id}")
def replay(run_id: str, body: dict):
    checkpoint_id = body.get("checkpoint_id", "")
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    new_run_id = uuid.uuid4().hex[:8]
    new_rs = RunState(new_run_id, RUNS[run_id].scenario_id)
    RUNS[new_run_id] = new_rs
    threading.Thread(
        target=_replay_thread, args=(new_rs, run_id, checkpoint_id), daemon=True
    ).start()
    return {"run_id": new_run_id}

@app.get("/api/crash-resume/{run_id}")
def crash_resume(run_id: str):
    thread_id = f"monitor-{run_id}"
    import sqlite3 as _sq
    conn = _sq.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    fresh_graph = build_graph(checkpointer=SqliteSaver(conn=conn))
    config = {"configurable": {"thread_id": thread_id}}
    snap = fresh_graph.get_state(config)
    if not snap or not snap.values:
        raise HTTPException(404, "No checkpoint found in DB")
    v = snap.values
    return {
        "thread_id": thread_id,
        "db_path": DB_PATH,
        "route": v.get("route", ""),
        "final_answer": v.get("final_answer", ""),
        "nodes_visited": [e.get("node") for e in (v.get("events") or [])],
        "events_count": len(v.get("events") or []),
        "resume_success": True,
    }

class ApprovalPayload(BaseModel):
    approved: bool
    reviewer: str = "human-reviewer"
    comment: str = ""

@app.post("/api/approve/{run_id}")
def approve(run_id: str, payload: ApprovalPayload):
    rs = RUNS.get(run_id)
    if not rs:
        raise HTTPException(404, "Run not found")
    rs.decision = payload.model_dump()
    rs.decision_event.set()
    return {"ok": True}

# Serve static files last (catch-all)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True,
                app_dir=str(Path(__file__).parent))
