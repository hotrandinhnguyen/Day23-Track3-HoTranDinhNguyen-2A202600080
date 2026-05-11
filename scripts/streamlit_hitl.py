"""Bonus 6 -- Streamlit HITL Approval UI.

Runs the LangGraph support-ticket agent with real interrupt() and provides
a web UI for a human reviewer to approve or reject risky actions.

Usage:
    streamlit run scripts/streamlit_hitl.py
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

os.environ["LANGGRAPH_INTERRUPT"] = "true"

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.state import AgentState, initial_state, Scenario, Route  # noqa: E402

DB_PATH = "outputs/streamlit_hitl.db"

st.set_page_config(page_title="Support Ticket Agent -- HITL Approval", layout="centered")
st.title("Support Ticket Agent")
st.caption("Human-in-the-Loop Approval Demo")


# --- Session state init ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "phase" not in st.session_state:
    st.session_state.phase = "input"       # input | pending_approval | done
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final_result" not in st.session_state:
    st.session_state.final_result = None


def get_graph():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return build_graph(checkpointer=SqliteSaver(conn=conn))


graph = get_graph()


# --- Phase: input ---
if st.session_state.phase == "input":
    st.subheader("Submit a support request")
    query = st.text_input("Query", value="Refund this customer and send confirmation email")
    scenario_id = st.text_input("Scenario ID", value=f"demo-{uuid.uuid4().hex[:6]}")

    if st.button("Submit", type="primary"):
        thread_id = f"thread-{scenario_id}"
        st.session_state.thread_id = thread_id

        scenario = Scenario(
            id=scenario_id,
            query=query,
            expected_route=Route.RISKY,
            requires_approval=True,
        )
        state: AgentState = initial_state(scenario)
        run_cfg = {"configurable": {"thread_id": thread_id}}

        graph.invoke(state, config=run_cfg)

        # Check if graph paused at interrupt
        pending = graph.get_state(config=run_cfg)
        interrupts = pending.tasks[0].interrupts if pending.tasks else []

        if interrupts:
            st.session_state.interrupt_payload = interrupts[0].value
            st.session_state.phase = "pending_approval"
        else:
            # No interrupt (simple/tool route) -- already done
            final = graph.get_state(config=run_cfg)
            st.session_state.final_result = final.values
            st.session_state.phase = "done"

        st.rerun()


# --- Phase: pending approval ---
elif st.session_state.phase == "pending_approval":
    payload = st.session_state.interrupt_payload
    st.subheader("Approval Required")
    st.warning("Graph is paused and waiting for your decision.")

    with st.container(border=True):
        st.markdown("**Proposed Action**")
        st.write(payload.get("proposed_action", "N/A"))
        st.markdown(f"**Risk Level**: `{payload.get('risk_level', 'unknown')}`")

    reviewer = st.text_input("Reviewer name", value="human-reviewer")
    comment = st.text_area("Comment (optional)", value="")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve", type="primary", use_container_width=True):
            run_cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            decision = {"approved": True, "reviewer": reviewer, "comment": comment}
            final = graph.invoke(Command(resume=decision), config=run_cfg)
            st.session_state.final_result = final
            st.session_state.phase = "done"
            st.rerun()

    with col2:
        if st.button("Reject", type="secondary", use_container_width=True):
            run_cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            decision = {"approved": False, "reviewer": reviewer, "comment": comment or "Rejected"}
            final = graph.invoke(Command(resume=decision), config=run_cfg)
            st.session_state.final_result = final
            st.session_state.phase = "done"
            st.rerun()


# --- Phase: done ---
elif st.session_state.phase == "done":
    result = st.session_state.final_result or {}
    approval = result.get("approval") or {}

    st.subheader("Request Completed")

    status = "Approved" if approval.get("approved") else "Rejected"
    color = "green" if approval.get("approved") else "red"
    st.markdown(f"**Status**: :{color}[{status}]")

    st.markdown(f"**Route**: `{result.get('route', 'N/A')}`")
    st.markdown("**Final Answer**")
    st.info(result.get("final_answer") or result.get("pending_question") or "No answer")

    if approval:
        with st.expander("Approval details"):
            st.json(approval)

    with st.expander("Audit events"):
        events = result.get("events", [])
        for ev in events:
            st.write(f"`{ev.get('node')}` — {ev.get('event_type')}: {ev.get('message')}")

    if st.button("Start new request"):
        st.session_state.phase = "input"
        st.session_state.thread_id = None
        st.session_state.interrupt_payload = None
        st.session_state.final_result = None
        st.rerun()
