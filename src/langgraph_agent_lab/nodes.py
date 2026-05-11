"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event

# Keywords used for routing (checked in priority order: risky > tool > missing_info > error > simple)
_RISKY_KEYWORDS = {"refund", "delete", "send", "cancel", "remove", "revoke"}
_TOOL_KEYWORDS = {"status", "order", "lookup", "check", "track", "find", "search"}
_ERROR_KEYWORDS = {"timeout", "fail", "failure", "error", "crash", "unavailable"}
_VAGUE_PRONOUNS = {"it", "this", "that", "them", "these", "those"}


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation into word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields."""
    query = state.get("query", "").strip()
    # Basic normalization: collapse whitespace
    query = re.sub(r"\s+", " ", query)
    return {
        "query": query,
        "messages": [f"intake:{query[:60]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using keyword heuristics.

    Priority: risky > tool > missing_info > error > simple.
    Uses word-boundary tokenization to avoid substring false positives.
    """
    tokens = set(_tokenize(state.get("query", "")))
    route = Route.SIMPLE
    risk_level = "low"

    if tokens & _RISKY_KEYWORDS:
        route = Route.RISKY
        risk_level = "high"
    elif tokens & _TOOL_KEYWORDS:
        route = Route.TOOL
    elif len(tokens) < 5 and tokens & _VAGUE_PRONOUNS:
        route = Route.MISSING_INFO
    elif tokens & _ERROR_KEYWORDS:
        route = Route.ERROR

    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = (
        f"Your request '{query}' is unclear. "
        "Could you provide more details, such as an order ID, account number, or a description of the issue?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={scenario_id}"
    else:
        result = f"mock-tool-result scenario={scenario_id} attempt={attempt}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval."""
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    proposed = (
        f"Proposed action: '{query}'. "
        f"Risk level: {risk_level}. "
        "This action is irreversible and requires explicit human approval before execution."
    )
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")

    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt and log bounded backoff metadata."""
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    errors = [f"transient failure attempt={attempt} of {max_attempts}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in tool_results and approval where relevant."""
    tool_results = state.get("tool_results") or []
    approval = state.get("approval") or {}

    if tool_results:
        answer = f"Result: {tool_results[-1]}"
        if approval.get("approved"):
            answer += f" (approved by {approval.get('reviewer', 'reviewer')})"
    else:
        answer = "Your request has been processed successfully."

    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops."""
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "needs_retry", "tool result indicates failure, retry needed")],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "success", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    """
    attempt = state.get("attempt", 0)
    scenario_id = state.get("scenario_id", "unknown")
    errors = state.get("errors") or []
    return {
        "final_answer": (
            f"Request '{scenario_id}' could not be completed after {attempt} attempt(s). "
            "Escalated to manual review queue."
        ),
        "errors": [f"dead_letter: max retries exhausted after {attempt} attempts, errors={errors}"],
        "events": [
            make_event(
                "dead_letter",
                "escalated",
                "max retries exceeded",
                scenario_id=scenario_id,
                attempt=attempt,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
