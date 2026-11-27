"""Human approval gate and simulated award recording."""

from collections.abc import Mapping
from typing import Any

from langgraph.types import interrupt

from app.schemas import EvaluationResult


def request_human_approval(state: Mapping[str, Any]) -> dict[str, object]:
    evaluation = EvaluationResult.model_validate(state["evaluation"])
    if evaluation.status != "AWAITING_APPROVAL" or evaluation.recommended_quote is None:
        raise ValueError("Human approval is only available for a recommendation")

    decision = interrupt(
        {
            "type": "human_approval",
            "rfq_id": evaluation.rfq_id,
            "recommended_quote": evaluation.recommended_quote,
            "message": "Approve this recommendation before recording a simulated award.",
        }
    )
    if not isinstance(decision, Mapping) or type(decision.get("approved")) is not bool:
        raise ValueError("Resume with an approval decision containing a boolean 'approved' field")
    status = "APPROVED" if decision["approved"] else "REJECTED"
    return {
        "approval_decision": dict(decision),
        "approval_status": status,
        "contract_awarded": False,
        "simulated_award": None,
    }


def record_simulated_award(state: Mapping[str, Any]) -> dict[str, object]:
    if state.get("approval_status") != "APPROVED":
        raise ValueError("A simulated award requires explicit human approval")
    decision = state.get("approval_decision")
    if not isinstance(decision, Mapping) or decision.get("approved") is not True:
        raise ValueError("A simulated award requires an affirmative human decision")
    evaluation = EvaluationResult.model_validate(state["evaluation"])
    if evaluation.status != "AWAITING_APPROVAL" or evaluation.recommended_quote is None:
        raise ValueError("A simulated award requires a valid recommendation")
    return {
        "contract_awarded": True,
        "simulated_award": {
            "rfq_id": evaluation.rfq_id,
            "quote_id": evaluation.recommended_quote,
            "approver": decision.get("approver"),
            "simulated": True,
        },
    }


def route_after_approval(state: Mapping[str, Any]) -> str:
    return "award" if state.get("approval_status") == "APPROVED" else "end"