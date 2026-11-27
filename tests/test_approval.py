import pytest

from app.approval import record_simulated_award, route_after_approval
from app.rules import evaluate_rfq
from app.data_loader import load_data


def test_award_node_requires_approved_state():
    evaluation = evaluate_rfq(load_data(), "RFQ-101")
    state = {
        "evaluation": evaluation,
        "approval_status": "APPROVED",
        "approval_decision": {"approved": True, "approver": "buyer"},
    }
    assert record_simulated_award(state)["contract_awarded"] is True
    assert route_after_approval(state) == "award"

    state["approval_status"] = "REJECTED"
    assert route_after_approval(state) == "end"
    with pytest.raises(ValueError, match="explicit human approval"):
        record_simulated_award(state)

    state["approval_status"] = "APPROVED"
    state["approval_decision"] = {"approved": False}
    with pytest.raises(ValueError, match="affirmative human decision"):
        record_simulated_award(state)