"""LangGraph state shared by procurement workflow nodes."""

from typing import TypedDict


class ProcurementState(TypedDict, total=False):
    rfq_id: str
    procurement_facts: dict[str, object]
    evaluation: dict[str, object]
    approval_decision: dict[str, object]
    approval_status: str
    contract_awarded: bool
    simulated_award: dict[str, object] | None
    crewai_review: dict[str, object]
    autogen_review: dict[str, object]