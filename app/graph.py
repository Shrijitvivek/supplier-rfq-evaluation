"""LangGraph procurement evaluation and checkpointed human approval workflow."""

from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.approval import record_simulated_award, request_human_approval, route_after_approval
from app.data_loader import ProcurementData, RFQFacts, load_data
from app.rules import evaluate_rfq
from app.schemas import EvaluationResult
from app.state import ProcurementState


def _facts_payload(facts: RFQFacts) -> dict[str, object]:
    return {
        "rfq_id": facts.rfq_id,
        "rfq": facts.rfq.model_dump(mode="json") if facts.rfq else None,
        "quotes": [
            {
                "quote": item.quote.model_dump(mode="json"),
                "supplier": item.supplier.model_dump(mode="json") if item.supplier else None,
                "performance": (
                    item.performance.model_dump(mode="json") if item.performance else None
                ),
            }
            for item in facts.quotes
        ],
        "missing_records": list(facts.missing_records),
    }


def _evaluate_node(data: ProcurementData, state: Mapping[str, Any]) -> dict[str, object]:
    rfq_id = state.get("rfq_id")
    if not isinstance(rfq_id, str) or not rfq_id.strip():
        raise ValueError("LangGraph input must contain a non-empty rfq_id")
    facts = data.for_rfq(rfq_id)
    evaluation = evaluate_rfq(data, rfq_id)
    return {
        "rfq_id": rfq_id,
        "procurement_facts": _facts_payload(facts),
        "evaluation": evaluation.model_dump(mode="json"),
        "contract_awarded": False,
        "simulated_award": None,
    }


def route_after_evaluation(state: Mapping[str, Any]) -> str:
    evaluation = EvaluationResult.model_validate(state["evaluation"])
    return "approval" if evaluation.status == "AWAITING_APPROVAL" else "end"


def route_after_advisory_reviews(state: Mapping[str, Any]) -> str:
    return route_after_evaluation(state)


def build_graph(
    data: ProcurementData | None = None,
    *,
    checkpointer: Any | None = None,
    include_advisory_reviews: bool = False,
    crew: Any | None = None,
    team: Any | None = None,
):
    """Compile the workflow, optionally including both advisory model reviews.

    The review-enabled graph requires ``ainvoke`` because the AutoGen node is async.
    Pass a durable checkpointer to persist checkpoints across processes.
    """
    procurement_data = data if data is not None else load_data()
    workflow = StateGraph(ProcurementState)
    workflow.add_node("evaluate", lambda state: _evaluate_node(procurement_data, state))
    workflow.add_node("approval", request_human_approval)
    workflow.add_node("award", record_simulated_award)
    if include_advisory_reviews:
        from app.autogen_adapter import create_autogen_adapter
        from app.crewai_adapter import create_crewai_adapter

        workflow.add_node("crewai_advisory", create_crewai_adapter(crew))
        workflow.add_node("autogen_advisory", create_autogen_adapter(team))

    workflow.set_entry_point("evaluate")
    if include_advisory_reviews:
        workflow.add_edge("evaluate", "crewai_advisory")
        workflow.add_edge("crewai_advisory", "autogen_advisory")
        workflow.add_conditional_edges(
            "autogen_advisory",
            route_after_advisory_reviews,
            {"approval": "approval", "end": END},
        )
    else:
        workflow.add_conditional_edges(
            "evaluate",
            route_after_evaluation,
            {"approval": "approval", "end": END},
        )
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {"award": "award", "end": END},
    )
    workflow.add_edge("award", END)
    return workflow.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )