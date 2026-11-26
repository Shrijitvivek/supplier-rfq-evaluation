"""CrewAI procurement reviewers and their LangGraph node adapter."""

import os
from collections.abc import Callable, Mapping
from typing import Any, Literal

from app.review_utils import serialize_review_context
from app.schemas import NonBlank, ProcurementModel


class BuyerReview(ProcurementModel):
    summary: NonBlank
    cost_concerns: tuple[NonBlank, ...] = ()
    recommendation_check: Literal["SUPPORTS", "CHALLENGES", "INCONCLUSIVE"]
    contract_awarded: Literal[False] = False


class SupplierRiskReview(ProcurementModel):
    summary: NonBlank
    risk_concerns: tuple[NonBlank, ...] = ()
    eligibility_check: Literal["SUPPORTS", "CHALLENGES", "INCONCLUSIVE"]
    contract_awarded: Literal[False] = False


def build_procurement_crew(llm: Any | None = None) -> Any:
    """Build the Buyer then Supplier-Risk Analyst CrewAI review team."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as exc:
        raise ImportError("CrewAI is required; install the project dependencies with pip.") from exc

    model = llm if llm is not None else os.getenv("CREWAI_MODEL", "openai/gpt-4o-mini")
    shared_agent_options = {
        "llm": model,
        "allow_delegation": False,
        "max_iter": 3,
        "verbose": False,
    }
    buyer = Agent(
        role="Buyer",
        goal="Independently scrutinize commercial fit and the deterministic recommendation.",
        backstory=(
            "You are a procurement buyer. You check quote totals, offered quantities, delivery "
            "terms, and the reported recommendation. Deterministic findings remain authoritative."
        ),
        **shared_agent_options,
    )
    risk_analyst = Agent(
        role="Supplier-Risk Analyst",
        goal="Identify supplier eligibility, tax, risk, performance, and evidence concerns.",
        backstory=(
            "You are a cautious supplier-risk analyst. Use only supplied facts, flag missing "
            "evidence, and treat deterministic business-rule findings as authoritative."
        ),
        **shared_agent_options,
    )
    buyer_task = Task(
        description=(
            "Review this validated procurement result and its source facts as an advisory Buyer. "
            "Check commercial details and challenge any apparent inconsistency, but do not "
            "recalculate or replace the deterministic ranking. Do not contact a supplier, approve "
            "an award, or claim an award occurred. Return the required structured review.\n"
            "{review_context_json}"
        ),
        expected_output=(
            "A BuyerReview with a concise summary, any cost concerns, a recommendation check, "
            "and contract_awarded=false."
        ),
        agent=buyer,
        output_pydantic=BuyerReview,
    )
    risk_task = Task(
        description=(
            "Independently review the validated supplier and procurement facts, using the Buyer "
            "review only as context. Flag risk, eligibility, tax, performance, or missing-evidence "
            "concerns without changing deterministic classifications. This is advisory only: do "
            "not contact a supplier, approve an award, or claim an award occurred. Return the "
            "required structured review.\n"
            "{review_context_json}"
        ),
        expected_output=(
            "A SupplierRiskReview with a concise summary, any risk concerns, an eligibility "
            "check, and contract_awarded=false."
        ),
        agent=risk_analyst,
        context=[buyer_task],
        output_pydantic=SupplierRiskReview,
    )
    return Crew(
        agents=[buyer, risk_analyst],
        tasks=[buyer_task, risk_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        share_crew=False,
    )


def _parse_output(model: type[ProcurementModel], output: Any) -> ProcurementModel:
    structured = getattr(output, "pydantic", None)
    if structured is not None:
        return model.model_validate(structured)
    raw = getattr(output, "raw", output)
    if isinstance(raw, Mapping):
        return model.model_validate(raw)
    if not isinstance(raw, str):
        raise TypeError("CrewAI review output must be structured data or JSON text")
    return model.model_validate_json(raw)


def run_crewai_review(
    evaluation: Any,
    procurement_facts: Any,
    *,
    crew: Any | None = None,
) -> dict[str, object]:
    """Run both CrewAI reviewers and return an advisory-only graph state update."""
    validated_evaluation, context_json = serialize_review_context(evaluation, procurement_facts)
    review_crew = crew if crew is not None else build_procurement_crew()
    result = review_crew.kickoff(inputs={"review_context_json": context_json})
    outputs = getattr(result, "tasks_output", None)
    if outputs is None or len(outputs) != 2:
        raise RuntimeError("CrewAI must return one structured output from each reviewer")

    buyer_review = _parse_output(BuyerReview, outputs[0])
    risk_review = _parse_output(SupplierRiskReview, outputs[1])
    return {
        "crewai_review": {
            "rfq_id": validated_evaluation.rfq_id,
            "buyer": buyer_review.model_dump(mode="json"),
            "supplier_risk": risk_review.model_dump(mode="json"),
            "contract_awarded": False,
        }
    }


def create_crewai_adapter(
    crew: Any | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, object]]:
    """Create a synchronous LangGraph node accepting evaluation and procurement_facts."""
    review_crew = crew if crew is not None else build_procurement_crew()

    def node(state: Mapping[str, Any]) -> dict[str, object]:
        if "evaluation" not in state or "procurement_facts" not in state:
            raise KeyError("LangGraph state requires evaluation and procurement_facts")
        return run_crewai_review(
            state["evaluation"],
            state["procurement_facts"],
            crew=review_crew,
        )

    return node