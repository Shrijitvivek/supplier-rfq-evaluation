import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from langgraph.graph import END, StateGraph

from app.data_loader import load_data
from app.crewai_adapter import (
    BuyerReview,
    SupplierRiskReview,
    build_procurement_crew,
    create_crewai_adapter,
    run_crewai_review,
)
from app.review_utils import serialize_review_context
from app.schemas import EvaluationResult


@pytest.fixture
def evaluation():
    return EvaluationResult(
        rfq_id="RFQ-101",
        status="NEEDS_INFORMATION",
        information_needed=("supplier tax record",),
    )


class FakeCrew:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = None

    def kickoff(self, *, inputs):
        self.inputs = inputs
        return SimpleNamespace(tasks_output=self.outputs)


def test_adapter_returns_structured_advice_without_changing_evaluation(evaluation):
    crew = FakeCrew(
        [
            SimpleNamespace(
                pydantic=BuyerReview(
                    summary="The missing tax record prevents a cost recommendation.",
                    recommendation_check="INCONCLUSIVE",
                )
            ),
            SimpleNamespace(
                pydantic=SupplierRiskReview(
                    summary="Verify the supplier tax record before proceeding.",
                    risk_concerns=("Tax record is missing.",),
                    eligibility_check="INCONCLUSIVE",
                )
            ),
        ]
    )
    state = {
        "evaluation": evaluation,
        "procurement_facts": {"rfq": {"rfq_id": "RFQ-101"}, "quotes": []},
    }

    workflow = StateGraph(dict)
    workflow.add_node("review", create_crewai_adapter(crew))
    workflow.set_entry_point("review")
    workflow.add_edge("review", END)
    update = workflow.compile().invoke(state)

    assert update["crewai_review"]["rfq_id"] == "RFQ-101"
    assert update["crewai_review"]["buyer"]["recommendation_check"] == "INCONCLUSIVE"
    assert update["crewai_review"]["supplier_risk"]["risk_concerns"] == [
        "Tax record is missing."
    ]
    assert update["crewai_review"]["contract_awarded"] is False
    assert evaluation.contract_awarded is False
    context = json.loads(crew.inputs["review_context_json"])
    assert context["evaluation"]["rfq_id"] == "RFQ-101"
    assert context["procurement_facts"]["rfq"]["rfq_id"] == "RFQ-101"


def test_builder_creates_the_two_sequential_crew_tasks():
    crew = build_procurement_crew(llm="openai/gpt-4o-mini")

    assert [agent.role for agent in crew.agents] == ["Buyer", "Supplier-Risk Analyst"]
    assert [task.output_pydantic for task in crew.tasks] == [BuyerReview, SupplierRiskReview]


def test_adapter_requires_both_reviewer_outputs(evaluation):
    crew = FakeCrew([SimpleNamespace(raw="{}")])

    with pytest.raises(RuntimeError, match="one structured output from each reviewer"):
        run_crewai_review(evaluation, {}, crew=crew)


def test_adapter_rejects_an_award_claim_in_a_review_model():
    with pytest.raises(ValidationError):
        BuyerReview(
            summary="Award the contract.",
            recommendation_check="SUPPORTS",
            contract_awarded=True,
        )


def test_adapter_requires_expected_langgraph_state():
    node = create_crewai_adapter(FakeCrew([]))
    with pytest.raises(KeyError, match="evaluation and procurement_facts"):
        node({})


def test_adapter_rejects_source_facts_for_a_different_rfq(evaluation):
    crew = FakeCrew([])
    with pytest.raises(ValueError, match="same RFQ"):
        run_crewai_review(
            evaluation,
            {"rfq": {"rfq_id": "RFQ-102"}},
            crew=crew,
        )
    assert crew.inputs is None


def test_review_context_serializes_source_data_records():
    evaluation = EvaluationResult(
        rfq_id="RFQ-999",
        status="NEEDS_INFORMATION",
        information_needed=("rfqs.csv:RFQ-999",),
    )
    _, context_json = serialize_review_context(evaluation, load_data().for_rfq("RFQ-999"))
    context = json.loads(context_json)

    assert context["procurement_facts"]["rfq_id"] == "RFQ-999"
    assert context["procurement_facts"]["missing_records"] == ["rfqs.csv:RFQ-999"]


def test_review_context_rejects_omitted_source_quotes(evaluation):
    with pytest.raises(ValueError, match="preserve every source quote"):
        serialize_review_context(
            evaluation,
            {
                "rfq_id": "RFQ-101",
                "quotes": [{"quote": {"quote_id": "Q-101"}}],
            },
        )


def test_review_context_preserves_all_six_rfqs():
    data = load_data()
    examples = Path(__file__).resolve().parents[1] / "examples"
    records = [
        record
        for path in sorted(examples.glob("*.json"))
        for record in json.loads(path.read_text(encoding="utf-8"))
    ]

    assert len(records) == 6
    for record in records:
        evaluation = EvaluationResult.model_validate(record["evaluation"])
        _, context_json = serialize_review_context(
            evaluation,
            data.for_rfq(evaluation.rfq_id),
        )
        context = json.loads(context_json)
        assert [item["quote"]["quote_id"] for item in context["procurement_facts"]["quotes"]] == [
            item["quote"]["quote_id"] for item in record["evaluation"]["quotes"]
        ]