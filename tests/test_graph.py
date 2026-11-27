import asyncio
from dataclasses import replace
from types import SimpleNamespace

from langgraph.types import Command

from app.autogen_adapter import ComplianceReview, CostReview
from app.crewai_adapter import BuyerReview, SupplierRiskReview
from app.data_loader import load_data
from app.graph import build_graph


def test_graph_checkpoints_and_resumes_after_human_approval():
    app = build_graph()
    config = {"configurable": {"thread_id": "approval-test"}}
    paused = app.invoke({"rfq_id": "RFQ-101"}, config)
    assert paused["evaluation"]["recommended_quote"] == "Q-101"
    assert paused["contract_awarded"] is False
    assert paused.get("__interrupt__")

    completed = app.invoke(
        Command(resume={"approved": True, "approver": "buyer"}),
        config,
    )
    assert completed["approval_status"] == "APPROVED"
    assert completed["contract_awarded"] is True
    assert completed["procurement_facts"]["rfq"]["rfq_id"] == "RFQ-101"
    assert completed["simulated_award"] == {
        "rfq_id": "RFQ-101",
        "quote_id": "Q-101",
        "approver": "buyer",
        "simulated": True,
    }


def test_rejection_and_non_recommendation_never_record_an_award():
    app = build_graph()
    config = {"configurable": {"thread_id": "rejection-test"}}
    app.invoke({"rfq_id": "RFQ-101"}, config)
    rejected = app.invoke(Command(resume={"approved": False}), config)
    assert rejected["approval_status"] == "REJECTED"
    assert rejected["contract_awarded"] is False
    assert rejected["simulated_award"] is None

    no_bid_data = replace(load_data(), quotes={})
    no_bid = build_graph(no_bid_data).invoke(
        {"rfq_id": "RFQ-101"},
        {"configurable": {"thread_id": "no-bid-test"}},
    )
    assert no_bid["evaluation"]["status"] == "NO_COMPLIANT_BID"
    assert not no_bid.get("__interrupt__")
    assert no_bid["contract_awarded"] is False


def test_optional_graph_runs_crewai_and_autogen_before_approval():
    class FakeCrew:
        def kickoff(self, *, inputs):
            return SimpleNamespace(
                tasks_output=[
                    SimpleNamespace(
                        pydantic=BuyerReview(
                            summary="Commercial review complete.",
                            recommendation_check="SUPPORTS",
                        )
                    ),
                    SimpleNamespace(
                        pydantic=SupplierRiskReview(
                            summary="Supplier review complete.",
                            eligibility_check="SUPPORTS",
                        )
                    ),
                ]
            )

    class FakeTeam:
        async def reset(self):
            pass

        async def run(self, *, task):
            return SimpleNamespace(
                messages=[
                    SimpleNamespace(
                        source="cost_reviewer",
                        content=CostReview(
                            summary="Cost review complete.",
                            recommendation_check="SUPPORTS",
                        ),
                    ),
                    SimpleNamespace(
                        source="compliance_reviewer",
                        content=ComplianceReview(
                            summary="Compliance review complete.",
                            eligibility_check="SUPPORTS",
                        ),
                    ),
                ]
            )

    app = build_graph(
        include_advisory_reviews=True,
        crew=FakeCrew(),
        team=FakeTeam(),
    )
    state = asyncio.run(
        app.ainvoke(
            {"rfq_id": "RFQ-101"},
            {"configurable": {"thread_id": "advisory-review-test"}},
        )
    )

    assert state["crewai_review"]["buyer"]["recommendation_check"] == "SUPPORTS"
    assert state["autogen_review"]["compliance_reviewer"]["eligibility_check"] == "SUPPORTS"
    assert state["contract_awarded"] is False
    assert state.get("__interrupt__")