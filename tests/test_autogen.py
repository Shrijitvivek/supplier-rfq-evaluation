import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from langgraph.graph import END, StateGraph
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from app.autogen_adapter import (
    ComplianceReview,
    CostReview,
    build_challenge_team,
    create_autogen_adapter,
    run_autogen_review,
)
from app.schemas import EvaluationResult


@pytest.fixture
def evaluation():
    return EvaluationResult(
        rfq_id="RFQ-101",
        status="NEEDS_INFORMATION",
        information_needed=("supplier tax record",),
    )


class FakeTeam:
    def __init__(self, messages):
        self.messages = messages
        self.reset_calls = 0
        self.task = None

    async def reset(self):
        self.reset_calls += 1

    async def run(self, *, task):
        self.task = task
        return SimpleNamespace(messages=self.messages)


def test_adapter_returns_both_challenges_without_changing_evaluation(evaluation):
    team = FakeTeam(
        [
            SimpleNamespace(
                source="cost_reviewer",
                content=CostReview(
                    summary="The cost review is inconclusive without tax details.",
                    recommendation_check="INCONCLUSIVE",
                ).model_dump_json(),
            ),
            SimpleNamespace(
                source="compliance_reviewer",
                content=ComplianceReview(
                    summary="Obtain the supplier tax record.",
                    compliance_concerns=("Tax record is missing.",),
                    eligibility_check="INCONCLUSIVE",
                ).model_dump_json(),
            ),
        ]
    )
    state = {
        "evaluation": evaluation,
        "procurement_facts": {"rfq": {"rfq_id": "RFQ-101"}, "quotes": []},
    }

    workflow = StateGraph(dict)
    workflow.add_node("review", create_autogen_adapter(team))
    workflow.set_entry_point("review")
    workflow.add_edge("review", END)
    update = asyncio.run(workflow.compile().ainvoke(state))

    assert update["autogen_review"]["rfq_id"] == "RFQ-101"
    assert update["autogen_review"]["cost_reviewer"]["recommendation_check"] == "INCONCLUSIVE"
    assert update["autogen_review"]["compliance_reviewer"]["compliance_concerns"] == [
        "Tax record is missing."
    ]
    assert update["autogen_review"]["contract_awarded"] is False
    assert evaluation.contract_awarded is False
    assert team.reset_calls == 1
    context = json.loads(team.task.split("\n", 1)[1])
    assert context["evaluation"]["rfq_id"] == "RFQ-101"
    assert context["procurement_facts"]["rfq"]["rfq_id"] == "RFQ-101"


def test_builder_creates_an_autogen_round_robin_team():
    client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="local-test-key")
    try:
        team = build_challenge_team(client)
        assert isinstance(team, RoundRobinGroupChat)
        assert team._participant_names == ["cost_reviewer", "compliance_reviewer"]
    finally:
        asyncio.run(client.close())


def test_adapter_rejects_incomplete_team_output(evaluation):
    team = FakeTeam(
        [
            SimpleNamespace(
                source="cost_reviewer",
                content=CostReview(
                    summary="No conclusion.",
                    recommendation_check="INCONCLUSIVE",
                ).model_dump_json(),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="did not return reviews from: compliance_reviewer"):
        asyncio.run(run_autogen_review(evaluation, {}, team=team))


def test_adapter_rejects_an_award_claim_in_a_review_model():
    with pytest.raises(ValidationError):
        ComplianceReview(
            summary="Award the contract.",
            eligibility_check="SUPPORTS",
            contract_awarded=True,
        )


def test_adapter_requires_expected_langgraph_state():
    node = create_autogen_adapter(FakeTeam([]))
    with pytest.raises(KeyError, match="evaluation and procurement_facts"):
        asyncio.run(node({}))


def test_adapter_rejects_source_facts_for_a_different_rfq(evaluation):
    team = FakeTeam([])
    with pytest.raises(ValueError, match="same RFQ"):
        asyncio.run(
            run_autogen_review(
                evaluation,
                {"rfq": {"rfq_id": "RFQ-102"}},
                team=team,
            )
        )
    assert team.reset_calls == 0