from autogen_agentchat.messages import StructuredMessage
"""AutoGen challenge reviewers and their asynchronous LangGraph node adapter."""

import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from app.review_utils import serialize_review_context
from app.schemas import NonBlank, ProcurementModel


class CostReview(ProcurementModel):
    summary: NonBlank
    cost_concerns: tuple[NonBlank, ...] = ()
    recommendation_check: Literal["SUPPORTS", "CHALLENGES", "INCONCLUSIVE"]
    contract_awarded: Literal[False] = False


class ComplianceReview(ProcurementModel):
    summary: NonBlank
    compliance_concerns: tuple[NonBlank, ...] = ()
    eligibility_check: Literal["SUPPORTS", "CHALLENGES", "INCONCLUSIVE"]
    contract_awarded: Literal[False] = False


def build_challenge_team(model_client: Any | None = None) -> Any:
    """Build a bounded Cost Reviewer / Compliance Reviewer AutoGen team."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as exc:
        raise ImportError(
            "AutoGen is required; install autogen-agentchat and autogen-ext[openai]."
        ) from exc

    client = (
        model_client
        if model_client is not None
        else OpenAIChatCompletionClient(model=os.getenv("AUTOGEN_MODEL", "gpt-4o-mini"))
    )
    cost_reviewer = AssistantAgent(
        "cost_reviewer",
        model_client=client,
        system_message=(
            "You are the Cost Reviewer. Independently check the supplied quote totals and "
            "deterministic recommendation for apparent inconsistencies. Use only supplied facts. "
            "This is an advisory challenge, not a recalculation or authority to change the result. "
            "Return a JSON object matching the required CostReview schema. Set contract_awarded "
            "to false; never authorize an award or contact a supplier."
        ),
        output_content_type=CostReview,
    )
    compliance_reviewer = AssistantAgent(
        "compliance_reviewer",
        model_client=client,
        system_message=(
            "You are the Compliance Reviewer. Independently challenge eligibility, policy "
            "evidence, and missing facts using only the supplied context. Deterministic business "
            "rules remain authoritative. Return a JSON object matching the required "
            "ComplianceReview schema. Set contract_awarded to false; never authorize an award or "
            "contact a supplier."
        ),
        output_content_type=ComplianceReview,
    )
    return RoundRobinGroupChat(
        [cost_reviewer, compliance_reviewer],
        termination_condition=MaxMessageTermination(max_messages=3),
    
        custom_message_types=[StructuredMessage[CostReview], StructuredMessage[ComplianceReview]],)


def _parse_output(model: type[ProcurementModel], content: Any) -> ProcurementModel:
    if isinstance(content, model):
        return content
    if isinstance(content, Mapping):
        return model.model_validate(content)
    if not isinstance(content, str):
        raise TypeError("AutoGen review output must be structured data or JSON text")
    return model.model_validate_json(content)


async def run_autogen_review(
    evaluation: Any,
    procurement_facts: Any,
    *,
    team: Any | None = None,
) -> dict[str, object]:
    """Run both AutoGen reviewers and return an advisory-only graph state update."""
    validated_evaluation, context_json = serialize_review_context(evaluation, procurement_facts)
    review_team = team if team is not None else build_challenge_team()
    reset = getattr(review_team, "reset", None)
    if reset is not None:
        await reset()
    result = await review_team.run(
        task=(
            "Challenge the following pre-approval procurement evaluation. Each reviewer must "
            "produce only its schema-validated advisory review. The deterministic result remains "
            "authoritative; no supplier contact or simulated award is allowed.\n"
            f"{context_json}"
        )
    )
    messages = getattr(result, "messages", None)
    if messages is None:
        raise RuntimeError("AutoGen did not return a message list")

    reviews: dict[str, ProcurementModel] = {}
    expected = {
        "cost_reviewer": CostReview,
        "compliance_reviewer": ComplianceReview,
    }
    for message in messages:
        source = getattr(message, "source", None)
        if source not in expected:
            continue
        if source in reviews:
            raise RuntimeError(f"AutoGen returned multiple final reviews from {source}")
        reviews[source] = _parse_output(expected[source], getattr(message, "content", None))
    missing = expected.keys() - reviews.keys()
    if missing:
        raise RuntimeError(f"AutoGen did not return reviews from: {', '.join(sorted(missing))}")

    return {
        "autogen_review": {
            "rfq_id": validated_evaluation.rfq_id,
            "cost_reviewer": reviews["cost_reviewer"].model_dump(mode="json"),
            "compliance_reviewer": reviews["compliance_reviewer"].model_dump(mode="json"),
            "contract_awarded": False,
        }
    }


def create_autogen_adapter(
    team: Any | None = None,
) -> Callable[[Mapping[str, Any]], Awaitable[dict[str, object]]]:
    """Create an async LangGraph node accepting evaluation and procurement_facts."""
    review_team = team if team is not None else build_challenge_team()

    async def node(state: Mapping[str, Any]) -> dict[str, object]:
        if "evaluation" not in state or "procurement_facts" not in state:
            raise KeyError("LangGraph state requires evaluation and procurement_facts")
        return await run_autogen_review(
            state["evaluation"],
            state["procurement_facts"],
            team=review_team,
        )

    return node