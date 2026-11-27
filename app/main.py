"""Command-line entry point for evaluating an RFQ and handling human approval."""

import argparse
import asyncio
import json
from typing import Sequence

from langgraph.types import Command
from dotenv import load_dotenv

from app.graph import build_graph
from app.schemas import EvaluationResult


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate supplier quotations for an RFQ.")
    parser.add_argument("rfq_id")
    parser.add_argument("--thread-id", help="Checkpoint thread ID used to resume the workflow.")
    parser.add_argument("--decision", choices=("approve", "reject"))
    parser.add_argument(
        "--advisory-reviews",
        action="store_true",
        help="Run CrewAI and AutoGen advisory reviews using configured provider credentials.",
    )
    args = parser.parse_args(argv)

    if args.advisory_reviews:
        load_dotenv()
    app = build_graph(include_advisory_reviews=args.advisory_reviews)
    config = {"configurable": {"thread_id": args.thread_id or args.rfq_id}}

    def invoke(payload):
        if args.advisory_reviews:
            return asyncio.run(app.ainvoke(payload, config))
        return app.invoke(payload, config)

    state = invoke({"rfq_id": args.rfq_id})
    if state.get("__interrupt__"):
        evaluation = EvaluationResult.model_validate(state["evaluation"])
        print(
            f"Recommended quote {evaluation.recommended_quote} for {args.rfq_id}."
        )
        decision = args.decision
        if decision is None:
            decision = input("Approve simulated award? [y/N] ").strip().lower()
            approved = decision in ("y", "yes")
        else:
            approved = decision == "approve"
        state = invoke(Command(resume={"approved": approved, "approver": "human"}))

    output = {
        "evaluation": EvaluationResult.model_validate(state["evaluation"]).model_dump(
            mode="json"
        ),
        "approval_status": state.get("approval_status"),
        "contract_awarded": state.get("contract_awarded", False),
        "simulated_award": state.get("simulated_award"),
    }
    for key in ("crewai_review", "autogen_review"):
        if key in state:
            output[key] = state[key]
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())