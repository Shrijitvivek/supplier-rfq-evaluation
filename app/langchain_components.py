"""Compose deterministic procurement findings into a structured, traceable explanation."""

from langchain_core.runnables import Runnable, RunnableLambda

from app.data_loader import ProcurementData, RFQFacts
from app.schemas import (
    EvaluationResult,
    ExplainedEvaluation,
    QuoteExplanation,
    StructuredExplanation,
)


def _explain(result: EvaluationResult, facts: RFQFacts) -> ExplainedEvaluation:
    quotes_by_id = {item.quote.quote_id: item for item in facts.quotes}
    evaluated_quotes = {item.quote.quote_id: item.quote for item in result.quotes}
    if evaluated_quotes != {quote_id: item.quote for quote_id, item in quotes_by_id.items()}:
        raise ValueError("Evaluation must preserve every source quote for the requested RFQ")
    if (facts.rfq is None or any(item.supplier is None for item in facts.quotes)) and (
        result.status != "NEEDS_INFORMATION"
    ):
        raise ValueError("A missing RFQ or supplier record requires NEEDS_INFORMATION")

    explanations = []
    for finding in result.quotes:
        quote = finding.quote
        joined = quotes_by_id[quote.quote_id]
        evidence = [f"supplier_quotes.csv:{quote.quote_id}"]
        if facts.rfq is not None:
            evidence.append(f"rfqs.csv:{result.rfq_id}")
        if joined.supplier is not None:
            evidence.append(f"suppliers.csv:{quote.supplier_id}")
        if joined.performance is not None:
            evidence.append(f"supplier_performance.csv:{quote.supplier_id}")
        narrative = (
            f"{quote.quote_id} was classified {finding.classification}. "
            f"Its tax-inclusive total is INR {finding.tax_inclusive_total_inr:.2f}."
        )
        if finding.reasons:
            narrative += f" Recorded reasons: {', '.join(finding.reasons)}."
        if finding.duplicate_of:
            evidence.append(f"supplier_quotes.csv:{finding.duplicate_of}")
            narrative += f" Possible duplicate of {finding.duplicate_of}. Flag for investigation."
        explanations.append(
            QuoteExplanation(quote_id=quote.quote_id, evidence_ids=evidence, narrative=narrative)
        )

    citations = {clause for finding in result.quotes for clause in finding.policy_citations}
    if result.status == "AWAITING_APPROVAL":
        winner = quotes_by_id[result.recommended_quote].quote
        summary = (
            f"Recommend {winner.quote_id} from supplier {winner.supplier_id} for {result.rfq_id}."
        )
        next_action = "Obtain human procurement approval before awarding or contacting a supplier."
        citations.update(("R5", "R6", "R9"))
    elif result.status == "NO_COMPLIANT_BID":
        summary = f"No compliant bid was found for {result.rfq_id}."
        next_action = "Request a revised RFQ or a human exception decision."
        citations.add("R7")
    else:
        summary = f"Evaluation of {result.rfq_id} requires more information."
        next_action = f"Obtain the missing information: {', '.join(result.information_needed)}."
        citations.add("R1")

    return ExplainedEvaluation(
        evaluation=result,
        explanation=StructuredExplanation(
            rfq_id=result.rfq_id,
            summary=summary,
            next_action=next_action,
            policy_citations=sorted(citations),
            quote_explanations=explanations,
        ),
    )


def build_explanation_chain(
    data: ProcurementData,
) -> Runnable[EvaluationResult, ExplainedEvaluation]:
    """Build an LCEL chain for the graph to call after deterministic evaluation."""

    def validate(payload: EvaluationResult) -> EvaluationResult:
        return EvaluationResult.model_validate(payload)

    def collect_facts(result: EvaluationResult) -> tuple[EvaluationResult, RFQFacts]:
        return result, data.for_rfq(result.rfq_id)

    def render(context: tuple[EvaluationResult, RFQFacts]) -> ExplainedEvaluation:
        return _explain(*context)

    return (
        RunnableLambda(validate, name="validate_evaluation")
        | RunnableLambda(collect_facts, name="collect_source_evidence")
        | RunnableLambda(render, name="render_procurement_explanation")
    )
