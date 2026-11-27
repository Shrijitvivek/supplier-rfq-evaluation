"""Deterministic procurement eligibility, validation, tax, and ranking rules."""

from decimal import Decimal, ROUND_HALF_UP

from app.data_loader import ProcurementData, QuoteFacts, RFQFacts
from app.schemas import EvaluationResult, QuoteEvaluation, SupplierQuote


CENT = Decimal("0.01")


def calculate_tax_inclusive_total(quote: SupplierQuote) -> Decimal:
    """Return the offered quantity's tax-inclusive total rounded to paise."""
    multiplier = Decimal("1") + quote.tax_percent / Decimal("100")
    return (quote.unit_price_inr * quote.offered_qty * multiplier).quantize(
        CENT, rounding=ROUND_HALF_UP
    )


def is_supplier_eligible(facts: QuoteFacts) -> bool:
    supplier = facts.supplier
    return bool(
        supplier
        and supplier.onboarding_status == "approved"
        and supplier.tax_verified == "yes"
    )


def _duplicate_key(facts: QuoteFacts) -> tuple[object, ...]:
    quote = facts.quote
    return (
        quote.rfq_id,
        quote.supplier_id,
        quote.unit_price_inr,
        quote.offered_qty,
        quote.promised_delivery,
    )


def _evaluate_quote(
    facts: QuoteFacts,
    rfq_facts: RFQFacts,
    duplicate_of: str | None,
    *,
    is_recommended: bool = False,
) -> QuoteEvaluation:
    quote = facts.quote
    total = calculate_tax_inclusive_total(quote)
    if rfq_facts.rfq is None:
        return QuoteEvaluation(
            quote=quote,
            classification="NEEDS_INFORMATION",
            tax_inclusive_total_inr=total,
            reasons=("MISSING_RFQ_RECORD",),
            duplicate_of=duplicate_of,
            policy_citations=("R1", "R4", *(("R8",) if duplicate_of else ())),
        )
    if facts.supplier is None:
        return QuoteEvaluation(
            quote=quote,
            classification="NEEDS_INFORMATION",
            tax_inclusive_total_inr=total,
            reasons=("MISSING_SUPPLIER_RECORD",),
            duplicate_of=duplicate_of,
            policy_citations=("R1", "R4", *(("R8",) if duplicate_of else ())),
        )

    citations = ["R1", "R2", "R3", "R4"]
    reasons = []
    if not is_supplier_eligible(facts):
        classification = "INELIGIBLE"
        reasons.append("INELIGIBLE_SUPPLIER")
        citations = ["R1", "R2", "R4"]
    else:
        rfq = rfq_facts.rfq
        if quote.offered_qty < rfq.requested_qty:
            reasons.append("INSUFFICIENT_QUANTITY")
        if quote.promised_delivery > rfq.required_by:
            reasons.append("LATE_DELIVERY")
        if quote.unit_price_inr > rfq.max_unit_budget_inr:
            reasons.append("OVER_BUDGET")
        if reasons:
            classification = "NON_COMPLIANT"
        elif facts.performance is None:
            classification = "NEEDS_INFORMATION"
            reasons.append("MISSING_SUPPLIER_PERFORMANCE")
            citations.append("R5")
        else:
            classification = "COMPLIANT"
            if is_recommended:
                citations.extend(("R5", "R6", "R9"))

    if duplicate_of:
        citations.append("R8")
    return QuoteEvaluation(
        quote=quote,
        classification=classification,
        tax_inclusive_total_inr=total,
        reasons=tuple(reasons),
        duplicate_of=duplicate_of,
        policy_citations=tuple(citations),
    )


def _recommendation_key(item: QuoteEvaluation, facts_by_quote: dict[str, QuoteFacts]):
    performance = facts_by_quote[item.quote.quote_id].performance
    assert performance is not None
    return (
        item.tax_inclusive_total_inr,
        -performance.on_time_delivery_pct,
        performance.defect_rate_pct,
        item.quote.quote_id,
    )


def evaluate_rfq(data: ProcurementData, rfq_id: str) -> EvaluationResult:
    """Evaluate every source quote for an RFQ and return a pre-approval result."""
    facts = data.for_rfq(rfq_id)
    first_quote_by_key: dict[tuple[object, ...], str] = {}
    duplicate_ids: dict[str, str] = {}
    for item in facts.quotes:
        key = _duplicate_key(item)
        if key in first_quote_by_key:
            duplicate_ids[item.quote.quote_id] = first_quote_by_key[key]
        else:
            first_quote_by_key[key] = item.quote.quote_id

    findings = [
        _evaluate_quote(item, facts, duplicate_ids.get(item.quote.quote_id))
        for item in facts.quotes
    ]
    facts_by_quote = {item.quote.quote_id: item for item in facts.quotes}
    information_needed = []
    if facts.rfq is None:
        information_needed.append(f"rfqs.csv:{rfq_id}")
    for item, finding in zip(facts.quotes, findings):
        if finding.classification == "NEEDS_INFORMATION":
            if item.supplier is None:
                information_needed.append(f"suppliers.csv:{item.quote.supplier_id}")
            elif item.performance is None and finding.reasons == ("MISSING_SUPPLIER_PERFORMANCE",):
                information_needed.append(f"supplier_performance.csv:{item.quote.supplier_id}")

    if information_needed:
        return EvaluationResult(
            rfq_id=rfq_id,
            status="NEEDS_INFORMATION",
            quotes=tuple(findings),
            information_needed=tuple(dict.fromkeys(information_needed)),
        )

    compliant = [
        item
        for item in findings
        if item.classification == "COMPLIANT" and item.duplicate_of is None
    ]
    if not compliant:
        return EvaluationResult(rfq_id=rfq_id, status="NO_COMPLIANT_BID", quotes=tuple(findings))

    winner = min(compliant, key=lambda item: _recommendation_key(item, facts_by_quote))
    findings = [
        _evaluate_quote(
            item,
            facts,
            duplicate_ids.get(item.quote.quote_id),
            is_recommended=item.quote.quote_id == winner.quote.quote_id,
        )
        for item in facts.quotes
    ]
    return EvaluationResult(
        rfq_id=rfq_id,
        status="AWAITING_APPROVAL",
        recommended_quote=winner.quote.quote_id,
        quotes=tuple(findings),
    )