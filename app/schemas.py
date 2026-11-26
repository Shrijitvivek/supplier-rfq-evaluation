"""Validated procurement records and the pre-approval explanation contract."""

from datetime import date, datetime
from decimal import Decimal
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, model_validator


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Money = Annotated[Decimal, Field(ge=0, decimal_places=2, allow_inf_nan=False)]
TaxPercent = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Percent = Annotated[Decimal, Field(ge=0, le=100, allow_inf_nan=False)]
PolicyClause = Literal["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]


def _validate_date(value: object) -> date:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return date.fromisoformat(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    raise ValueError("Expected a date in YYYY-MM-DD format")


ISODate = Annotated[date, BeforeValidator(_validate_date)]


class ProcurementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RFQRequest(ProcurementModel):
    rfq_id: NonBlank
    note: str | None = None


class RFQ(ProcurementModel):
    rfq_id: NonBlank
    category: NonBlank
    requested_qty: Annotated[int, Field(gt=0)]
    max_unit_budget_inr: Money
    required_by: ISODate


class Supplier(ProcurementModel):
    supplier_id: NonBlank
    supplier_name: NonBlank
    onboarding_status: Literal["approved", "pending", "suspended"]
    risk: Literal["low", "medium", "high"]
    tax_verified: Literal["yes", "no"]


class SupplierPerformance(ProcurementModel):
    supplier_id: NonBlank
    on_time_delivery_pct: Percent
    defect_rate_pct: Percent
    last_review_date: ISODate


class SupplierQuote(ProcurementModel):
    quote_id: NonBlank
    rfq_id: NonBlank
    supplier_id: NonBlank
    unit_price_inr: Money
    offered_qty: Annotated[int, Field(gt=0)]
    promised_delivery: ISODate
    tax_percent: TaxPercent


class QuoteEvaluation(ProcurementModel):
    quote: SupplierQuote
    classification: Literal["COMPLIANT", "INELIGIBLE", "NON_COMPLIANT", "NEEDS_INFORMATION"]
    tax_inclusive_total_inr: Money
    reasons: tuple[NonBlank, ...] = ()
    duplicate_of: NonBlank | None = None
    policy_citations: Annotated[tuple[PolicyClause, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_findings(self) -> Self:
        if self.classification != "COMPLIANT" and not self.reasons:
            raise ValueError("A rejected or incomplete quote requires a reason")
        if self.duplicate_of and "R8" not in self.policy_citations:
            raise ValueError("A possible duplicate requires policy citation R8")
        return self


class EvaluationResult(ProcurementModel):
    """Deterministic findings supplied by the workflow before human approval."""

    rfq_id: NonBlank
    status: Literal["AWAITING_APPROVAL", "NO_COMPLIANT_BID", "NEEDS_INFORMATION"]
    recommended_quote: NonBlank | None = None
    quotes: tuple[QuoteEvaluation, ...] = ()
    information_needed: tuple[NonBlank, ...] = ()
    contract_awarded: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        quotes_by_id = {item.quote.quote_id: item for item in self.quotes}
        if len(quotes_by_id) != len(self.quotes):
            raise ValueError("Each evaluated quote ID must appear once")
        for item in self.quotes:
            if item.quote.rfq_id != self.rfq_id:
                raise ValueError("All evaluated quotes must belong to the requested RFQ")
            if item.duplicate_of:
                original = quotes_by_id.get(item.duplicate_of)
                if original is None or original is item or original.duplicate_of:
                    raise ValueError("Duplicate must reference a distinct original quote in this RFQ")

        if self.status == "AWAITING_APPROVAL":
            winner = quotes_by_id.get(self.recommended_quote)
            if winner is None or winner.classification != "COMPLIANT" or winner.duplicate_of:
                raise ValueError("Approval requires a distinct compliant recommended quote")
        elif self.recommended_quote is not None:
            raise ValueError("Only AWAITING_APPROVAL can carry a recommendation")

        if self.status == "NO_COMPLIANT_BID" and any(
            item.classification in ("COMPLIANT", "NEEDS_INFORMATION") for item in self.quotes
        ):
            raise ValueError("A no-bid result requires resolved, non-compliant or ineligible quotes")
        if self.status == "NEEDS_INFORMATION" and not self.information_needed:
            raise ValueError("Identify the information required to continue")
        return self


class QuoteExplanation(ProcurementModel):
    quote_id: NonBlank
    evidence_ids: tuple[NonBlank, ...]
    narrative: NonBlank


class StructuredExplanation(ProcurementModel):
    rfq_id: NonBlank
    summary: NonBlank
    next_action: NonBlank
    policy_citations: tuple[PolicyClause, ...]
    quote_explanations: tuple[QuoteExplanation, ...]


class ExplainedEvaluation(ProcurementModel):
    evaluation: EvaluationResult
    explanation: StructuredExplanation
