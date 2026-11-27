from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import pytest

from app.data_loader import load_data
from app.rules import calculate_tax_inclusive_total, evaluate_rfq, is_supplier_eligible
from app.schemas import EvaluationResult, SupplierQuote


@pytest.fixture
def data():
    return load_data()


def test_tax_calculation_uses_decimal_and_rounds_to_two_places():
    quote = SupplierQuote(
        quote_id="Q",
        rfq_id="R",
        supplier_id="S",
        unit_price_inr="0.05",
        offered_qty=1,
        promised_delivery="2026-01-01",
        tax_percent="10",
    )
    assert calculate_tax_inclusive_total(quote) == Decimal("0.06")


def test_examples_have_deterministic_recommendations_and_duplicates(data):
    expected = {
        "RFQ-101": "Q-101",
        "RFQ-102": "Q-104",
        "RFQ-103": "Q-106",
        "RFQ-104": "Q-108",
        "RFQ-105": "Q-110",
        "RFQ-106": "Q-112",
    }
    for rfq_id, quote_id in expected.items():
        result = evaluate_rfq(data, rfq_id)
        assert result.status == "AWAITING_APPROVAL"
        assert result.recommended_quote == quote_id
        assert result.contract_awarded is False
    duplicate = evaluate_rfq(data, "RFQ-105")
    assert (
        next(item for item in duplicate.quotes if item.quote.quote_id == "Q-114").duplicate_of
        == "Q-110"
    )


def test_rule_results_match_all_supplied_evaluations(data):
    examples = Path(__file__).resolve().parents[1] / "examples"
    for path in examples.glob("*.json"):
        for record in json.loads(path.read_text(encoding="utf-8")):
            expected = EvaluationResult.model_validate(record["evaluation"])
            assert evaluate_rfq(data, expected.rfq_id) == expected


def test_ranking_ties_use_delivery_then_defect_performance(data):
    quotes = dict(data.quotes)
    quotes["Q-111"] = quotes["Q-111"].model_copy(update={"unit_price_inr": Decimal("17800")})
    performance = dict(data.performance)
    performance["S-04"] = performance["S-04"].model_copy(
        update={"on_time_delivery_pct": Decimal("98")}
    )
    tied = replace(data, quotes=quotes, performance=performance)
    assert evaluate_rfq(tied, "RFQ-105").recommended_quote == "Q-111"

    performance["S-04"] = performance["S-04"].model_copy(
        update={
            "on_time_delivery_pct": Decimal("97"),
            "defect_rate_pct": Decimal("0.5"),
        }
    )
    tied = replace(tied, performance=performance)
    assert evaluate_rfq(tied, "RFQ-105").recommended_quote == "Q-111"


def test_eligibility_requires_approved_onboarding_and_verified_tax(data):
    quotes = data.for_rfq("RFQ-101").quotes
    assert is_supplier_eligible(quotes[0])
    assert not is_supplier_eligible(quotes[2])
    result = evaluate_rfq(data, "RFQ-101")
    ineligible = next(item for item in result.quotes if item.quote.quote_id == "Q-103")
    assert ineligible.classification == "INELIGIBLE"
    assert "INELIGIBLE_SUPPLIER" in ineligible.reasons


def test_missing_supplier_data_requests_information(data):
    incomplete = replace(data, suppliers={key: value for key, value in data.suppliers.items() if key != "S-01"})
    result = evaluate_rfq(incomplete, "RFQ-101")
    assert result.status == "NEEDS_INFORMATION"
    assert "suppliers.csv:S-01" in result.information_needed
    assert result.recommended_quote is None


def test_unknown_rfq_and_no_quotes_have_safe_results(data):
    unknown = evaluate_rfq(data, "RFQ-999")
    assert unknown.status == "NEEDS_INFORMATION"
    assert unknown.information_needed == ("rfqs.csv:RFQ-999",)
    no_quotes = evaluate_rfq(replace(data, quotes={}), "RFQ-101")
    assert no_quotes.status == "NO_COMPLIANT_BID"