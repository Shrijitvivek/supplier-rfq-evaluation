from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.data_loader import load_data
from app.langchain_components import build_explanation_chain
from app.schemas import EvaluationResult, ExplainedEvaluation


@pytest.fixture
def chain():
    return build_explanation_chain(load_data())


def test_examples_cover_all_six_rfqs_through_the_chain(chain):
    examples = Path(__file__).resolve().parents[1] / "examples"
    files = sorted(examples.glob("*.json"))
    assert len(files) == 5
    recommendations = {}
    for path in files:
        for record in json.loads(path.read_text(encoding="utf-8")):
            expected = ExplainedEvaluation.model_validate(record)
            result = chain.invoke(record["evaluation"])
            assert result == expected
            for finding in result.evaluation.quotes:
                quote = finding.quote
                total = quote.offered_qty * quote.unit_price_inr * (1 + quote.tax_percent / 100)
                assert finding.tax_inclusive_total_inr == total.quantize(Decimal("0.01"))
                if quote.quote_id == "Q-114":
                    assert finding.duplicate_of == "Q-110"
                    assert "R8" in finding.policy_citations
            assert result.evaluation.contract_awarded is False
            assert "R6" in result.explanation.policy_citations
            assert result.evaluation.rfq_id not in recommendations
            recommendations[result.evaluation.rfq_id] = result.evaluation.recommended_quote
    assert recommendations == {
        "RFQ-101": "Q-101",
        "RFQ-102": "Q-104",
        "RFQ-103": "Q-106",
        "RFQ-104": "Q-108",
        "RFQ-105": "Q-110",
        "RFQ-106": "Q-112",
    }


def test_unknown_rfq_requests_information_without_inventing_evidence(chain):
    result = chain.invoke(
        EvaluationResult(
            rfq_id="UNKNOWN",
            status="NEEDS_INFORMATION",
            information_needed=("rfqs.csv:UNKNOWN",),
        )
    )
    assert result.explanation.policy_citations == ("R1",)
    assert result.explanation.quote_explanations == ()
    assert "rfqs.csv:UNKNOWN" in result.explanation.next_action
    assert result.evaluation.recommended_quote is None


def test_no_quotes_requests_a_revised_rfq():
    data = load_data()
    chain = build_explanation_chain(replace(data, quotes={}))
    result = chain.invoke(EvaluationResult(rfq_id="RFQ-101", status="NO_COMPLIANT_BID"))
    assert result.explanation.policy_citations == ("R7",)
    assert "revised RFQ" in result.explanation.next_action
    assert result.evaluation.recommended_quote is None


def test_preapproval_contract_rejects_an_award(chain):
    with pytest.raises(ValidationError, match="contract_awarded"):
        chain.invoke(
            {"rfq_id": "RFQ-101", "status": "NO_COMPLIANT_BID", "contract_awarded": True}
        )


def test_chain_requires_every_source_quote(chain):
    with pytest.raises(ValueError, match="preserve every source quote"):
        chain.invoke(EvaluationResult(rfq_id="RFQ-101", status="NO_COMPLIANT_BID"))
