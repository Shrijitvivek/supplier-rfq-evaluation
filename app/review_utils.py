"""Shared serialization for advisory multi-agent review inputs."""

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
import json
from typing import Any, Mapping

from pydantic import BaseModel

from app.schemas import EvaluationResult


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported review context value: {type(value).__name__}")


def serialize_review_context(
    evaluation: EvaluationResult | Mapping[str, Any],
    procurement_facts: Any,
) -> tuple[EvaluationResult, str]:
    """Validate the pre-approval result and serialize it with validated source facts."""
    parsed = EvaluationResult.model_validate(evaluation)
    if procurement_facts is None:
        raise ValueError("procurement_facts are required for an advisory review")
    facts = _json_value(procurement_facts)
    if isinstance(facts, Mapping):
        source_rfq_id = facts.get("rfq_id")
        if source_rfq_id is None and isinstance(facts.get("rfq"), Mapping):
            source_rfq_id = facts["rfq"].get("rfq_id")
        if source_rfq_id is not None and source_rfq_id != parsed.rfq_id:
            raise ValueError("Evaluation and procurement_facts must refer to the same RFQ")
        if "quotes" in facts:
            source_quotes = facts["quotes"]
            if not isinstance(source_quotes, list) or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("quote"), Mapping)
                or "quote_id" not in item["quote"]
                or not isinstance(item["quote"]["quote_id"], str)
                for item in source_quotes
            ):
                raise ValueError("procurement_facts quotes must contain source quote records")
            source_quote_ids = sorted(item["quote"]["quote_id"] for item in source_quotes)
            evaluated_quote_ids = sorted(item.quote.quote_id for item in parsed.quotes)
            if source_quote_ids != evaluated_quote_ids:
                raise ValueError("Evaluation must preserve every source quote for the RFQ")
    context = {
        "evaluation": parsed.model_dump(mode="json"),
        "procurement_facts": facts,
    }
    return parsed, json.dumps(context, indent=2, sort_keys=True, allow_nan=False)
