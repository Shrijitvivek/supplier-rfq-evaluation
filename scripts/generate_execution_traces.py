"""Generate deterministic, pre-approval LangGraph traces for all example RFQs."""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.data_loader import load_data
from app.graph import build_graph
from app.schemas import EvaluationResult


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "execution-traces.json"


def main() -> None:
    app = build_graph(load_data())
    traces = []
    for path in sorted((ROOT / "examples").glob("*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            rfq_id = record["evaluation"]["rfq_id"]
            updates = app.stream(
                {"rfq_id": rfq_id},
                {"configurable": {"thread_id": f"trace-{rfq_id}"}},
                stream_mode="updates",
            )
            events = []
            for update in updates:
                if "evaluate" in update:
                    evaluation = EvaluationResult.model_validate(update["evaluate"]["evaluation"])
                    events.append(
                        {
                            "node": "evaluate",
                            "status": evaluation.status,
                            "recommendation": evaluation.recommended_quote,
                            "quotes": [
                                {
                                    "quote_id": item.quote.quote_id,
                                    "classification": item.classification,
                                    "duplicate_of": item.duplicate_of,
                                }
                                for item in evaluation.quotes
                            ],
                            "contract_awarded": False,
                        }
                    )
                if "__interrupt__" in update:
                    events.append(
                        {
                            "node": "approval",
                            "status": "AWAITING_HUMAN",
                            "contract_awarded": False,
                        }
                    )
            traces.append(
                {
                    "rfq_id": rfq_id,
                    "advisory_reviews": "not invoked; deterministic offline trace",
                    "events": events,
                }
            )

    OUTPUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "approval_policy": "No approval was supplied; no award was recorded.",
                "traces": traces,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(traces)} traces to {OUTPUT}")


if __name__ == "__main__":
    main()
