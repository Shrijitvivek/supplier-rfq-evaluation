## LangChain and data validation

Call `load_data()` from `app.data_loader` to read and validate the four supplied CSVs. Use `data.for_rfq(request.rfq_id)` after validating the input with `RFQRequest`. It returns the RFQ, its joined quote records, and source IDs for missing records.

Pass the rule engine's pre-approval `EvaluationResult` to `build_explanation_chain(data).invoke(result)`. The chain validates the payload, checks it against the source quotes, and returns an `ExplainedEvaluation` with a narrative and row-level evidence. It makes zero model calls and preserves the rule engine's classifications, totals and recommendation. Use `model_dump(mode="json")` for serialization. INR amounts are decimal strings.

Run `python -m pytest tests/test_data_loader.py tests/test_langchain.py` from the repository root. The five JSON example files contain six evaluation records. The supplier eligibility file groups the network-switch and headset RFQs.

### Data and explanation failure handling

Malformed CSV headers or values raise `DataValidationError` with the source path and, for row errors, the row number. Correct the input before retrying. Missing join records are reported by source ID so the workflow can request information when a decision depends on it. Invalid evaluation payloads or quote records that differ from the source stop explanation generation before any approval handoff. Re-running the same valid inputs produces the same explanation.

### LangChain benchmark and demo

Median elapsed time over 100 warm runs on Windows with Python 3.12. Each run validates the supplied evaluation, collects source evidence, and renders the explanation. The timing excludes CSV and JSON reads and uses prepared rule findings.

| Example | RFQs | Median | Model calls |
| --- | --- | ---: | ---: |
| Laptops | RFQ-101 | 0.403 ms | 0 |
| Chairs | RFQ-102 | 0.490 ms | 0 |
| Monitors | RFQ-103 | 0.393 ms | 0 |
| Supplier eligibility | RFQ-104, RFQ-106 | 0.777 ms | 0 |
| Printers | RFQ-105 | 0.403 ms | 0 |

To demonstrate duplicate evidence and the pre-approval handoff, run this from the repository root.

```python
import json
from pathlib import Path

from app.data_loader import load_data
from app.langchain_components import build_explanation_chain

example = json.loads(Path("examples/05-printers.json").read_text(encoding="utf-8"))[0]
result = build_explanation_chain(load_data()).invoke(example["evaluation"])
print(result.model_dump_json(indent=2))
```

## CrewAI and AutoGen advisory reviews

The Adarsh adapters add independent advisory reviews to a LangGraph workflow without changing
the deterministic evaluation or invoking approval. Both nodes accept a state containing the
validated pre-approval `evaluation` and its `procurement_facts`, and return their result under
`crewai_review` or `autogen_review`.

```python
from app.autogen_adapter import create_autogen_adapter
from app.crewai_adapter import create_crewai_adapter

crew_node = create_crewai_adapter()
autogen_node = create_autogen_adapter()

async def review(state):
    crew_update = crew_node(state)
    autogen_update = await autogen_node(state)
    return crew_update, autogen_update
```

The CrewAI team runs the Buyer and Supplier-Risk Analyst tasks sequentially. AutoGen runs the
Cost Reviewer and Compliance Reviewer in a bounded round-robin chat. Both adapters validate
structured outputs, reject missing reviewer results, and set `contract_awarded` to `false`.
Their findings are advisory: the deterministic rules remain authoritative and a human must
approve before any simulated award.

Install the requirements and set `OPENAI_API_KEY` before running live reviews. CrewAI uses
`CREWAI_MODEL` (default `openai/gpt-4o-mini`); AutoGen uses `AUTOGEN_MODEL` (default
`gpt-4o-mini`). Tests use fake framework runs and do not call a model:

```powershell
python -m pytest tests/test_crewai.py tests/test_autogen.py
```

See [docs/architecture.png](docs/architecture.png) for the application flow and the separation
between advisory reviews and the human approval/award boundary.
