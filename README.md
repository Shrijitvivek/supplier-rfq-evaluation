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
