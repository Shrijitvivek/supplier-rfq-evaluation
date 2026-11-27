# Project Implementation and Usage

## 1. Project Overview

The Supplier RFQ Evaluation and Award Recommendation system automates the evaluation of supplier quotations against procurement requirements and generates explainable award recommendations.

The application demonstrates how four major agentic AI frameworks and orchestration approaches can work together in a single procurement use case:

- **LangChain** – Builds a deterministic explanation chain with source-level evidence.
- **LangGraph** – Orchestrates the end-to-end procurement workflow and manages application state.
- **CrewAI** – Provides buyer and supplier-risk advisory reviews through specialized agents.
- **AutoGen** – Provides independent cost and compliance reviews through a multi-agent conversation.

The system follows a hybrid architecture: deterministic business rules make the actual procurement decision, while AI agents provide additional advisory perspectives. Human approval is required before a simulated contract award can be recorded.

The architecture is designed to maintain separation between evaluation, advisory review, and award approval. AI-generated recommendations cannot independently approve suppliers or award contracts.

## 2. End-to-End Workflow

The application follows this sequence:

1. **RFQ input:** The user supplies an RFQ identifier through the command-line interface.
2. **Input validation:** The request is validated using the Pydantic request schema.
3. **Data loading:** The application loads the four supplied CSV datasets and validates their structure and values.
4. **Data joining:** The RFQ is matched with its supplier quotations and related supplier information.
5. **Deterministic evaluation:** Procurement rules classify quotations, calculate totals, identify issues, and determine the recommended compliant quotation.
6. **Explanation generation:** The LangChain component generates a narrative explanation supported by source records and policy citations.
7. **Advisory reviews:** When enabled, CrewAI and AutoGen independently review the pre-approval evaluation.
8. **Human approval:** The workflow presents the recommendation for explicit human approval.
9. **Simulated award:** An award is recorded only when the human approves the recommendation. Rejection leaves the contract unawarded.

LangGraph controls the workflow transitions and passes the application state between the relevant nodes.

## 3. Deterministic Procurement Evaluation

The procurement rule engine is implemented independently of the AI advisory frameworks. It remains the authoritative source for quotation classifications and award recommendations.

The evaluation process considers the supplied RFQ requirements, quotation details, supplier information, and procurement policies.

Its responsibilities include:

- Validating RFQ and quotation information.
- Checking supplier eligibility.
- Evaluating quotation compliance.
- Checking delivery commitments against RFQ requirements.
- Evaluating quoted prices against procurement constraints.
- Calculating tax-inclusive quotation totals.
- Identifying duplicate quotations.
- Recording reasons for non-compliance and ineligibility.
- Attaching applicable procurement policy citations.
- Selecting the recommended compliant quotation.
- Identifying cases where additional information is required.

Quotation evaluations use classifications such as:

- `COMPLIANT` – The quotation satisfies the applicable procurement requirements.
- `NON_COMPLIANT` – The quotation violates one or more procurement requirements.
- `INELIGIBLE` – The supplier does not meet the applicable eligibility requirements.

The evaluation produces a structured result containing the RFQ identifier, quotation-level classifications, calculated totals, reasons, duplicate references, policy citations, information requirements, and recommended quotation.

The rule engine does not depend on an AI model to make procurement decisions. This ensures that the same valid inputs and procurement rules produce consistent evaluation results.

## 4. LangGraph Workflow Orchestration

LangGraph is responsible for coordinating the application workflow.

The graph manages the shared procurement state and connects the deterministic evaluation, optional advisory reviews, and human approval stages.

The workflow includes:

- State management across workflow nodes.
- Deterministic evaluation.
- Optional CrewAI advisory review.
- Optional AutoGen advisory review.
- Conditional workflow transitions.
- Human approval handling.
- Final award-status management.

The advisory reviews are optional and can be enabled through the `--advisory-reviews` command-line argument.

When advisory reviews are enabled, the workflow executes the CrewAI review followed by the AutoGen review before reaching the approval stage.

The deterministic evaluation remains unchanged by the advisory agents. Their findings are stored separately in the workflow state.

## 5. LangChain: Data Validation and Explainable Evaluation

The LangChain implementation provides a structured explanation layer for the deterministic procurement evaluation.

It consists of two main components:

### Data loading and validation

Call `load_data()` from `app.data_loader` to load and validate the four supplied CSV files.

The data loader validates the input datasets and provides the data required by the evaluation and explanation components.

Use `RFQRequest` to validate the incoming RFQ request, then call:

```python
data.for_rfq(request.rfq_id)
```

This returns the RFQ, its joined quotation records, and source IDs for missing records.

### Evidence-based explanation chain

Pass the rule engine's pre-approval `EvaluationResult` to:

```python
build_explanation_chain(data).invoke(result)
```

The chain validates the evaluation payload, verifies quotation information against the source data, and returns an `ExplainedEvaluation` containing:

- A narrative explanation of the evaluation.
- Row-level source evidence.
- Supporting procurement information.
- Policy citations.
- The original deterministic evaluation findings.

The explanation chain makes zero model calls. It does not independently change quotation classifications, calculated totals, or the recommended quotation.

Use `model_dump(mode="json")` for serialization. INR monetary amounts are represented as decimal strings.

### Data and explanation failure handling

Malformed CSV headers or values raise `DataValidationError` with the source path and, for row-level errors, the corresponding row number.

Missing join records are reported through their source IDs. When missing information affects a procurement decision, the workflow can request additional information rather than proceeding with an unsupported decision.

Invalid evaluation payloads or quotation records that differ from the source data stop explanation generation before the approval handoff.

Re-running the same valid inputs produces the same explanation.

## 6. CrewAI: Buyer and Supplier-Risk Advisory Reviews

CrewAI adds a specialized advisory review layer to the procurement workflow.

The implementation uses two specialized reviewer roles:

**Buyer**

Reviews the procurement recommendation from the buyer's perspective, including:

- Whether the recommended quotation aligns with procurement requirements.
- Potential cost concerns.
- Whether the recommendation should be challenged.
- Whether the evaluation provides sufficient justification.

**Supplier-Risk Analyst**

Reviews supplier-related risks, including:

- Supplier eligibility.
- Supplier risk information.
- Potential supplier reliability concerns.
- Risks associated with non-compliant or ineligible quotations.

The two tasks execute sequentially within the CrewAI team.

The adapter validates the structured reviewer outputs and returns the results under `crewai_review` in the LangGraph state.

CrewAI findings are advisory only. They cannot modify the authoritative deterministic evaluation or approve a contract award.

## 7. AutoGen: Cost and Compliance Advisory Reviews

AutoGen provides another independent review layer through a bounded round-robin multi-agent conversation.

The implementation uses two reviewer roles:

**Cost Reviewer**

Examines the procurement evaluation from a cost perspective, including:

- Quotation price differences.
- Cost-related concerns.
- The relationship between price and compliance.
- Potential issues with the recommended quotation.

**Compliance Reviewer**

Examines the procurement evaluation from a compliance perspective, including:

- Supplier eligibility.
- Quotation compliance.
- Delivery-related concerns.
- Potential inconsistencies requiring further verification.

The reviewers exchange information within a bounded round-robin chat.

The adapter validates the structured results and returns them under `autogen_review` in the LangGraph state.

Like CrewAI, AutoGen is an advisory component. Its findings do not override deterministic procurement classifications or authorize an award.

## 8. Human Approval and Simulated Award

Human approval is an essential control in the procurement workflow.

The system separates the recommendation process from the final award decision.

After deterministic evaluation and any enabled advisory reviews, the workflow presents the recommended quotation for human approval.

The approval component handles:

- Approval or rejection of the recommendation.
- Recording the approval outcome.
- Maintaining the contract-award status.
- Preventing an award when approval is rejected.

If the recommendation is rejected:

- `approval_status` becomes `REJECTED`.
- `contract_awarded` remains `false`.
- No simulated award is recorded.

If approved, the system records a simulated award according to the approval workflow.

This is a demonstration of procurement decision control, not an actual purchase order or contract execution.

## 9. Running the Application

### Prerequisites

- Python 3.12.
- A virtual environment.
- Project dependencies installed from the requirements file.
- An OpenAI API key for live CrewAI and AutoGen advisory reviews.

Activate the virtual environment and install the dependencies:

```bash
python -m venv my_venv
source my_venv/bin/activate
pip install -r requirements.txt
```

Configure the environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
CREWAI_MODEL=openai/gpt-4o-mini
AUTOGEN_MODEL=gpt-4o-mini
```

The model variables are configurable and use the documented defaults when not explicitly supplied.

### Run deterministic evaluation

From the repository root:

```bash
python -m app.main RFQ-101
```

This executes the procurement workflow without enabling the optional advisory reviews.

### Run with CrewAI and AutoGen reviews

```bash
python -m app.main RFQ-101 --advisory-reviews
```

This enables both advisory review stages before human approval.

The application displays the recommended quotation and requests approval through the command line.

Enter `y` to approve or `n` to reject the simulated award.

## 10. Example Scenarios

The repository includes five JSON example files containing six evaluation records.

| Example | RFQ | Scenario |
| --- | --- | --- |
| `01-laptops.json` | RFQ-101 | Laptop quotation evaluation |
| `02-chairs.json` | RFQ-102 | Chair quotation evaluation |
| `03-monitors.json` | RFQ-103 | Monitor quotation evaluation |
| `04-supplier-eligibility.json` | RFQ-104, RFQ-106 | Supplier eligibility scenarios |
| `05-printers.json` | RFQ-105 | Printer evaluation and duplicate quotation evidence |

The examples demonstrate procurement evaluation across different RFQs, including supplier eligibility and duplicate quotation handling.

## 11. Testing and Validation

The project includes tests for the deterministic procurement logic, workflow orchestration, human approval, data validation, LangChain explanations, and the CrewAI and AutoGen adapters.

The framework adapter tests use simulated framework runs and do not require live model calls.

Run the complete test suite from the repository root:

```bash
python -m pytest -v
```

Run specific component tests:

```bash
python -m pytest tests/test_rules.py tests/test_graph.py tests/test_approval.py
```

```bash
python -m pytest tests/test_data_loader.py tests/test_langchain.py
```

```bash
python -m pytest tests/test_crewai.py tests/test_autogen.py
```

The verified project test run completed with **42 passed**.

Live advisory review execution was also verified with RFQ-101. Both CrewAI and AutoGen returned their structured reviews, and the human rejection was correctly reflected in the final workflow result.

## 12. LangChain Performance Benchmark

The LangChain benchmark measures the execution time of the deterministic explanation component.

The reported benchmark uses 100 warm runs on Windows with Python 3.12. Each run validates the supplied evaluation, collects source evidence, and renders the explanation. CSV and JSON file reads are excluded, and prepared rule findings are used.

| Example | RFQs | Median execution time | Model calls |
| --- | --- | ---: | ---: |
| Laptops | RFQ-101 | 0.403 ms | 0 |
| Chairs | RFQ-102 | 0.490 ms | 0 |
| Monitors | RFQ-103 | 0.393 ms | 0 |
| Supplier eligibility | RFQ-104, RFQ-106 | 0.777 ms | 0 |
| Printers | RFQ-105 | 0.403 ms | 0 |

These measurements are environment-specific and should not be interpreted as performance guarantees on other systems.

See `docs/framework-benchmark.json` for the project's benchmark results.

## 13. Project Documentation and Deliverables

The repository includes supporting documentation and example artifacts:

| File | Purpose |
| --- | --- |
| `docs/architecture.md` | Written description of the system architecture |
| `docs/architecture.png` | Visual representation of the application workflow |
| `docs/execution-traces.json` | Recorded workflow execution traces |
| `docs/failure-analysis.md` | Analysis of failure scenarios and handling |
| `docs/framework-benchmark.json` | Framework benchmark results |
| `examples/` | JSON evaluation examples |
| `benchmarks/` | Benchmark-related project files |
| `scripts/` | Supporting project scripts |

Refer to the architecture diagram for the relationship between deterministic evaluation, framework advisory reviews, and the human approval boundary.

## 14. Design Principles

The implementation follows these principles:

- **Deterministic decision-making:** Procurement classifications and recommendations are produced by explicit business rules.
- **Separation of responsibilities:** LangChain, LangGraph, CrewAI, and AutoGen each have distinct responsibilities.
- **Explainability:** Evaluation results include reasons, evidence, and procurement policy citations.
- **Data validation:** Invalid or inconsistent source data is detected before it can influence the decision.
- **Independent advisory reviews:** CrewAI and AutoGen provide additional perspectives without replacing the rule engine.
- **Human control:** A simulated award requires explicit human approval.
- **Testability:** Components are tested independently, alongside the integrated workflow.
- **Reproducibility:** Deterministic evaluation and explanation behavior can be repeated using the same valid inputs.

The result is an explainable supplier quotation evaluation system that combines deterministic procurement rules, structured data validation, multi-framework agent orchestration, independent advisory reviews, and human-controlled award approval.