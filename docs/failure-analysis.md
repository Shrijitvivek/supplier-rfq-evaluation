# Failure analysis

## Input and deterministic evaluation

- Malformed CSV headers, rows, or values raise `DataValidationError` with the source
  path and row when available. The workflow stops before evaluation.
- Missing RFQ or supplier information, or missing supplier performance needed to
  rank an otherwise compliant quote, produces `NEEDS_INFORMATION`; no approval or
  award node is reached.
- An RFQ with no distinct compliant quote produces `NO_COMPLIANT_BID`. It does not
  interrupt for approval and does not record an award.
- Duplicate quotations are linked to their first matching quote and excluded from
  ranking after the first instance.

## Advisory framework failures

- Advisory reviews are opt-in. The default graph makes no provider calls.
- The CrewAI and AutoGen nodes validate the evaluation and source facts before
  invoking a framework. A mismatched RFQ or incomplete source quote set raises an
  error before the model call.
- Missing reviewer outputs, duplicate AutoGen reviewer messages, and invalid
  structured output raise explicit `RuntimeError`, `TypeError`, or Pydantic
  validation errors. The workflow fails before the human approval interrupt.
- Provider authentication, network, quota, and framework exceptions propagate; the
  application does not replace them with a successful-looking review. No retry or
  backoff policy is configured.
- The advisory data is sent to configured model providers when
  `--advisory-reviews` is used. Keep provider credentials in environment variables
  or a local ignored `.env`; do not commit or log them.

## Approval, checkpoint, and award

- A review-enabled graph must be driven with `ainvoke()`; the CLI does this when
  `--advisory-reviews` is selected.
- An invalid resume payload raises `ValueError`. A rejected decision leaves
  `contract_awarded` false. The award node independently requires both
  `approval_status=APPROVED` and an affirmative decision.
- The default `MemorySaver` checkpoint is process-local. Use a durable
  checkpointer and the same `thread_id` for recovery after a process restart.
- The approval interrupt is not an approval. If execution stops there, the state
  contains a recommendation but no simulated award.

## Validation evidence

The offline unit and integration test suite exercises deterministic decisions,
checkpoint/resume, both advisory adapters with fake framework outputs, malformed
review data, and the approval guard. Live provider/network failure modes require
provider-specific integration tests and are not represented by those fakes.
