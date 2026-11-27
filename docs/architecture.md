# Workflow architecture

```mermaid
flowchart TD
    A[Validated CSV data] --> B[LangGraph: deterministic evaluation]
    B --> C{Advisory reviews enabled?}
    C -- No --> F{Recommendation available?}
    C -- Yes --> D[CrewAI: Buyer then Supplier-Risk Analyst]
    D --> E[AutoGen: Cost Reviewer then Compliance Reviewer]
    E --> F
    F -- No --> G[Return findings / request information]
    F -- Yes --> H[Checkpointed human approval interrupt]
    H -- Rejected --> I[Return without an award]
    H -- Approved --> J[Record simulated award]
```

`build_graph()` defaults to the offline deterministic path. Setting
`include_advisory_reviews=True` adds both reviewers in sequence before the approval
gate; invoke that graph with `ainvoke()` because the AutoGen node is asynchronous.
Both reviewers receive the same validated evaluation and source facts, return
structured advisory output, and cannot change the deterministic recommendation or
record an award.

The approval node pauses with a LangGraph interrupt. Resume the same thread with an
explicit `{"approved": true}` decision. `MemorySaver` is the default checkpointer;
pass a durable LangGraph checkpointer to `build_graph(checkpointer=...)` when a
workflow must resume across process restarts. An award is only recorded by the node
after an affirmative human decision.

The companion [architecture image](./architecture.png) is a high-level overview;
this document describes the executable graph and its ordering.
