"""Measure the local deterministic evaluation explanation path for five examples."""

import json
import platform
import statistics
import time
from pathlib import Path

from app.data_loader import load_data
from app.langchain_components import build_explanation_chain


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUTPUT = ROOT / "docs" / "framework-benchmark.json"
WARMUP_RUNS = 10
MEASURED_RUNS = 100


def main() -> None:
    data = load_data()
    chain = build_explanation_chain(data)
    measurements = []
    for path in sorted(EXAMPLES.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        evaluations = [record["evaluation"] for record in records]

        def run_case() -> None:
            for evaluation in evaluations:
                chain.invoke(evaluation)

        for _ in range(WARMUP_RUNS):
            run_case()
        samples = []
        for _ in range(MEASURED_RUNS):
            start = time.perf_counter_ns()
            run_case()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        measurements.append(
            {
                "case": path.stem,
                "rfqs": len(evaluations),
                "median_ms": round(statistics.median(samples), 3),
                "model_calls": 0,
            }
        )

    report = {
        "framework": "LangChain LCEL local explanation chain",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmup_runs_per_case": WARMUP_RUNS,
        "measured_runs_per_case": MEASURED_RUNS,
        "measurement": "Median wall time per case; source CSV/JSON loading excluded.",
        "cases": measurements,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
