"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report_stub(metrics: MetricsReport) -> str:
    """Return a minimal report stub.

    TODO(student): replace with a richer report using the template in reports/.
    """
    return f"""# Day 08 Lab Report

## Metrics summary

- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.2%}
- Average nodes visited: {metrics.avg_nodes_visited:.2f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}

## TODO(student)

Explain your architecture, state schema, failure modes, and improvement plan.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite a manually written report (longer than the generated stub).
    stub = render_report_stub(metrics)
    if path.exists() and len(path.read_text(encoding="utf-8")) > len(stub):
        return
    path.write_text(stub, encoding="utf-8")
