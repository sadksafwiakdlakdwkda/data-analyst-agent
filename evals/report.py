"""Aggregate evaluation results without counting unreviewed answers as correct."""

import math
import statistics
from collections import defaultdict


def rate(values):
    checked = [v for v in values if v is not None]
    return sum(checked) / len(checked) if checked else None


def summarize(records):
    times = sorted(r["elapsed_seconds"] for r in records)
    sql_calls = sum(r["metrics"]["sql_calls"] for r in records)
    sql_errors = sum(r["metrics"]["sql_errors"] for r in records)
    recoverable = [r for r in records if r["metrics"]["tool_errors"] > 0
                   and r["score"]["automated_passed"] is not None]
    usage_known = [r for r in records if r["metrics"]["total_tokens"] is not None]
    return {
        "cases": len(records),
        "automatically_scored_cases": sum(r["score"]["automated_passed"] is not None for r in records),
        "automated_pass_rate": rate([r["score"]["automated_passed"] for r in records]),
        "result_pass_rate": rate([r["score"]["result_passed"] for r in records]),
        "inspection_evidence_passes": sum(r["score"].get("evidence_source") == "inspect_row_count"
                                          and r["score"]["automated_passed"] is True for r in records),
        "scalar_answer_pass_rate": rate([r["score"]["answer_passed"] for r in records]),
        "manual_review_cases": sum(r["score"]["review_required"] for r in records),
        "run_errors": sum(r.get("error") is not None for r in records),
        "sql_calls": sql_calls,
        "sql_success_rate": (sql_calls - sql_errors) / sql_calls if sql_calls else None,
        "observed_recovery_cases": len(recoverable),
        "observed_recovery_rate": rate([r["score"]["automated_passed"] for r in recoverable]),
        "mean_tool_calls": statistics.mean(r["metrics"]["tool_calls"] for r in records) if records else None,
        "latency_p50_seconds": statistics.median(times) if times else None,
        "latency_p95_seconds": times[max(0, math.ceil(len(times) * 0.95) - 1)] if times else None,
        "total_tokens_reported": sum(r["metrics"]["total_tokens"] for r in usage_known),
        "token_usage_coverage": len(usage_known) / len(records) if records else None,
    }


def build_report(records, metadata):
    groups = defaultdict(list)
    for record in records:
        groups[(record["mode"], record["category"])].append(record)
    modes = sorted({r["mode"] for r in records})
    return {
        "metadata": metadata,
        "summary": {mode: summarize([r for r in records if r["mode"] == mode]) for mode in modes},
        # Query-only comparison excludes chart tasks, which baseline cannot do,
        # and manual cases, which have no automated score.
        "query_comparison": {mode: summarize([r for r in records if r["mode"] == mode
                             and r["check"] not in ("chart", "manual")]) for mode in modes},
        "by_category": {f"{mode}/{category}": summarize(items)
                        for (mode, category), items in sorted(groups.items())},
        "records": records,
    }


def markdown_report(report):
    def display(value):
        if value is None:
            return "N/A"
        return f"{value:.1%}"

    lines = ["# Evaluation report", "",
             "Automated scores check query data, scalar final values and chart bindings.",
             "Opted-in row-count cases may use structured inspect_dataset evidence without SQL.",
             "Table/chart prose and unavailable-information answers require manual review.",
             "Recovery is observational; no deterministic fault injection is performed.", "",
             "| Mode (query tasks only) | Cases | Automated pass | SQL success | P95 seconds |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for mode, summary in report["query_comparison"].items():
        latency = summary["latency_p95_seconds"]
        latency_text = f"{latency:.2f}" if latency is not None else "N/A"
        lines.append(f"| {mode} | {summary['cases']} | {display(summary['automated_pass_rate'])} | "
                     f"{display(summary['sql_success_rate'])} | {latency_text} |")
    lines.extend(["", "## Failed checks and reviews", ""])
    for record in report["records"]:
        if record["score"]["automated_passed"] is not True or record["score"]["review_required"]:
            lines.append(f"- {record['mode']}/{record['case_id']} (repeat {record['repeat']}): "
                         f"{record['score']['reason']}; review={record['score']['review_required']}")
    return "\n".join(lines) + "\n"
