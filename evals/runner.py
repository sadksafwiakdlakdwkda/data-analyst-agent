"""Run with `python -m evals.runner`; live model calls require --live."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import (CSV_PATH, create_dataset_connection, create_tools, get_models,
                   json_value, latest_query_result, run_agent)
from evals.report import build_report, markdown_report
from evals.scoring import rows_match, score

ROOT = Path(__file__).resolve().parent
ERROR_PREFIXES = ("SQL Validation Error:", "SQL Execution Error:",
                  "Dataset Inspection Error:", "Tool Error:", "Chart Error:")


def load_suite(path=ROOT / "cases.json"):
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if digest != suite["dataset_sha256"]:
        raise ValueError("Dataset fingerprint differs from the frozen evaluation suite.")
    ids = [case["id"] for case in suite["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation case IDs.")
    return suite


def verify_references(suite):
    """No LLM: check frozen answers and SQL compatibility with current tools."""
    connection = create_dataset_connection(CSV_PATH)
    try:
        execute = {t.name: t for t in create_tools(connection)}["execute_sql"]
        for case in suite["cases"]:
            if case["check"] == "manual":
                continue
            message = execute.invoke({"name": "execute_sql", "args": {"query": case["reference_sql"]},
                                      "id": case["id"], "type": "tool_call"})
            artifact = message.artifact
            if not artifact or artifact["columns"] != case["expected_columns"] or not rows_match(
                artifact["rows"], case["expected_rows"], ordered=case["ordered"],
                tolerance=case["tolerance"]
            ) or artifact["result_truncated"]:
                raise ValueError(f"Reference check failed: {case['id']}")
    finally:
        connection.close()


def question_for(case):
    question = case["question"]
    if case["check"] == "scalar":
        question += ("\nEnd with a standalone FINAL_ANSWER: <value> line. "
                     "Use only the requested number, name, or ISO date, without units or a trailing period.")
    return question


def baseline(question):
    """One model call with schema, at most one validated SQL call, no repair."""
    connection = create_dataset_connection(CSV_PATH)
    try:
        tools = {t.name: t for t in create_tools(connection)}
        schema = tools["inspect_dataset"].invoke({})
        messages = [SystemMessage(content=(
            "Answer using one execute_sql call on sales. Only SELECT is supported; "
            "CTEs and subqueries are forbidden. Do not invent unavailable fields. "
            "You cannot create charts. Schema and row count:\n" + schema
        )), HumanMessage(content=question)]
        model = get_models().bind_tools([tools["execute_sql"]], parallel_tool_calls=False)
        response = model.invoke(messages)
        messages.append(response)
        if response.tool_calls:
            call = response.tool_calls[0]
            messages.append(tools["execute_sql"].invoke({**call, "type": "tool_call"}))
            result = latest_query_result(messages)
            # No second LLM call: render the SQL result deterministically. The
            # scalar answer here is a value rendering, not prose-generation quality.
            if result["sql"] is not None:
                rows = result["rows"]
                text = (f"FINAL_ANSWER: {rows[0][0]}" if len(rows) == 1 and len(rows[0]) == 1
                        else json.dumps(rows, ensure_ascii=False))
                messages.append(AIMessage(content=text))
        return {"messages": messages, **latest_query_result(messages), "chart": None}
    finally:
        connection.close()


def tool_failed(message):
    return message.status == "error" or str(message.content).startswith(ERROR_PREFIXES)


def trace_metrics(messages):
    calls = [m for m in messages if isinstance(m, ToolMessage)]
    sql = [m for m in calls if m.name == "execute_sql"]
    models = [m for m in messages if isinstance(m, AIMessage)]
    usage = [m.usage_metadata for m in models if m.usage_metadata]
    # Missing usage is unknown, not a zero-token invocation. Locally rendered
    # baseline answers have no usage and are excluded from provider-call counting.
    provider_messages = [m for m in models if m.usage_metadata or m.response_metadata or m.tool_calls]
    known = bool(usage) and len(usage) == len(provider_messages)
    metrics = {
        "tool_calls": len(calls), "sql_calls": len(sql),
        "sql_errors": sum(tool_failed(m) or not isinstance(m.artifact, dict) for m in sql),
        "tool_errors": sum(tool_failed(m) for m in calls),
        "total_tokens": sum(u["total_tokens"] for u in usage) if known else None,
        "reported_models": sorted({str(m.response_metadata.get("model_name")) for m in models
                                   if m.response_metadata.get("model_name")}),
    }
    trace = []
    for message in messages:
        entry = {"type": message.type, "content": message.content}
        if isinstance(message, AIMessage):
            entry.update(tool_calls=message.tool_calls, usage=message.usage_metadata,
                         response_metadata=message.response_metadata)
        if isinstance(message, ToolMessage):
            entry.update(name=message.name, tool_call_id=message.tool_call_id,
                         status=message.status, artifact=message.artifact)
        trace.append(entry)
    return trace, metrics


def run_case(case, mode, repeat=1, run=None):
    started = time.perf_counter()
    error = None
    try:
        result = (run or (baseline if mode == "baseline" else run_agent))(question_for(case))
        messages = result.get("messages", [])
        answer = str(messages[-1].content) if messages else ""
        result = {**result, "answer": answer}
        scored = score(case, result)
    except Exception as exception:
        error = type(exception).__name__
        result, messages = {}, []
        scored = {"result_passed": False if case["check"] != "manual" else None,
                  "answer_passed": False if case["check"] == "scalar" else None,
                  "automated_passed": False if case["check"] != "manual" else None,
                  "review_required": case["check"] != "scalar", "reason": "run_error"}
    trace, metrics = trace_metrics(messages)
    return {"case_id": case["id"], "category": case["category"], "check": case["check"],
            "question": case["question"], "mode": mode, "repeat": repeat,
            "elapsed_seconds": round(time.perf_counter() - started, 6), "error": error,
            "score": scored, "metrics": metrics,
            "result": {k: v for k, v in result.items() if k != "messages"},
            "review_rubric": case.get("review_rubric"), "trace": trace}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable paid model calls; default only verifies references")
    parser.add_argument("--mode", choices=("agent", "baseline", "both"), default="agent")
    parser.add_argument("--category", choices=("basic", "compound", "time", "edge", "chart"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeat and --limit must be positive")
    suite = load_suite()
    verify_references(suite)
    print(f"Verified {len(suite['cases'])} cases (45 SQL references; 5 manual review cases).", flush=True)
    if not args.live:
        print("No model calls made. Add --live to run the selected evaluation.")
        return
    cases = [c for c in suite["cases"] if not args.category or c["category"] == args.category]
    if args.limit:
        cases = cases[:args.limit]
    modes = ["baseline", "agent"] if args.mode == "both" else [args.mode]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = args.output / run_id
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    # Flush every completed case so an interrupted evaluation retains evidence.
    with (destination / "records.jsonl").open("w", encoding="utf-8") as stream:
        for repeat in range(1, args.repeat + 1):
            for case in cases:
                for mode in modes:
                    record = run_case(case, mode, repeat)
                    stream.write(json.dumps(record, ensure_ascii=False, default=json_value, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    print(f"{mode}/{case['id']} #{repeat}: {record['score']['reason']}", flush=True)
    metadata = {"run_id": run_id, "suite_version": suite["version"],
                "dataset_sha256": suite["dataset_sha256"],
                "suite_sha256": hashlib.sha256((ROOT / "cases.json").read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                  [("agent.py", ROOT.parent / "agent.py"),
                                   ("runner.py", Path(__file__)),
                                   ("scoring.py", ROOT / "scoring.py")]},
                "category": args.category, "limit": args.limit, "repeat": args.repeat,
                "baseline": "one model call with schema; one validated SQL execution; deterministic value rendering",
                "numeric_absolute_tolerance": 0.01}
    report = build_report(records, metadata)
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                      default=json_value, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Reports: {destination}")


if __name__ == "__main__":
    main()
