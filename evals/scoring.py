"""Deterministic result scoring; prose judgments remain explicit manual reviews."""

import re
from decimal import Decimal, InvalidOperation

from langchain_core.messages import ToolMessage


def inspection_count_matches(case, result):
    """Only explicitly opted-in scalar cases may use typed inspection evidence."""
    if case["check"] != "scalar" or "inspect_row_count" not in case.get("allowed_evidence", []):
        return False
    expected = case["expected_rows"]
    if len(expected) != 1 or len(expected[0]) != 1 or type(expected[0][0]) is not int:
        return False
    messages = [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]
    # Never hide an attempted SQL query's missing or incorrect result with
    # inspection metadata. Existing SQL result rules continue to apply.
    if result.get("sql") is not None or any(m.name == "execute_sql" for m in messages):
        return False
    for message in reversed(messages):
        if message.name != "inspect_dataset":
            continue
        artifact = message.artifact
        return (
            message.status == "success"
            and not str(message.content).startswith("Dataset Inspection Error:")
            and isinstance(artifact, dict)
            and type(artifact.get("row_count")) is int
            and artifact["row_count"] >= 0
            and artifact["row_count"] == expected[0][0]
        )
    return False


def value_matches(actual, expected, tolerance=0.01):
    if expected is None or actual is None:
        return actual is expected
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float, Decimal)):
        try:
            number = Decimal(str(actual))
            target = Decimal(str(expected))
            return number.is_finite() and abs(number - target) <= Decimal(str(tolerance))
        except (InvalidOperation, ValueError, TypeError):
            return False
    return isinstance(actual, str) and actual == str(expected)


def rows_match(actual, expected, *, ordered=False, tolerance=0.01):
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False

    def same(left, right):
        return isinstance(left, (list, tuple)) and len(left) == len(right) and all(
            value_matches(a, e, tolerance) for a, e in zip(left, right)
        )

    if ordered:
        return all(same(a, e) for a, e in zip(actual, expected))
    # Maximum bipartite matching preserves duplicates and handles overlapping
    # numeric tolerances without relying on a greedy row ordering.
    matched = {}

    def assign(index, visited):
        for other, row in enumerate(expected):
            if other in visited or not same(actual[index], row):
                continue
            visited.add(other)
            if other not in matched or assign(matched[other], visited):
                matched[other] = index
                return True
        return False

    return all(assign(index, set()) for index in range(len(actual)))


def final_value(answer):
    text = str(answer).strip()
    if not text:
        return ""
    match = re.fullmatch(r"FINAL_ANSWER:\s*(.+)", text.splitlines()[-1], re.I)
    return match.group(1).strip() if match else text


def score(case, result):
    if case["check"] == "manual":
        return {"result_passed": None, "answer_passed": None,
                "automated_passed": None, "review_required": True,
                "reason": "manual_review_required"}
    columns = result.get("columns", [])
    expected_columns = case["expected_columns"]
    # Aliases may differ. Column positions and values are contractual; exact
    # reference aliases are not. Empty results still require the right shape.
    table_ok = (
        result.get("sql") is not None
        and not result.get("result_truncated", False)
        and len(columns) == len(expected_columns)
        and rows_match(result.get("rows", []), case["expected_rows"],
                       ordered=case.get("ordered", False),
                       tolerance=case.get("tolerance", 0.01))
    )
    inspection_ok = inspection_count_matches(case, result)
    result_ok = table_ok or inspection_ok
    evidence_source = "sql" if table_ok else "inspect_row_count" if inspection_ok else None
    answer_ok = None
    if case["check"] == "scalar":
        expected = case["expected_rows"][0][0]
        actual = final_value(result.get("answer", ""))
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            # Only allow valid thousands grouping, never arbitrary prose.
            numeric = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
            answer_ok = bool(re.fullmatch(numeric, actual)) and value_matches(
                actual.replace(",", ""), expected, case.get("tolerance", 0.01)
            )
        else:
            answer_ok = actual.casefold() == str(expected).casefold()
    chart_ok = None
    if case["check"] == "chart":
        chart = result.get("chart") or {}
        wanted = case["expected_chart"]
        chart_ok = (
            len(columns) == len(expected_columns)
            and chart.get("type") == wanted["type"]
            and chart.get("x") == columns[wanted["x_index"]]
            and chart.get("y") == columns[wanted["y_index"]]
        )
    passed = result_ok and answer_ok is not False and chart_ok is not False
    reason = ("result_mismatch" if not result_ok else
              "answer_mismatch" if answer_ok is False else
              "chart_mismatch" if chart_ok is False else "passed")
    return {"result_passed": result_ok, "sql_result_passed": table_ok,
            "evidence_source": evidence_source, "answer_passed": answer_ok,
            "chart_passed": chart_ok, "automated_passed": passed,
            "review_required": case["check"] != "scalar", "reason": reason}
