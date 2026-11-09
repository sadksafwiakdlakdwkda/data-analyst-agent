"""Row-count answers require actual typed tool evidence, not matching text alone."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import agent
from evals.runner import load_suite, run_case, trace_metrics, verify_references
from evals.scoring import score
from evals.report import summarize


@pytest.fixture
def case():
    return next(c for c in load_suite()["cases"] if c["id"] == "basic_002")


def inspection(artifact=None, *, status="success", content="Total rows: 10000"):
    return ToolMessage(name="inspect_dataset", tool_call_id="inspect",
                       content=content, status=status, artifact=artifact)


def result_with(*messages, answer="FINAL_ANSWER: 10000"):
    return {"messages": list(messages), "answer": answer,
            "sql": None, "columns": [], "rows": [], "result_truncated": False}


def test_inspection_preserves_text_and_adds_tool_owned_artifact(dataset_tools):
    inspect = dataset_tools["inspect_dataset"]
    plain = inspect.invoke({})
    assert isinstance(plain, str)
    assert "Total rows: 10000" in plain
    message = inspect.invoke({"name": "inspect_dataset", "args": {},
                              "id": "inspect", "type": "tool_call"})
    assert message.content == plain
    assert message.artifact == {"row_count": 10000}
    assert message.status == "success"
    assert "row_count" not in inspect.tool_call_schema.model_fields


def test_inspection_failure_has_no_artifact():
    connection = Mock()
    connection.execute.side_effect = RuntimeError("failed")
    inspect = agent.create_tools(connection)[0]
    message = inspect.invoke({"name": "inspect_dataset", "args": {},
                              "id": "inspect", "type": "tool_call"})
    assert message.artifact is None
    assert message.content.startswith("Dataset Inspection Error:")


def test_count_can_pass_without_inventing_sql_result(case):
    result = result_with(inspection({"row_count": 10000}))
    scored = score(case, result)
    assert scored["automated_passed"]
    assert scored["result_passed"]
    assert scored["answer_passed"]
    assert not scored["sql_result_passed"]
    assert scored["evidence_source"] == "inspect_row_count"
    assert result["sql"] is None and result["rows"] == []
    assert trace_metrics(result["messages"])[1]["sql_calls"] == 0


@pytest.mark.parametrize("artifact", [None, {}, {"row_count": "10000"},
                                      {"row_count": 10000.0}, {"row_count": True},
                                      {"row_count": -1}, {"row_count": 9999}])
def test_missing_malformed_or_incorrect_evidence_fails(case, artifact):
    scored = score(case, result_with(inspection(artifact)))
    assert scored["answer_passed"]
    assert not scored["automated_passed"]
    assert scored["evidence_source"] is None


def test_correct_answer_without_tool_call_fails(case):
    result = result_with(AIMessage(content="FINAL_ANSWER: 10000",
                                   additional_kwargs={"row_count": 10000}))
    result["row_count"] = 10000
    assert not score(case, result)["automated_passed"]


@pytest.mark.parametrize("message", [
    inspection({"row_count": 10000}, status="error"),
    inspection({"row_count": 10000}, content="Dataset Inspection Error: failed"),
])
def test_failed_inspection_cannot_supply_evidence(case, message):
    assert not score(case, result_with(message))["automated_passed"]


def test_latest_failed_or_wrong_inspection_clears_prior_evidence(case):
    good = inspection({"row_count": 10000})
    bad = inspection(None, status="error")
    assert not score(case, result_with(good, bad))["automated_passed"]
    assert not score(case, result_with(good, inspection({"row_count": 2})))["automated_passed"]
    assert score(case, result_with(bad, good))["automated_passed"]


def test_correct_evidence_cannot_hide_wrong_answer(case):
    scored = score(case, result_with(inspection({"row_count": 10000}), answer="FINAL_ANSWER: 5"))
    assert scored["result_passed"]
    assert not scored["answer_passed"]
    assert scored["reason"] == "answer_mismatch"


def test_only_opted_in_case_can_use_inspection(case):
    unmarked = deepcopy(case)
    del unmarked["allowed_evidence"]
    result = result_with(inspection({"row_count": 10000}))
    assert not score(unmarked, result)["automated_passed"]
    suite = load_suite()
    assert suite["version"] == 2
    assert [c["id"] for c in suite["cases"]
            if "inspect_row_count" in c.get("allowed_evidence", [])] == ["basic_002"]


def test_failed_or_incorrect_sql_is_not_rescued(case):
    message = ToolMessage(name="execute_sql", tool_call_id="sql", content="SQL Execution Error: bad")
    result = result_with(inspection({"row_count": 10000}), message)
    assert not score(case, result)["automated_passed"]
    result.update(sql="SELECT 1", columns=["n"], rows=[[1]])
    assert not score(case, result)["automated_passed"]
    result["rows"] = [[10000]]
    scored = score(case, result)
    assert scored["automated_passed"]
    assert scored["sql_result_passed"]
    assert scored["evidence_source"] == "sql"


def test_real_graph_inspection_only_path_is_scored_correctly(monkeypatch, case):
    # Reproduce the observed live failure using the real graph and DuckDB tools.
    model, bound = Mock(), Mock()
    model.bind_tools.return_value = bound
    bound.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{"name": "inspect_dataset", "args": {}, "id": "inspect"}]),
        AIMessage(content="FINAL_ANSWER: 10000"),
    ]
    monkeypatch.setattr(agent, "get_models", lambda: model)
    record = run_case(case, "agent")
    assert record["score"]["automated_passed"]
    assert record["score"]["evidence_source"] == "inspect_row_count"
    assert record["result"]["sql"] is None
    assert record["metrics"]["sql_calls"] == 0
    summary = summarize([record])
    assert summary["inspection_evidence_passes"] == 1
    assert summary["sql_success_rate"] is None


def test_all_frozen_sql_references_still_match():
    verify_references(load_suite())
