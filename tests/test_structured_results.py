import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from threading import Barrier, Event
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

import agent
import api


EMPTY = {"sql": None, "columns": [], "rows": []}


def sql_call(query, call_id="sql"):
    return {"name": "execute_sql", "args": {"query": query}, "id": call_id}


def invoke_sql(tool, query):
    return tool.invoke({**sql_call(query), "type": "tool_call"})


def mock_model(monkeypatch, batches):
    model = Mock()
    model.bind_tools.return_value.invoke.side_effect = [
        *(AIMessage(content="", tool_calls=batch) for batch in batches),
        AIMessage(content="Analysis complete."),
    ]
    monkeypatch.setattr(agent, "get_models", lambda: model)
    return model


def test_structured_tool_result_executes_once(dataset_connection):
    connection = Mock(wraps=dataset_connection)
    _, tool = agent.create_tools(connection)
    message = invoke_sql(tool, "SELECT brand, revenue FROM sales ORDER BY revenue DESC LIMIT 1")
    assert isinstance(message, ToolMessage)
    assert message.artifact["columns"] == ["brand", "revenue"]
    assert message.artifact["rows"] == [["Toyota", 15000]]
    assert "Toyota" in message.content
    executed = connection.execute.call_args.args[0]
    assert message.artifact["sql"] == executed
    assert "LIMIT 100" in executed
    connection.execute.assert_called_once()


@pytest.mark.parametrize("query,columns,rows", [
    ("SELECT brand, revenue FROM sales WHERE FALSE", ["brand", "revenue"], []),
    ("SELECT 7 AS count, TRUE AS flag, NULL AS missing", ["count", "flag", "missing"], [[7, True, None]]),
    ("SELECT DATE '2026-10-26' AS day, TIMESTAMP '2026-10-26 12:34:56' AS stamp",
     ["day", "stamp"], [["2026-10-26T00:00:00", "2026-10-26T12:34:56"]]),
    ("SELECT 12.50::DECIMAL(10,2) AS amount, 'NaN'::DOUBLE AS nan, 'Infinity'::DOUBLE AS inf",
     ["amount", "nan", "inf"], [[12.5, None, None]]),
    ("SELECT [1, 2, NULL] AS items", ["items"], [[[1, 2, None]]]),
])
def test_duckdb_types_and_empty_results(execute_sql, query, columns, rows):
    result = invoke_sql(execute_sql, query).artifact
    assert result["columns"] == columns
    assert result["rows"] == rows
    json.dumps(result, allow_nan=False)


def test_pandas_numpy_and_temporal_serialization():
    frame = pd.DataFrame({
        "integer": pd.Series([np.int64(42), pd.NA], dtype="Int64"),
        "float": [np.float64(1.25), np.nan],
        "flag": pd.Series([np.bool_(True), pd.NA], dtype="boolean"),
        "date": [date(2026, 10, 26), None],
        "timestamp": [datetime(2026, 10, 26, 12, 30), pd.NaT],
        "decimal": [Decimal("12.50"), None],
        "time": [time(12, 30), None],
        "duration": [timedelta(seconds=3), pd.NaT],
    })
    result = agent.structured_result(frame, "SELECT values")
    assert result["rows"] == [
        [42, 1.25, True, "2026-10-26", "2026-10-26T12:30:00",
         "12.50", "12:30:00", "P0DT0H0M3S"],
        [None] * 8,
    ]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value,expected", [
    (np.datetime64("2026-10-26T12:30:00.123456789"), "2026-10-26T12:30:00.123456789"),
    (np.datetime64("NaT", "ns"), None),
    (np.timedelta64(3, "s"), "P0DT0H0M3S"),
    (np.timedelta64("NaT", "ns"), None),
    (pd.Timestamp("2026-10-26T12:30:00", tz="America/Toronto"), "2026-10-26T12:30:00-04:00"),
])
def test_numpy_dates_and_timezone(value, expected):
    assert agent.json_value(value) == expected


@pytest.mark.parametrize("query,prefix", [
    ("DELETE FROM sales", "SQL Validation Error:"),
    ("SELECT missing FROM sales", "SQL Execution Error:"),
])
def test_sql_failures_have_no_artifact(execute_sql, query, prefix):
    assert invoke_sql(execute_sql, "SELECT 1 AS previous").artifact is not None
    message = invoke_sql(execute_sql, query)
    assert message.content.startswith(prefix)
    assert message.artifact is None
    assert agent.latest_query_result([message]) == EMPTY


def test_structured_row_and_output_limits(execute_sql):
    message = invoke_sql(execute_sql, "SELECT a.brand FROM sales a CROSS JOIN sales b CROSS JOIN sales c")
    assert len(message.artifact["rows"]) == agent.MAX_RESULT_ROWS
    message = invoke_sql(execute_sql, "SELECT repeat('x', 3000) AS value FROM sales")
    assert 0 < len(message.artifact["rows"]) < 5
    assert len(json.dumps(message.artifact, ensure_ascii=False, allow_nan=False)) <= agent.MAX_OUTPUT_CHARS
    assert len(message.content) <= agent.MAX_OUTPUT_CHARS
    assert all(row == ["x" * 3000] for row in message.artifact["rows"])


def test_oversized_cell_is_omitted_whole(execute_sql):
    message = invoke_sql(execute_sql, "SELECT repeat('x', 20000) AS value")
    assert message.artifact["columns"] == ["value"]
    assert message.artifact["rows"] == []


def test_oversized_metadata_suppresses_result(execute_sql):
    message = invoke_sql(execute_sql, f'SELECT 1 AS "{"x" * agent.MAX_OUTPUT_CHARS}"')
    assert message.artifact is None


@pytest.mark.parametrize("same_batch", [False, True])
def test_last_call_uses_message_order(monkeypatch, same_batch):
    calls = [sql_call("SELECT 1 AS first", "a"), sql_call("SELECT 2 AS last", "b")]
    mock_model(monkeypatch, [calls] if same_batch else [[call] for call in calls])
    result = agent.run_agent("test")
    assert result["columns"] == ["last"]
    assert result["rows"] == [[2]]
    assert result["messages"][-1].content == "Analysis complete."


def test_parallel_batch_ignores_completion_order(monkeypatch):
    original = agent.structured_result
    second_finished = Event()

    def serialize(frame, sql):
        if frame.columns[0] == "first":
            assert second_finished.wait(timeout=15)
        result = original(frame, sql)
        if frame.columns[0] == "last":
            second_finished.set()
        return result

    monkeypatch.setattr(agent, "structured_result", serialize)
    mock_model(monkeypatch, [[sql_call("SELECT 1 AS first", "a"), sql_call("SELECT 2 AS last", "b")]])
    result = agent.run_agent("test")
    assert second_finished.is_set()
    assert result["columns"] == ["last"]
    assert result["rows"] == [[2]]


@pytest.mark.parametrize("last_call", [
    sql_call("DELETE FROM sales", "bad"),
    sql_call("SELECT missing FROM sales", "bad"),
    {"name": "execute_sql", "args": {}, "id": "bad"},
])
def test_last_failed_call_clears_previous_result(monkeypatch, last_call):
    mock_model(monkeypatch, [[sql_call("SELECT 1 AS previous", "good")], [last_call]])
    result = agent.run_agent("test")
    assert {key: result[key] for key in EMPTY} == EMPTY


def test_success_after_failed_call_and_inspection(monkeypatch):
    mock_model(monkeypatch, [
        [sql_call("SELECT missing FROM sales", "bad")],
        [sql_call("SELECT 3 AS recovered", "good")],
        [{"name": "inspect_dataset", "args": {}, "id": "inspect"}],
    ])
    result = agent.run_agent("test")
    assert result["rows"] == [[3]]


@pytest.mark.parametrize("batches", [[], [[{"name": "inspect_dataset", "args": {}, "id": "inspect"}]]])
def test_no_sql_has_empty_result(monkeypatch, batches):
    mock_model(monkeypatch, batches)
    result = agent.run_agent("test")
    assert {key: result[key] for key in EMPTY} == EMPTY


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    with TestClient(api.app) as client:
        yield client


def upload(client, content="brand,revenue\nToyota,11\n"):
    response = client.post("/upload", files={"file": ("sales.csv", content, "text/csv")})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def test_upload_and_ask_full_graph(client, monkeypatch):
    mock_model(monkeypatch, [[sql_call("SELECT brand, revenue FROM sales")]])
    dataset_id = upload(client)
    response = client.post("/ask", json={"question": "Revenue?", "dataset_id": dataset_id})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Analysis complete.",
        "sql": "SELECT * FROM (SELECT brand, revenue FROM sales) AS limited_result LIMIT 100",
        "columns": ["brand", "revenue"], "rows": [["Toyota", 11]],
    }


def test_legacy_agent_response_and_request_compatibility(client, monkeypatch):
    monkeypatch.setattr(api, "run_dataset_agent", lambda **_: {"messages": [AIMessage(content="old answer")]})
    response = client.post("/ask", json={"question": "Revenue?", "dataset_id": upload(client)})
    assert response.status_code == 200
    assert response.json() == {"answer": "old answer", **EMPTY}


def test_api_json_types(client, monkeypatch):
    mock_model(monkeypatch, [[sql_call(
        "SELECT DATE '2026-10-26' AS day, 12.50::DECIMAL(10,2) AS amount, NULL AS missing"
    )]])
    response = client.post("/ask", json={"question": "types", "dataset_id": upload(client)})
    assert response.status_code == 200
    assert response.json()["rows"] == [["2026-10-26T00:00:00", 12.5, None]]


@pytest.mark.parametrize("query", ["DELETE FROM sales", "SELECT missing FROM sales", None])
def test_api_empty_result(client, monkeypatch, query):
    mock_model(monkeypatch, [[sql_call(query)]] if query else [])
    response = client.post("/ask", json={"question": "test", "dataset_id": upload(client)})
    assert response.status_code == 200
    assert response.json() == {"answer": "Analysis complete.", **EMPTY}


def test_api_request_errors(client):
    assert client.post("/ask", json={"question": "test", "dataset_id": "00000000-0000-0000-0000-000000000000"}).status_code == 404
    assert client.post("/ask", json={"question": "test"}).status_code == 422
    assert client.post("/ask", json={"question": "", "dataset_id": upload(client)}).status_code == 422
    assert client.post("/upload", files={"file": ("data.txt", "x", "text/plain")}).status_code == 400


def test_concurrent_api_requests_are_isolated(client, monkeypatch):
    barrier = Barrier(4)

    class BoundModel:
        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="done")
            barrier.wait(timeout=15)
            query = "SELECT missing FROM sales" if messages[-1].content == "fail" else "SELECT brand, revenue FROM sales"
            return AIMessage(content="", tool_calls=[sql_call(query)])

    model = Mock()
    model.bind_tools.side_effect = lambda _: BoundModel()
    monkeypatch.setattr(agent, "get_models", lambda: model)
    first = upload(client, "brand,revenue\nFirst,11\n")
    second = upload(client, "brand,revenue\nSecond,73\n")
    requests = [(first, "ok"), (second, "ok"), (first, "fail"), (first, "ok")]

    def ask(request):
        dataset_id, question = request
        response = client.post("/ask", json={"dataset_id": dataset_id, "question": question})
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ask, requests))
    assert results[0]["rows"] == [["First", 11]]
    assert results[1]["rows"] == [["Second", 73]]
    assert results[2] == {"answer": "done", **EMPTY}
    assert results[3]["rows"] == [["First", 11]]


@pytest.mark.parametrize("rows", [[], [["Toyota", 11]]])
def test_streamlit_displays_and_preserves_results(monkeypatch, rows):
    import requests
    from streamlit.testing.v1 import AppTest

    response = Mock()
    response.json.return_value = {
        "answer": "done", "sql": "SELECT brand, revenue FROM sales",
        "columns": ["brand", "revenue"], "rows": rows,
    }
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    ui = AppTest.from_file(str(agent.PROJECT_DIR / "app.py"))
    ui.session_state["dataset_id"] = "test-dataset"
    ui.run()
    ui.chat_input[0].set_value("Revenue?").run()
    assert not ui.exception
    assert ui.code[0].value == response.json.return_value["sql"]
    assert list(ui.dataframe[0].value.columns) == ["brand", "revenue"]
    assert ui.dataframe[0].value.values.tolist() == rows
    ui.run()
    assert not ui.exception
    assert len(ui.code) == len(ui.dataframe) == 1


def test_streamlit_legacy_history_has_no_table():
    from streamlit.testing.v1 import AppTest

    ui = AppTest.from_file(str(agent.PROJECT_DIR / "app.py"))
    ui.session_state["dataset_id"] = "test-dataset"
    ui.session_state["messages"] = [{"role": "assistant", "content": "old answer"}]
    ui.run()
    assert not ui.exception
    assert not ui.code
    assert not ui.dataframe
