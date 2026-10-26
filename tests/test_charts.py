import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

import agent
from test_structured_results import client, mock_model, sql_call, upload


def chart_call(kind="bar", x="brand", y="total", title="Revenue", call_id="chart", **extra):
    return {"name": "create_chart", "args": {
        "chart_type": kind, "x": x, "y": y, "title": title, **extra,
    }, "id": call_id}


QUERY = "SELECT brand, SUM(revenue) AS total FROM sales GROUP BY brand ORDER BY total DESC"


def chart_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "create_chart"]


@pytest.mark.parametrize("kind,query,x", [
    ("bar", QUERY, "brand"),
    ("line", "SELECT revenue AS x, revenue AS total FROM sales ORDER BY revenue DESC", "x"),
    ("scatter", "SELECT revenue AS x, revenue AS total FROM sales", "x"),
])
def test_chart_types_reuse_sql_without_execution(monkeypatch, dataset_connection, kind, query, x):
    connection = Mock(wraps=dataset_connection)
    monkeypatch.setattr(agent, "create_dataset_connection", lambda _: connection)
    mock_model(monkeypatch, [[sql_call(query)], [chart_call(kind, x)]])
    result = agent.run_agent("Chart please")
    assert result["chart"] == {"type": kind, "x": x, "y": "total", "title": "Revenue"}
    assert chart_messages(result)[0].content.startswith(f"Created {kind}")
    connection.execute.assert_called_once()
    if kind == "bar":
        assert result["rows"] == [["Toyota", 36500], ["Honda", 15500]]


def test_chart_state_is_hidden_from_model():
    schema = agent.create_chart.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"chart_type", "x", "y", "title"}


@pytest.mark.parametrize("query,call", [
    (QUERY, chart_call("pie")),
    (QUERY, chart_call(x="missing")),
    (QUERY, chart_call(y="missing")),
    (QUERY, chart_call("scatter")),
    ("SELECT brand, brand AS total FROM sales", chart_call()),
    ("SELECT brand, TRUE AS total FROM sales", chart_call()),
    ("SELECT TRUE AS x, revenue AS total FROM sales", chart_call("scatter", "x")),
    ("SELECT [1, 2] AS x, revenue AS total FROM sales", chart_call("bar", "x")),
    ("SELECT brand, revenue AS total FROM sales WHERE FALSE", chart_call()),
    ("SELECT NULL AS brand, revenue AS total FROM sales", chart_call()),
    ("SELECT brand, NULL AS total FROM sales", chart_call()),
    (QUERY, chart_call(title="x" * 201)),
    (QUERY, {"name": "create_chart", "args": {}, "id": "invalid"}),
])
def test_chart_validation_errors(monkeypatch, query, call):
    mock_model(monkeypatch, [[sql_call(query)], [call]])
    result = agent.run_agent("Chart")
    assert result["chart"] is None
    message = chart_messages(result)[0]
    assert message.status == "error" or message.content.startswith("Chart Error:")
    assert result["sql"] is not None


@pytest.mark.parametrize("query", [None, "DELETE FROM sales", "SELECT missing FROM sales"])
def test_no_valid_sql_cannot_chart(monkeypatch, query):
    batches = [[sql_call(query)]] if query else []
    model = mock_model(monkeypatch, [*batches, [chart_call()]])
    model.invoke.return_value = AIMessage(content="Unable to chart")
    result = agent.run_agent("Chart")
    assert result["chart"] is None
    assert "No valid SQL result" in chart_messages(result)[0].content


def test_null_axis_rows_are_excluded_without_changing_table(monkeypatch):
    query = "SELECT brand, CASE WHEN revenue = 15000 THEN NULL ELSE revenue END AS total FROM sales"
    mock_model(monkeypatch, [[sql_call(query)], [chart_call()]])
    result = agent.run_agent("Chart")
    assert result["chart"] is not None
    assert len(result["rows"]) == 5
    assert "excluded 1 null-axis rows" in chart_messages(result)[0].content


@pytest.mark.parametrize("next_call", [
    sql_call("SELECT 2 AS next", "next"),
    sql_call("SELECT missing FROM sales", "next"),
    {"name": "execute_sql", "args": {}, "id": "next"},
    chart_call("scatter", call_id="next"),
])
def test_new_sql_or_failed_chart_invalidates_previous_chart(monkeypatch, next_call):
    mock_model(monkeypatch, [[sql_call(QUERY)], [chart_call()], [next_call]])
    result = agent.run_agent("Chart")
    assert chart_messages(result)[0].artifact is not None
    assert result["chart"] is None


@pytest.mark.parametrize("same_batch", [False, True])
def test_last_chart_call_wins(monkeypatch, same_batch):
    calls = [chart_call(title="First", call_id="a"), chart_call("line", title="Last", call_id="b")]
    mock_model(monkeypatch, [[sql_call(QUERY)], *([calls] if same_batch else [[c] for c in calls])])
    result = agent.run_agent("Chart")
    assert result["chart"]["title"] == "Last"
    assert result["chart"]["type"] == "line"


@pytest.mark.parametrize("reverse", [False, True])
def test_sql_and_chart_in_same_batch_rejected_even_with_previous_result(monkeypatch, reverse):
    calls = [sql_call(QUERY, "new"), chart_call()]
    mock_model(monkeypatch, [[sql_call(QUERY, "old")], calls[::-1] if reverse else calls])
    result = agent.run_agent("Chart")
    assert result["chart"] is None
    assert "separate batch" in chart_messages(result)[0].content


def test_chart_retry_after_batch_error_uses_new_query(monkeypatch):
    mock_model(monkeypatch, [
        [sql_call("SELECT 1 AS brand, 2 AS total"), chart_call()],
        [chart_call()],
    ])
    result = agent.run_agent("Chart")
    assert result["chart"] is not None
    assert result["rows"] == [[1, 2]]


def test_chart_recreated_after_new_query(monkeypatch):
    mock_model(monkeypatch, [
        [sql_call(QUERY, "first")], [chart_call(call_id="first-chart")],
        [sql_call("SELECT 2 AS x, 4 AS total", "second")],
        [chart_call("scatter", "x", call_id="second-chart")],
        [{"name": "inspect_dataset", "args": {}, "id": "inspect"}],
    ])
    result = agent.run_agent("Chart")
    assert result["chart"]["type"] == "scatter"
    assert result["rows"] == [[2, 4]]


def test_chart_errors_count_toward_existing_error_budget(monkeypatch):
    model = mock_model(monkeypatch, [[chart_call(call_id="one")], [chart_call(call_id="two")]])
    model.invoke.return_value = AIMessage(content="Unable to chart")
    result = agent.run_agent("Chart")
    assert result["messages"][-1].content == "Unable to chart"
    model.invoke.assert_called_once()


def test_model_cannot_override_injected_chart_state(monkeypatch):
    fake_state = {"messages": []}
    mock_model(monkeypatch, [[sql_call(QUERY)], [chart_call(state=fake_state)]])
    result = agent.run_agent("Chart")
    assert result["chart"] is not None
    assert result["rows"] == [["Toyota", 36500], ["Honda", 15500]]


@pytest.mark.parametrize("query,expected", [
    (QUERY, False),
    ("SELECT a.brand, a.revenue AS total FROM sales a CROSS JOIN sales b CROSS JOIN sales c", True),
    ("SELECT repeat('x', 3000) AS brand, revenue AS total FROM sales", True),
])
def test_chart_truncation_status(client, monkeypatch, query, expected):
    mock_model(monkeypatch, [[sql_call(query)], [chart_call()]])
    response = client.post("/ask", json={"question": "Chart", "dataset_id": upload(client, agent.CSV_PATH.read_text())})
    assert response.status_code == 200
    data = response.json()
    assert data["chart"] == {"type": "bar", "x": "brand", "y": "total", "title": "Revenue"}
    assert data["result_truncated"] is expected
    assert len(data["rows"]) <= 100


def test_concurrent_chart_requests_are_isolated(client, monkeypatch):
    barrier = Barrier(4)

    class BoundModel:
        def invoke(self, messages):
            last = messages[-1]
            if isinstance(last, ToolMessage) and last.name == "create_chart":
                return AIMessage(content="Done")
            if isinstance(last, ToolMessage):
                barrier.wait(timeout=15)
                question = next(m.content for m in messages if m.type == "human")
                return AIMessage(content="", tool_calls=[chart_call("scatter" if question == "fail" else "bar", title=question)])
            return AIMessage(content="", tool_calls=[sql_call("SELECT brand, revenue AS total FROM sales")])

    model = Mock()
    model.bind_tools.side_effect = lambda _: BoundModel()
    monkeypatch.setattr(agent, "get_models", lambda: model)
    first = upload(client, "brand,revenue\nFirst,11\n")
    second = upload(client, "brand,revenue\nSecond,73\n")
    inputs = [(first, "One"), (second, "Two"), (first, "fail"), (first, "Four")]

    def ask(item):
        dataset_id, question = item
        response = client.post("/ask", json={"dataset_id": dataset_id, "question": question})
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ask, inputs))
    assert [r["rows"] for r in results] == [[["First", 11]], [["Second", 73]], [["First", 11]], [["First", 11]]]
    assert [r["chart"]["title"] if r["chart"] else None for r in results] == ["One", "Two", None, "Four"]


@pytest.mark.parametrize("kind", ["bar", "line", "scatter"])
def test_streamlit_chart_render_and_history(monkeypatch, kind):
    import requests

    response = Mock()
    response.json.return_value = {
        "answer": "Done", "sql": "SELECT ...", "columns": ["x.field", "y[value]"],
        "rows": [[2, 4], [None, 3], [1, 2]], "result_truncated": True,
        "chart": {"type": kind, "x": "x.field", "y": "y[value]", "title": "Revenue"},
    }
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    ui = AppTest.from_file(str(agent.PROJECT_DIR / "app.py"))
    ui.session_state["dataset_id"] = "test"
    ui.run()
    ui.chat_input[0].set_value("Chart please").run()
    assert not ui.exception
    assert ui.subheader[0].value == "Revenue"
    assert len(ui.warning) == 1
    assert "Excluded 1" in ui.caption[0].value
    charts = ui.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    plotted = convert_arrow_bytes_to_pandas_df(charts[0].proto.data.data)
    assert plotted["axis_x"].tolist() == [2, 1]
    assert plotted["axis_y"].tolist() == [4, 2]
    assert plotted["row_order"].tolist() == [0, 1]
    assert spec["mark"] == {"bar": "bar", "line": "line", "scatter": "point"}[kind]
    assert spec["encoding"]["x"]["sort"] is None
    assert spec["encoding"]["y"]["stack"] is None
    if kind == "line":
        assert spec["encoding"]["order"]["field"] == "row_order"
    ui.run()
    assert not ui.exception
    assert len(ui.get("vega_lite_chart")) == 1
    assert len(ui.dataframe[0].value) == 3


def test_streamlit_failed_chart_still_displays_answer_and_table():
    ui = AppTest.from_file(str(agent.PROJECT_DIR / "app.py"))
    ui.session_state["dataset_id"] = "test"
    ui.session_state["messages"] = [{
        "role": "assistant", "content": "Cannot chart", "sql": "SELECT ...",
        "columns": ["brand"], "rows": [["Toyota"]], "chart": None,
    }]
    ui.run()
    assert not ui.exception
    assert len(ui.dataframe) == 1
    assert len(ui.code) == 1
    assert not ui.get("vega_lite_chart")
