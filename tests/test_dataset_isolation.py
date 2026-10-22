from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from unittest.mock import Mock

import duckdb
import pytest
from langchain_core.messages import AIMessage, ToolMessage

import agent as app


def test_tools_keep_datasets_isolated_during_concurrent_queries(tmp_path):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text("brand,revenue\nFirst,11\n")
    second_path.write_text("brand,revenue\nSecond,73\nSecond,27\n")
    with app.create_dataset_connection(first_path) as first_connection, \
            app.create_dataset_connection(second_path) as second_connection:
        first_inspect, first_sql = app.create_tools(first_connection)
        second_inspect, second_sql = app.create_tools(second_connection)

        def query(index):
            inspect, execute, count, brand, total = (
                (first_inspect, first_sql, 1, "First", "11") if index % 2 == 0
                else (second_inspect, second_sql, 2, "Second", "100")
            )
            assert f"Total rows: {count}" in inspect.invoke({})
            result = execute.invoke({
                "query": "SELECT brand, SUM(revenue) AS total FROM sales GROUP BY brand"
            })
            values = result.split()
            assert values[:3] == ["brand", "total", brand]
            assert len(values) == 4
            assert Decimal(values[3]) == Decimal(total)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(query, range(20)))


def test_each_request_binds_and_executes_its_own_dataset_tools(monkeypatch, tmp_path):
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for path, total in zip(paths, (11, 73)):
        path.write_text(f"revenue\n{total}\n")
    model = Mock()
    bound_models = [Mock(), Mock()]
    for bound in bound_models:
        bound.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{
                "name": "execute_sql",
                "args": {"query": "SELECT SUM(revenue) AS total FROM sales"},
                "id": "sql",
            }]),
            AIMessage(content="done"),
        ]
    model.bind_tools.side_effect = bound_models
    monkeypatch.setattr(app, "get_models", lambda: model)
    for path, total in zip(paths, (11, 73)):
        result = app.run_agent("Total?", dataset_path=path)
        tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_results) == 1
        values = tool_results[0].content.split()
        assert values[0] == "total"
        assert len(values) == 2
        assert Decimal(values[1]) == total
    assert model.bind_tools.call_count == 2
    first_tools, second_tools = [call.args[0] for call in model.bind_tools.call_args_list]
    assert all(first is not second for first, second in zip(first_tools, second_tools))


@pytest.mark.parametrize("stage", ["success", "initialization", "binding", "model", "limit", "unexpected"])
def test_request_closes_connection_on_every_exit(monkeypatch, stage):
    connection = app.create_dataset_connection(app.CSV_PATH)
    monkeypatch.setattr(app, "create_dataset_connection", lambda _: connection)
    model = Mock()
    bound = model.bind_tools.return_value
    bound.invoke.return_value = AIMessage(content="done")
    monkeypatch.setattr(app, "get_models", lambda: model)
    error = RuntimeError("synthetic failure")
    if stage == "initialization":
        monkeypatch.setattr(app, "get_models", Mock(side_effect=error))
    elif stage == "binding":
        model.bind_tools.side_effect = error
    elif stage == "model":
        bound.invoke.side_effect = error
    elif stage == "limit":
        bound.invoke.side_effect = lambda _: AIMessage(content="", tool_calls=[{
            "name": "inspect_dataset", "args": {}, "id": "loop",
        }])
    elif stage == "unexpected":
        graph = Mock()
        graph.invoke.side_effect = error
        monkeypatch.setattr(app, "create_agent_graph", lambda _: graph)

    if stage in ("initialization", "binding", "unexpected"):
        with pytest.raises(RuntimeError, match="synthetic failure"):
            app.run_agent("test")
    else:
        result = app.run_agent("test", recursion_limit=6)
        answer = result["messages"][-1].content
        assert {"success": "done", "model": "model service failed", "limit": "workflow step limit"}[stage] in answer
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_failed_dataset_load_closes_connection(monkeypatch, tmp_path):
    connection = duckdb.connect(":memory:")
    monkeypatch.setattr(app.duckdb, "connect", lambda _: connection)
    with pytest.raises(duckdb.IOException):
        app.create_dataset_connection(tmp_path / "missing.csv")
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_dataset_wrapper_forwards_path_and_recursion_limit(monkeypatch, tmp_path):
    runner = Mock()
    monkeypatch.setattr(app, "run_agent", runner)
    path = tmp_path / "dataset.csv"
    assert app.run_dataset_agent("Total?", path, recursion_limit=8) is runner.return_value
    runner.assert_called_once_with("Total?", dataset_path=path, recursion_limit=8)
