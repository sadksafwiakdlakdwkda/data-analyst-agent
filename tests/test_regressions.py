from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import agent as app


@pytest.mark.parametrize("copy_project", [False, True])
def test_import_and_sql_without_credentials_from_other_directory(tmp_path, copy_project):
    project = Path(app.__file__).resolve().parent
    if copy_project:
        # Also exercise spaces and apostrophes in the dataset's absolute path.
        project = tmp_path / "analyst's project"
        project.mkdir()
        for name in ("agent.py", "sales.csv"):
            shutil.copy2(Path(app.__file__).parent / name, project / name)
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_ADMIN_KEY", None)
    env["PYTHONPATH"] = str(project)
    script = """
from unittest.mock import patch
with patch('dotenv.load_dotenv', side_effect=AssertionError('eager dotenv')), \\
     patch('langchain_openai.ChatOpenAI', side_effect=AssertionError('eager LLM')):
    import agent
    assert agent.llm is None and agent.llm_with_tools is None
    assert 'Total rows: 5' in agent.inspect_dataset.invoke({})
    assert '52000' in agent.execute_sql.invoke({'query': 'SELECT SUM(revenue) FROM sales'})
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_models_initialized_once_on_first_use(monkeypatch):
    monkeypatch.setattr(app, "llm", None)
    monkeypatch.setattr(app, "llm_with_tools", None)
    dotenv = Mock()
    constructor = Mock()
    monkeypatch.setattr(app, "load_dotenv", dotenv)
    monkeypatch.setattr(app, "ChatOpenAI", constructor)
    first = app.get_models()
    assert app.get_models() == first
    constructor.assert_called_once()
    dotenv.assert_called_once_with(app.PROJECT_DIR / ".env")
    constructor.return_value.bind_tools.assert_called_once_with(app.tools)


@pytest.mark.parametrize("query", [
    "SELECT SUM(revenue) AS total FROM sales -- trailing comment",
    "SELECT SUM(revenue) AS total FROM sales; -- trailing comment",
    "SELECT SUM(revenue) AS total FROM sales; /* trailing comment */",
    "SELECT /* embedded comment */ SUM(revenue) AS total FROM sales;\n",
])
def test_sql_is_generated_from_ast(query):
    result = app.execute_sql.invoke({"query": query})
    assert not result.startswith("SQL ")
    assert "52000" in result


@pytest.mark.parametrize("query", [
    "", "-- comment only", "SELECT (", "SELECT * FROM main.sales",
    "SELECT * FROM memory.main.sales", "SELECT * FROM (SELECT * FROM sales)",
    "WITH x AS (SELECT * FROM sales) SELECT * FROM sales",
    "SELECT brand FROM sales UNION SELECT brand FROM sales",
    "SELECT * FROM read_text('sales.csv')",
    "SELECT * FROM read_parquet('sales.parquet')",
    "SELECT * FROM glob('*')", "SELECT getenv('PATH')",
    "UPDATE sales SET revenue = 0", "DROP TABLE sales",
    "SELECT 1; SELECT 2",
])
def test_rejected_sql_never_reaches_database(monkeypatch, query):
    connection = Mock()
    monkeypatch.setattr(app, "conn", connection)
    assert app.execute_sql.invoke({"query": query}).startswith("SQL Validation Error:")
    connection.execute.assert_not_called()


def test_row_limit():
    result = app.execute_sql.invoke({
        "query": "SELECT a.revenue FROM sales a CROSS JOIN sales b CROSS JOIN sales c"
    })
    assert len(result.splitlines()) == app.MAX_RESULT_ROWS + 1


def test_output_character_limit():
    result = app.execute_sql.invoke({
        "query": f"SELECT repeat('x', {app.MAX_OUTPUT_CHARS * 2}) AS value"
    })
    assert len(result) == app.MAX_OUTPUT_CHARS
    assert result.endswith("[Output truncated; refine the query.]")


@pytest.mark.parametrize("tool,args,prefix", [
    (app.inspect_dataset, {}, "Dataset Inspection Error:"),
    (app.execute_sql, {"query": "SELECT * FROM sales"}, "SQL Execution Error:"),
])
def test_database_failures_are_bounded(monkeypatch, tool, args, prefix):
    connection = Mock()
    connection.execute.side_effect = RuntimeError("x" * (app.MAX_OUTPUT_CHARS * 2))
    monkeypatch.setattr(app, "conn", connection)
    result = tool.invoke(args)
    assert result.startswith(prefix)
    assert len(result) == app.MAX_OUTPUT_CHARS
    assert not app.db_lock.locked()


def test_shared_connection_lock_covers_execute_and_fetch(monkeypatch):
    real_connection = app.conn

    class CheckedResult:
        def fetchdf(self):
            assert app.db_lock.locked(), "fetchdf must be inside the lock"
            return real_connection.fetchdf()

        def fetchone(self):
            assert app.db_lock.locked(), "fetchone must be inside the lock"
            return real_connection.fetchone()

    class CheckedConnection:
        def execute(self, query):
            assert app.db_lock.locked(), "execute must be inside the lock"
            real_connection.execute(query)
            # Yield after execute: without a lock a different tool can replace results.
            time.sleep(0.005)
            return CheckedResult()

    monkeypatch.setattr(app, "conn", CheckedConnection())

    def run(index):
        if index % 3 == 0:
            assert "Total rows: 5" in app.inspect_dataset.invoke({})
        else:
            result = app.execute_sql.invoke({"query": f"SELECT {1000 + index} AS marker"})
            assert result.split() == ["marker", str(1000 + index)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(15)))


def tool_call(name="execute_sql", args=None, call_id="call"):
    if args is None:
        args = {"query": "SELECT missing FROM sales"}
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def models(monkeypatch):
    plain, bound = Mock(), Mock()
    monkeypatch.setattr(app, "get_models", lambda: (plain, bound))
    return plain, bound


def test_graph_success(models):
    plain, bound = models
    bound.invoke.side_effect = [
        tool_call("inspect_dataset", {}, "inspect"),
        tool_call(args={"query": "SELECT SUM(revenue) AS total FROM sales"}, call_id="sql"),
        AIMessage(content="The total is 52000."),
    ]
    result = app.run_agent("Total revenue?")
    assert result["messages"][-1].content == "The total is 52000."
    assert app.count_sql_errors(result) == 0
    plain.invoke.assert_not_called()


def test_graph_recovers_after_one_sql_error(models):
    plain, bound = models
    bound.invoke.side_effect = [
        tool_call(call_id="bad"),
        tool_call(args={"query": "SELECT SUM(revenue) FROM sales"}, call_id="good"),
        AIMessage(content="52000"),
    ]
    result = app.run_agent("Total revenue?")
    assert result["messages"][-1].content == "52000"
    assert app.count_sql_errors(result) == 1
    plain.invoke.assert_not_called()


@pytest.mark.parametrize("kind", ["sql", "inspection", "arguments"])
def test_graph_stops_after_two_tool_errors(models, monkeypatch, kind):
    plain, bound = models
    if kind == "inspection":
        connection = Mock()
        connection.execute.side_effect = RuntimeError("synthetic failure")
        monkeypatch.setattr(app, "conn", connection)
        calls = [tool_call("inspect_dataset", {}, str(i)) for i in range(2)]
    elif kind == "arguments":
        calls = [tool_call(args={}, call_id=str(i)) for i in range(2)]
    else:
        calls = [tool_call(call_id=str(i)) for i in range(2)]
    bound.invoke.side_effect = calls
    plain.invoke.return_value = AIMessage(content="Could not complete the analysis.")
    result = app.run_agent("test")
    assert result["messages"][-1].content == "Could not complete the analysis."
    assert bound.invoke.call_count == 2
    plain.invoke.assert_called_once()
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_results) == 2
    if kind == "arguments":
        assert all(m.status == "error" for m in tool_results)


@pytest.mark.parametrize("stage", ["initialization", "model", "final_failure"])
def test_model_errors_are_graceful_and_do_not_echo_provider_data(models, monkeypatch, stage):
    plain, bound = models
    error = RuntimeError("sensitive-provider-request-data")
    if stage == "initialization":
        monkeypatch.setattr(app, "get_models", Mock(side_effect=error))
    elif stage == "model":
        bound.invoke.side_effect = error
    else:
        bound.invoke.side_effect = [tool_call(call_id="a"), tool_call(call_id="b")]
        plain.invoke.side_effect = error
    result = app.run_agent("test")
    answer = result["messages"][-1].content
    assert "model service failed" in answer
    assert "sensitive-provider-request-data" not in answer


def test_recursion_limit_is_graceful(models):
    plain, bound = models
    bound.invoke.side_effect = lambda _: tool_call("inspect_dataset", {}, "loop")
    result = app.run_agent("test", recursion_limit=6)
    assert "workflow step limit" in result["messages"][-1].content
    plain.invoke.assert_not_called()
