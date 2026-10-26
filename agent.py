"""Dataset-isolated SQL analysis agent using DuckDB and LangGraph."""

import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from threading import Lock

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlglot import exp, parse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "sales.csv"
MAX_OUTPUT_CHARS = 10_000
MAX_RESULT_ROWS = 100
MAX_SQL_ERRORS = 2

SYSTEM_PROMPT = (
    "You are a reliable data analysis assistant. "
    "Always inspect the dataset before generating SQL. "
    "Only query the sales table. "
    "Use execute_sql for calculations. "
    "If SQL fails, inspect the error message and correct the query before retrying. "
    "Never repeat the exact same failed SQL. "
    "Never invent query results. "
    "If analysis fails, explain the problem."
)


def limit_output(text: str) -> str:
    """Limit tool output, including the truncation notice."""
    notice = "\n[Output truncated; refine the query.]"
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[: MAX_OUTPUT_CHARS - len(notice)] + notice


def create_dataset_connection(dataset_path: Path) -> duckdb.DuckDBPyConnection:
    """Load one CSV into a new, independent in-memory DuckDB connection."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE sales AS SELECT * FROM read_csv_auto(?)",
            [str(Path(dataset_path).resolve())],
        )
        return connection
    except Exception:
        connection.close()
        raise


def _validated_sql(query: str) -> tuple[exp.Select | None, str | None]:
    """Return a validated SQL AST or a readable validation error."""
    try:
        statements = [
            statement
            for statement in parse(query, read="duckdb")
            if statement is not None and not isinstance(statement, exp.Semicolon)
        ]
    except Exception as error:
        return None, f"Invalid SQL syntax: {error}"

    if len(statements) != 1:
        return None, "Only one SQL statement is allowed."

    ast = statements[0]
    if not isinstance(ast, exp.Select):
        return None, "Only SELECT queries are allowed."

    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)
    if any(isinstance(node, forbidden) for node in ast.walk()):
        return None, "SQL contains a forbidden operation."

    for table in ast.find_all(exp.Table):
        if table.name.lower() != "sales" or table.db or table.catalog:
            return None, f"Unauthorized table: {table.sql()}"

    if any(
        isinstance(node, (exp.Subquery, exp.CTE, exp.Anonymous))
        for node in ast.walk()
    ):
        return None, "Subqueries, CTEs and custom functions are not allowed."

    return ast, None


def validate_sql(query: str) -> str | None:
    """Return None for accepted SQL, otherwise a validation error."""
    return _validated_sql(query)[1]


def json_value(value):
    """Normalize DataFrame cells, including DuckDB nested/masked arrays."""
    if value is None or value is pd.NA or value is pd.NaT or value is np.ma.masked:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, np.timedelta64):
        return None if np.isnat(value) else pd.Timedelta(value).isoformat()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    # Decimal and uncommon object values retain a JSON-safe text representation.
    return str(value)


def structured_result(frame: pd.DataFrame, sql: str) -> dict | None:
    """Serialize the fetched frame, keeping complete rows within the output cap."""
    result = {
        "sql": sql,
        "columns": [str(column) for column in frame.columns],
        "rows": [
            [json_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }
    while len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > MAX_OUTPUT_CHARS:
        if not result["rows"]:
            # Even the SQL/column metadata cannot fit the output budget.
            return None
        result["rows"].pop()
    return result


def latest_query_result(messages) -> dict:
    """Use the last SQL call in graph message order, including failed calls."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "execute_sql":
            if message.status != "error" and isinstance(message.artifact, dict):
                return message.artifact
            break
    return {"sql": None, "columns": [], "rows": []}


def create_tools(connection: duckdb.DuckDBPyConnection):
    """Build tools whose closures share only this dataset connection."""
    connection_lock = Lock()

    @tool
    def inspect_dataset() -> str:
        """Return the schema and row count of the sales dataset."""
        try:
            with connection_lock:
                schema = connection.execute("DESCRIBE sales").fetchdf()
                count = connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
            return limit_output(
                f"Schema:\n{schema.to_string(index=False)}\n\nTotal rows: {count}"
            )
        except Exception as error:
            return limit_output(f"Dataset Inspection Error: {error}")

    @tool(response_format="content_and_artifact")
    def execute_sql(query: str) -> tuple[str, dict | None]:
        """Execute a validated SQL SELECT query on the sales table."""
        ast, error = _validated_sql(query)
        if error:
            return limit_output(f"SQL Validation Error: {error}"), None

        try:
            safe_query = (
                exp.select("*")
                .from_(ast.subquery("limited_result"))
                .limit(MAX_RESULT_ROWS)
                .sql(dialect="duckdb", comments=False)
            )
            with connection_lock:
                result = connection.execute(safe_query).fetchdf()
            return (
                limit_output(result.to_string(index=False)),
                structured_result(result, safe_query),
            )
        except Exception as error:
            return limit_output(f"SQL Execution Error: {error}"), None

    return [inspect_dataset, execute_sql]


_model = None
_model_lock = Lock()


def get_models():
    """Lazily initialize the base LLM; tools are bound per request."""
    global _model
    with _model_lock:
        if _model is None:
            load_dotenv(PROJECT_DIR / ".env")
            _model = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0,
                timeout=30,
                max_retries=2,
            )
    return _model


def model_failure():
    """Do not expose provider exceptions, which may contain user data."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "Analysis could not be completed because the model service failed. "
                    "Check OPENAI_API_KEY, network access, and API availability."
                )
            )
        ]
    }


def after_tools(state: MessagesState):
    """Stop after the configured number of tool errors."""
    errors = sum(
        1
        for message in state["messages"]
        if isinstance(message, ToolMessage)
        and (
            message.status == "error"
            or str(message.content).startswith(
                (
                    "SQL Validation Error:",
                    "SQL Execution Error:",
                    "Dataset Inspection Error:",
                    "Tool Error:",
                )
            )
        )
    )
    return "failure" if errors >= MAX_SQL_ERRORS else "llm"


def failure_node(state: MessagesState):
    """Tell the final model call to stop using tools."""
    return {
        "messages": [
            HumanMessage(
                content=(
                    "The tool error limit has been reached. "
                    "Do not call any more tools. "
                    "Briefly explain why the analysis could not be completed."
                )
            )
        ]
    }


def handle_tool_error(error: Exception) -> str:
    """Hide unexpected exception details from the model."""
    return f"Tool Error: {type(error).__name__}. Check the tool arguments and retry."


def create_agent_graph(dataset_tools):
    """Compile a graph whose LLM and ToolNode use the same request tools."""
    model = get_models()
    model_with_tools = model.bind_tools(dataset_tools)

    def call_model(state: MessagesState):
        try:
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception:
            return model_failure()

    def final_failure(state: MessagesState):
        try:
            response = model.invoke(state["messages"])
            return {"messages": [response]}
        except Exception:
            return model_failure()

    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_model)
    graph.add_node(
        "tools", ToolNode(dataset_tools, handle_tool_errors=handle_tool_error)
    )
    graph.add_node("failure", failure_node)
    graph.add_node("final_failure", final_failure)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm", tools_condition, {"tools": "tools", "__end__": END}
    )
    graph.add_conditional_edges(
        "tools", after_tools, {"llm": "llm", "failure": "failure"}
    )
    graph.add_edge("failure", "final_failure")
    graph.add_edge("final_failure", END)
    return graph.compile()


def run_agent(
    question: str,
    *,
    dataset_path: Path | None = None,
    recursion_limit: int = 20,
):
    """Run one analysis with a dedicated connection, tools, and graph."""
    path = CSV_PATH if dataset_path is None else Path(dataset_path)
    connection = create_dataset_connection(path)
    try:
        dataset_tools = create_tools(connection)
        dataset_agent = create_agent_graph(dataset_tools)
        try:
            result = dataset_agent.invoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=question),
                    ]
                },
                config={"recursion_limit": recursion_limit},
            )
        except GraphRecursionError:
            result = {
                "messages": [
                    AIMessage(
                        content=(
                            "Analysis stopped because the workflow step limit was reached. "
                            "Try a simpler or more specific question."
                        )
                    )
                ]
            }
        return {**result, **latest_query_result(result["messages"])}
    finally:
        connection.close()


def run_dataset_agent(
    question: str, dataset_path: Path, *, recursion_limit: int = 20
):
    """Compatibility wrapper for the existing FastAPI /ask endpoint."""
    return run_agent(
        question, dataset_path=dataset_path, recursion_limit=recursion_limit
    )


if __name__ == "__main__":
    question = input("Ask a question about sales: ")
    result = run_agent(question)
    print("\n========== FINAL ANSWER ==========")
    print(result["messages"][-1].content)
