"""Dataset-isolated SQL analysis agent using DuckDB and LangGraph."""

import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlglot import exp, parse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import Field
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition

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
    "When the user asks for a chart, use create_chart after execute_sql completes. "
    "Never call execute_sql and create_chart in the same tool-call batch. "
    "Charts use only the latest returned SQL rows; aggregate and order in SQL first. "
    "Use bar for category comparisons, line for ordered trends, scatter for numeric relationships. "
    "Do not create charts unless useful or requested. "
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
        "result_truncated": len(frame) >= MAX_RESULT_ROWS,
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
        result["result_truncated"] = True
    return result


def latest_query_result(messages) -> dict:
    """Use the last SQL call in graph message order, including failed calls."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "execute_sql":
            if message.status != "error" and isinstance(message.artifact, dict):
                return message.artifact
            break
    return {"sql": None, "columns": [], "rows": []}


def latest_chart(messages) -> dict | None:
    """A new SQL call invalidates any chart; the last chart call wins."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if message.name == "execute_sql":
            return None
        if message.name == "create_chart":
            return message.artifact if message.status != "error" else None
    return None


@tool(response_format="content_and_artifact")
def create_chart(
    chart_type: Literal["bar", "line", "scatter"],
    x: str,
    y: str,
    state: Annotated[dict, InjectedState],
    title: Annotated[str | None, Field(max_length=200)] = None,
) -> tuple[str, dict | None]:
    """Create one chart from the latest completed SQL result. Run SQL separately first.

    Use exact column names. Bar/line require numeric y; scatter requires numeric
    x and y. Rows with null axes are excluded. SQL must do aggregation/ordering.
    """
    messages = state["messages"]
    batch = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if batch and any(call["name"] == "execute_sql" for call in batch.tool_calls):
        return "Chart Error: Run execute_sql in a separate batch before create_chart.", None
    result = latest_query_result(messages)
    if result["sql"] is None:
        return "Chart Error: No valid SQL result. Execute a query first.", None
    columns = result["columns"]
    if columns.count(x) != 1 or columns.count(y) != 1:
        return "Chart Error: Axis names must each identify exactly one result column.", None
    xi, yi = columns.index(x), columns.index(y)
    rows = [row for row in result["rows"] if row[xi] is not None and row[yi] is not None]
    if not rows:
        return "Chart Error: No plottable rows; both axes need non-null values.", None
    if any(type(row[yi]) not in (int, float) for row in rows):
        return "Chart Error: The y axis must contain numeric values, excluding booleans.", None
    if chart_type == "scatter":
        if any(type(row[xi]) not in (int, float) for row in rows):
            return "Chart Error: Scatter x must contain numeric values, excluding booleans.", None
    elif any(not isinstance(row[xi], (str, int, float, bool)) for row in rows):
        return "Chart Error: The x axis must contain scalar values.", None
    chart = {"type": chart_type, "x": x, "y": y, "title": title}
    skipped = len(result["rows"]) - len(rows)
    notice = " Result may be partial because output limits were reached." if result.get("result_truncated") else ""
    return f"Created {chart_type} chart with {len(rows)} rows; excluded {skipped} null-axis rows.{notice}", chart


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
                    "Chart Error:",
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
    dataset_tools = [*dataset_tools, create_chart.model_copy()]
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
        query = latest_query_result(result["messages"])
        return {
            **result, **query,
            "result_truncated": query.get("result_truncated", False),
            "chart": latest_chart(result["messages"]),
        }
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
