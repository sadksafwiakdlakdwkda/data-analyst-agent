from pathlib import Path
from threading import Lock

import duckdb
from dotenv import load_dotenv
from sqlglot import parse, exp

from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError


# ==========================================
# 1. Project paths and output limits
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "sales.csv"
MAX_OUTPUT_CHARS = 10_000
MAX_RESULT_ROWS = 100


def limit_output(text: str) -> str:
    """Bound tool response length, including the truncation notice."""
    notice = "\n[Output truncated; refine the query.]"
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS - len(notice)] + notice


# ==========================================
# 2. Initialize DuckDB
# ==========================================

conn = duckdb.connect(":memory:")
db_lock = Lock()

with db_lock:
    conn.read_csv(str(CSV_PATH)).create_view("sales")


# ==========================================
# 3. SQL Safety Validation
# ==========================================

def _validated_sql(query: str) -> tuple[exp.Select | None, str | None]:
    """Return the validated AST, so execution uses exactly what was checked."""

    try:
        # SQLGlot may represent a comment after ';' as an empty Semicolon node.
        statements = [
            s for s in parse(query, read="duckdb")
            if s is not None and not isinstance(s, exp.Semicolon)
        ]

    except Exception as e:
        return None, f"Invalid SQL syntax: {e}"

    # Allow only one statement
    if len(statements) != 1:
        return None, "Only one SQL statement is allowed."

    ast = statements[0]

    # Only allow SELECT
    if not isinstance(ast, exp.Select):
        return None, "Only SELECT queries are allowed."

    # Block dangerous operations
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Command,
    )

    if any(
        isinstance(node, forbidden)
        for node in ast.walk()
    ):
        return None, "SQL contains a forbidden operation."

    # Restrict accessible tables
    for table in ast.find_all(exp.Table):
        if (
            table.name.lower() != "sales"
            or table.db
            or table.catalog
        ):
            return None, f"Unauthorized table: {table.sql()}"

    # Conservative restriction for this MVP:
    # disallow subqueries, CTEs and custom/unknown functions
    if any(
        isinstance(node, (exp.Subquery, exp.CTE, exp.Anonymous))
        for node in ast.walk()
    ):
        return None, "Subqueries, CTEs and custom functions are not allowed."

    return ast, None


def validate_sql(query: str) -> str | None:
    """Validate SQL before execution; return None when accepted."""
    return _validated_sql(query)[1]


# ==========================================
# 4. Define Tools
# ==========================================

@tool
def inspect_dataset() -> str:
    """Return the schema and row count of the sales dataset."""

    try:
        with db_lock:
            schema = conn.execute("DESCRIBE sales").fetchdf()
            count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        return limit_output(
            f"Schema:\n{schema.to_string(index=False)}\n\nTotal rows: {count}"
        )
    except Exception as e:
        return limit_output(f"Dataset Inspection Error: {e}")


@tool
def execute_sql(query: str) -> str:
    """Execute a validated SQL SELECT query on the sales table."""

    # Validate before execution
    ast, error = _validated_sql(query)

    if error:
        return limit_output(f"SQL Validation Error: {error}")

    try:
        # Restrict output size at the SQL level
        safe_query = (
            exp.select("*")
            .from_(ast.subquery("limited_result"))
            .limit(MAX_RESULT_ROWS)
            .sql(dialect="duckdb", comments=False)
        )
        with db_lock:
            result = conn.execute(safe_query).fetchdf()
        return limit_output(result.to_string(index=False))

    except Exception as e:
        return limit_output(f"SQL Execution Error: {e}")


# ==========================================
# 5. Lazy LLM initialization
# ==========================================

tools = [inspect_dataset, execute_sql]

llm = None
llm_with_tools = None
llm_lock = Lock()


def get_models():
    """Initialize credentials and models only when analysis needs the LLM."""
    global llm, llm_with_tools
    with llm_lock:
        if llm is None or llm_with_tools is None:
            load_dotenv(PROJECT_DIR / ".env")
            model = ChatOpenAI(
                model="gpt-4.1-mini", temperature=0, timeout=30, max_retries=2,
            )
            bound_model = model.bind_tools(tools)
            llm, llm_with_tools = model, bound_model
    return llm, llm_with_tools


def model_failure():
    # Provider exception text can contain request data; do not echo it.
    return {"messages": [AIMessage(content=(
        "Analysis could not be completed because the model service failed. "
        "Check OPENAI_API_KEY, network access, and API availability."
    ))]}


# ==========================================
# 6. Define LangGraph Nodes
# ==========================================

MAX_SQL_ERRORS = 2


def call_model(state: MessagesState):
    """Call the LLM with conversation history."""

    try:
        _, model = get_models()
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    except Exception:
        return model_failure()


def count_sql_errors(state: MessagesState) -> int:
    """Count SQL errors in tool responses."""

    return sum(
        1
        for msg in state["messages"]
        if isinstance(msg, ToolMessage)
        and (
            str(msg.content).startswith(
                "SQL Validation Error:"
            )
            or str(msg.content).startswith(
                "SQL Execution Error:"
            )
        )
    )


def after_tools(state: MessagesState):
    """Decide whether to retry or stop."""

    # Include inspection and ToolNode argument errors in the same retry budget.
    errors = sum(
        1 for msg in state["messages"]
        if isinstance(msg, ToolMessage) and (
            msg.status == "error" or str(msg.content).startswith((
                "SQL Validation Error:", "SQL Execution Error:",
                "Dataset Inspection Error:", "Tool Error:",
            ))
        )
    )

    if errors >= MAX_SQL_ERRORS:
        return "failure"

    return "llm"


def failure_node(state: MessagesState):
    """Return a clear failure message."""

    return {
        "messages": [
            HumanMessage(
                content=(
                    "The tool error limit has been reached. "
                    "Do not call any more tools. "
                    "Briefly explain why the analysis could "
                    "not be completed."
                )
            )
        ]
    }


def final_failure(state: MessagesState):
    """Generate a final explanation without tool calls."""

    try:
        model, _ = get_models()
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    except Exception:
        return model_failure()


def handle_tool_error(error: Exception) -> str:
    """Keep unexpected tool errors bounded and avoid exposing exception data."""
    return f"Tool Error: {type(error).__name__}. Check the tool arguments and retry."


# ==========================================
# 7. Build LangGraph
# ==========================================

graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("llm", call_model)
graph.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_error))
graph.add_node("failure", failure_node)
graph.add_node("final_failure", final_failure)

# Start
graph.add_edge(START, "llm")

# LLM -> Tools or END
graph.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

# Tools -> LLM or Failure
graph.add_conditional_edges(
    "tools",
    after_tools,
    {
        "llm": "llm",
        "failure": "failure",
    },
)

# Failure -> Final explanation -> END
graph.add_edge("failure", "final_failure")
graph.add_edge("final_failure", END)

# Compile
agent = graph.compile()


# ==========================================
# 8. Run Agent
# ==========================================

def run_agent(question: str, *, recursion_limit: int = 20):
    """Run the shared CLI/evaluation workflow with a graceful loop limit."""
    try:
        return agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are a reliable data analysis assistant. "
                            "Always inspect the dataset before generating SQL. "
                            "Only query the sales table. "
                            "Use execute_sql for calculations. "
                            "If SQL fails, inspect the error message "
                            "and correct the query before retrying. "
                            "Never repeat the exact same failed SQL. "
                            "Never invent query results. "
                            "If analysis fails, explain the problem."
                        )
                    ),
                    HumanMessage(content=question),
                ]
            },
            config={"recursion_limit": recursion_limit},
        )
    except GraphRecursionError:
        return {"messages": [AIMessage(content=(
            "Analysis stopped because the workflow step limit was reached. "
            "Try a simpler or more specific question."
        ))]}


if __name__ == "__main__":
    question = input("Ask a question about sales: ")
    result = run_agent(question)

    print("\n========== FINAL ANSWER ==========")
    print(result["messages"][-1].content)
