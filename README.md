# Data Analyst Agent

A CSV analysis agent built with LangGraph, OpenAI tool calling, DuckDB, and
SQLGlot. `agent.py` provides the Python interface and CLI; `api.py` provides a
local FastAPI service for uploading datasets and asking questions.

## What it does

- Answers natural-language questions about the included `sales.csv` or another CSV.
- Creates an independent in-memory database, tool set, and graph for each run.
- Builds two tools per dataset: `inspect_dataset()` and `execute_sql(query: str)`.
- Adds a state-injected `create_chart` tool to each graph for bar, line and scatter charts.
- Validates SQL before execution and generates the executed query from that AST.
- Returns at most 100 SQL result rows and 10,000 characters per tool response.
  Longer text includes a truncation notice; refine the query when this happens.
- Lets the model correct tool failures. After two cumulative tool errors in a
  run, it requests a final explanation using the model without bound tools.
- Returns readable messages for model invocation failures and workflow step
  exhaustion. Dataset loading, model initialization, and tool-binding failures
  propagate to the caller.

The included five-row CSV has `region`, `brand`, and `revenue` columns. Its total
revenue is 52,000; Toyota totals 36,500, Honda 15,500, Ontario 35,500, and Quebec
16,500. The application is a local prototype, not a hosted service.

## Setup

The verified environment is Windows with Python 3.14.2. Other Python versions
and operating systems have not been verified. Runtime dependencies are pinned
to versions tested in this environment; transitive dependencies are not fully
locked.

From the repository root, create and install into a virtual environment.
PowerShell commands (activation is optional):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development and tests, install `requirements-dev.txt` instead; it includes
the runtime requirements and Pytest:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For real agent runs, copy the example and edit the new `.env`:

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, set `OPENAI_API_KEY` in your shell. Existing environment variables
take precedence over `.env`. The file is loaded from the directory containing
`agent.py`, only when the LLM is first needed. Importing the module and running
SQL tools/tests do not require a key. Keep `.env` private; `.env.example` must
contain placeholders only.

The model is `gpt-4.1-mini` with temperature zero. Actual agent runs require
network access, API access to that model, and an account with available quota.
Prompts, conversation history, tool schemas, dataset schema, query results, and
tool error messages can be sent to OpenAI. Do not substitute sensitive data
without considering that data flow. Real runs and evaluation can incur API fees.

## Run

```powershell
.\.venv\Scripts\python.exe agent.py
```

Example questions:

- What is the total revenue?
- Which brand has the highest total revenue?
- What is the total revenue in Quebec?

`sales.csv` is resolved relative to `agent.py`, not the current working directory.
Keep the CSV beside that file for default CLI runs. Importing the module does
not load a dataset or initialize the model. Each `run_agent()` call loads its
CSV into a dedicated in-memory `sales` table and closes the connection on exit,
including when graph creation or execution fails.

Python usage:

```python
from pathlib import Path
from agent import run_agent, run_dataset_agent

result = run_agent("Which brand has the highest total revenue?")
print(result["messages"][-1].content)

# Select another CSV; each run gets its own database and tools.
result = run_agent("What is the total revenue?", dataset_path=Path("my_sales.csv"))
print(result["messages"][-1].content)

# Compatibility interface used by the API.
result = run_dataset_agent("How many records are there?", Path("my_sales.csv"))
```

`run_agent()` supplies the standard system prompt and catches LangGraph recursion
limits (20 steps by default, adjustable with `recursion_limit=`). The base model
is initialized lazily and cached; each graph binds that model to its own dataset
tools. There is no global compiled agent or global dataset connection.

To inspect or query a dataset without calling the model:

```python
from agent import CSV_PATH, create_dataset_connection, create_tools

connection = create_dataset_connection(CSV_PATH)
try:
    tools = {tool.name: tool for tool in create_tools(connection)}
    print(tools["inspect_dataset"].invoke({}))
    print(tools["execute_sql"].invoke({
        "query": "SELECT SUM(revenue) AS total FROM sales"
    }))
finally:
    connection.close()
```

Each tool set shares a lock spanning database execution and result fetching.
Calls using the same tool set are serialized; separate datasets have independent
connections and locks. Advanced callers can build a graph with
`create_agent_graph(dataset_tools)` and must manage their connection's lifetime,
initial messages, and graph exceptions themselves.

## Local API

API and Streamlit packages are included in `requirements.txt`. Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1
```

Open `http://127.0.0.1:8000/docs` to try the endpoints:

- `GET /health` returns `{"status": "ok"}`.
- `POST /upload` accepts a multipart `file` with a `.csv` filename, up to 5 MB.
  The CSV must parse and contain at least one data row. The response includes
  `dataset_id`, `filename`, `size`, `row_count`, and `columns`.
- `POST /ask` accepts a JSON question and the `dataset_id` returned by upload:

  ```json
  {"question": "What is the total revenue?", "dataset_id": "<dataset UUID>"}
  ```

  It returns the answer plus structured query results, for example:

  ```json
  {
    "answer": "Total revenue is 52000.",
    "sql": "SELECT * FROM (SELECT SUM(revenue) AS total FROM sales) AS limited_result LIMIT 100",
    "columns": ["total"],
    "rows": [[52000]],
    "result_truncated": false,
    "chart": null
  }
  ```

  Questions must contain 1–2,000 characters;
  a missing dataset returns 404. Actual analysis requires model credentials.

In a second terminal, start the chat interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`, upload `sales.csv`, click **Load Dataset**, and ask
"Which brand has the highest total revenue?". Check the answer, SQL block and
table, then ask another question to confirm previous tables remain in chat.
For an API-only check, upload `sales.csv` using `/docs`, copy its `dataset_id`
into `/ask`, and inspect the response fields. Ask for a filter that matches
no rows to verify that a successful empty table still displays its column names.

### Structured result flow and policy

The validated, row-limited query executes once under the existing connection
lock. The fetched DataFrame supplies both the bounded text for the model and a
JSON-safe artifact. LangChain's `content_and_artifact` tool format attaches the
artifact to its `ToolMessage`; the model still receives text. Ordinary
`execute_sql.invoke({"query": ...})` calls still return a string. Advanced callers
can invoke with a full tool call to access the message's `artifact`.

`run_agent()` retains `messages` and adds `sql`, `columns`, and `rows` from the
last `execute_sql` ToolMessage in graph message order. For parallel calls in a
single batch, the model's tool-call order determines the last result, regardless
of completion order. Inspection calls do not replace SQL results. A failed last
SQL call (including invalid arguments) clears the exposed result; a later
successful retry replaces it. No SQL call, or recursion-limit termination,
returns `sql: null`, `columns: []`, `rows: []`. A model-service failure after a
successful SQL call retains that query's result alongside the failure answer.
Every execution has its own connection, tools, graph messages and artifacts;
there is no shared result cache between requests or datasets.

FastAPI validates the additional fields with `QuestionResponse`. Streamlit
stores them with each assistant message and uses `st.code()` and `st.dataframe()`
for both new answers and chat history. Empty successful results retain columns
and display an empty table. Existing request formats and `answer` are unchanged;
clients must tolerate the additional response fields. Older answer-only
responses and chat messages still display normally.

`sql` is the actual SQL sent to DuckDB, including the outer `LIMIT 100`, rather
than the model's original SQL text. Structured output has its own 10,000-character
JSON budget for `sql`, `columns`, `rows`, and `result_truncated` (measured with Python `json.dumps`,
`ensure_ascii=False`); complete trailing rows are removed to fit. A large cell
can therefore leave an empty table. If SQL and column metadata alone exceed the
budget, no structured result is exposed. `result_truncated` is true when complete
rows were removed for the size budget or the fetched frame reaches 100 rows.
The latter is conservative: exactly 100 rows might also be the complete result.
No extra count query is executed. Streamlit warns that the result may be partial;
the table and chart use the same returned rows. The existing text limit and
truncation notice are preserved independently.

NumPy scalars become Python values, dates/times/timestamps and durations become
ISO strings, and missing values (including masked nested array values), NaN and
infinities become JSON `null`. Decimal objects and uncommon object values become
strings. DuckDB's `fetchdf()` may already convert DECIMAL values to floats; this
does not provide exact arbitrary-precision decimal transport. Large integers
remain JSON numbers and may exceed JavaScript's exact integer range. The caps
bound returned data, not query execution or serialization memory usage.

### Chart tool

Ask "Show total revenue by brand as a bar chart." The model should inspect the
dataset, execute an aggregate query, then call:

```python
create_chart(chart_type="bar", x="brand", y="total_revenue", title="Revenue by brand")
```

The tool accepts only `bar`, `line`, or `scatter`, two exact unambiguous column
names, and an optional title of at most 200 characters. It reads the most recent
completed SQL artifact through LangGraph `InjectedState`; state and data are not
model-supplied arguments. Ordinary questions do not require charts. Bar/line
require numeric Y values; scatter requires numeric X and Y values. Booleans do
not count as numeric. Bar/line X values must be scalar; ISO date strings work as
ordered category labels. Rows with a null axis are excluded only from the chart,
with a count in tool feedback and a frontend caption. Empty results, entirely
null axes, invalid columns/types and unsupported charts produce tool errors,
which count toward the existing two-error budget. Query tables remain intact.

SQL and charts must be called in separate model tool batches. If they appear in
one batch, chart creation returns a recoverable error, even if an earlier SQL
result exists. A later chart call can use the newly completed query. Any new SQL
call invalidates an earlier chart, including failed/invalid SQL calls. Multiple
chart calls use the last call in message order; a failed last chart call clears
the chart. Inspection calls do not invalidate charts. On recursion-limit
termination no chart is exposed. There is no mutable global query/chart cache.
`create_tools(connection)` still returns the two dataset tools; `create_agent_graph`
adds a separate instance of the state-injected chart tool to each graph.

`/ask` adds `chart` (null when absent/invalid) and `result_truncated` (false by
default). A chart references the same response's columns and rows; it never
duplicates data or executes another query:

```json
{
  "answer": "Toyota has the highest total revenue.",
  "sql": "SELECT * FROM (SELECT brand, SUM(revenue) AS total_revenue FROM sales GROUP BY brand ORDER BY total_revenue DESC) AS limited_result LIMIT 100",
  "columns": ["brand", "total_revenue"],
  "rows": [["Toyota", 36500], ["Honda", 15500]],
  "result_truncated": false,
  "chart": {"type": "bar", "x": "brand", "y": "total_revenue", "title": "Revenue by brand"}
}
```

Streamlit displays one chart after the answer, SQL and table, and retains it in
chat history. It uses its built-in `st.vega_lite_chart` with a fixed specification
and field aliases, preserving input order without aggregation, stacking or axis
sorting. SQL must do any necessary grouping and ordering; duplicate categories
can overlap. Charts only cover returned rows and do not fetch full data. Loading
a new dataset clears the existing chat and charts. Old responses/messages without
the new fields still render normally. No plotting dependency or generated code
execution was added. Multi-series charts, pie charts, dual axes and image export
are outside this version's scope.

Uploads are saved under `uploads/` with server-generated UUID filenames. This
directory is ignored by Git. Uploaded files persist on disk until removed;
the per-analysis in-memory connection is closed after each run. The API has no
authentication or automatic upload cleanup and is intended for local use.

## Tests and evaluation

Run all offline tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

If the existing `.pytest_tmp` belongs to another Windows account, use a fresh
temporary directory rather than changing its permissions:

```powershell
$testTemp = Join-Path $env:TEMP ("analyst-tests-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp $testTemp
```

Tests exercise SQL validation and execution, file-path resolution without API
credentials, concurrent database access, row/text limits, model failures,
tool-error recovery, recursion limits, dataset isolation, per-request tool binding,
connection cleanup, and evaluation matching. Structured-result tests also cover
JSON types, empty/multiple-column results, multiple and parallel SQL calls,
failed-call clearing, concurrent API requests across the same/different datasets,
upload/ask compatibility, and Streamlit display/history. Chart tests cover all
three types, validation, null filtering, SQL/chart batch dependencies, invalidation,
retries, tool error limits, hidden state injection, concurrent API requests,
truncation warnings and Streamlit chart data/order/history. Pytest collects `tests/` by default.
Model calls in
tests are mocked; tests do not validate real OpenAI connectivity or model quality.

Run the five-question live evaluation separately:

```powershell
.\.venv\Scripts\python.exe evaluation.py
```

Evaluation uses the same workflow as the CLI and asks for a final standalone
`FINAL_ANSWER: <value>` line. Numbers are compared by exact decimal value,
including valid thousands separators; categorical values are compared exactly
after case/whitespace normalization. Thus `15` does not match `5`, and prose
merely mentioning `Toyota` does not match `Toyota`. An answer consisting only of
the exact value is also accepted. Missing/invalid formatting can fail an otherwise
correct response. The evaluator checks the declared final value, not the reasoning
or tool-use quality. Failed cases do not abort remaining cases.

## SQL restrictions and security limits

The validator permits one top-level `SELECT`, restricts explicit table references
to unqualified `sales`, and rejects writes, CTEs, subqueries, `UNION`, and functions
that SQLGlot classifies as anonymous/custom. Recognized built-in functions are
not exhaustively allowlisted. Queries without table references can be accepted.
SQL comments are discarded during AST generation before applying the outer row
limit. This is a deliberately narrow query policy; not all valid DuckDB SQL is
supported.

**This is not a security sandbox for hostile SQL.** AST validation and tested
file-reader rejection do not prove that every DuckDB capability is inaccessible.
DuckDB external access is not disabled. The CSV is materialized into an in-memory
table during dataset loading.
There is no OS/process isolation, database query timeout, or explicit CPU/memory
budget. Large expressions, large individual values, or expensive joins can still
consume resources before the result is fetched and formatted. The character cap
limits the returned text; it does not bound query execution or formatting memory.
The graph step limit also cannot interrupt a single long-running SQL query.

Use only with trusted local data and inputs. Before exposing it to untrusted
users, add database capability restrictions, query cancellation, resource limits,
and process isolation. Any external-access restrictions must still allow the
initial CSV load.

The instructions to inspect data first, avoid repeated failed SQL, and avoid
invented answers are model prompts rather than enforced state-machine guarantees.
The error budget is cumulative across the messages in a run, including errors
separated by successful tool calls; it is checked after a tool batch finishes.
Model calls use a 30-second request timeout and up to two client retries; these
are not an overall analysis deadline. SQL/inspection error text may reveal local
paths or schema details. Model invocation errors are replaced with a generic
message; initialization and tool-binding exceptions propagate to the caller.

## Project files and Git hygiene

```text
agent.py                 Tools, SQL validation, lazy LLM setup, graph, CLI
api.py                   Local FastAPI upload and question endpoints
evaluation.py            Live evaluation and exact-answer matching
sales.csv                Included five-row dataset
requirements.txt         Pinned direct runtime dependencies
requirements-dev.txt     Runtime requirements plus Pytest
.env.example             Credential placeholder
.gitignore               Secrets, environments, caches, databases, generated files
tests/                   Offline tests
uploads/                 Local uploaded datasets (ignored by Git)
```

Before the first commit, review the staged file list. Commit source, tests,
documentation, dependency files, the example environment file, and the intended
sample CSV. Exclude `.env`, virtual environments, caches, logs, and generated
databases. Ignore rules do not remove files that Git already tracks. If a real
key has ever been published, revoke/rotate it; adding an ignore rule is insufficient.

No license has been selected in this repository. Choose an appropriate license
and confirm permission to redistribute the dataset before inviting public reuse.
