# Data Analyst Agent

A small command-line sales analysis agent built with LangGraph, OpenAI tool
calling, DuckDB, and SQLGlot. The agent code stays in `agent.py`.

## What it does

- Answers natural-language questions about the included `sales.csv` dataset.
- Exposes two tools: `inspect_dataset()` and `execute_sql(query: str)`.
- Validates SQL before execution and generates the executed query from that AST.
- Returns at most 100 SQL result rows and 10,000 characters per tool response.
  Longer text includes a truncation notice; refine the query when this happens.
- Lets the model correct tool failures. After two cumulative tool errors in a
  run, it requests a final explanation using the model without bound tools.
- Handles model initialization/API failures and workflow step exhaustion with
  a readable final message through `run_agent()`.

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
Keep the CSV beside that file; the module initializes its in-memory DuckDB view
when imported. Dataset loading is still required at import time.

Python usage:

```python
from agent import execute_sql, inspect_dataset, run_agent

print(inspect_dataset.invoke({}))
print(execute_sql.invoke({"query": "SELECT SUM(revenue) AS total FROM sales"}))

result = run_agent("Which brand has the highest total revenue?")
print(result["messages"][-1].content)
```

`run_agent()` supplies the standard system prompt and catches LangGraph recursion
limits (20 steps by default, adjustable with `recursion_limit=`). The compiled
`agent` remains available for advanced use; direct `agent.invoke()` callers must
provide their own prompt and handle recursion exceptions.

The shared DuckDB connection is protected by a lock spanning each entire tool's
database work, including result fetching. Database operations are serialized,
even when LangGraph dispatches tool calls in parallel.

## Tests and evaluation

Run all offline tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

Tests exercise SQL validation and execution, file-path resolution without API
credentials, concurrent database access, row/text limits, model failures,
tool-error recovery, recursion limits, and evaluation matching. Model calls in
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
DuckDB external access is not disabled, and the CSV remains a file-backed view.
There is no OS/process isolation, database query timeout, or explicit CPU/memory
budget. Large expressions, large individual values, or expensive joins can still
consume resources before the result is fetched and formatted. The character cap
limits the returned text; it does not bound query execution or formatting memory.
The graph step limit also cannot interrupt a single long-running SQL query.

Use only with trusted local data and inputs. Before exposing it to untrusted
users, add database capability restrictions, query cancellation, resource limits,
and process isolation. Disabling external access would require changing the
file-backed view to a materialized table or deliberately allowing the CSV path.

The instructions to inspect data first, avoid repeated failed SQL, and avoid
invented answers are model prompts rather than enforced state-machine guarantees.
The error budget is cumulative across the messages in a run, including errors
separated by successful tool calls; it is checked after a tool batch finishes.
Model calls use a 30-second request timeout and up to two client retries; these
are not an overall analysis deadline. SQL/inspection error text may reveal local
paths or schema details. Provider exception bodies are not echoed to the user.

## Project files and Git hygiene

```text
agent.py                 Tools, SQL validation, lazy LLM setup, graph, CLI
evaluation.py            Live evaluation and exact-answer matching
sales.csv                Included five-row dataset
requirements.txt         Pinned direct runtime dependencies
requirements-dev.txt     Runtime requirements plus Pytest
.env.example             Credential placeholder
.gitignore               Secrets, environments, caches, databases, generated files
tests/                   Offline tests
```

Before the first commit, review the staged file list. Commit source, tests,
documentation, dependency files, the example environment file, and the intended
sample CSV. Exclude `.env`, virtual environments, caches, logs, and generated
databases. Ignore rules do not remove files that Git already tracks. If a real
key has ever been published, revoke/rotate it; adding an ignore rule is insufficient.

No license has been selected in this repository. Choose an appropriate license
and confirm permission to redistribute the dataset before inviting public reuse.
