from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from uuid import uuid4, UUID
from typing import Any
from agent import run_dataset_agent
import duckdb


app = FastAPI(
    title="Data Analyst Agent",
    version="0.1.0",
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def validate_csv(file_path: Path):
    conn = duckdb.connect(":memory:")

    try:
        conn.execute(
            "CREATE TABLE sales AS SELECT * FROM read_csv_auto(?)",
            [str(file_path)]
        )

        columns = [
            row[0]
            for row in conn.execute("DESCRIBE sales").fetchall()
        ]

        row_count = conn.execute(
            "SELECT COUNT(*) FROM sales"
        ).fetchone()[0]

        if row_count == 0:
            raise ValueError("CSV must contain at least one data row")

        return {
            "row_count": row_count,
            "columns": columns
        }

    finally:
        conn.close()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    dataset_id: UUID


class QuestionResponse(BaseModel):
    answer: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    dataset_path = UPLOAD_DIR / f"{request.dataset_id}.csv"

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    result = run_dataset_agent(
        question=request.question,
        dataset_path=dataset_path
    )

    answer = str(result["messages"][-1].content)

    return QuestionResponse(
        answer=answer,
        sql=result.get("sql"),
        columns=result.get("columns", []),
        rows=result.get("rows", []),
    )
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    max_size = 5 * 1024 * 1024
    content = await file.read(max_size + 1)

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail="File is too large (max 5 MB)"
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    # Generate unique dataset ID
    dataset_id = str(uuid4())

    # Save CSV using a server-generated filename
    file_path = UPLOAD_DIR / f"{dataset_id}.csv"
    file_path.write_bytes(content)

    try:
        dataset_info = validate_csv(file_path)

    except (duckdb.Error, ValueError):
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file or no data rows"
        )

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "size": len(content),
        "row_count": dataset_info["row_count"],
        "columns": dataset_info["columns"]
    }
