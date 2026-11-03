import pytest


def test_total_revenue(execute_sql):
    result = execute_sql.invoke({
        "query": "SELECT SUM(revenue) AS total FROM sales"
    })
    assert float(result.splitlines()[1]) == pytest.approx(7139074.56, abs=0.001, rel=0)


def test_group_by_product(execute_sql):
    result = execute_sql.invoke({
        "query": """
            SELECT product, ROUND(SUM(revenue), 2) AS total
            FROM sales
            GROUP BY product
            ORDER BY total DESC
        """
    })
    expected = [
        ("Aurora Lamp", 1469425.71), ("Pebble Speaker", 1244861.83),
        ("Tide Mug", 1002624.54), ("Cedar Shelf", 968397.46),
        ("Meadow Chair", 760067.57), ("Linen Throw", 715744.29),
        ("Glow Bulb", 493853.17), ("Drift Clock", 484099.99),
    ]
    rows = result.splitlines()[1:]
    assert len(rows) == len(expected)
    for row, (product, total) in zip(rows, expected):
        name, value = row.rsplit(maxsplit=1)
        assert name.strip() == product
        assert float(value) == pytest.approx(total, abs=0.001, rel=0)


def test_current_sales_schema_and_row_count(dataset_connection):
    columns = dataset_connection.execute("DESCRIBE sales").fetchall()
    assert [column[0] for column in columns] == [
        "date", "region", "product", "channel", "units", "unit_price", "revenue",
    ]
    assert dataset_connection.execute("SELECT COUNT(*) FROM sales").fetchone() == (10000,)


def test_invalid_column(execute_sql):
    result = execute_sql.invoke({
        "query": "SELECT price FROM sales"
    })
    assert result.startswith("SQL Execution Error:")


def test_unauthorized_table(execute_sql):
    result = execute_sql.invoke({
        "query": "SELECT * FROM customers"
    })
    assert result.startswith("SQL Validation Error:")
