from agent import execute_sql


def test_total_revenue():
    result = execute_sql.invoke({
        "query": "SELECT SUM(revenue) AS total FROM sales"
    })
    assert "52000" in result


def test_group_by_brand():
    result = execute_sql.invoke({
        "query": """
            SELECT brand, SUM(revenue) AS total
            FROM sales
            GROUP BY brand
        """
    })
    assert "Toyota" in result
    assert "36500" in result
    assert "Honda" in result
    assert "15500" in result


def test_invalid_column():
    result = execute_sql.invoke({
        "query": "SELECT price FROM sales"
    })
    assert result.startswith("SQL Execution Error:")


def test_unauthorized_table():
    result = execute_sql.invoke({
        "query": "SELECT * FROM customers"
    })
    assert result.startswith("SQL Validation Error:")