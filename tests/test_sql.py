from agent import validate_sql


def test_valid_select():
    query = "SELECT * FROM sales"
    assert validate_sql(query) is None


def test_reject_delete():
    query = "DELETE FROM sales"
    assert validate_sql(query) is not None


def test_reject_unknown_table():
    query = "SELECT * FROM customers"
    assert validate_sql(query) is not None


def test_reject_multiple_statements():
    query = "SELECT * FROM sales; DROP TABLE sales"
    assert validate_sql(query) is not None


def test_reject_file_access():
    query = "SELECT * FROM read_csv('/private/data.csv')"
    assert validate_sql(query) is not None