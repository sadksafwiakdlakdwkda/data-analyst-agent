import pytest

import agent as app


@pytest.fixture
def dataset_connection():
    connection = app.create_dataset_connection(app.CSV_PATH)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def dataset_tools(dataset_connection):
    return {tool.name: tool for tool in app.create_tools(dataset_connection)}


@pytest.fixture
def execute_sql(dataset_tools):
    return dataset_tools["execute_sql"]
