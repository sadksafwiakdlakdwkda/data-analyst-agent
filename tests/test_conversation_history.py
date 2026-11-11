"""Complete-turn persistence, follow-up context and turn-local execution state."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock
from uuid import uuid4
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict

import agent
import api
import conversations as store


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def sql_turn():
    artifact = {"sql": "SELECT 7 AS n", "columns": ["n"], "rows": [[7]], "result_truncated": False}
    return [HumanMessage(content="previous question"),
            tool_call("execute_sql", {"query": "SELECT 7 AS n"}, "old-sql"),
            ToolMessage(content="7", name="execute_sql", tool_call_id="old-sql", artifact=artifact),
            tool_call("create_chart", {"chart_type": "bar", "x": "n", "y": "n"}, "old-chart"),
            ToolMessage(content="chart", name="create_chart", tool_call_id="old-chart",
                        artifact={"type": "bar", "x": "n", "y": "n"}),
            AIMessage(content="previous answer")]


@pytest.fixture
def stored(tmp_path):
    path = tmp_path / "conversations.db"
    conversation = store.resolve_conversation(path, uuid4(), None)
    return path, conversation


def set_model(monkeypatch, responses):
    model, bound = Mock(), Mock()
    model.bind_tools.return_value = bound
    bound.invoke.side_effect = responses
    monkeypatch.setattr(agent, "get_models", lambda: model)
    return model, bound


def test_history_roundtrip_preserves_tools_artifacts_ids_and_order(stored):
    path, conversation = stored
    turn = sql_turn()
    store.save_turn(path, conversation, 0, [SystemMessage(content="transient"), *turn])
    loaded, revision = store.load_history(path, conversation)
    assert revision > 0
    assert messages_to_dict(loaded) == messages_to_dict(turn)
    # Reopening the database uses no process-local message cache.
    assert store.load_history(path, conversation)[0][2].artifact["rows"] == [[7]]


def test_turn_retention_and_character_budget_do_not_split_tool_chains(stored, monkeypatch):
    path, conversation = stored
    for index in range(store.MAX_HISTORY_TURNS + 2):
        _, revision = store.load_history(path, conversation)
        store.save_turn(path, conversation, revision, [HumanMessage(content=f"q{index}"), AIMessage(content="done")])
    history, _ = store.load_history(path, conversation)
    assert len(history) == 2 * store.MAX_HISTORY_TURNS
    assert history[0].content == "q2"
    last_turn_chars = len(json.dumps(messages_to_dict(history[-2:]), ensure_ascii=False))
    monkeypatch.setattr(store, "MAX_HISTORY_CHARS", last_turn_chars)
    assert len(store.load_history(path, conversation)[0]) == 2
    monkeypatch.setattr(store, "MAX_HISTORY_CHARS", 1)
    assert store.load_history(path, conversation)[0] == []


def test_stale_revision_cannot_overwrite_a_completed_turn(stored):
    path, conversation = stored
    store.save_turn(path, conversation, 0, [HumanMessage(content="first"), AIMessage(content="answer")])
    with pytest.raises(store.ConversationConflict):
        store.save_turn(path, conversation, 0, [HumanMessage(content="stale")])
    assert store.load_history(path, conversation)[0][0].content == "first"


def test_old_results_are_context_but_not_returned_as_current_results(monkeypatch):
    _, bound = set_model(monkeypatch, [AIMessage(content="Please clarify")])
    history = sql_turn()
    result = agent.run_agent("Which channel?", history=history)
    received = bound.invoke.call_args.args[0]
    assert any(m.content == "previous question" for m in received)
    assert any(isinstance(m, ToolMessage) and m.artifact for m in received)
    assert received[-1].content == "Which channel?"
    assert result["sql"] is None and result["rows"] == [] and result["chart"] is None
    assert [m.content for m in result["turn_messages"]] == ["Which channel?", "Please clarify"]


def test_old_errors_do_not_consume_new_turn_error_budget(monkeypatch):
    history = [HumanMessage(content="old failure")]
    for index in range(2):
        history.extend([tool_call("execute_sql", {"query": "bad"}, f"old-{index}"),
                        ToolMessage(content="SQL Execution Error: bad", name="execute_sql", tool_call_id=f"old-{index}")])
    history.append(AIMessage(content="Unable to finish"))
    model, bound = set_model(monkeypatch, [
        tool_call("execute_sql", {"query": "SELECT missing FROM sales"}, "new-bad"),
        tool_call("execute_sql", {"query": "SELECT COUNT(*) AS n FROM sales"}, "new-good"),
        AIMessage(content="10000"),
    ])
    result = agent.run_agent("Try again", history=history)
    assert result["rows"] == [[10000]]
    assert bound.invoke.call_count == 3
    model.invoke.assert_not_called()


def test_chart_cannot_reuse_previous_turn_sql(monkeypatch):
    set_model(monkeypatch, [tool_call("create_chart", {"chart_type": "bar", "x": "n", "y": "n"}, "new-chart"),
                            AIMessage(content="Need a fresh query")])
    result = agent.run_agent("Draw it", history=sql_turn())
    tool = next(m for m in result["turn_messages"] if isinstance(m, ToolMessage))
    assert "No valid SQL result" in tool.content
    assert result["chart"] is None


def test_new_turn_still_stops_after_its_own_two_errors(monkeypatch):
    model, bound = set_model(monkeypatch, [
        tool_call("execute_sql", {"query": "SELECT missing FROM sales"}, "bad1"),
        tool_call("execute_sql", {"query": "SELECT other_missing FROM sales"}, "bad2"),
    ])
    model.invoke.return_value = AIMessage(content="Unable to finish")
    result = agent.run_agent("bad queries", history=sql_turn())
    assert bound.invoke.call_count == 2
    model.invoke.assert_called_once()
    assert result["messages"][-1].content == "Unable to finish"


def test_recursion_failure_keeps_complete_history_and_no_orphan_tools(monkeypatch):
    _, bound = set_model(monkeypatch, [])
    bound.invoke.side_effect = lambda _: tool_call("inspect_dataset", {}, str(uuid4()))
    result = agent.run_agent("loop", history=sql_turn(), recursion_limit=4)
    assert "workflow step limit" in result["messages"][-1].content
    assert len(result["turn_messages"]) == 2
    assert isinstance(result["turn_messages"][0], HumanMessage)
    assert not result["turn_messages"][-1].tool_calls
    assert result["sql"] is None and result["chart"] is None


def test_api_replays_context_after_reopening_and_isolates_new_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    seen = []

    class Bound:
        def invoke(self, messages):
            seen.append(list(messages))
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="Analysis complete")
            if messages[-1].content == "North revenue?":
                return tool_call("execute_sql", {"query": "SELECT SUM(revenue) AS total FROM sales WHERE region = 'North'"}, str(uuid4()))
            if messages[-1].content == "Only Online?":
                assert any(m.content == "North revenue?" for m in messages)
                assert any(isinstance(m, ToolMessage) and m.artifact for m in messages)
                return tool_call("execute_sql", {"query": "SELECT SUM(revenue) AS total FROM sales WHERE region = 'North' AND channel = 'Online'"}, str(uuid4()))
            return AIMessage(content="Follow-up received")

    model = Mock()
    model.bind_tools.return_value = Bound()
    monkeypatch.setattr(agent, "get_models", lambda: model)
    with TestClient(api.app) as client:
        dataset = client.post("/upload", files={"file": ("data.csv", "region,channel,revenue\nNorth,Online,1\nNorth,Retail,2\nSouth,Online,4\n", "text/csv")}).json()["dataset_id"]
        first = client.post("/ask", json={"question": "North revenue?", "dataset_id": dataset})
        assert first.status_code == 200
        assert first.json()["rows"] == [[3]]
        conversation = first.json()["conversation_id"]
    # A new API client still retrieves the SQLite-backed tool chain.
    with TestClient(api.app) as client:
        next_response = client.post("/ask", json={"question": "Only Online?", "dataset_id": dataset,
                                                  "conversation_id": conversation})
        assert next_response.status_code == 200
        assert next_response.json()["rows"] == [[1]]
        assert "Online" in next_response.json()["sql"]
        assert any(m.content == "North revenue?" for m in seen[-1])
        assert any(isinstance(m, ToolMessage) and m.artifact for m in seen[-1])
        clarification = client.post("/ask", json={"question": "What do you need?", "dataset_id": dataset,
                                                  "conversation_id": conversation})
        assert clarification.json()["sql"] is None
        client.post("/ask", json={"question": "New conversation", "dataset_id": dataset})
        assert not any(m.content == "North revenue?" for m in seen[-1])


def test_same_conversation_serializes_requests_and_different_conversations_continue(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    blocked, release = Event(), Event()
    seen = {}

    def run(question, dataset_path, history):
        seen[question] = [m.content for m in history]
        if question == "slow":
            blocked.set()
            assert release.wait(10)
        return {"messages": [AIMessage(content=question + " answer")]}

    monkeypatch.setattr(api, "run_dataset_agent", run)
    with TestClient(api.app) as client:
        dataset = client.post("/upload", files={"file": ("data.csv", "revenue\n1\n", "text/csv")}).json()["dataset_id"]
        conversation = client.post("/ask", json={"question": "start", "dataset_id": dataset}).json()["conversation_id"]
        def ask(question, cid):
            return client.post("/ask", json={"question": question, "dataset_id": dataset, "conversation_id": cid})
        with ThreadPoolExecutor(max_workers=3) as pool:
            slow = pool.submit(ask, "slow", conversation)
            assert blocked.wait(10)
            follow = pool.submit(ask, "follow", conversation)
            other = pool.submit(ask, "other", None)
            try:
                assert other.result(timeout=10).status_code == 200
                assert "follow" not in seen
            finally:
                release.set()
            assert slow.result(timeout=10).status_code == follow.result(timeout=10).status_code == 200
        assert "slow answer" in seen["follow"]
        assert seen["other"] == []
    assert not store._locks


def test_uncaught_model_setup_error_does_not_save_partial_turn(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(api, "run_dataset_agent", Mock(side_effect=RuntimeError("setup failed")))
    with TestClient(api.app, raise_server_exceptions=False) as client:
        dataset = client.post("/upload", files={"file": ("data.csv", "revenue\n1\n", "text/csv")}).json()["dataset_id"]
        # Bind a separate ID to the actual uploaded dataset.
        from uuid import UUID
        cid = store.resolve_conversation(tmp_path / "conversations.db", UUID(dataset), None)
        response = client.post("/ask", json={"question": "broken", "dataset_id": dataset, "conversation_id": str(cid)})
        assert response.status_code == 500
        assert store.load_history(tmp_path / "conversations.db", cid) == ([], 0)
    assert not store._locks
